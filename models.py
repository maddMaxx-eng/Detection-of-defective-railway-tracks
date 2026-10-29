"""Model construction, freezing/unfreezing helpers and checkpoint I/O.

Both architectures come from torchvision and share the same structure
(``features`` -> global average pool -> ``classifier``), so they receive an
identical custom classification head and identical training code.

Checkpoint format (``models/<model_id>_best.pt``) is a plain dict that contains
the weights AND everything inference needs (class names, image size, mean/std),
which guarantees that inference uses the training-time class mapping.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models as tvm

from . import config

ARCH_INFO = {
    "mobilenet_v2": {"display": "MobileNetV2", "weights": "MobileNet_V2_Weights.IMAGENET1K_V1"},
    "efficientnet_b0": {"display": "EfficientNetB0", "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1"},
}


def _head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(256, num_classes),
    )


def build_model(
    model_id: str,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create the network with ImageNet weights and a fresh 5-class head."""
    if model_id == "mobilenet_v2":
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.mobilenet_v2(weights=weights)
        in_features = net.classifier[1].in_features
    elif model_id == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
    else:
        raise ValueError(f"Unknown model '{model_id}'. Choose from {config.MODEL_IDS}.")
    net.classifier = _head(in_features, num_classes, dropout)
    return net


def head_parameters(net: nn.Module):
    return net.classifier.parameters()


def set_backbone_trainable(net: nn.Module, trainable: bool, from_block: int = 0) -> int:
    """Freeze/unfreeze ``net.features[from_block:]``; earlier blocks stay frozen.

    Returns the number of trainable backbone parameters.
    """
    for p in net.features.parameters():
        p.requires_grad = False
    if trainable:
        for block in list(net.features.children())[from_block:]:
            for p in block.parameters():
                p.requires_grad = True
    return sum(p.numel() for p in net.features.parameters() if p.requires_grad)


def set_bn_eval(net: nn.Module) -> None:
    """Keep BatchNorm running statistics fixed (stable for small datasets/batches)."""
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def count_parameters(net: nn.Module) -> dict:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_checkpoint(net: nn.Module, path: Path, model_id: str, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def plain(v):  # keep checkpoints loadable with torch.load(weights_only=True)
        if isinstance(v, (bool, int, float, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [plain(i) for i in v]
        if isinstance(v, dict):
            return {str(k): plain(i) for k, i in v.items()}
        return str(v)

    extra = {k: plain(v) for k, v in extra.items()}
    payload = {
        "model_id": model_id,
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "class_names": list(config.CLASS_NAMES),
        "num_classes": config.NUM_CLASSES,
        "image_size": config.IMAGE_SIZE,
        "mean": list(config.IMAGENET_MEAN),
        "std": list(config.IMAGENET_STD),
        **extra,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint written by :func:`save_checkpoint`. Returns (eval-mode model, metadata)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_id", "state_dict", "class_names", "image_size", "mean", "std"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint {path.name} is missing '{key}'.")
    if list(ckpt["class_names"]) != config.CLASS_NAMES:
        raise ValueError(
            f"Checkpoint class mapping {ckpt['class_names']} differs from the application's "
            f"{config.CLASS_NAMES}. Retrain the model."
        )
    net = build_model(ckpt["model_id"], num_classes=len(ckpt["class_names"]), pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return net, meta
