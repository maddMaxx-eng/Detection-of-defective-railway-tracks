"""Inference: loads trained checkpoints and classifies PIL images.

Uses the SAME eval transform (resize -> tensor -> normalise) and the class order
stored inside the checkpoint, so predictions match what was evaluated.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from . import config
from .data import build_eval_transform
from .models import load_checkpoint


class ModelUnavailable(RuntimeError):
    """Raised when a requested model has no (loadable) checkpoint."""


class Predictor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._models: dict[str, tuple[torch.nn.Module, dict, object]] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    # -- availability ------------------------------------------------------- #
    def checkpoint_file(self, model_id: str) -> Path:
        return config.checkpoint_path(model_id)

    def is_available(self, model_id: str) -> bool:
        return self.checkpoint_file(model_id).is_file() and model_id not in self._errors

    def error_for(self, model_id: str) -> str | None:
        return self._errors.get(model_id)

    def reload(self) -> None:
        with self._lock:
            self._models.clear()
            self._errors.clear()

    def _get(self, model_id: str):
        if model_id not in config.MODEL_IDS:
            raise ModelUnavailable(f"Unknown model '{model_id}'.")
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
            path = self.checkpoint_file(model_id)
            if not path.is_file():
                raise ModelUnavailable(
                    f"{config.MODEL_NAMES[model_id]} has not been trained yet "
                    f"(missing {path.name}). Run: python -m scripts.train --model {model_id}"
                )
            try:
                net, meta = load_checkpoint(path, self.device)
            except Exception as exc:  # corrupted / incompatible checkpoint
                self._errors[model_id] = f"{type(exc).__name__}: {exc}"
                raise ModelUnavailable(
                    f"{config.MODEL_NAMES[model_id]} checkpoint could not be loaded: {exc}"
                ) from exc
            tfm = build_eval_transform(meta["image_size"], meta["mean"], meta["std"])
            self._models[model_id] = (net, meta, tfm)
            return self._models[model_id]

    def metadata(self, model_id: str) -> dict:
        return self._get(model_id)[1]

    # -- inference ---------------------------------------------------------- #
    def predict(self, image, model_id: str, low_conf: float | None = None) -> dict:
        net, meta, tfm = self._get(model_id)
        class_names: list[str] = meta["class_names"]
        threshold = config.low_confidence_threshold() if low_conf is None else low_conf

        x = tfm(image.convert("RGB")).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            probs = torch.softmax(net(x), dim=1)[0].cpu().tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000

        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top = order[0]
        margin = probs[order[0]] - probs[order[1]]
        confidence = probs[top]
        low = confidence < threshold or margin < config.MIN_TOP2_MARGIN
        warning = None
        if low:
            warning = (
                f"Low confidence ({confidence:.0%}; runner-up is "
                f"{class_names[order[1]]} at {probs[order[1]]:.0%}). Do not rely on this "
                "result - retake the photo or have the track inspected manually."
            )
        return {
            "model_id": model_id,
            "model_name": config.MODEL_NAMES[model_id],
            "predicted_class": class_names[top],
            "confidence": confidence,
            "top3": [{"class": class_names[i], "probability": probs[i]} for i in order[:3]],
            "probabilities": {class_names[i]: probs[i] for i in range(len(probs))},
            "low_confidence": low,
            "warning": warning,
            "inference_ms": round(elapsed_ms, 2),
        }
