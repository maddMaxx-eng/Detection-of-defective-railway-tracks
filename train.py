"""Train MobileNetV2 and/or EfficientNetB0 with two-phase transfer learning.

    python -m scripts.train --model all
    python -m scripts.train --model mobilenet_v2
    python -m scripts.train --model efficientnet_b0 --head-epochs 5 --finetune-epochs 10

Phase 1 trains only the new classification head (backbone frozen).
Phase 2 unfreezes the last backbone blocks and fine-tunes with a small LR.
The checkpoint with the best VALIDATION macro-F1 (ties: lower val loss) is kept.
The test split is never touched here - use ``python -m scripts.evaluate``.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from railguard import __version__, config, data, models
from railguard.metrics import compute_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def run_epoch(net, loader, criterion, device, optimizer=None, freeze_bn=False, backbone_frozen=False, scaler=None):
    training = optimizer is not None
    net.train(training)
    if training:
        if backbone_frozen:
            net.features.eval()  # no BN-stat drift while frozen
        if freeze_bn:
            models.set_bn_eval(net)
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = net(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    n = max(len(y_true), 1)
    m = compute_metrics(y_true, y_pred)
    return total_loss / n, m["accuracy"], m["macro_f1"]


def train_one(model_id: str, cfg: dict, splits: dict, args, device: torch.device) -> dict:
    mcfg = cfg["models"][model_id]
    root = config.data_dir() if args.data_dir is None else Path(args.data_dir)
    seed_everything(cfg["seed"])

    train_recs, val_recs = data.records_for(splits, "train"), data.records_for(splits, "val")
    if not train_recs or not val_recs:
        raise SystemExit("Train or validation split is empty - add images and re-run prepare_data.")
    counts = data.class_counts(train_recs)
    if min(c for c in counts if c > 0) == 0:
        raise SystemExit("Empty class in training split.")

    size = cfg["image_size"]
    train_ds = data.make_dataset(train_recs, root, data.build_train_transform(size))
    val_ds = data.make_dataset(val_recs, root, data.build_eval_transform(size))
    gen = torch.Generator().manual_seed(cfg["seed"])
    sampler = None
    if cfg["imbalance"] == "sampler":
        w = [1.0 / counts[r["label"]] for r in train_recs]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=gen)
    loader_kw = dict(batch_size=cfg["batch_size"], num_workers=args.num_workers,
                     pin_memory=device.type == "cuda", worker_init_fn=_worker_init,
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, generator=gen, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    weight = None
    if cfg["imbalance"] == "class_weights":
        present = [c for c in counts if c > 0]
        total = sum(counts)
        weight = torch.tensor([total / (len(present) * c) if c > 0 else 0.0 for c in counts],
                              dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])

    net = models.build_model(model_id, pretrained=not args.no_pretrained, dropout=cfg["dropout"]).to(device)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    total_params = models.count_parameters(net)["total"]

    print(f"\n{'=' * 70}\n{config.MODEL_NAMES[model_id]}  |  device={device}  |  params={total_params:,}")
    print(f"train={len(train_recs)} val={len(val_recs)}  class counts(train)={counts}")
    if args.no_pretrained:
        print("!! --no-pretrained: random weights. Use ONLY for smoke tests, never for real results.")
    print("=" * 70)

    history: list[dict] = []
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0, "metrics": None}
    ckpt_path = config.checkpoint_path(model_id)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    epoch_no = 0
    phases = [
        ("head", args.head_epochs if args.head_epochs is not None else cfg["head_epochs"]),
        ("finetune", args.finetune_epochs if args.finetune_epochs is not None else cfg["finetune_epochs"]),
    ]
    phase_info = {}
    for phase, n_epochs in phases:
        if n_epochs <= 0:
            continue
        if phase == "head":
            trainable_backbone = models.set_backbone_trainable(net, False)
            params = list(models.head_parameters(net))
            lr = mcfg["head_lr"]
        else:
            trainable_backbone = models.set_backbone_trainable(net, True, mcfg["finetune_from_block"])
            params = [p for p in net.parameters() if p.requires_grad]
            lr = mcfg["finetune_lr"]
        phase_info[phase] = {"epochs": n_epochs, "lr": lr, "trainable_backbone_params": trainable_backbone,
                             "trainable_params": sum(p.numel() for p in params)}
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        stale = 0
        print(f"\n-- Phase '{phase}' | epochs={n_epochs} | lr={lr} | trainable params={phase_info[phase]['trainable_params']:,}")
        for _ in range(n_epochs):
            epoch_no += 1
            es = time.perf_counter()
            tr_loss, tr_acc, _ = run_epoch(net, train_loader, criterion, device, opt, cfg["freeze_bn"],
                                           backbone_frozen=phase == "head", scaler=scaler)
            va_loss, va_acc, va_f1 = run_epoch(net, val_loader, criterion, device)
            sched.step()
            row = {"epoch": epoch_no, "phase": phase, "lr": opt.param_groups[0]["lr"],
                   "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                   "val_acc": va_acc, "val_macro_f1": va_f1, "seconds": round(time.perf_counter() - es, 2)}
            history.append(row)
            improved = va_f1 > best["f1"] + 1e-9 or (abs(va_f1 - best["f1"]) <= 1e-9 and va_loss < best["loss"])
            print(f"epoch {epoch_no:02d} [{phase:8}] train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} val_macroF1={va_f1:.3f}"
                  + ("  <- best" if improved else ""))
            if improved:
                best.update(f1=va_f1, loss=va_loss, epoch=epoch_no, metrics={"val_acc": va_acc, "val_macro_f1": va_f1, "val_loss": va_loss})
                stale = 0
                models.save_checkpoint(net, ckpt_path, model_id, {
                    "best_epoch": epoch_no,
                    "val_macro_f1": va_f1,
                    "val_accuracy": va_acc,
                    "pretrained_imagenet": not args.no_pretrained,
                    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "split_fingerprint": splits["fingerprint"],
                    "railguard_version": __version__,
                    "torch_version": torch.__version__,
                })
            else:
                stale += 1
                if stale >= cfg["early_stopping_patience"]:
                    print(f"early stopping (no improvement for {stale} epochs in phase '{phase}')")
                    break

    elapsed = time.perf_counter() - t0
    summary = {
        "model_id": model_id,
        "model_name": config.MODEL_NAMES[model_id],
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(elapsed, 1),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or platform.machine(),
        "pretrained_imagenet": not args.no_pretrained,
        "best_epoch": best["epoch"],
        "best_val": best["metrics"],
        "epochs_run": epoch_no,
        "total_parameters": total_params,
        "phases": phase_info,
        "split_fingerprint": splits["fingerprint"],
        "train_images": len(train_recs),
        "val_images": len(val_recs),
        "class_counts_train": dict(zip(config.CLASS_NAMES, counts)),
        "config": {k: v for k, v in cfg.items() if k != "models"} | {"model": mcfg},
        "versions": {"railguard": __version__, "torch": torch.__version__, "python": platform.python_version()},
        "smoke_test": bool(args.no_pretrained),
        "history": history,
    }
    out = config.artifacts_dir() / model_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # A new training run invalidates any older test metrics for this model.
    stale_metrics = out / "metrics.json"
    if stale_metrics.exists():
        stale_metrics.unlink()
    print(f"\nSaved best checkpoint (epoch {best['epoch']}, val macro-F1 {best['f1']:.4f}) -> {ckpt_path}")
    print(f"Training time: {elapsed / 60:.1f} min   History -> {out / 'history.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=[*config.MODEL_IDS, "all"], default="all")
    ap.add_argument("--config", type=Path, default=config.ROOT / "configs" / "train.yaml")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None, help="path to splits.json")
    ap.add_argument("--head-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--no-pretrained", action="store_true", help="SMOKE TESTS ONLY: random init")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg["image_size"] != config.IMAGE_SIZE:
        print("image_size in the YAML must equal railguard.config.IMAGE_SIZE", file=sys.stderr)
        return 1
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is None:
        args.num_workers = cfg["num_workers"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    try:
        splits = data.load_splits(args.splits)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ids = list(config.MODEL_IDS) if args.model == "all" else [args.model]
    for mid in ids:
        train_one(mid, cfg, splits, args, device)
    print("\nNext: python -m scripts.evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
