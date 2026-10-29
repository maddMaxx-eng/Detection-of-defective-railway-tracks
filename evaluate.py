"""Evaluate every trained model on the SAME held-out test split and compare them.

    python -m scripts.evaluate

For each available checkpoint it writes ``artifacts/<model>/metrics.json`` with
accuracy, per-class precision/recall/F1, macro-F1, confusion matrix, model size,
parameter count, CPU inference latency and training time. It then writes
``artifacts/comparison.json`` including the winner (only when both models have
been evaluated on the same split).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from railguard import __version__, config, data, models
from railguard.metrics import compute_metrics


def measure_latency(net, size: int, warmup: int, runs: int) -> dict:
    """Single-image forward-pass latency on CPU (what a CPU-hosted API experiences)."""
    net = net.to("cpu").eval()
    x = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            net(x)
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 2),
        "runs": runs,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
    }


def evaluate_model(model_id: str, splits: dict, root: Path, cfg: dict, device, workers: int) -> dict:
    ckpt = config.checkpoint_path(model_id)
    net, meta = models.load_checkpoint(ckpt, device)
    test_recs = data.records_for(splits, "test")
    if not test_recs:
        raise SystemExit("Test split is empty - add more images and re-run prepare_data.")
    if meta.get("split_fingerprint") != splits["fingerprint"]:
        print(f"WARNING: {model_id} was trained on a different split than the current splits.json. "
              "Re-run training so both models use the same split.")
    ds = data.make_dataset(test_recs, root, data.build_eval_transform(meta["image_size"], meta["mean"], meta["std"]))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=workers)
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            y_pred.extend(net(x.to(device)).argmax(1).cpu().tolist())
            y_true.extend(y.tolist())
    m = compute_metrics(y_true, y_pred)

    ev = cfg["evaluation"]
    latency = measure_latency(net, meta["image_size"], ev["latency_warmup"], ev["latency_runs"])
    params = models.count_parameters(net)
    hist_path = config.artifacts_dir() / model_id / "history.json"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {}
    result = {
        "model_id": model_id,
        "model_name": config.MODEL_NAMES[model_id],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "test",
        "split_fingerprint": splits["fingerprint"],
        "trained_on_fingerprint": meta.get("split_fingerprint"),
        "test_images": len(test_recs),
        "metrics": m,
        "model_size_mb": round(ckpt.stat().st_size / 1e6, 2),
        "parameters": params["total"],
        "latency": latency,
        "training_seconds": hist.get("training_seconds"),
        "training_device": hist.get("device_name"),
        "best_epoch": meta.get("best_epoch"),
        "pretrained_imagenet": meta.get("pretrained_imagenet"),
        "smoke_test": bool(hist.get("smoke_test", False)),
        "versions": {"railguard": __version__, "torch": torch.__version__},
    }
    out = config.artifacts_dir() / model_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def pick_winner(results: dict[str, dict], tolerance: float) -> dict | None:
    if len(results) < 2:
        return None
    a, b = (results[i] for i in config.MODEL_IDS)
    if a["split_fingerprint"] != b["split_fingerprint"] or a["trained_on_fingerprint"] != b["trained_on_fingerprint"]:
        return {"model_id": None, "decided_by": "not_comparable",
                "explanation": "The models were not trained and tested on the same split; retrain both before comparing."}
    fa, fb = a["metrics"]["macro_f1"], b["metrics"]["macro_f1"]
    gap = fa - fb
    if abs(gap) > tolerance:
        w, l = (a, b) if gap > 0 else (b, a)
        return {"model_id": w["model_id"], "decided_by": "macro_f1",
                "explanation": (f"{w['model_name']} wins on classification quality: macro-F1 "
                                f"{w['metrics']['macro_f1']:.3f} vs {l['metrics']['macro_f1']:.3f} "
                                f"(gap {abs(gap):.3f} > tolerance {tolerance}).")}
    la, lb = a["latency"]["median_ms"], b["latency"]["median_ms"]
    if la == lb:
        return {"model_id": None, "decided_by": "tie",
                "explanation": "Macro-F1 and latency are effectively identical; no clear winner."}
    w, l = (a, b) if la < lb else (b, a)
    return {"model_id": w["model_id"], "decided_by": "latency",
            "explanation": (f"Macro-F1 is within the {tolerance} tolerance ({fa:.3f} vs {fb:.3f}), so the more "
                            f"efficient model wins: {w['model_name']} median CPU latency "
                            f"{w['latency']['median_ms']} ms vs {l['latency']['median_ms']} ms.")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=config.ROOT / "configs" / "train.yaml")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--device", default="cpu", help="device for test-set inference (latency is always CPU)")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        splits = data.load_splits(args.splits)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    root = Path(args.data_dir) if args.data_dir else config.data_dir()

    results: dict[str, dict] = {}
    missing: list[str] = []
    for mid in config.MODEL_IDS:
        if not config.checkpoint_path(mid).is_file():
            missing.append(mid)
            print(f"[skip] {config.MODEL_NAMES[mid]}: no checkpoint - run `python -m scripts.train --model {mid}`")
            continue
        print(f"[eval] {config.MODEL_NAMES[mid]} on {len(data.records_for(splits, 'test'))} test images ...")
        results[mid] = evaluate_model(mid, splits, root, cfg, torch.device(args.device), args.num_workers)
        r = results[mid]
        print(f"       accuracy={r['metrics']['accuracy']:.4f}  macro-F1={r['metrics']['macro_f1']:.4f}  "
              f"size={r['model_size_mb']} MB  latency(median)={r['latency']['median_ms']} ms")

    if not results:
        print("No trained models found. Train first: python -m scripts.train --model all", file=sys.stderr)
        return 1

    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluated": list(results),
        "not_trained": missing,
        "macro_f1_tolerance": cfg["evaluation"]["macro_f1_tolerance"],
        "winner": pick_winner(results, cfg["evaluation"]["macro_f1_tolerance"]),
        "smoke_test": any(r["smoke_test"] for r in results.values()),
        "summary": {mid: {
            "model_name": r["model_name"],
            "accuracy": r["metrics"]["accuracy"],
            "macro_f1": r["metrics"]["macro_f1"],
            "macro_precision": r["metrics"]["macro_precision"],
            "macro_recall": r["metrics"]["macro_recall"],
            "model_size_mb": r["model_size_mb"],
            "latency_median_ms": r["latency"]["median_ms"],
            "training_seconds": r["training_seconds"],
            "parameters": r["parameters"],
        } for mid, r in results.items()},
    }
    (config.artifacts_dir() / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    if comparison["winner"]:
        print("\nWinner:", comparison["winner"]["explanation"])
    else:
        print("\nOnly one model evaluated - no winner can be declared yet.")
    print(f"Wrote {config.artifacts_dir() / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
