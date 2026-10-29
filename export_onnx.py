"""Convert the trained PyTorch checkpoints to ONNX for in-browser inference (Vercel).

    python -m scripts.export_onnx                # exports every checkpoint found in models/
    python -m scripts.export_onnx --model efficientnet_b0

The checkpoints are loaded with ``railguard.models.load_checkpoint`` (the same code used by
training/evaluation), exported in eval mode, then the ONNX file is run with onnxruntime and
compared with PyTorch. The measured differences are stored in ``public/models/manifest.json``.
Nothing is hardcoded: class names, mean/std and input size come from the checkpoint itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from railguard import config
from railguard.models import load_checkpoint

OUT = config.ROOT / "public" / "models"
INPUT, OUTPUT, OPSET = "input", "logits", 17


def export_one(model_id: str) -> dict:
    import onnx
    import onnxruntime as ort

    ckpt = config.checkpoint_path(model_id)
    net, meta = load_checkpoint(ckpt)
    size = meta["image_size"]
    dummy = torch.zeros(1, 3, size, size)

    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{model_id}.tmp.onnx"
    kwargs = dict(input_names=[INPUT], output_names=[OUTPUT], opset_version=OPSET)
    try:  # legacy TorchScript exporter: single self-contained file, well supported by onnxruntime-web
        torch.onnx.export(net, dummy, str(tmp), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(net, dummy, str(tmp), **kwargs)
    onnx.checker.check_model(onnx.load(str(tmp)))

    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    final = OUT / f"{model_id}.{digest[:8]}.onnx"
    for old in OUT.glob(f"{model_id}.????????.onnx"):  # remove superseded exports (hashed names only)
        if old != final:
            old.unlink()
    tmp.replace(final)

    # ---- parity check: ONNX Runtime vs PyTorch on random + structured inputs -----------------
    sess = ort.InferenceSession(str(final), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal((1, 3, size, size)).astype(np.float32) for _ in range(8)]
    inputs.append(np.zeros((1, 3, size, size), np.float32))
    max_logit, max_prob, agree = 0.0, 0.0, 0
    for x in inputs:
        with torch.inference_mode():
            t = net(torch.from_numpy(x))
        o = sess.run([OUTPUT], {INPUT: x})[0]
        max_logit = max(max_logit, float(np.abs(t.numpy() - o).max()))
        max_prob = max(max_prob, float(np.abs(torch.softmax(t, 1).numpy() - _softmax(o)).max()))
        agree += int(t.argmax(1).item() == int(o.argmax(1)[0]))
    print(f"{model_id}: {final.name}  {final.stat().st_size / 1e6:.2f} MB  "
          f"max|logit diff|={max_logit:.2e}  max|prob diff|={max_prob:.2e}  argmax agree {agree}/{len(inputs)}")
    if max_prob > 1e-3 or agree != len(inputs):
        raise SystemExit(f"ONNX export of {model_id} does not match PyTorch - aborting.")

    keep = ("best_epoch", "trained_at", "val_accuracy", "val_macro_f1", "pretrained_imagenet",
            "split_fingerprint", "torch_version", "railguard_version")
    return {
        "id": model_id,
        "name": config.MODEL_NAMES[model_id],
        "file": final.name,
        "sha256": digest,
        "size_bytes": final.stat().st_size,
        "input_name": INPUT,
        "output_name": OUTPUT,
        "image_size": size,
        "class_names": meta["class_names"],
        "mean": meta["mean"],
        "std": meta["std"],
        "checkpoint": {k: meta.get(k) for k in keep},
        "onnx_parity": {"opset": OPSET, "max_abs_logit_diff": max_logit, "max_abs_prob_diff": max_prob,
                        "argmax_agreement": f"{agree}/{len(inputs)}", "n_inputs": len(inputs)},
    }


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=[*config.MODEL_IDS, "all"], default="all")
    args = ap.parse_args()
    ids = list(config.MODEL_IDS) if args.model == "all" else [args.model]
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}
    for mid in ids:
        if not config.checkpoint_path(mid).is_file():
            print(f"[skip] {mid}: {config.checkpoint_path(mid)} not found")
            continue
        manifest["models"][mid] = export_one(mid)
    if not manifest["models"]:
        print("Nothing exported.", file=sys.stderr)
        return 1
    manifest["class_names"] = config.CLASS_NAMES
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
