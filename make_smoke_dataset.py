"""Create a SYNTHETIC toy dataset to smoke-test the pipeline (not real railway data).

    python -m scripts.make_smoke_dataset --out data/_smoke --per-class 40
    set RAILGUARD_DATA_DIR=data\\_smoke  (Windows)   then run prepare_data/train/evaluate

The images are procedurally drawn coloured rails. Any accuracy obtained on them
is MEANINGLESS. Use it only to check that the code runs end-to-end, and delete
the generated artifacts/models afterwards.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from railguard import config


def draw(label: str, rng: random.Random, size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), (rng.randint(40, 110),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(25):  # random background texture so images are distinct
        x, y, w = rng.randint(0, size), rng.randint(0, size), rng.randint(10, 80)
        d.rectangle([x, y, x + w, y + rng.randint(5, 60)], fill=tuple(rng.randint(20, 160) for _ in range(3)))
    for x in (size * 0.3, size * 0.7):  # two rails
        d.rectangle([x - 8, 0, x + 8, size], fill=(150, 150, 160))
    for y in range(10, size, 40):  # sleepers
        d.rectangle([size * 0.15, y, size * 0.85, y + 10], fill=(90, 70, 50))
    if label == "Broken Track":
        y = rng.randint(60, 200)
        d.rectangle([size * 0.3 - 12, y, size * 0.3 + 12, y + 10], fill=(10, 10, 10))
    elif label == "Rust":
        for _ in range(60):
            x, y = rng.randint(60, 200), rng.randint(0, size)
            d.ellipse([x, y, x + 8, y + 8], fill=(150 + rng.randint(0, 40), 70, 20))
    elif label == "Loose Bolts":
        for y in range(20, size, 70):
            d.ellipse([size * 0.3 - 6 + rng.randint(-12, 12), y, size * 0.3 + 6, y + 12], fill=(220, 220, 40))
    elif label == "Missing Parts":
        for y in range(20, size, 70):
            d.rectangle([size * 0.3 - 5, y, size * 0.3 + 5, y + 10], fill=(0, 120, 200))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.ROOT / "data" / "_smoke")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    print("!! Generating SYNTHETIC images for pipeline testing only. Results are meaningless.")
    for name in config.CLASS_NAMES:
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            draw(name, rng).save(folder / f"synthetic_{i:03d}.jpg", quality=92)
    print(f"Created {args.per_class * len(config.CLASS_NAMES)} images in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
