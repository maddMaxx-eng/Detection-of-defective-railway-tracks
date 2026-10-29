"""Validate the dataset and create the leakage-free train/val/test split.

    python -m scripts.prepare_data
    python -m scripts.prepare_data --data-dir D:\\railway_images --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from railguard import config, data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None, help="dataset root (default: data/raw)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--allow-missing-classes", action="store_true",
                    help="continue even if some class folders are empty/missing (experiments only)")
    args = ap.parse_args(argv)

    try:
        splits, report = data.prepare(args.data_dir, args.val_ratio, args.test_ratio, args.seed,
                                      args.allow_missing_classes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    sp, rp = data.write_reports(splits, report)
    print("\nDataset check")
    print("-" * 64)
    print(f"{'Class':<16}{'found':>8}{'usable':>8}{'train':>8}{'val':>8}{'test':>8}")
    for c in config.CLASS_NAMES:
        v = report["per_class_split"][c]
        print(f"{c:<16}{report['found_per_class'][c]:>8}{report['usable_per_class'][c]:>8}"
              f"{v['train']:>8}{v['val']:>8}{v['test']:>8}")
    print("-" * 64)
    t = report["split_totals"]
    print(f"{'TOTAL':<16}{report['total_found']:>8}{report['total_usable']:>8}"
          f"{t['train']:>8}{t['val']:>8}{t['test']:>8}")
    d = report["duplicates"]
    print(f"\nDuplicate groups: {d['duplicate_groups']} "
          f"(each kept entirely inside ONE split) | unreadable: {len(report['unreadable_files'])}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nWrote {sp}\nWrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
