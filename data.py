"""Dataset discovery, validation, leakage-free splitting and PyTorch datasets.

Pipeline (run by ``python -m scripts.prepare_data``):

1. ``scan_dataset``      - find class folders, validate every image, hash it.
2. ``group_duplicates``  - exact (MD5) and near-duplicate (dHash) images are
                           grouped so that they can never straddle two splits.
3. ``build_splits``      - stratified, group-aware train/val/test split with a
                           fixed seed.
4. ``write_reports``     - ``splits.json`` + ``dataset_report.json``.

Training/evaluation read ``splits.json`` so that BOTH models use exactly the
same images in each split.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

Image.MAX_IMAGE_PIXELS = 100_000_000  # decompression-bomb guard

# dHash distance (of 64 bits) treated as "same picture". Railway photos look alike, so keep this
# strict: 0-1 catches re-saved / resized / re-compressed copies without merging distinct shots.
NEAR_DUP_HAMMING = 1


# --------------------------------------------------------------------------- #
# Scanning & validation
# --------------------------------------------------------------------------- #
@dataclass
class ImageRecord:
    path: str  # POSIX path relative to the dataset root
    label: int
    md5: str
    dhash: int
    width: int
    height: int
    group: int = -1
    split: str = ""


@dataclass
class ScanResult:
    root: str
    class_folders: dict[str, str] = field(default_factory=dict)  # canonical -> folder
    unknown_folders: list[str] = field(default_factory=list)
    missing_classes: list[str] = field(default_factory=list)
    duplicate_folder_mappings: list[str] = field(default_factory=list)
    records: list[ImageRecord] = field(default_factory=list)
    unreadable: list[dict] = field(default_factory=list)
    skipped_non_images: list[str] = field(default_factory=list)
    found_per_class: dict[str, int] = field(default_factory=dict)


def _dhash(img: Image.Image) -> int:
    small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> tuple[int, int, int]:
    """Fully decode an image. Returns (width, height, dhash) or raises."""
    with Image.open(path) as im:
        im.verify()  # structural check
    with Image.open(path) as im:  # verify() invalidates the handle; reopen
        im = ImageOps.exif_transpose(im)
        im.load()  # forces decoding -> catches truncated files
        if im.width < 16 or im.height < 16:
            raise ValueError(f"image too small ({im.width}x{im.height})")
        return im.width, im.height, _dhash(im)


def scan_dataset(root: Path | None = None) -> ScanResult:
    """Discover class folders and validate every image file."""
    root = Path(root) if root else config.data_dir()
    result = ScanResult(root=str(root))
    if not root.is_dir():
        result.missing_classes = list(config.CLASS_NAMES)
        return result

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        canonical = config.canonical_class(child.name)
        if canonical is None:
            result.unknown_folders.append(child.name)
        elif canonical in result.class_folders:
            result.duplicate_folder_mappings.append(
                f"'{child.name}' also maps to '{canonical}' (already using "
                f"'{result.class_folders[canonical]}') - ignored"
            )
        else:
            result.class_folders[canonical] = child.name

    result.missing_classes = [c for c in config.CLASS_NAMES if c not in result.class_folders]

    for canonical, folder in result.class_folders.items():
        label = config.CLASS_NAMES.index(canonical)
        files = sorted(p for p in (root / folder).rglob("*") if p.is_file())
        count = 0
        for p in files:
            rel = p.relative_to(root).as_posix()
            if p.name.startswith(".") or p.suffix.lower() == "":
                if p.name != ".gitkeep":
                    result.skipped_non_images.append(rel)
                continue
            if p.suffix.lower() not in config.DATASET_EXTENSIONS:
                result.skipped_non_images.append(rel)
                continue
            count += 1
            try:
                if p.stat().st_size == 0:
                    raise ValueError("empty file")
                w, h, dh = inspect_image(p)
                result.records.append(
                    ImageRecord(path=rel, label=label, md5=_md5(p), dhash=dh, width=w, height=h)
                )
            except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
                result.unreadable.append({"path": rel, "reason": f"{type(exc).__name__}: {exc}"})
        result.found_per_class[canonical] = count
    return result


# --------------------------------------------------------------------------- #
# Duplicate grouping (prevents train/test leakage)
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_duplicates(records: list[ImageRecord]) -> dict:
    """Assign ``record.group``; identical or near-identical images share a group."""
    n = len(records)
    uf = _UnionFind(n)
    by_md5: dict[str, int] = {}
    for i, r in enumerate(records):
        if r.md5 in by_md5:
            uf.union(i, by_md5[r.md5])
        else:
            by_md5[r.md5] = i

    if n:
        hashes = np.array([r.dhash for r in records], dtype=np.uint64)
        for i in range(n):  # O(n^2) bitwise compare - fine for <= ~20k images
            xor = hashes[i + 1 :] ^ hashes[i]
            if xor.size == 0:
                continue
            dist = np.unpackbits(xor.view(np.uint8)).reshape(-1, 64).sum(axis=1)
            for j in np.nonzero(dist <= NEAR_DUP_HAMMING)[0]:
                uf.union(i, i + 1 + int(j))

    roots: dict[int, int] = {}
    for i, r in enumerate(records):
        r.group = roots.setdefault(uf.find(i), len(roots))

    members: dict[int, list[ImageRecord]] = defaultdict(list)
    for r in records:
        members[r.group].append(r)
    dup_groups = [g for g in members.values() if len(g) > 1]
    cross = [g for g in dup_groups if len({m.label for m in g}) > 1]
    return {
        "duplicate_groups": len(dup_groups),
        "images_in_duplicate_groups": sum(len(g) for g in dup_groups),
        "cross_class_duplicate_groups": len(cross),
        "cross_class_examples": [[m.path for m in g][:4] for g in cross[:10]],
    }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def build_splits(
    records: list[ImageRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ImageRecord]:
    """Stratified, group-aware split. Mutates and returns ``records``.

    Every duplicate group goes entirely to one split. Groups whose members carry
    different labels are ambiguous and must be removed beforehand.
    """
    if not 0 < val_ratio < 0.5 or not 0 < test_ratio < 0.5 or val_ratio + test_ratio >= 0.6:
        raise ValueError("val_ratio/test_ratio must each be in (0, 0.5) and sum to < 0.6")
    rng = random.Random(seed)
    train_ratio = 1.0 - val_ratio - test_ratio
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    for label in sorted({r.label for r in records}):
        groups: dict[int, list[ImageRecord]] = defaultdict(list)
        for r in records:
            if r.label == label:
                groups[r.group].append(r)
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        total = sum(len(groups[g]) for g in group_ids)
        targets = {s: ratios[s] * total for s in ratios}
        current = {s: 0 for s in ratios}

        def assign(gid: int, split: str) -> None:
            for rec in groups[gid]:
                rec.split = split
            current[split] += len(groups[gid])

        # Seed each split with one group when possible so none is empty.
        seeds = ["test", "val", "train"]
        remaining = list(group_ids)
        if len(remaining) >= 3:
            for s in seeds:
                assign(remaining.pop(0), s)
        for gid in remaining:
            best = max(ratios, key=lambda s: (targets[s] - current[s]) / max(targets[s], 1e-9))
            assign(gid, best)
        if len(group_ids) < 3:
            for gid in group_ids:  # too few groups: everything to train, flagged in report
                for rec in groups[gid]:
                    rec.split = "train"
    return records


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _fingerprint(records: list[ImageRecord]) -> str:
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r.path):
        h.update(f"{r.path}:{r.md5}:{r.label}:{r.split}\n".encode())
    return h.hexdigest()


def prepare(
    root: Path | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    allow_missing_classes: bool = False,
) -> tuple[dict, dict]:
    """Scan + validate + split. Returns (splits_doc, report_doc). Raises on fatal problems."""
    scan = scan_dataset(root)
    root_path = Path(scan.root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root_path}")
    if scan.missing_classes and not allow_missing_classes:
        raise ValueError(
            "Missing class folder(s): "
            + ", ".join(scan.missing_classes)
            + f"\nExpected sub-folders inside {root_path}: "
            + ", ".join(config.CLASS_NAMES)
            + "\n(Use --allow-missing-classes only for experiments.)"
        )

    records = scan.records
    dup_info = group_duplicates(records)

    # Remove images whose duplicate group spans several classes (ambiguous label).
    by_group: dict[int, set[int]] = defaultdict(set)
    for r in records:
        by_group[r.group].add(r.label)
    ambiguous = {g for g, labels in by_group.items() if len(labels) > 1}
    removed_ambiguous = [r.path for r in records if r.group in ambiguous]
    records = [r for r in records if r.group not in ambiguous]

    empty = [c for c in config.CLASS_NAMES if c in scan.class_folders
             and not any(r.label == config.CLASS_NAMES.index(c) for r in records)]
    if empty and not allow_missing_classes:
        raise ValueError("No valid images for class(es): " + ", ".join(empty))

    build_splits(records, val_ratio, test_ratio, seed)

    per_class_split: dict[str, dict[str, int]] = {c: {"train": 0, "val": 0, "test": 0} for c in config.CLASS_NAMES}
    for r in records:
        per_class_split[config.CLASS_NAMES[r.label]][r.split] += 1

    counts = {c: sum(v.values()) for c, v in per_class_split.items()}
    nonzero = [v for v in counts.values() if v > 0]
    imbalance = (max(nonzero) / min(nonzero)) if nonzero else None

    warnings: list[str] = []
    for c, v in per_class_split.items():
        if 0 < counts[c] < 30:
            warnings.append(f"'{c}' has only {counts[c]} usable images - metrics will be unreliable.")
        if counts[c] > 0 and (v["val"] == 0 or v["test"] == 0):
            warnings.append(f"'{c}' has no images in the validation or test split - add more images.")
    if imbalance and imbalance > 3:
        warnings.append(f"Class imbalance ratio {imbalance:.1f}:1 - class weights are applied during training.")
    if ambiguous:
        warnings.append(f"{len(removed_ambiguous)} images removed: identical pictures appear in several class folders.")
    if scan.unreadable:
        warnings.append(f"{len(scan.unreadable)} unreadable image file(s) were excluded.")
    for c in scan.missing_classes:
        warnings.append(f"Class folder for '{c}' is missing.")

    sizes = np.array([[r.width, r.height] for r in records]) if records else np.zeros((0, 2))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fingerprint = _fingerprint(records)

    splits_doc = {
        "created_at": now,
        "seed": seed,
        "ratios": {"train": round(1 - val_ratio - test_ratio, 4), "val": val_ratio, "test": test_ratio},
        "class_names": config.CLASS_NAMES,
        "fingerprint": fingerprint,
        "total": len(records),
        "records": [
            {"path": r.path, "label": r.label, "split": r.split, "md5": r.md5,
             "width": r.width, "height": r.height, "group": r.group}
            for r in sorted(records, key=lambda r: r.path)
        ],
    }
    report_doc = {
        "generated_at": now,
        "dataset_dir_name": root_path.name,
        "seed": seed,
        "fingerprint": fingerprint,
        "class_names": config.CLASS_NAMES,
        "found_per_class": {c: scan.found_per_class.get(c, 0) for c in config.CLASS_NAMES},
        "usable_per_class": counts,
        "per_class_split": per_class_split,
        "split_totals": {s: sum(v[s] for v in per_class_split.values()) for s in ("train", "val", "test")},
        "total_usable": len(records),
        "total_found": sum(scan.found_per_class.values()),
        "imbalance_ratio": round(imbalance, 3) if imbalance else None,
        "missing_classes": scan.missing_classes,
        "unknown_folders": scan.unknown_folders,
        "folder_mapping": scan.class_folders,
        "unreadable_files": scan.unreadable,
        "skipped_non_image_files": scan.skipped_non_images[:50],
        "duplicates": {**dup_info, "removed_ambiguous_images": removed_ambiguous[:50]},
        "image_size_stats": {
            "min_width": int(sizes[:, 0].min()) if len(sizes) else None,
            "max_width": int(sizes[:, 0].max()) if len(sizes) else None,
            "median_width": float(np.median(sizes[:, 0])) if len(sizes) else None,
            "min_height": int(sizes[:, 1].min()) if len(sizes) else None,
            "max_height": int(sizes[:, 1].max()) if len(sizes) else None,
            "median_height": float(np.median(sizes[:, 1])) if len(sizes) else None,
        },
        "warnings": warnings,
    }
    return splits_doc, report_doc


def write_reports(splits_doc: dict, report_doc: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out = Path(out_dir) if out_dir else config.artifacts_dir()
    out.mkdir(parents=True, exist_ok=True)
    sp, rp = out / "splits.json", out / "dataset_report.json"
    sp.write_text(json.dumps(splits_doc, indent=1), encoding="utf-8")
    rp.write_text(json.dumps(report_doc, indent=2), encoding="utf-8")
    return sp, rp


def load_splits(path: Path | None = None) -> dict:
    p = Path(path) if path else config.artifacts_dir() / "splits.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found. Run `python -m scripts.prepare_data` first to create the "
            "train/val/test split."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def records_for(splits_doc: dict, split: str) -> list[dict]:
    return [r for r in splits_doc["records"] if r["split"] == split]


def class_counts(records: list[dict], num_classes: int = config.NUM_CLASSES) -> list[int]:
    c = Counter(r["label"] for r in records)
    return [c.get(i, 0) for i in range(num_classes)]


# --------------------------------------------------------------------------- #
# Transforms & torch dataset (torch imported lazily so the API/data tools stay light)
# --------------------------------------------------------------------------- #
def build_train_transform(size: int = config.IMAGE_SIZE):
    from torchvision import transforms as T

    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.85, 1.2)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        T.ToTensor(),
        T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def build_eval_transform(size: int = config.IMAGE_SIZE, mean=None, std=None):
    """Deterministic transform used for validation, testing AND inference."""
    from torchvision import transforms as T

    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean or config.IMAGENET_MEAN, std or config.IMAGENET_STD),
    ])


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def make_dataset(records: list[dict], root: Path, transform):
    import torch
    from torch.utils.data import Dataset

    class RailDataset(Dataset):
        def __init__(self):
            self.records, self.root, self.transform = records, Path(root), transform

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            rec = self.records[idx]
            img = load_rgb(self.root / rec["path"])
            return self.transform(img), torch.tensor(rec["label"], dtype=torch.long)

    return RailDataset()
