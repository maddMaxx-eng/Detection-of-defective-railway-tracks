"""Central configuration: class names, folder aliases and filesystem paths.

Paths are resolved lazily from environment variables so tests and deployments
can redirect them without editing code.

Environment variables (all optional):
    RAILGUARD_DATA_DIR       dataset root          (default: <repo>/data/raw)
    RAILGUARD_MODELS_DIR     trained checkpoints   (default: <repo>/models)
    RAILGUARD_ARTIFACTS_DIR  JSON reports          (default: <repo>/artifacts)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The five categories. The ORDER defines the class index used everywhere
# (training, checkpoints, inference, charts). Never reorder without retraining.
CLASS_NAMES: list[str] = [
    "Broken Track",
    "Loose Bolts",
    "Missing Parts",
    "Rust",
    "Normal Track",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_SLUGS: dict[str, str] = {
    "Broken Track": "broken-track",
    "Loose Bolts": "loose-bolts",
    "Missing Parts": "missing-parts",
    "Rust": "rust",
    "Normal Track": "normal-track",
}
SLUG_TO_CLASS = {v: k for k, v in CLASS_SLUGS.items()}

# Folder-name aliases (compared after lower-casing and removing non-alphanumerics),
# so "Broken", "broken_track" and "Broken Track" all map to "Broken Track".
_ALIASES: dict[str, str] = {
    "brokentrack": "Broken Track",
    "broken": "Broken Track",
    "crack": "Broken Track",
    "loosebolts": "Loose Bolts",
    "loosebolt": "Loose Bolts",
    "loose": "Loose Bolts",
    "missingparts": "Missing Parts",
    "missingpart": "Missing Parts",
    "missing": "Missing Parts",
    "rust": "Rust",
    "rusty": "Rust",
    "corrosion": "Rust",
    "normaltrack": "Normal Track",
    "normal": "Normal Track",
    "good": "Normal Track",
    "nodefect": "Normal Track",
}

# Image formats accepted when scanning the dataset folders.
DATASET_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Formats accepted for uploads through the API / UI.
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_IDS = ("mobilenet_v2", "efficientnet_b0")
MODEL_NAMES = {"mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNetB0"}

# Predictions below this top-1 probability are flagged as low-confidence.
# NOTE: softmax scores are not calibrated probabilities; this is a safety
# heuristic, not a statistical guarantee. Override with LOW_CONFIDENCE_THRESHOLD.
DEFAULT_LOW_CONFIDENCE = 0.60
MIN_TOP2_MARGIN = 0.10


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_class(folder_name: str) -> str | None:
    """Map a folder name to a canonical class name, or None if unknown."""
    return _ALIASES.get(normalize_name(folder_name))


def _path_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    p = Path(raw).expanduser() if raw else default
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def data_dir() -> Path:
    return _path_from_env("RAILGUARD_DATA_DIR", ROOT / "data" / "raw")


def models_dir() -> Path:
    return _path_from_env("RAILGUARD_MODELS_DIR", ROOT / "models")


def artifacts_dir() -> Path:
    return _path_from_env("RAILGUARD_ARTIFACTS_DIR", ROOT / "artifacts")


def checkpoint_path(model_id: str) -> Path:
    return models_dir() / f"{model_id}_best.pt"


def low_confidence_threshold() -> float:
    try:
        return float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", DEFAULT_LOW_CONFIDENCE))
    except ValueError:
        return DEFAULT_LOW_CONFIDENCE
