import shutil
from pathlib import Path

import pytest
from PIL import Image

from railguard import config, data
from scripts.make_smoke_dataset import draw
import random


@pytest.fixture()
def dataset(tmp_path):
    rng = random.Random(1)
    root = tmp_path / "raw"
    for name in config.CLASS_NAMES:
        (root / name).mkdir(parents=True)
        for i in range(24):
            draw(name, rng).save(root / name / f"{i}.jpg")
    return root


def test_aliases():
    assert config.canonical_class("Broken") == "Broken Track"
    assert config.canonical_class("missing_parts") == "Missing Parts"
    assert config.canonical_class("NORMAL") == "Normal Track"
    assert config.canonical_class("banana") is None


def test_missing_class_is_reported(dataset):
    shutil.rmtree(dataset / "Rust")
    with pytest.raises(ValueError, match="Rust"):
        data.prepare(dataset)


def test_missing_directory():
    with pytest.raises(FileNotFoundError):
        data.prepare(Path("/definitely/not/here"))


def test_unreadable_files_detected(dataset):
    (dataset / "Rust" / "bad.jpg").write_bytes(b"not an image")
    splits, report = data.prepare(dataset)
    assert any(u["path"].endswith("bad.jpg") for u in report["unreadable_files"])
    assert all(not r["path"].endswith("bad.jpg") for r in splits["records"])


def test_no_leakage_with_duplicates(dataset):
    # copy images (exact duplicates) inside a class
    for i in range(6):
        shutil.copy(dataset / "Rust" / f"{i}.jpg", dataset / "Rust" / f"dup_{i}.jpg")
    splits, report = data.prepare(dataset, seed=7)
    assert report["duplicates"]["duplicate_groups"] >= 6
    split_by_group = {}
    for r in splits["records"]:
        assert split_by_group.setdefault(r["group"], r["split"]) == r["split"], "group straddles splits"
    md5_split = {}
    for r in splits["records"]:
        assert md5_split.setdefault(r["md5"], r["split"]) == r["split"]


def test_cross_class_duplicates_removed(dataset):
    shutil.copy(dataset / "Rust" / "0.jpg", dataset / "Loose Bolts" / "from_rust.jpg")
    splits, report = data.prepare(dataset)
    assert report["duplicates"]["cross_class_duplicate_groups"] >= 1
    removed = report["duplicates"]["removed_ambiguous_images"]
    assert any(p.endswith("from_rust.jpg") for p in removed) and any(p.endswith("Rust/0.jpg") for p in removed)
    assert all(not r["path"].endswith("from_rust.jpg") for r in splits["records"])


def test_split_is_reproducible_and_stratified(dataset):
    a, ra = data.prepare(dataset, seed=3)
    b, _ = data.prepare(dataset, seed=3)
    assert [r["split"] for r in a["records"]] == [r["split"] for r in b["records"]]
    assert a["fingerprint"] == b["fingerprint"]
    for c in config.CLASS_NAMES:
        s = ra["per_class_split"][c]
        assert s["train"] > 0 and s["val"] > 0 and s["test"] > 0
