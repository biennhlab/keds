"""File/ID validation only; these fixtures are never used for app retrieval."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("import_fashioniq", Path("scripts/import_fashioniq.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resolve_keeps_official_ids_and_order_and_ignores_extra_files(tmp_path):
    (tmp_path / "category").mkdir()
    first = tmp_path / "category/B000AAA001.jpg"
    second = tmp_path / "B000AAA002.png"
    first.touch()
    second.touch()
    (tmp_path / "unrelated.jpg").touch()
    assert module.resolve_images(tmp_path, [second.stem, first.stem]) == [second, first]


def test_missing_images_cannot_be_silently_dropped(tmp_path):
    with pytest.raises(ValueError, match="Missing 2/2"):
        module.resolve_images(tmp_path, ["B000AAA001", "B000AAA002"])
    with pytest.raises(ValueError, match="unique"):
        module.resolve_images(tmp_path, ["B000AAA001", "B000AAA001"])


def test_duplicate_stems_are_ambiguous_even_with_different_extensions(tmp_path):
    (tmp_path / "B000AAA001.jpg").touch()
    (tmp_path / "B000AAA001.png").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        module.resolve_images(tmp_path, ["B000AAA001"])


def test_upload_manifest_rejects_corrupted_dataset_bytes(tmp_path):
    photo = tmp_path / "image.jpg"
    photo.write_bytes(b"unit test checksum bytes")
    records = [{"path": photo.name, "size": photo.stat().st_size,
                "sha256": hashlib.sha256(photo.read_bytes()).hexdigest()}]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    (tmp_path / "upload_manifest.json").write_text(json.dumps({"files": records, "fingerprint": fingerprint}))
    assert module.verify_upload_manifest(tmp_path) == fingerprint
    photo.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify_upload_manifest(tmp_path)
