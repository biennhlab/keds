"""Deployment contracts; small byte fixtures never serve as retrieval data."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_modal", ROOT / "scripts/prepare_modal_artifacts.py")
prepare_modal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_modal)


def test_modal_uses_best_without_changing_architecture(tmp_path):
    original = (ROOT / "configs/model.yaml").read_bytes()
    config = prepare_modal.runtime_config(tmp_path, "cuda")
    assert config["bkp"] == yaml.safe_load(original)["bkp"]
    assert config["knowledge"]["expected_count"] == 500000
    assert config["rag"]["profile"] == "E0"
    assert config["model"]["local_files_only"] is True
    assert config["model"]["device"] == "cuda"
    for phase, path in config["checkpoints"].items():
        assert Path(path).is_relative_to(tmp_path)
        assert Path(path).name == f"{phase}_best_author_modules.pt"
        assert prepare_modal.BEST[1] in path
    assert (ROOT / "configs/model.yaml").read_bytes() == original
    destination = tmp_path / "configs/runtime.yaml"
    prepare_modal.write_config(tmp_path, destination, device="cpu")
    assert yaml.safe_load(destination.read_text())["model"]["device"] == "cpu"


def download_fixture(data):
    sha = hashlib.sha256(data).hexdigest()
    item = SimpleNamespace(rfilename="weights.pt", lfs=SimpleNamespace(sha256=sha))
    api = SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(siblings=[item]))
    return api, sha


def test_verified_existing_weights_are_never_replaced(tmp_path, monkeypatch):
    data = b"immutable unit test bytes"
    target = tmp_path / "weights.pt"
    target.write_bytes(data)
    api, expected = download_fixture(data)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Valid existing artifact must not be downloaded or replaced")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", unexpected_download)
    record = prepare_modal.download_verified("repo", "revision", "weights.pt", target, False, api)
    assert record["sha256"] == expected
    assert target.read_bytes() == data


def test_conflicting_existing_weights_are_preserved(tmp_path):
    target = tmp_path / "weights.pt"
    target.write_bytes(b"older bytes")
    api, _ = download_fixture(b"different bytes")
    with pytest.raises(ValueError, match="preserving"):
        prepare_modal.download_verified("repo", "revision", "weights.pt", target, False, api)
    assert target.read_bytes() == b"older bytes"


def test_download_verifies_bytes_before_publishing(tmp_path, monkeypatch):
    cached = tmp_path / "cached"
    cached.write_bytes(b"downloaded bytes")
    api, expected = download_fixture(cached.read_bytes())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(cached))
    target = tmp_path / "new/weights.pt"
    record = prepare_modal.download_verified("repo", "revision", "weights.pt", target, False, api)
    assert record["sha256"] == expected
    assert target.read_bytes() == cached.read_bytes()
    target.unlink()
    cached.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_modal.download_verified("repo", "revision", "weights.pt", target, False, api)
    assert not target.exists()
