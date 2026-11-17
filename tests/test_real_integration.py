"""Opt-in tests use real downloaded CLIP, CC3M 500K, v21 weights and image files.

No replacement encoders, mocked predictions or hardcoded retrieval outputs.
"""
import io
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from keds.api.main import create_app
from keds.gallery.repository import Gallery, image_digest
from keds.inference.engine import KEDsEngine
from keds.utils.io import read_json, write_json

pytestmark = [pytest.mark.artifacts, pytest.mark.skipif(
    not os.environ.get("KEDS_TEST_ARTIFACTS"), reason="Set KEDS_TEST_ARTIFACTS=1 for actual full pipeline")]


@pytest.fixture(scope="module")
def real_engine():
    engine = KEDsEngine("configs/model.yaml")
    engine.load_gallery("artifacts/gallery/demo")
    return engine


def test_real_composition_regression_and_image_modes(real_engine):
    image = Image.open("data/demo/dog.jpg").convert("RGB")
    modes = [image, image.convert("RGBA"), image.convert("L")]
    encoded = real_engine.clip.encode_images(modes)
    assert encoded.shape == (3, 768) and torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded[0], encoded[1], atol=1e-6, rtol=1e-6)
    text = "with two dogs instead of one"
    first = real_engine.encode_query(image, text)
    second = real_engine.encode_query(image, text)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    assert first.shape == (768,) and first.dtype == torch.float32
    assert first.norm().item() == pytest.approx(1.0, abs=1e-6)
    # A one-token identity replacement must reproduce CLIP's native text path.
    with torch.no_grad():
        marker = real_engine.clip.clip.text_model.embeddings.token_embedding.weight[
            real_engine.clip.placeholder_id].reshape(1, 1, 768)
        identity = real_engine.clip.compose(["a photo of *"], marker)
        native = real_engine.clip.encode_texts(["a photo of *"])
    torch.testing.assert_close(identity, native, rtol=1e-6, atol=1e-6)
    gallery = real_engine.gallery
    rows = gallery.search(first.unsqueeze(0), 3, exclude_hash=image_digest(image))
    old = read_json("tests/fixtures/e0_regression.json")
    assert [r["image_id"] for r in rows] == [r["image_id"] for r in old["results"]]
    np.testing.assert_allclose([r["score"] for r in rows], [r["score"] for r in old["results"]], atol=1e-5)
    assert all(Path(row["image_path"]).is_file() for row in rows)
    # Exercise actual frozen CLIP backward into the trained BKP without taking
    # an optimizer step or modifying downloaded weights.
    stream = real_engine.streams.phi_m.requires_grad_(True)
    ni = real_engine.database.rows("image", old["details"]["neighbors"]["image"]["indices"])
    nt = real_engine.database.rows("text", old["details"]["neighbors"]["text"]["indices"])
    pseudo = stream(encoded[:1], ni, nt)
    composed = real_engine.clip.compose(["a photo of *"], pseudo)
    (1 - (composed * encoded[:1]).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stream.parameters())
    assert all(p.grad is None for p in real_engine.clip.parameters())
    stream.zero_grad(set_to_none=True)
    stream.requires_grad_(False)
    # Store a numeric regression artifact with the complete checkpoint/config provenance.
    np.save("reports/regression_query.npy", first.numpy(), allow_pickle=False)
    write_json("reports/regression_verified.json", {"deterministic": True, "shape": [768],
               "dtype": "float32", "gallery_count": len(gallery.records),
               "model": real_engine.info(), "ranked_ids": [r["image_id"] for r in rows]})
    for invalid in ("", "  ", "has * marker"):
        with pytest.raises(ValueError):
            real_engine.encode_query(image, invalid)


def test_real_api_search_upload_and_file_validation(real_engine, tmp_path, monkeypatch):
    c = dict(real_engine.config)
    root = tmp_path / "gallery"
    shutil.copytree(real_engine.gallery.path, root / "demo")
    c["gallery"] = {"root": str(root)}
    config_file = tmp_path / "model.yaml"
    config_file.write_text(yaml.safe_dump(c), encoding="utf-8")
    monkeypatch.setenv("KEDS_ADMIN_TOKEN", "unit-integration-token")
    with TestClient(create_app(config_file, engine=real_engine)) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/api/v1/model/info").json()["knowledge_count"] == 500000
        payload = Path("data/demo/dog.jpg").read_bytes()
        request = {"modification": "with two dogs instead of one", "gallery_id": "demo", "top_k": 3}
        response = client.post("/api/v1/search", data=request,
                               files={"image": ("reference.jpg", payload, "image/jpeg")})
        assert response.status_code == 200, response.text
        assert response.headers["x-request-id"] == response.json()["request_id"]
        output = response.json()
        assert len(output["results"]) == 3
        assert all("image_path" not in row for row in output["results"])
        for row in output["results"]:
            photo = client.get(row["image_url"])
            assert photo.status_code == 200
            with Image.open(io.BytesIO(photo.content)) as opened:
                opened.verify()
        write_json("reports/api_integration_verified.json", output)
        bad = client.post("/api/v1/search", data=request,
                          files={"image": ("looks-valid.png", b"not an image", "image/png")})
        assert bad.status_code == 422
        oversized = client.post("/api/v1/search", data=request,
                               files={"image": ("big.png", b"x" * (10 * 1024 * 1024 + 1))})
        assert oversized.status_code == 413
        traversal = client.post("/api/v1/search", data={**request, "gallery_id": "../outside"},
                                files={"image": ("x.jpg", payload)})
        assert traversal.status_code == 422
        assert client.post("/api/v1/gallery/images", data={"gallery_id": "new"},
                           files={"image": ("x.jpg", payload)}).status_code == 401
        added = client.post("/api/v1/gallery/images", data={"gallery_id": "new"},
                            files={"image": ("x.jpg", payload)},
                            headers={"x-admin-token": "unit-integration-token"})
        assert added.status_code == 200, added.text
        assert Gallery(root / "new").validate()["count"] == 1
        # Explicit opt-in must traverse hybrid knowledge, reranking, adaptive
        # streams and gallery reranking with the same real frozen checkpoints.
        advanced = client.post("/api/v1/search", data={**request, "rag_profile": "combined_experimental"},
                               files={"image": ("reference.jpg", payload, "image/jpeg")})
        assert advanced.status_code == 200, advanced.text
        advanced_output = advanced.json()
        assert advanced_output["details"]["knowledge_k_used"] in (4, 8, 16, 32)
        assert 0.1 <= advanced_output["details"]["alpha_used"] <= 0.9
        assert all(r["score_type"] == "reranking_relevance" for r in advanced_output["results"])
        write_json("reports/rag_api_verified.json", advanced_output)
        for repeat in range(2):
            cached = client.post("/api/v1/search", data={**request, "rag_profile": "exact_cache"},
                files={"image": ("reference.jpg", payload, "image/jpeg")}).json()
            assert [r["image_id"] for r in cached["results"]] == [r["image_id"] for r in output["results"]]
        assert cached["details"]["composed_cache_hit"]

