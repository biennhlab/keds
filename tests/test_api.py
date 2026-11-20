import numpy as np
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from keds.api.main import create_app
from keds.gallery.repository import commit_gallery, image_digest, remove_images
from keds.utils.config import load_config


def test_health_readiness_and_no_fake_service_without_artifacts():
    with TestClient(create_app(load_on_startup=False)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/api/v1/model/info").status_code == 503
        response = client.post("/api/v1/search", data={"modification": "blue", "gallery_id": "demo"},
                               files={"image": ("x.png", b"invalid", "image/png")})
        assert response.status_code == 503
        assert client.post("/api/v1/gallery/images").status_code in (401, 403)


def test_gallery_images_are_bound_to_the_ranked_snapshot(tmp_path):
    config = load_config()
    root = tmp_path / "gallery"
    config["gallery"]["root"] = str(root)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    path = root / "valid"
    (path / "images").mkdir(parents=True)
    records = []
    for name, color in (("first", "blue"), ("second", "red")):
        image = Image.new("RGB", (4, 4), color)
        image.save(path / "images" / f"{name}.png")
        records.append({"image_id": name, "image_path": f"images/{name}.png", "pixel_sha256": image_digest(image)})
    original = commit_gallery(path, records, np.eye(2, dtype=np.float32),
                              config["model"]["backbone"], config["model"]["revision"])
    (root / "broken").mkdir()
    (root / "broken/CURRENT.json").write_text("invalid", encoding="utf-8")
    with TestClient(create_app(config_path, load_on_startup=False)) as client:
        assert [row["gallery_id"] for row in client.get("/api/v1/gallery").json()] == ["valid"]
        endpoint = "/api/v1/gallery/valid/images/first"
        assert client.get(endpoint).status_code == 200
        remove_images(path, ["first"])
        assert client.get(endpoint).status_code == 404
        pinned = client.get(endpoint, params={"version": original.version})
        assert pinned.status_code == 200
        assert pinned.content == original.image_path("first").read_bytes()

