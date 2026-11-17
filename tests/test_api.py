from fastapi.testclient import TestClient

from keds.api.main import create_app


def test_health_readiness_and_no_fake_service_without_artifacts():
    with TestClient(create_app(load_on_startup=False)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/api/v1/model/info").status_code == 503
        response = client.post("/api/v1/search", data={"modification": "blue", "gallery_id": "demo"},
                               files={"image": ("x.png", b"invalid", "image/png")})
        assert response.status_code == 503
        assert client.post("/api/v1/gallery/images").status_code in (401, 403)

