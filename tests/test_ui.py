import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from keds.api.proxy import GradioUploadProgressMiddleware, PublicSchemeMiddleware


def test_gradio_interface_builds_without_loading_a_second_model():
    path = Path("app/gradio_app.py")
    spec = importlib.util.spec_from_file_location("keds_gradio", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = module.create_demo("http://127.0.0.1:8000")
    config = demo.get_config_file()
    assert sum(c["type"] == "gallery" for c in config["components"]) == 2
    assert any(c["type"] == "image" for c in config["components"])
    reference = next(c for c in config["components"] if c["type"] == "image")
    assert reference["props"]["type"] == "filepath" and reference["props"].get("image_mode") is None
    demo.close()


def test_orphaned_upload_progress_does_not_intercept_valid_upload():
    app = FastAPI()
    app.add_middleware(GradioUploadProgressMiddleware)

    @app.get("/ui/gradio_api/upload_progress")
    def progress(upload_id: str):
        return {"tracked": upload_id}

    with TestClient(app) as client:
        path = "/ui/gradio_api/upload_progress"
        orphaned = client.get(path, params={"upload_id": "undefined"})
        assert orphaned.status_code == 200
        assert orphaned.headers["content-type"].startswith("text/event-stream")
        assert orphaned.text.startswith(":") and "done" not in orphaned.text
        assert client.get(path, params={"upload_id": "live"}).json() == {"tracked": "live"}


def test_example_and_refresh_recover_without_existing_gradio_session(tmp_path, monkeypatch):
    import gradio as gr

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("keds_gradio_presets", Path("app/gradio_app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    png = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(png, format="PNG")
    galleries = [{"gallery_id": "dress", "count": 3817}]
    examples = [{"label": "dress 2", "gallery_id": "dress", "reference_id": "real",
                 "modification": "is lighter colored"},
                {"label": "missing", "gallery_id": "dress", "reference_id": "missing",
                 "modification": "another caption"}]

    def backend(request):
        match request.url.path:
            case "/api/v1/gallery":
                return httpx.Response(200, json=galleries)
            case "/api/v1/model/info":
                return httpx.Response(200, json={"model_version": "test"})
            case "/api/v1/gallery/dress/images/real":
                return httpx.Response(200, content=png.getvalue())
            case "/api/v1/gallery/dress/images/missing":
                return httpx.Response(404, json={"detail": "not found"})
        raise AssertionError(request.url)

    monkeypatch.setattr(module, "httpx", SimpleNamespace(
        Client=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(backend), **kwargs),
        HTTPError=httpx.HTTPError, TimeoutException=httpx.TimeoutException))
    demo = module.create_demo(examples=examples)
    app = gr.mount_gradio_app(FastAPI(), demo, path="/ui")
    with TestClient(app) as client:
        # The browser was already displaying `dress`; the new worker has empty choices.
        refreshed = client.post("/ui/gradio_api/run/refresh",
                                json={"data": ["dress"], "session_hash": "old-browser"})
        assert refreshed.status_code == 200, refreshed.text
        assert refreshed.json()["data"][0]["value"] == "dress"
        preset = client.post("/ui/gradio_api/run/load_example", json={"data": ["dress 2"]})
        assert preset.status_code == 200, preset.text
        preview, caption, gallery, status, *_ = preset.json()["data"]
        assert Path(preview["path"]).is_file()
        assert caption == "is lighter colored"
        assert gallery["value"] == "dress" and gallery["choices"] == [["dress · 3,817 ảnh", "dress"]]
        assert "Đã tải ví dụ" in status
        invalid = client.post("/ui/gradio_api/run/load_example", json={"data": ["missing"]})
        assert invalid.status_code == 200, invalid.text
        data = invalid.json()["data"]
        assert all(data[i] == {"__type__": "update"} for i in (0, 1, 2))
        assert "Chưa tải được ví dụ" in data[3]
    demo.close()


@pytest.mark.parametrize("forwarded_host", [False, True])
def test_proxy_https_config_and_real_gradio_upload(tmp_path, monkeypatch, forwarded_host):
    """Reproduce Modal's plain HTTP hop and verify Gradio uses HTTPS URLs."""
    import gradio as gr

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("keds_gradio_proxy", Path("app/gradio_app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = module.create_demo()
    app = gr.mount_gradio_app(FastAPI(), demo, path="/ui")
    app.add_middleware(PublicSchemeMiddleware, scheme="https")
    headers = {"x-forwarded-proto": "http"}
    if forwarded_host:
        headers["x-forwarded-host"] = "public.example"
    with TestClient(app, base_url="http://public.example") as client:
        config = client.get("/ui/config", headers=headers).json()
        assert config["root"] == "https://public.example/ui"
        assert client.get("/ui/", headers=headers).status_code == 200
        content = io.BytesIO()
        Image.new("RGB", (8, 8), color="blue").save(content, format="PNG")
        uploaded = client.post("/ui/gradio_api/upload", headers=headers,
                               files={"files": ("reference.png", content.getvalue(), "image/png")})
        assert uploaded.status_code == 200, uploaded.text
        uploaded_path = uploaded.json()[0]
        response = client.get("/ui/gradio_api/file=" + uploaded_path, headers=headers)
        assert response.status_code == 200
        assert response.content == content.getvalue()
    demo.close()

