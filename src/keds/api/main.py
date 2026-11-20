from __future__ import annotations

import io
import json
import logging
import os
import secrets
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from keds.api.schemas import SearchResponse
from keds.gallery.repository import Gallery, add_images, list_galleries, remove_images, safe_id
from keds.inference.engine import KEDsEngine
from keds.utils.config import load_config
from keds.utils.io import read_json

logger = logging.getLogger("keds.api")


def available_galleries(config):
    return [entry for entry in list_galleries(config["gallery"]["root"], skip_invalid=True)
            if entry.get("backbone") == config["model"]["backbone"]
            and entry.get("clip_revision") == config["model"]["revision"]]


def create_app(config_path=None, engine=None, load_on_startup=None):
    config = load_config(config_path)
    load_dotenv(Path(config["config_path"]).parent.parent / ".env")
    lock = threading.RLock()
    gallery_lock = threading.Lock()
    state = {"engine": engine, "error": None}
    gallery_cache = OrderedDict()

    @asynccontextmanager
    async def lifespan(app):
        if (load_on_startup if load_on_startup is not None else config["api"]["load_on_startup"]):
            if state["engine"] is None:
                try:
                    state["engine"] = KEDsEngine(config["config_path"])
                except Exception as e:
                    memory_limit = isinstance(e, MemoryError) or getattr(e, "winerror", None) == 1455
                    state["error"] = "memory_limit" if memory_limit else type(e).__name__
                    logger.exception("Model startup failed; readiness remains false")
        yield

    app = FastAPI(title="KEDs Composed Image Retrieval", version="0.1.0", lifespan=lifespan)
    app.state.runtime = state

    @app.middleware("http")
    async def request_logging(request, call_next):
        if request.url.path != "/api/v1/search":
            return await call_next(request)
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        if response.status_code >= 400:
            logger.warning(json.dumps({"request_id": request.state.request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(), "success": False,
                "status_code": response.status_code}))
        return response

    def require_engine():
        if state["engine"] is None:
            if state["error"] == "memory_limit":
                raise HTTPException(503, "Memory limit; close other model workers and restart the demo")
            raise HTTPException(503, "Model unavailable; validate artifacts and restart service")
        return state["engine"]

    def admin(x_admin_token: str | None = Header(default=None)):
        expected = os.environ.get("KEDS_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(403, "Gallery administration disabled; set KEDS_ADMIN_TOKEN")
        if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
            raise HTTPException(401, "Invalid administrator token")

    def gallery_path(gallery_id):
        try:
            safe_id(gallery_id)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        root = Path(config["gallery"]["root"]).resolve()
        path = (root / gallery_id).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(422, "Invalid gallery")
        return path

    def get_gallery(gallery_id, version=None):
        path = gallery_path(gallery_id)
        if version is not None:
            safe_id(version)
        try:
            with gallery_lock:
                generation = version or read_json(path / "CURRENT.json")["generation"]
                key = (gallery_id, generation)
                if key not in gallery_cache:
                    gallery_cache[key] = Gallery(path, version=generation)
                    if len(gallery_cache) > 4:
                        gallery_cache.popitem(last=False)
                gallery_cache.move_to_end(key)
                return gallery_cache[key]
        except FileNotFoundError as e:
            raise HTTPException(404, "Gallery or generation not found") from e
        except (OSError, KeyError, TypeError, ValueError) as e:
            logger.exception("Cannot load gallery %s", gallery_id)
            raise HTTPException(503, "Gallery index or metadata is invalid; validate or rebuild it") from e

    def decode_upload(upload):
        data = upload.file.read(config["api"]["max_upload_mb"] * 1024 * 1024 + 1)
        if len(data) > config["api"]["max_upload_mb"] * 1024 * 1024:
            raise HTTPException(413, "Image exceeds upload limit")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as checked:
                    if checked.format not in ("JPEG", "PNG", "WEBP", "BMP"):
                        raise ValueError("Unsupported image format")
                    if checked.width * checked.height > config["api"]["max_image_pixels"]:
                        raise HTTPException(413, "Image exceeds pixel limit")
                    checked.verify()
                with Image.open(io.BytesIO(data)) as checked:
                    from keds.models.clip_encoder import rgb_image
                    return rgb_image(checked)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
            raise HTTPException(413, "Image exceeds pixel limit") from e
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise HTTPException(422, "Invalid or unsupported image") from e

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        model_ready = state["engine"] is not None
        galleries = available_galleries(config)
        ready = False
        if model_ready:
            for entry in galleries:
                try:
                    get_gallery(entry["gallery_id"], entry["version"])
                except HTTPException:
                    continue
                ready = True
                break
        return JSONResponse(status_code=200 if ready else 503,
                            content={"ready": ready, "model": model_ready,
                                     "galleries": len(galleries), "error": state["error"]})

    @app.get("/api/v1/model/info")
    def model_info(model=Depends(require_engine)):
        with lock:
            return model.info()

    @app.get("/api/v1/gallery")
    def galleries():
        return available_galleries(config)

    @app.post("/api/v1/search", response_model=SearchResponse)
    def search(request: Request, image: UploadFile = File(...), modification: str = Form(...),
               gallery_id: str = Form(...), top_k: int = Form(10), mode: str = Form("keds"),
               alpha: float | None = Form(None), knowledge_k: int | None = Form(None),
               rag_profile: str | None = Form(None),
               reference_id: str | None = Form(None), model=Depends(require_engine)):
        request_id, start = request.state.request_id, time.perf_counter()
        if not 1 <= top_k <= config["api"]["max_top_k"]:
            raise HTTPException(422, "top_k out of range")
        if not modification.strip() or len(modification) > config["inference"]["max_text_length"]:
            raise HTTPException(422, "Invalid modification length")
        decoded = decode_upload(image)
        gallery = get_gallery(gallery_id)
        with lock:
            output = model.retrieve(decoded, modification, top_k=top_k, gallery=gallery,
                                    exclude_ids=[reference_id] if reference_id else [],
                                    mode=mode, alpha=alpha, knowledge_k=knowledge_k, rag_profile=rag_profile)
        for row in output["results"]:
            row.pop("image_path")
            row["image_url"] = f"/api/v1/gallery/{gallery_id}/images/{row['image_id']}?version={gallery.version}"
        output.update(request_id=request_id, latency_ms=(time.perf_counter() - start) * 1000)
        event = {k: output[k] for k in (
            "request_id", "model_version", "gallery_version", "knowledge_version", "latency_ms")}
        event.update(timestamp=datetime.now(timezone.utc).isoformat(), success=True,
                     inference_ms=output["details"]["inference_ms"], search_ms=output["details"]["search_ms"])
        logger.info(json.dumps(event))
        return output

    @app.post("/api/v1/gallery/images", dependencies=[Depends(admin)])
    def upload_image(image: UploadFile = File(...), gallery_id: str = Form(...),
                     model=Depends(require_engine)):
        decoded = decode_upload(image)
        with lock:
            gallery = add_images(gallery_path(gallery_id), [decoded], model.clip)
        return gallery.manifest

    @app.delete("/api/v1/gallery/{gallery_id}/images/{image_id}", dependencies=[Depends(admin)])
    def delete_image(gallery_id: str, image_id: str):
        with lock:
            gallery = get_gallery(gallery_id)
            if image_id not in gallery.by_id:
                raise HTTPException(404, "Image not found")
            return remove_images(gallery.path, [image_id]).manifest

    @app.get("/api/v1/gallery/{gallery_id}/images/{image_id}")
    def gallery_image(gallery_id: str, image_id: str, version: str | None = None):
        gallery = get_gallery(gallery_id, version)
        if image_id not in gallery.by_id:
            raise HTTPException(404, "Image not found")
        return FileResponse(gallery.image_path(image_id))

    return app


app = create_app()

