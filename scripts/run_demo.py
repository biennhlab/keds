import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import gradio as gr
import torch
import uvicorn
from fastapi.responses import RedirectResponse

from app.gradio_app import create_demo
from keds.api.main import available_galleries, create_app
from keds.api.proxy import GradioUploadProgressMiddleware, PublicSchemeMiddleware
from keds.utils.config import load_config
from keds.utils.io import read_json


def demo_examples(report_path):
    if not report_path.is_file():
        return []
    report = read_json(report_path)
    examples = []
    for category, entry in report["categories"].items():
        for position, case in enumerate(entry["methods"]["keds"]["cases"][:5], start=1):
            examples.append({"label": f"{category} {position} · {case['modification']}",
                             "gallery_id": entry["gallery"]["gallery_id"], "reference_id": case["reference_id"],
                             "modification": case["modification"]})
    return examples

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--examples-report", type=Path, default=ROOT / "reports/demo_quality.json")
    p.add_argument("--show-experimental", action="store_true", help="Expose unvalidated RAG profiles in the UI")
    p.add_argument("--public-scheme", choices=["http", "https"],
                   help="External scheme when TLS terminates at a reverse proxy")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    torch.set_num_threads(2)
    config = load_config(args.config)
    options = {"alpha": config["inference"]["alpha"], "knowledge_k": config["knowledge"]["top_k"]}
    limit = config["api"]["max_upload_mb"]
    api = create_app(args.config)
    demo = create_demo(f"http://127.0.0.1:{args.port}", upload_mb=limit, defaults=options,
                       examples=demo_examples(args.examples_report), experimental=args.show_experimental,
                       initial_galleries=available_galleries(config))
    application = gr.mount_gradio_app(api, demo, path="/ui", max_file_size=limit * 1024**2)

    @application.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return RedirectResponse("/ui/favicon.ico")

    application.add_middleware(GradioUploadProgressMiddleware)
    if args.public_scheme:
        application.add_middleware(PublicSchemeMiddleware, scheme=args.public_scheme)
    uvicorn.run(application, host=args.host, port=args.port, workers=1)

