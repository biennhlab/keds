import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import gradio as gr
import torch
import uvicorn

from app.gradio_app import create_demo
from keds.api.main import create_app

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    torch.set_num_threads(2)
    api = create_app(args.config)
    application = gr.mount_gradio_app(api, create_demo(f"http://127.0.0.1:{args.port}"), path="/ui")
    uvicorn.run(application, host=args.host, port=args.port, workers=1)

