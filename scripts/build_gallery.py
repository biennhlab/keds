from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from keds.gallery.repository import add_images
from keds.models.clip_encoder import CLIPEncoder
from keds.utils.config import load_config
from keds.utils.io import read_json


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build/append a real image gallery using frozen CLIP")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ids", help="JSON array of benchmark image IDs; preserves IDs and order")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    c = load_config(args.config)
    ids = read_json(args.ids) if args.ids else None
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(x for x in Path(args.input).rglob("*") if x.suffix.lower() in extensions)
    if ids is not None:
        by_name = {}
        for path in paths:
            if path.stem in by_name:
                raise ValueError(f"Ambiguous benchmark ID: {path.stem}")
            by_name[path.stem] = path
        missing = [i for i in ids if i not in by_name]
        if missing:
            raise ValueError(f"Missing benchmark gallery images: {missing[:10]}")
        paths = [by_name[i] for i in ids]
    encoder = CLIPEncoder(**c["model"])
    gallery = add_images(args.output, paths, encoder, ids, args.batch_size)
    print(gallery.validate())

