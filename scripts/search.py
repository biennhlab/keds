import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from keds.inference.engine import MODES, KEDsEngine
from keds.utils.io import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Image + modification → actual gallery Top-K")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--image", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--gallery", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--mode", choices=MODES, default="keds")
    p.add_argument("--alpha", type=float)
    p.add_argument("--knowledge-k", type=int)
    p.add_argument("--output")
    args = p.parse_args()
    torch.set_num_threads(2)
    engine = KEDsEngine(args.config)
    engine.load_gallery(args.gallery)
    output = engine.retrieve(args.image, args.text, args.top_k, mode=args.mode,
                             alpha=args.alpha, knowledge_k=args.knowledge_k)
    if args.output:
        write_json(args.output, output)
    print(json.dumps(output, indent=2))

