"""Functional/latency ablation on real user-supplied images, without accuracy claims.

For Recall/MRR use evaluate.py with complete benchmark images and fixed IDs.
This script never assigns ground-truth labels to the six-image smoke gallery.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from keds.inference.engine import KEDsEngine
from keds.utils.io import read_json, write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--image", default="data/demo/dog.jpg")
    p.add_argument("--text", default="with two dogs instead of one")
    p.add_argument("--gallery", default="artifacts/gallery/demo")
    p.add_argument("--profiles", default="E0,hybrid_weighted,hybrid_rrf,knowledge_mmr,adaptive_k,adaptive_fusion,gallery_rerank,combined_experimental,exact_cache")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--output", default="reports/rag_functional_profile.json")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(999)
    engine = KEDsEngine(args.config)
    gallery = engine.load_gallery(args.gallery)
    report = {"purpose": "functional_smoke_and_latency_only", "is_benchmark": False,
              "seed": 999, "image": args.image, "modification": args.text,
              "gallery_count": len(gallery.records), "model": engine.info(), "profiles": {}}
    oracle = read_json("tests/fixtures/e0_regression.json")
    for profile in args.profiles.split(","):
        times, runs = [], []
        engine.caches.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(args.repeats):
            start = time.perf_counter()
            output = engine.retrieve(args.image, args.text, top_k=3, rag_profile=profile)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
            assert len(output["results"]) == 3
            assert all(Path(r["image_path"]).is_file() for r in output["results"])
            if profile in ("E0", "exact_cache") and args.image == oracle["reference"] and args.text == oracle["modification"]:
                assert [r["image_id"] for r in output["results"]] == [r["image_id"] for r in oracle["results"]]
                np.testing.assert_allclose([r["score"] for r in output["results"]],
                                           [r["score"] for r in oracle["results"]], atol=1e-5)
            runs.append({"ranked_ids": [r["image_id"] for r in output["results"]],
                         "scores": [r["score"] for r in output["results"]], "details": output["details"]})
        report["profiles"][profile] = {"runs": runs, "latency_ms": {
            "samples": times, "p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))},
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "accuracy_metrics": None, "accuracy_status": "not_run_missing_labeled_full_benchmark"}
        write_json(args.output, report)
        print(profile, "actual Top-3 OK", report["profiles"][profile]["latency_ms"], flush=True)

