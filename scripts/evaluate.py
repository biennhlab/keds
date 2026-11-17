from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from keds.evaluation.datasets import cirr, fashioniq
from keds.evaluation.metrics import category_macro
from keds.evaluation.runner import evaluate_queries
from keds.gallery.repository import Gallery
from keds.inference.engine import MODES, KEDsEngine
from keds.utils.io import write_json


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full-gallery Fashion-IQ/CIRR evaluation; never samples gallery")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--dataset", choices=("fashioniq", "cirr"), required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--gallery-root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--mode", choices=MODES, default="keds")
    p.add_argument("--alpha", type=float)
    p.add_argument("--knowledge-k", type=int)
    p.add_argument("--ablation", action="store_true", help="6 baselines/streams + alpha/K sweeps on val only")
    p.add_argument("--output", default="reports/evaluation_report.json")
    args = p.parse_args()
    if args.ablation and args.split != "val":
        p.error("Ablations/tuning are allowed on validation only")
    torch.manual_seed(999)
    torch.set_num_threads(2)
    # Parse/verify data first so missing benchmark data never triggers expensive model loading.
    groups = {}
    for name in (("dress", "shirt", "toptee") if args.dataset == "fashioniq" else ("cirr",)):
        queries, ids, provenance = (fashioniq(args.data_root, name, args.split) if args.dataset == "fashioniq"
                                   else cirr(args.data_root, args.split))
        gallery = Gallery(Path(args.gallery_root) / name)
        groups[name] = (queries, ids, provenance, gallery)
    engine = KEDsEngine(args.config)
    experiments = [(args.mode, args.alpha, args.knowledge_k)]
    if args.ablation:
        experiments = [(m, 0.5, 16) for m in MODES]
        experiments += [("keds", a, 16) for a in (0.0, 0.25, 0.75, 1.0)]
        experiments += [("keds", 0.5, k) for k in (4, 8, 32)]
    report = {"dataset": args.dataset, "split": args.split, "seed": 999,
              "model": engine.info(), "config": engine.config, "experiments": [],
              "overlap_audit": "Not verified: benchmark hashes unavailable in knowledge metadata"}
    for mode, alpha, k in experiments:
        result = {"mode": mode, "alpha": alpha, "knowledge_k": k, "categories": {}}
        for name, (queries, ids, provenance, gallery) in groups.items():
            values = evaluate_queries(engine, gallery, queries, ids, mode, alpha, k)
            values["provenance"] = provenance
            result["categories"][name] = values
        if args.dataset == "fashioniq":
            result["macro_metrics"] = category_macro({n: v["metrics"] for n, v in result["categories"].items()})
        report["experiments"].append(result)
        write_json(args.output, report)
        if args.dataset == "cirr" and args.split == "test1":
            values = result["categories"]["cirr"]
            for label, field in (("global", "predictions"), ("subset", "subset_predictions")):
                write_json(Path(args.output).with_name(f"cirr_{label}_submission.json"),
                           {"version": "rc2", "metric": "recall_subset" if label == "subset" else "recall",
                            **values[field]})
    print(f"Wrote actual evaluation output to {args.output}")

