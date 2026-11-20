from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from keds.evaluation.datasets import cirr, fashioniq
from keds.evaluation.candidates import CandidateManifest
from keds.evaluation.metrics import category_macro
from keds.evaluation.runner import evaluate_queries
from keds.gallery.repository import Gallery
from keds.inference.engine import MODES, KEDsEngine
from keds.utils.io import read_json, sha256, write_json


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
    p.add_argument("--batch-size", type=int, default=1, help="Batch CLIP and fixed-K E0 on full galleries")
    p.add_argument("--ablation", action="store_true", help="6 baselines/streams + alpha/K sweeps on val only")
    p.add_argument("--rag-profile", default="E0")
    p.add_argument("--rag-profiles", help="Comma-separated research profiles; development partition required")
    p.add_argument("--protocol", choices=("full", "candidate1500"), default="full")
    p.add_argument("--candidates", help="Frozen JSONL: query_id and exactly 1500 unique candidate_ids")
    p.add_argument("--partitions", help="Fixed development/holdout query-ID partition manifest")
    p.add_argument("--phase", choices=("all", "development", "holdout"), default="all")
    p.add_argument("--output", default="reports/evaluation_report.json")
    args = p.parse_args()
    if args.ablation and args.split != "val":
        p.error("Ablations/tuning are allowed on validation only")
    profiles = args.rag_profiles.split(",") if args.rag_profiles else [args.rag_profile]
    if (args.ablation or len(profiles) > 1) and (args.phase != "development" or not args.partitions):
        p.error("Hyperparameter/technique sweeps require a fixed development partition")
    if args.phase != "all" and not args.partitions:
        p.error("Partitioned evaluation requires --partitions")
    if args.protocol == "candidate1500" and (args.dataset != "fashioniq" or not args.candidates):
        p.error("Candidate-1500 requires Fashion-IQ and an externally supplied frozen --candidates manifest")
    torch.manual_seed(999)
    torch.set_num_threads(2)
    # Parse/verify data first so missing benchmark data never triggers expensive model loading.
    groups = {}
    for name in (("dress", "shirt", "toptee") if args.dataset == "fashioniq" else ("cirr",)):
        queries, ids, provenance = (fashioniq(args.data_root, name, args.split) if args.dataset == "fashioniq"
                                   else cirr(args.data_root, args.split))
        gallery = Gallery(Path(args.gallery_root) / name)
        groups[name] = (queries, ids, provenance, gallery)
    all_queries = [q.query_id for queries, _, _, _ in groups.values() for q in queries]
    candidates = None
    if args.protocol == "candidate1500":
        galleries = {}
        for queries, ids, _, _ in groups.values():
            allowed = set(ids)
            galleries.update({q.query_id: allowed for q in queries})
        candidates = CandidateManifest(args.candidates, all_queries, galleries)
    partition = read_json(args.partitions) if args.partitions else None
    if partition:
        if set(partition["assignments"]) != set(all_queries):
            p.error("Partition IDs do not exactly match official query IDs")
        if not set(partition["assignments"].values()) <= {"development", "holdout"}:
            p.error("Unknown partition label")
        if args.phase != "all":
            groups = {name: ([q for q in queries if partition["assignments"][q.query_id] == args.phase],
                             ids, provenance, gallery) for name, (queries, ids, provenance, gallery) in groups.items()}
    engine = KEDsEngine(args.config)
    batch_parity = None
    if args.batch_size > 1 and args.rag_profile == "E0" and candidates is None:
        queries, _, _, gallery = next(iter(groups.values()))
        first = queries[:2]
        paths = [gallery.image_path(q.reference_id) for q in first]
        texts = [q.modification for q in first]
        options = {"mode": args.mode, "alpha": args.alpha, "knowledge_k": args.knowledge_k, "rag_profile": "E0"}
        batched = engine.encode_queries(paths, texts, **options)
        single = torch.stack([engine.encode_query(path, text, **options) for path, text in zip(paths, texts)])
        torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-4)
        def rank_ids(embedding):
            return [r["image_id"] for r in gallery.search(embedding.unsqueeze(0), 10)]

        for batch_vector, one_vector in zip(batched, single):
            if rank_ids(batch_vector) != rank_ids(one_vector):
                raise ValueError("Batch inference altered preflight top-10 ranking")
        batch_parity = {"query_ids": [q.query_id for q in first], "top10_equal": True,
                        "max_abs_error": float((batched - single).abs().max())}
        print(f"Batch parity: {batch_parity}", flush=True)
    experiments = [(args.mode, args.alpha, args.knowledge_k)]
    if args.ablation:
        experiments = [(m, 0.5, 16) for m in MODES]
        experiments += [("keds", a, 16) for a in (0.0, 0.25, 0.75, 1.0)]
        experiments += [("keds", 0.5, k) for k in (4, 8, 32)]
    experiments = [(profile, *e) for profile in profiles for e in experiments]
    report = {"dataset": args.dataset, "split": args.split, "seed": 999,
              "protocol": args.protocol, "phase": args.phase,
              "partitions_sha256": sha256(args.partitions) if args.partitions else None,
              "historical_tuning_overlap_verified": partition.get("historical_tuning_overlap_verified", False) if partition else False,
              "query_total": len(all_queries),
              "model": engine.info(), "config": engine.config, "experiments": [], "batch_parity": batch_parity,
              "overlap_audit": "Not verified: benchmark hashes unavailable in knowledge metadata"}
    for profile, mode, alpha, k in experiments:
        if profile not in engine.info()["available_rag_profiles"]:
            p.error(f"Unknown RAG profile: {profile}")
        engine.caches.clear()
        result = {"rag_profile": profile, "rag_settings": engine.settings(profile),
                  "mode": mode, "alpha": alpha, "knowledge_k": k, "categories": {}}
        for name, (queries, ids, provenance, gallery) in groups.items():
            values = evaluate_queries(engine, gallery, queries, ids, mode, alpha, k, profile, candidates,
                                      batch_size=args.batch_size)
            values["provenance"] = provenance
            result["categories"][name] = values
            print(f"{name} {profile}/{mode}: {values['metrics']}", flush=True)
        if args.dataset == "fashioniq":
            result["macro_metrics"] = category_macro({n: v["metrics"] for n, v in result["categories"].items()})
        report["experiments"].append(result)
        report["last_cache_stats"] = engine.info()["cache_stats"]
        write_json(args.output, report)
        if args.dataset == "cirr" and args.split == "test1":
            values = result["categories"]["cirr"]
            for label, field in (("global", "predictions"), ("subset", "subset_predictions")):
                write_json(Path(args.output).with_name(f"cirr_{label}_submission.json"),
                           {"version": "rc2", "metric": "recall_subset" if label == "subset" else "recall",
                            **values[field]})
    print(f"Wrote actual evaluation output to {args.output}")

