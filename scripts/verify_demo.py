"""Build and measure a small Fashion-IQ demo using real images and BEST inference."""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from keds.evaluation.datasets import fashioniq, require_full_gallery
from keds.evaluation.metrics import ranking_metrics
from keds.gallery.repository import Gallery, add_images
from keds.inference.engine import KEDsEngine
from keds.utils.config import load_config
from keds.utils.io import write_json
from scripts.import_fashioniq import resolve_images

CATEGORIES = ("dress", "shirt", "toptee")


def pick_cases(queries, count):
    groups = defaultdict(list)
    for query in queries:
        groups[query.reference_id].append(query)
    ordered = list(groups.values())
    Random(999).shuffle(ordered)
    pairs = [group for group in ordered if len({q.target_id for q in group}) > 1]
    if not pairs:
        raise ValueError("No same-reference change pair in the annotations")
    pair = pairs[0]
    chosen = [pair[0], next(q for q in pair[1:] if q.target_id != pair[0].target_id)]
    for pattern in (r"\b(blue|red|white|black|green)\b", r"\b(sleeves?|longer|shorter|neck)\b",
                    r"\b(stripes?|floral|pattern|buttons?|print)\b"):
        query = next(q for group in ordered for q in group
                     if q.reference_id not in {x.reference_id for x in chosen}
                     and re.search(pattern, q.modification, flags=re.IGNORECASE))
        chosen.append(query)
    used = {q.reference_id for q in chosen}
    chosen.extend(group[0] for group in ordered if group[0].reference_id not in used)
    return chosen[:count]


def run(config_path, data_root, output, gallery_size, query_count, batch_size, full_gallery=False):
    config = load_config(config_path)
    engine = KEDsEngine(config_path)
    root = Path(config["gallery"]["root"])
    protocol = "full_gallery_sampled_queries" if full_gallery else "label_anchored_demo_subset_not_benchmark"
    report = {"protocol": protocol, "seed": 999,
              "model": engine.info(), "categories": {}}
    for category in CATEGORIES:
        queries, official_ids, provenance = fashioniq(data_root, category)
        queries = pick_cases(queries, query_count)
        if full_gallery:
            ids, location = official_ids, root / category
        else:
            required = sorted({v for q in queries for v in (q.reference_id, q.target_id)})
            others = sorted(set(official_ids) - set(required))
            ids = required + Random(999).sample(others, gallery_size - len(required))
            location = root / f"fashion-demo-{category}-{gallery_size}"
        print(f"{category}: checking fixed gallery of {len(ids)} real photos", flush=True)
        if full_gallery or (location / "CURRENT.json").exists():
            gallery = Gallery(location)
            require_full_gallery(gallery, ids)
        else:
            paths = resolve_images(Path(data_root) / "images", ids)
            gallery = add_images(location, paths, engine.clip, image_ids=ids, batch_size=batch_size)
        gallery_info = {"gallery_id": gallery.gallery_id, "count": len(ids), "version": gallery.version}
        if not full_gallery:
            gallery_info.update(gallery.validate())
        report["categories"][category] = {**provenance, "gallery": gallery_info, "methods": {}}
        for mode in ("keds", "image", "fusion"):
            rankings, cases = [], []
            for query in queries:
                result = engine.retrieve(gallery.image_path(query.reference_id), query.modification,
                                         gallery=gallery, top_k=len(ids), mode=mode, rag_profile="E0",
                                         exclude_reference=not full_gallery, include_image_paths=False)
                ranking = [row["image_id"] for row in result["results"]]
                rankings.append(ranking)
                cases.append({"query_id": query.query_id, "reference_id": query.reference_id,
                              "modification": query.modification, "target_id": query.target_id,
                              "target_rank": ranking.index(query.target_id) + 1,
                              "top_5": [{k: row[k] for k in ("rank", "image_id", "score")}
                                        for row in result["results"][:5]],
                              "inference_ms": result["details"]["inference_ms"],
                              "search_ms": result["details"]["search_ms"]})
            metrics = ranking_metrics(rankings, [q.target_id for q in queries],
                                      ks=(1, 5, 10, 50) if full_gallery else (1, 5, 10))
            report["categories"][category]["methods"][mode] = {
                "metrics": metrics, "same_reference_ranking_changed": rankings[0] != rankings[1],
                "same_reference_top_5_changed": rankings[0][:5] != rankings[1][:5], "cases": cases}
            print(f"{category} {mode}: {metrics}", flush=True)
        write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-root", default="data/fashioniq")
    parser.add_argument("--gallery-size", type=int, default=32)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--full-gallery", action="store_true", help="Use existing official full gallery indexes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="reports/demo_quality.json")
    args = parser.parse_args()
    if args.queries < 5 or (not args.full_gallery and args.gallery_size < 2 * args.queries):
        parser.error("Need at least five queries/category and room for all reference/target images")
    torch.set_num_threads(args.threads)
    run(args.config, args.data_root, args.output, args.gallery_size, args.queries, args.batch_size, args.full_gallery)
