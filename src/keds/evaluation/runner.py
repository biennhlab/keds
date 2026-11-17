from __future__ import annotations

import time

import numpy as np
import torch

from keds.evaluation.datasets import require_full_gallery
from keds.evaluation.metrics import ranking_metrics


def evaluate_queries(engine, gallery, queries, official_ids, mode="keds", alpha=None, knowledge_k=None):
    require_full_gallery(gallery, official_ids)
    global_rankings, subset_rankings, targets, latencies, diagnostics = [], [], [], [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for q in queries:
        if q.reference_id not in gallery.by_id or (q.target_id and q.target_id not in gallery.by_id):
            raise ValueError("Missing query reference/target in full gallery")
        if q.group_members and not set(q.group_members) <= gallery.by_id.keys():
            raise ValueError("CIRR group IDs missing from full gallery")
        start = time.perf_counter()
        # No target or group information enters the composed query.
        embedding = engine.encode_query(gallery.image_path(q.reference_id), q.modification,
                                         mode=mode, alpha=alpha, knowledge_k=knowledge_k)
        # Match official KEDs: CIRR removes reference ID; Fashion-IQ keeps the
        # complete category gallery (get_metrics_fashion does no self removal).
        ranking = gallery.search(embedding.unsqueeze(0), len(gallery.records),
                                 exclude_ids=[q.reference_id] if q.exclude_reference else [])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        ids = [r["image_id"] for r in ranking]
        global_rankings.append(ids)
        if q.group_members:
            members = set(q.group_members) - {q.reference_id}
            subset_rankings.append([i for i in ids if i in members])
        if q.target_id:
            targets.append(q.target_id)
        diagnostics.append({"query_id": q.query_id, "reference_id": q.reference_id,
                            "target_id": q.target_id, "top_10": ids[:10],
                            "target_rank": ids.index(q.target_id) + 1 if q.target_id else None})
    if not latencies:
        raise ValueError("No evaluation queries")
    metrics = ranking_metrics(global_rankings, targets) if len(targets) == len(queries) else None
    subset = ranking_metrics(subset_rankings, targets, ks=(1, 2, 3)) if subset_rankings and metrics else None
    return {"mode": mode, "alpha": alpha, "knowledge_k": knowledge_k,
            "metrics": metrics, "subset_metrics": subset, "diagnostics": diagnostics,
            "latency_ms": {"p50": float(np.percentile(latencies, 50)),
                           "p95": float(np.percentile(latencies, 95))},
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "gallery_version": gallery.version, "gallery_count": len(gallery.records),
            "reference_policy": "remove_by_id" if queries[0].exclude_reference else "keep_full_gallery",
            "predictions": {q.query_id: r[:50] for q, r in zip(queries, global_rankings)},
            "subset_predictions": {q.query_id: r[:3] for q, r in zip(queries, subset_rankings)}}

