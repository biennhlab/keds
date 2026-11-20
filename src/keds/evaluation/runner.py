from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from keds.evaluation.datasets import require_full_gallery
from keds.evaluation.metrics import ranking_metrics
from keds.models.clip_encoder import rgb_image


def _batched_embeddings(engine, gallery, queries, mode, alpha, knowledge_k, rag_profile, batch_size) -> Iterator:
    def decode(query):
        return rgb_image(gallery.image_path(query.reference_id))

    with ThreadPoolExecutor(max_workers=min(8, batch_size)) as executor:
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start+batch_size]
            begin = time.perf_counter()
            values = engine.encode_queries(list(executor.map(decode, batch)),
                                          [q.modification for q in batch], mode=mode, rag_profile=rag_profile,
                                          alpha=alpha, knowledge_k=knowledge_k)
            per_query_ms = 1000 * (time.perf_counter() - begin) / len(batch)
            yield from ((value.unsqueeze(0), per_query_ms) for value in values)


def evaluate_queries(engine, gallery, queries, official_ids, mode="keds", alpha=None, knowledge_k=None,
                     rag_profile=None, candidates=None, batch_size=1):
    require_full_gallery(gallery, official_ids)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    global_rankings, subset_rankings, targets, latencies, diagnostics = [], [], [], [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batches = (batch_size > 1 and rag_profile == "E0" and candidates is None)
    if batches and (gallery.manifest["backbone"] != engine.clip.backbone or
                    gallery.manifest["clip_revision"] != engine.clip.revision):
        raise ValueError("Gallery CLIP version mismatch")
    encoded = _batched_embeddings(engine, gallery, queries, mode, alpha, knowledge_k, rag_profile, batch_size)
    for q in queries:
        if q.reference_id not in gallery.by_id or (q.target_id and q.target_id not in gallery.by_id):
            raise ValueError("Missing query reference/target in full gallery")
        if q.group_members and not set(q.group_members) <= gallery.by_id.keys():
            raise ValueError("CIRR group IDs missing from full gallery")
        start = time.perf_counter()
        # No target or group information enters the composed query.
        candidate_ids = candidates.get(q.query_id) if candidates is not None else None
        if candidate_ids is not None and q.target_id not in candidate_ids:
            raise ValueError("Target missing from supplied Candidate-1500 pool; do not repair pools using labels")
        excluded = [q.reference_id] if q.exclude_reference else []
        encoding_ms = 0
        if batches:
            embedding, encoding_ms = next(encoded)
            start = time.perf_counter()
            ranking = gallery.search(embedding, len(gallery.records), excluded, include_image_paths=False)
        else:
            ranking = engine.retrieve(gallery.image_path(q.reference_id), q.modification,
                gallery=gallery, top_k=len(gallery.records), mode=mode, alpha=alpha, knowledge_k=knowledge_k,
                rag_profile=rag_profile, exclude_ids=excluded, exclude_reference=False,
                candidate_ids=candidate_ids, include_image_paths=False)["results"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(encoding_ms + (time.perf_counter() - start) * 1000)
        ids = [r["image_id"] for r in ranking]
        global_rankings.append(ids)
        if q.group_members:
            members = set(q.group_members) - {q.reference_id}
            subset_rankings.append([i for i in ids if i in members])
        if q.target_id:
            targets.append(q.target_id)
        diagnostics.append({"query_id": q.query_id, "reference_id": q.reference_id,
                            "target_id": q.target_id, "top_10": ids[:10],
                            "target_rank": ids.index(q.target_id) + 1 if q.target_id else None,
                            "latency_ms": latencies[-1]})
        if batches and len(diagnostics) % 256 == 0:
            print(f"{gallery.gallery_id} {mode}: {len(diagnostics)}/{len(queries)}", flush=True)
    if not latencies:
        raise ValueError("No evaluation queries")
    metrics = ranking_metrics(global_rankings, targets) if len(targets) == len(queries) else None
    subset = ranking_metrics(subset_rankings, targets, ks=(1, 2, 3)) if subset_rankings and metrics else None
    return {"mode": mode, "alpha": alpha, "knowledge_k": knowledge_k, "rag_profile": rag_profile,
            "protocol": "candidate1500" if candidates is not None else "full_gallery",
            "candidate_manifest_sha256": candidates.sha256 if candidates is not None else None,
            "metrics": metrics, "subset_metrics": subset, "diagnostics": diagnostics,
            "batch_size": batch_size if batches else 1,
            "latency_estimator": "amortized_batch_encoding_plus_search" if batches else "sequential_end_to_end",
            "latency_ms": {"p50": float(np.percentile(latencies, 50)),
                           "p95": float(np.percentile(latencies, 95))},
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "gallery_version": gallery.version, "gallery_count": len(gallery.records),
            "reference_policy": "remove_by_id" if queries[0].exclude_reference else "keep_full_gallery",
            "predictions": {q.query_id: r[:50] for q, r in zip(queries, global_rankings)},
            "subset_predictions": {q.query_id: r[:3] for q, r in zip(queries, subset_rankings)}}

