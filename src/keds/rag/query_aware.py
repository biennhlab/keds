from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from keds.utils.tensors import normalized_numpy
from .adaptive import choose_k
from .fusion import normalize_scores, reciprocal_rank_fusion
from .knowledge_reranking import KnowledgeReranker


@dataclass
class KnowledgeSelection:
    images: torch.Tensor  # [1,K,D]; no padding/mask changes to trained CrossFormer.
    texts: torch.Tensor
    details: dict


class QueryAwareKnowledge:
    def __init__(self, retriever, config):
        self.retriever, self.db, self.config = retriever, retriever.db, config

    @torch.no_grad()
    def retrieve(self, visual, modification, top_k=16):
        qv, qt = normalized_numpy(visual, self.db.dim), normalized_numpy(modification, self.db.dim)
        if qv.shape != qt.shape:
            raise ValueError("One modification embedding is required for each reference image")
        c = self.config
        hybrid = c.get("query_aware", {})
        rerank = c.get("knowledge_reranking", {})
        adaptive = c.get("adaptive_k", {})
        pool = min(int(hybrid.get("candidate_pool", 64)), self.db.size)
        required = 32 if adaptive.get("enabled", False) else top_k
        if pool < required:
            raise ValueError("Knowledge candidate pool must cover requested/adaptive K")
        start = perf_counter()
        routes = {}
        for modality in ("image", "text"):
            routes[(modality, "visual")] = self.retriever.search(visual, pool, modality)
            if hybrid.get("enabled", False):
                routes[(modality, "modification")] = self.retriever.search(modification, pool, modality)
        results = []
        for batch in range(len(qv)):
            candidates = {}
            for modality in ("image", "text"):
                visual_ids, visual_scores = routes[(modality, "visual")]
                ranked = [visual_ids[batch].tolist()]
                if (modality, "modification") in routes:
                    ranked.append(routes[(modality, "modification")][0][batch].tolist())
                ids = np.unique(np.concatenate(ranked))
                images = normalized_numpy(self.db.rows("image", ids))
                texts = normalized_numpy(self.db.rows("text", ids))
                channels = np.stack([images @ qv[batch], texts @ qv[batch],
                                     images @ qt[batch], texts @ qt[batch]], axis=1)
                if hybrid.get("enabled", False):
                    default = [0.55, 0.15, 0.2, 0.1] if modality == "image" else [0.15, 0.55, 0.1, 0.2]
                    weights = np.asarray(hybrid.get(f"{modality}_weights", default))
                    if weights.shape != (4,) or (weights < 0).any() or weights.sum() <= 0:
                        raise ValueError("Hybrid fusion requires four nonnegative channel weights")
                    normalized = np.stack([normalize_scores(channels[:, i], hybrid.get("normalization", "minmax"))
                                           for i in range(4)], 1)
                    weighted = normalized @ weights / weights.sum()
                    rrf = reciprocal_rank_fusion(ranked, hybrid.get("route_weights", [0.7, 0.3]),
                                                 hybrid.get("rrf_constant", 60))
                    rrf_scores = normalize_scores([rrf.get(int(i), 0.0) for i in ids])
                    method = hybrid.get("method", "hybrid")
                    if method == "weighted":
                        scores = weighted
                    elif method == "rrf":
                        scores = rrf_scores
                    elif method == "hybrid":
                        ratio = hybrid.get("rrf_weight", 0.5)
                        if not 0 <= ratio <= 1:
                            raise ValueError("rrf_weight must be in [0,1]")
                        scores = ratio * rrf_scores + (1-ratio) * weighted
                    else:
                        raise ValueError("Hybrid method must be weighted, rrf or hybrid")
                else:
                    scores = images @ qv[batch] if modality == "image" else texts @ qv[batch]
                order = np.lexsort((ids, -scores))
                candidates[modality] = (ids, images, texts, scores, order)
            k, confidence = (choose_k(candidates["image"][3][candidates["image"][4]], adaptive, self.db.size)
                             if adaptive.get("enabled", False) else (top_k, None))
            trace = {"k_used": k, "confidence": confidence, "candidate_pool": pool,
                     "query_aware": hybrid.get("enabled", False), "reranking": rerank.get("enabled", False)}
            tensors = []
            for modality in ("image", "text"):
                ids, images, texts, scores, order = candidates[modality]
                if rerank.get("enabled", False):
                    chosen, scores, _ = KnowledgeReranker(rerank).rank(
                        scores, images, texts, qv[batch], qt[batch], k, modality, ids)
                else:
                    chosen = order[:k]
                chosen_ids = ids[chosen]
                tensors.append(F.normalize(self.db.rows(modality, chosen_ids), dim=-1).unsqueeze(0).to(visual.device))
                trace[modality] = {"indices": [chosen_ids.tolist()], "scores": [scores[chosen].tolist()]}
                if self.db.captions is not None:
                    trace[modality]["captions"] = [[self.db.captions[int(i)] for i in chosen_ids]]
            trace["retrieval_ms"] = (perf_counter() - start) * 1000
            results.append(KnowledgeSelection(tensors[0], tensors[1], trace))
        return results

