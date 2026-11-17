import numpy as np

from .fusion import mmr_select, normalize_scores


class GalleryReranker:
    def __init__(self, config):
        self.config = config

    def rank(self, gallery, query, visual, text, top_k, exclude_ids=(), exclude_hash=None, candidate_ids=None):
        pool = max(top_k, self.config.get("candidate_pool", 100))
        initial = gallery.search(query, pool, exclude_ids, exclude_hash, candidate_ids)
        if not initial:
            return []
        n = min(len(initial), self.config.get("candidate_pool", 100))
        head, tail = initial[:n], initial[n:]
        vectors = np.asarray(gallery.embeddings[[gallery.row_by_id[r["image_id"]] for r in head]])
        components = {"composed": np.array([r["score"] for r in head]),
                      "visual": vectors @ visual, "text": vectors @ text}
        weights = self.config.get("weights", {"composed": 0.7, "visual": 0.1, "text": 0.2})
        if set(weights) != set(components) or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Invalid gallery reranking weights")
        scores = sum(weights[k] * normalize_scores(v, self.config.get("normalization", "minmax"))
                     for k, v in components.items()) / sum(weights.values())
        if self.config.get("mmr", False):
            order = mmr_select(scores, vectors, len(head), self.config.get("diversity_lambda", 0.9))
        else:
            order = np.lexsort((np.arange(len(head)), -scores))
        ranked = []
        for index in order:
            row = dict(head[index])
            row["cosine_score"] = row["score"]
            row["score"] = float(scores[index])
            row["score_type"] = "reranking_relevance"
            ranked.append(row)
        # Retain every remaining gallery image for full-ranking MRR. No target
        # or ground-truth value enters reranking or candidate selection.
        for row in tail:
            row = dict(row)
            row["cosine_score"] = row["score"]
            row["score_type"] = "cosine_tail"
            ranked.append(row)
        for i, row in enumerate(ranked):
            row["rank"] = i + 1
        return ranked[:top_k]

