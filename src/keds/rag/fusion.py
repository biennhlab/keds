import numpy as np


def normalize_scores(scores, method="minmax"):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite")
    if not len(scores):
        return scores
    if method == "minmax":
        span = np.ptp(scores)
        return (scores - scores.min()) / span if span > 1e-12 else np.zeros_like(scores)
    if method == "zscore":
        z = (scores - scores.mean()) / max(scores.std(), 1e-12)
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    if method == "none":
        return scores
    raise ValueError(f"Unknown score normalization: {method}")


def reciprocal_rank_fusion(rankings, weights=None, constant=60):
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("RRF weights must be aligned, nonnegative and nonzero")
    scores = {}
    for ranking, weight in zip(rankings, weights):
        if len(set(ranking)) != len(ranking):
            raise ValueError("A ranking cannot contain duplicate IDs")
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (constant + rank)
    return scores


def mmr_select(relevance, embeddings, k, diversity_lambda=0.8, identifiers=None):
    scores, features = np.asarray(relevance), np.asarray(embeddings, dtype=np.float32)
    if not 0 <= diversity_lambda <= 1 or not 1 <= k <= len(scores):
        raise ValueError("Invalid MMR lambda/K")
    identifiers = np.arange(len(scores)) if identifiers is None else np.asarray(identifiers)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    selected, redundancy = [], np.full(len(scores), -1.0)
    available = np.ones(len(scores), dtype=bool)
    for _ in range(k):
        objective = scores if not selected else diversity_lambda * scores - (1-diversity_lambda) * redundancy
        order = np.lexsort((identifiers, -np.where(available, objective, -np.inf)))
        chosen = int(order[0])
        selected.append(chosen)
        available[chosen] = False
        redundancy = np.maximum(redundancy, features @ features[chosen])
    return np.asarray(selected)

