from __future__ import annotations

import numpy as np


def ranking_metrics(rankings, targets, ks=(1, 5, 10, 50)):
    """Percent recall and full-ranking MRR. Targets must be present exactly once."""
    if not targets or len(rankings) != len(targets):
        raise ValueError("Non-empty aligned rankings/targets required")
    ranks = []
    for ranking, target in zip(rankings, targets):
        if len(set(ranking)) != len(ranking) or ranking.count(target) != 1:
            raise ValueError("Expected unique full ranking containing the target")
        ranks.append(ranking.index(target) + 1)
    values = np.asarray(ranks)
    return {**{f"R@{k}": float((values <= k).mean() * 100) for k in ks},
            "MRR": float((1.0 / values).mean()), "queries": len(targets)}


def category_macro(category_metrics):
    if set(category_metrics) != {"dress", "shirt", "toptee"}:
        raise ValueError("Fashion-IQ macro mean requires all three categories")
    return {k: float(np.mean([v[k] for v in category_metrics.values()]))
            for k in ("R@1", "R@5", "R@10", "R@50", "MRR")}

