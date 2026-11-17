import numpy as np


def paired_bootstrap(baseline_ranks, variant_ranks, groups=None, seed=999, repetitions=2000):
    baseline = np.asarray(baseline_ranks, dtype=np.float64)
    variant = np.asarray(variant_ranks, dtype=np.float64)
    if baseline.shape != variant.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("Paired bootstrap needs aligned non-empty query ranks")
    if (baseline < 1).any() or (variant < 1).any() or not np.isfinite(baseline + variant).all():
        raise ValueError("Ranks must be finite and positive")
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    groups = np.zeros(len(baseline), dtype=int) if groups is None else np.asarray(groups)
    if groups.shape != baseline.shape:
        raise ValueError("Bootstrap category labels must align with query IDs")
    strata = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    names = ["R@1", "R@5", "R@10", "R@50", "MRR"]
    delta = np.stack([(variant <= k).astype(float)*100 - (baseline <= k).astype(float)*100
                      for k in (1, 5, 10, 50)] + [1/variant - 1/baseline], 1)
    observed = np.mean([delta[s].mean(0) for s in strata], axis=0)
    random = np.random.default_rng(seed)
    samples = np.empty((repetitions, len(names)))
    for repeat in range(repetitions):
        samples[repeat] = np.mean([delta[random.choice(s, len(s), replace=True)].mean(0) for s in strata], 0)
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    return {name: {"delta": float(observed[i]), "ci95": [float(lower[i]), float(upper[i])],
                   "ci_excludes_zero": bool(lower[i] > 0 or upper[i] < 0),
                   "conclusion": "improvement" if lower[i] > 0 else "regression" if upper[i] < 0 else "uncertain"}
            for i, name in enumerate(names)}


def compare_experiments(baseline, variant, seed=999, repetitions=2000):
    if set(baseline["categories"]) != set(variant["categories"]):
        raise ValueError("Experiments must contain the same categories")
    ranks_a, ranks_b, groups = [], [], []
    for category in baseline["categories"]:
        a, b = baseline["categories"][category], variant["categories"][category]
        for key in ("gallery_version", "gallery_count", "provenance", "reference_policy", "protocol", "candidate_manifest_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Cannot pair different protocols/artifacts: {key}")
        rows_a = {q["query_id"]: q for q in a["diagnostics"]}
        rows_b = {q["query_id"]: q for q in b["diagnostics"]}
        if set(rows_a) != set(rows_b):
            raise ValueError("Paired comparison requires identical query IDs")
        for query_id in sorted(rows_a):
            qa, qb = rows_a[query_id], rows_b[query_id]
            if qa["reference_id"] != qb["reference_id"] or qa["target_id"] != qb["target_id"]:
                raise ValueError("Query/label identity differs")
            ranks_a.append(qa["target_rank"])
            ranks_b.append(qb["target_rank"])
            groups.append(category)
    return paired_bootstrap(ranks_a, ranks_b, groups, seed, repetitions)

