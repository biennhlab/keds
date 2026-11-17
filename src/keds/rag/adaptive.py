import numpy as np


def retrieval_confidence(scores, temperature=0.1):
    values = np.asarray(scores, dtype=np.float64)[:32]
    if len(values) < 2:
        return {"confidence": 1.0, "entropy": 0.0, "gap": 1.0}
    logits = (values - values.max()) / max(temperature, 1e-6)
    probability = np.exp(logits)
    probability /= probability.sum()
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12)))) / np.log(len(values))
    gap = float((values[0] - values[min(3, len(values)-1)]) / max(np.ptp(values), 1e-12))
    confidence = np.clip(0.6 * (1-entropy) + 0.4 * gap, 0, 1)
    return {"confidence": float(confidence), "entropy": float(entropy), "gap": gap}


def choose_k(scores, config, available):
    stats = retrieval_confidence(scores, config.get("temperature", 0.1))
    thresholds = config.get("thresholds", [0.75, 0.5, 0.25])
    choices = config.get("choices", [4, 8, 16, 32])
    if choices != [4, 8, 16, 32] or len(thresholds) != 3 or sorted(thresholds, reverse=True) != thresholds:
        raise ValueError("Adaptive K requires choices 4/8/16/32 and descending confidence thresholds")
    selected = choices[-1]
    for threshold, k in zip(thresholds, choices):
        if stats["confidence"] >= threshold:
            selected = k
            break
    if selected > available:
        selected = max((k for k in choices if k <= available), default=0)
    if selected == 0:
        raise ValueError("Adaptive K requires at least four available knowledge rows")
    return selected, stats


def adaptive_alpha(feature_m, feature_a, neighbors, config, prior=0.5):
    bank = np.asarray(neighbors, dtype=np.float64)
    cm = retrieval_confidence(np.sort(bank @ feature_m)[::-1], config.get("temperature", 0.1))
    ca = retrieval_confidence(np.sort(bank @ feature_a)[::-1], config.get("temperature", 0.1))
    low, high = config.get("bounds", [0.1, 0.9])
    if not 0 <= low <= high <= 1:
        raise ValueError("Adaptive alpha bounds must be in [0,1]")
    logit = np.log(np.clip(prior, 1e-6, 1-1e-6) / np.clip(1-prior, 1e-6, 1))
    logit += (cm["confidence"] - ca["confidence"]) / max(config.get("fusion_temperature", 0.2), 1e-6)
    alpha = float(np.clip(1/(1+np.exp(-np.clip(logit, -40, 40))), low, high))
    return alpha, {"phi_m": cm, "phi_a": ca, "prior": prior}

