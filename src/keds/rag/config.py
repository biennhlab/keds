import copy
import hashlib
import json


DEFAULTS = {
    "profile": "E0",
    "query_aware": {"enabled": False, "method": "hybrid", "candidate_pool": 64,
                    "normalization": "minmax", "rrf_constant": 60, "rrf_weight": 0.5,
                    "route_weights": [0.7, 0.3]},
    "knowledge_reranking": {"enabled": False, "normalization": "minmax", "mmr": False,
                            "diversity_lambda": 0.8},
    "adaptive_k": {"enabled": False, "choices": [4, 8, 16, 32],
                   "thresholds": [0.75, 0.5, 0.25], "temperature": 0.1},
    "adaptive_fusion": {"enabled": False, "bounds": [0.1, 0.9], "temperature": 0.1,
                        "fusion_temperature": 0.2},
    "gallery_reranking": {"enabled": False, "candidate_pool": 100, "mmr": False,
                           "diversity_lambda": 0.9, "normalization": "minmax",
                           "weights": {"composed": 0.7, "visual": 0.1, "text": 0.2}},
    "cache": {"enabled": False, "max_entries": 128, "max_mb": 64, "ttl_seconds": 900,
              "semantic": False, "semantic_threshold": 0.9995},
}


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rag_config(values=None):
    c = merge(DEFAULTS, values or {})
    unknown = set(c) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown RAG sections: {sorted(unknown)}")
    for section in DEFAULTS:
        if section != "profile" and not isinstance(c[section].get("enabled"), bool):
            raise ValueError(f"{section}.enabled must be boolean")
        if section != "profile":
            extra = {"image_weights", "text_weights"} if section == "query_aware" else {"weights"} if section == "knowledge_reranking" else set()
            if set(c[section]) - set(DEFAULTS[section]) - extra:
                raise ValueError(f"Unknown configuration fields in {section}")
    if c["query_aware"]["candidate_pool"] < 32:
        raise ValueError("Knowledge candidate_pool must be >=32 to support adaptive K")
    return c

