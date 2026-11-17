import numpy as np
import torch

from keds.rag.cache import SemanticCache
from keds.rag.config import rag_config


def test_cache_namespace_ttl_lru_and_mutation_isolation():
    now = [0.0]
    config = rag_config({"cache": {"enabled": True, "max_entries": 2, "ttl_seconds": 2}})["cache"]
    cache = SemanticCache(config, clock=lambda: now[0])
    cache.put("model-db-config-v1", "a", torch.ones(4))
    cached = cache.get("model-db-config-v1", "a")
    cached.zero_()
    assert cache.get("model-db-config-v1", "a").sum() == 4
    assert cache.get("model-db-config-v2", "a") is None
    cache.put("model-db-config-v1", "b", "b")
    cache.put("model-db-config-v1", "c", "c")
    assert cache.get("model-db-config-v1", "a") is None
    now[0] = 3
    assert cache.get("model-db-config-v1", "b") is None
    assert cache.bytes == 0


def test_semantic_reuse_is_opt_in_and_never_crosses_namespaces():
    config = rag_config({"cache": {"enabled": True, "semantic": True}})["cache"]
    cache = SemanticCache(config)
    cache.put("same-artifacts-params", "first", [1, 2], descriptor=np.array([1., 0.]))
    assert cache.get("same-artifacts-params", "second", descriptor=np.array([1., 0.])) == [1, 2]
    assert cache.get("different-checkpoint", "second", descriptor=np.array([1., 0.])) is None
    assert cache.semantic_hits == 1

