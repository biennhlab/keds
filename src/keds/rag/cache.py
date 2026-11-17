import copy
import threading
import time
from collections import OrderedDict

import numpy as np
import torch


def byte_size(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(byte_size(k) + byte_size(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(byte_size(v) for v in value)
    return len(value.encode()) if isinstance(value, str) else 32


class SemanticCache:
    """Bounded, expiring CPU cache. Semantic reuse is opt-in and separately logged.

    Exact keys always include a version/parameter namespace. A semantic hit is
    possible only within the same namespace, never across checkpoint/DB/config.
    """
    def __init__(self, config, clock=time.monotonic):
        self.config, self.clock = config, clock
        if config["max_entries"] < 1 or config["max_mb"] <= 0 or config["ttl_seconds"] <= 0:
            raise ValueError("Cache entry, size and TTL limits must be positive")
        if not 0 < config["semantic_threshold"] <= 1:
            raise ValueError("Invalid semantic similarity threshold")
        self.entries, self.bytes = OrderedDict(), 0
        self.hits, self.misses, self.semantic_hits = 0, 0, 0
        self.lock = threading.RLock()

    def _remove(self, key):
        self.bytes -= self.entries.pop(key)[2]

    def get(self, namespace, key, descriptor=None):
        if not self.config["enabled"]:
            return None
        with self.lock:
            now = self.clock()
            for old in list(self.entries):
                if self.entries[old][1] <= now:
                    self._remove(old)
            combined = (namespace, key)
            if combined in self.entries:
                self.entries.move_to_end(combined)
                self.hits += 1
                return copy.deepcopy(self.entries[combined][0])
            if descriptor is not None and self.config["semantic"]:
                descriptor = np.asarray(descriptor).reshape(-1)
                descriptor = descriptor / max(np.linalg.norm(descriptor), 1e-12)
                for candidate in reversed(self.entries):
                    value, _, _, stored = self.entries[candidate]
                    if candidate[0] == namespace and stored is not None:
                        if float(descriptor @ stored) >= self.config["semantic_threshold"]:
                            self.semantic_hits += 1
                            self.hits += 1
                            return copy.deepcopy(value)
            self.misses += 1
            return None

    def put(self, namespace, key, value, descriptor=None):
        if not self.config["enabled"]:
            return
        # Clone all tensors to CPU. Never retain CUDA graphs or mutable caller tensors.
        def cpu(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().clone()
            if isinstance(v, dict):
                return {k: cpu(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(cpu(x) for x in v)
            return copy.deepcopy(v)
        value = cpu(value)
        size = byte_size(value)
        if size > self.config["max_mb"] * 1024 * 1024:
            return
        with self.lock:
            combined = (namespace, key)
            if combined in self.entries:
                self._remove(combined)
            while self.entries and (len(self.entries) >= self.config["max_entries"] or
                                   self.bytes + size > self.config["max_mb"] * 1024 * 1024):
                self._remove(next(iter(self.entries)))
            descriptor = None if descriptor is None else np.asarray(descriptor, dtype=np.float32).reshape(-1).copy()
            if descriptor is not None:
                descriptor /= max(np.linalg.norm(descriptor), 1e-12)
            self.entries[combined] = value, self.clock() + self.config["ttl_seconds"], size, descriptor
            self.bytes += size

    def info(self):
        return {"entries": len(self.entries), "bytes": self.bytes, "hits": self.hits,
                "misses": self.misses, "semantic_hits": self.semantic_hits}

