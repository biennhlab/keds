from __future__ import annotations

import numpy as np
import torch

from keds.utils.io import sha256
from .storage import read_bank_archive


class KnowledgeDatabase:
    def __init__(self, path, expected_count=500000, dim=768):
        self.path = path
        self.sha256 = sha256(path)
        self.payload = read_bank_archive(path)
        self.images = self.payload["image_features"]
        self.texts = self.payload["text_features"]
        if self.images.ndim != 2 or self.images.shape != self.texts.shape or self.images.shape[1] != dim:
            raise ValueError("Knowledge image/text embeddings must have identical [N,768] shapes")
        self.size, self.dim = self.images.shape
        if expected_count is not None and self.size != expected_count:
            raise ValueError(f"Expected {expected_count} knowledge pairs, found {self.size}")
        self.metadata = self.payload.get("meta", self.payload.get("metadata", {}))
        self.ids = self.payload.get("image_ids", self.payload.get("ids"))
        self.captions = self.payload.get("captions")
        for name, values in (("ids", self.ids), ("captions", self.captions)):
            if values is not None and len(values) != self.size:
                raise ValueError(f"Knowledge {name} count differs from embeddings")

    def validate(self, chunk_size=4096):
        stats = {}
        for name, features in (("image", self.images), ("text", self.texts)):
            min_norm, max_norm = float("inf"), 0.0
            for offset in range(0, self.size, chunk_size):
                block = features[offset:offset+chunk_size]
                if not np.isfinite(block).all():
                    raise ValueError(f"Knowledge {name} embeddings contain NaN/Inf")
                norms = np.linalg.norm(block.astype(np.float32, copy=False), axis=-1)
                if (norms <= 1e-12).any():
                    raise ValueError(f"Knowledge {name} embeddings contain zero vectors")
                min_norm = min(min_norm, float(norms.min()))
                max_norm = max(max_norm, float(norms.max()))
            stats[name] = {"shape": list(features.shape), "dtype": str(features.dtype),
                           "norm_min": min_norm, "norm_max": max_norm}
        return {"count": self.size, "dim": self.dim, "sha256": self.sha256,
                "features": stats, "metadata": self.metadata,
                "has_ids": self.ids is not None, "has_captions": self.captions is not None}

    def rows(self, modality, indices):
        features = self.images if modality == "image" else self.texts
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        return torch.from_numpy(np.array(features[indices], dtype=np.float32, copy=True))

