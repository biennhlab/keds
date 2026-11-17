from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from keds.utils.io import read_json, sha256, write_json
from keds.utils.faiss_backend import faiss
from keds.utils.tensors import normalized_numpy


class KnowledgeRetriever:
    """Independent exact image/text cosine retrieval using bounded-memory FAISS.

    Each shard is IndexFlatIP. Global Top-K is merged without approximation.
    Flat indexes cannot memory-map their vectors; stream shards rather than
    allocating two 1.5 GB indexes plus reallocation and CLIP weights in RAM.
    """
    def __init__(self, database, index_dir, build=False, shard_size=16384):
        self.db = database
        self.index_dir = Path(index_dir)
        if build:
            self.build(shard_size)
        self.manifest = read_json(self.index_dir / "manifest.json")
        m = self.manifest
        if (m.get("format") != "sharded-flatip-v1" or m["database_sha256"] != database.sha256 or
                m["count"] != database.size or m["dim"] != database.dim):
            raise ValueError("Knowledge index does not match database format/bytes/shape/order")
        for name in ("image", "text"):
            offset = 0
            for item in m["indexes"][name]:
                path = self._shard_path(item)
                if item["offset"] != offset or sha256(path) != item["sha256"]:
                    raise ValueError("Knowledge shard ordering/checksum mismatch")
                index = self._read_shard(item)
                offset += index.ntotal
                del index
            if offset != database.size:
                raise ValueError("Knowledge shard counts differ from database")
        self.version = m["database_sha256"][:16]

    def _shard_path(self, item):
        path = (self.index_dir / item["file"]).resolve()
        if not path.is_relative_to(self.index_dir.resolve()):
            raise ValueError("Knowledge shard path outside index directory")
        return path

    def _read_shard(self, item):
        index = faiss.read_index(str(self._shard_path(item)))
        if (not isinstance(index, faiss.IndexFlatIP) or index.ntotal != item["count"] or
                index.d != self.db.dim):
            raise ValueError("Knowledge shard must be IndexFlatIP with matching shape")
        return index

    def build(self, shard_size):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.index_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("database_sha256") != self.db.sha256:
                raise ValueError("Existing index belongs to another DB; choose another directory")
            return
        folder = self.index_dir / f"shards-{self.db.sha256[:16]}-{shard_size}"
        folder.mkdir(exist_ok=True)
        indexes = {}
        for name, features in (("image", self.db.images), ("text", self.db.texts)):
            items = []
            for offset in range(0, self.db.size, shard_size):
                array = normalized_numpy(features[offset:offset+shard_size], self.db.dim)
                path = folder / f"{name}-{offset:08d}.faiss"
                if path.exists():
                    old = faiss.read_index(str(path))
                    if (old.d != self.db.dim or old.ntotal != len(array) or
                            not np.allclose(old.reconstruct_n(0, len(array)), array, atol=1e-6)):
                        raise ValueError(f"Existing partial shard invalid, preserving it: {path}")
                    del old
                else:
                    index = faiss.IndexFlatIP(self.db.dim)
                    index.add(array)
                    faiss.write_index(index, str(path))
                    del index
                items.append({"file": path.relative_to(self.index_dir).as_posix(),
                              "offset": offset, "count": len(array), "sha256": sha256(path)})
            indexes[name] = items
        write_json(manifest_path, {"format": "sharded-flatip-v1", "database_sha256": self.db.sha256,
                                   "count": self.db.size, "dim": self.db.dim,
                                   "retrieval": "independent", "metric": "cosine", "indexes": indexes})

    @torch.no_grad()
    def retrieve(self, query, top_k=16, exclude_indices=None):
        if not 1 <= top_k <= self.db.size - int(exclude_indices is not None):
            raise ValueError("Knowledge top_k exceeds available neighbors")
        q = normalized_numpy(query, self.db.dim)
        excluded = None
        if exclude_indices is not None:
            excluded = np.asarray(exclude_indices, dtype=np.int64)
            if excluded.shape != (len(q),) or ((excluded < 0) | (excluded >= self.db.size)).any():
                raise ValueError("Provide one valid excluded knowledge row per query")
        output, info = [], {}
        for name, features in (("image", self.db.images), ("text", self.db.texts)):
            best_scores = np.empty((len(q), 0), dtype=np.float32)
            best_ids = np.empty((len(q), 0), dtype=np.int64)
            for item in self.manifest["indexes"][name]:
                index = self._read_shard(item)
                scores, ids = index.search(q, min(item["count"], top_k + int(excluded is not None)))
                del index
                ids += item["offset"]
                if excluded is not None:
                    scores[ids == excluded[:, None]] = -np.inf
                scores, ids = np.concatenate([best_scores, scores], 1), np.concatenate([best_ids, ids], 1)
                order = np.lexsort((ids, -scores), axis=1)[:, :top_k]
                best_scores, best_ids = np.take_along_axis(scores, order, 1), np.take_along_axis(ids, order, 1)
            neighbors = F.normalize(features[torch.from_numpy(best_ids)].float(), dim=-1)
            output.append(neighbors.to(query.device))
            info[name] = {"indices": best_ids.tolist(), "scores": best_scores.tolist()}
            if self.db.ids is not None:
                info[name]["ids"] = [[str(self.db.ids[i]) for i in row] for row in best_ids]
            if self.db.captions is not None:
                info[name]["captions"] = [[str(self.db.captions[i]) for i in row] for row in best_ids]
        return output[0], output[1], info

    def validate_order(self):
        for name, features in (("image", self.db.images), ("text", self.db.texts)):
            for item in self.manifest["indexes"][name]:
                index = self._read_shard(item)
                offset, count = item["offset"], item["count"]
                actual = index.reconstruct_n(0, count)
                if not np.allclose(actual, normalized_numpy(features[offset:offset+count]), atol=1e-6):
                    raise ValueError(f"Knowledge {name} vector order mismatch at {offset}")
                del index
        return True

