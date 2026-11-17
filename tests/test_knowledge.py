import numpy as np
import pytest
import torch
from torch.nn import functional as F

from keds.knowledge.database import KnowledgeDatabase
from keds.knowledge.retriever import KnowledgeRetriever


def test_sharded_exact_cosine_matches_bruteforce_and_excludes_id(tmp_path):
    # Small numerical fixtures test the real FAISS implementation, not application predictions.
    images, texts = F.normalize(torch.randn(23, 8), dim=-1), F.normalize(torch.randn(23, 8), dim=-1)
    path = tmp_path / "db.pt"
    torch.save({"image_features": images, "text_features": texts}, path)
    db = KnowledgeDatabase(path, expected_count=23, dim=8)
    assert db.validate()["count"] == 23
    retriever = KnowledgeRetriever(db, tmp_path / "indexes", build=True, shard_size=5)
    query = images[[0, 8]]
    i, t, info = retriever.retrieve(query, 4, exclude_indices=[0, 8])
    for name, bank in (("image", images), ("text", texts)):
        scores = query @ bank.T
        scores[torch.arange(2), torch.tensor([0, 8])] = -float("inf")
        expected = scores.topk(4, dim=1).indices.tolist()
        assert info[name]["indices"] == expected
    assert i.shape == t.shape == (2, 4, 8)
    assert retriever.validate_order()
    assert info["image"]["indices"] != info["text"]["indices"]
    with pytest.raises(ValueError):
        retriever.retrieve(query, 23, exclude_indices=[0, 8])


def test_knowledge_invalid_and_corrupt_index(tmp_path):
    path = tmp_path / "db.pt"
    values = torch.eye(3)
    torch.save({"image_features": values, "text_features": values}, path)
    with pytest.raises(ValueError, match="Expected"):
        KnowledgeDatabase(path, expected_count=500000, dim=3)
    db = KnowledgeDatabase(path, expected_count=3, dim=3)
    retriever = KnowledgeRetriever(db, tmp_path / "index", build=True, shard_size=2)
    shard = retriever._shard_path(retriever.manifest["indexes"]["image"][0])
    shard.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        KnowledgeRetriever(db, tmp_path / "index")


def test_reject_zero_nan_vectors():
    from keds.utils.tensors import normalized_numpy
    for array in (np.zeros((2, 3)), np.full((2, 3), np.nan)):
        with pytest.raises(ValueError):
            normalized_numpy(array)

