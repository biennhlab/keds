import pickle
import zipfile

import numpy as np
import pytest
import torch

from keds.knowledge.database import KnowledgeDatabase
from keds.knowledge.retriever import KnowledgeRetriever
from keds.knowledge.storage import read_bank_archive
from keds.rag.config import rag_config
from keds.rag.fusion import normalize_scores, reciprocal_rank_fusion
from keds.rag.query_aware import QueryAwareKnowledge


def test_rrf_weighted_ranks_and_score_normalization():
    values = reciprocal_rank_fusion([[1, 2], [2, 3]], weights=[0.7, 0.3], constant=60)
    assert values[2] == pytest.approx(0.7/62 + 0.3/61)
    assert values[2] > values[1] > values[3]
    np.testing.assert_array_equal(normalize_scores([4, 4]), [0, 0])
    assert np.isfinite(normalize_scores([-1, 0, 1], "zscore")).all()


def test_readonly_tensor_archive_is_bitwise_equal_to_original(tmp_path):
    images = torch.randn(64, 8)
    path = tmp_path / "knowledge.pt"
    torch.save({"image_features": images, "text_features": images.clone(), "captions": ["caption"]*64}, path)
    loaded = read_bank_archive(path)
    assert isinstance(loaded["image_features"], np.memmap)
    assert not loaded["image_features"].flags.writeable
    np.testing.assert_array_equal(loaded["image_features"], images.numpy())
    # A global outside the explicit storage whitelist is rejected, never executed.
    bad = tmp_path / "bad.pt"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("x/data.pkl", pickle.dumps(eval))
    with pytest.raises(pickle.UnpicklingError):
        read_bank_archive(bad)


def test_modification_changes_hybrid_neighbors_without_any_target_argument(tmp_path):
    features = torch.eye(64)
    path = tmp_path / "bank.pt"
    torch.save({"image_features": features, "text_features": features}, path)
    db = KnowledgeDatabase(path, expected_count=64, dim=64)
    retriever = KnowledgeRetriever(db, tmp_path / "index", build=True, shard_size=16)
    c = rag_config({"query_aware": {"enabled": True, "method": "weighted", "candidate_pool": 32,
                                    "image_weights": [0, 0, 0, 1], "text_weights": [0, 0, 0, 1]}})
    advanced = QueryAwareKnowledge(retriever, c)
    first = advanced.retrieve(features[:1], features[1:2], top_k=4)[0]
    second = advanced.retrieve(features[:1], features[2:3], top_k=4)[0]
    assert first.details["image"]["indices"][0][0] == 1
    assert second.details["image"]["indices"][0][0] == 2
    assert first.images.shape == first.texts.shape == (1, 4, 64)
    baseline_ids, _ = retriever.search(features[:1], 4, "image")
    assert baseline_ids[0, 0] == 0


def test_variable_length_adaptive_knowledge_is_not_padded(tmp_path):
    features = torch.eye(64)
    path = tmp_path / "bank.pt"
    torch.save({"image_features": features, "text_features": features}, path)
    db = KnowledgeDatabase(path, expected_count=64, dim=64)
    retriever = KnowledgeRetriever(db, tmp_path / "index", build=True, shard_size=16)
    images = torch.stack([features[0], features[:32].sum(0)])
    advanced = QueryAwareKnowledge(retriever, rag_config({"adaptive_k": {"enabled": True}}))
    selections = advanced.retrieve(images, images)
    assert selections[0].images.shape == (1, 4, 64)
    assert selections[1].images.shape == (1, 32, 64)


