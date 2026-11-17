import numpy as np

from keds.rag.adaptive import adaptive_alpha, choose_k
from keds.rag.config import rag_config
from keds.rag.fusion import mmr_select
from keds.rag.knowledge_reranking import KnowledgeReranker


def test_mmr_promotes_diversity_and_lambda_one_is_relevance_order():
    vectors = np.array([[1., 0.], [1., 0.], [0., 1.]])
    relevance = np.array([1., 0.99, 0.8])
    assert mmr_select(relevance, vectors, 2, 0.5).tolist() == [0, 2]
    assert mmr_select(relevance, vectors, 3, 1.0).tolist() == [0, 1, 2]


def test_knowledge_consistency_and_modification_reranking():
    images = np.eye(3, dtype=np.float32)
    texts = images.copy()
    texts[0] = -images[0]
    reranker = KnowledgeReranker({"weights": {"initial": 0, "consistency": 0.5,
                                             "modification": 0.5, "visual": 0}})
    selected, scores, _ = reranker.rank(np.ones(3), images, texts, images[0], images[1], 2, "image", [0, 1, 2])
    assert selected[0] == 1 and scores[1] > scores[0]


def test_adaptive_k_confidence_and_alpha_bounds():
    config = rag_config()
    clear, confidence = choose_k([1] + [0]*31, config["adaptive_k"], 64)
    ambiguous, uncertain = choose_k([0.5]*32, config["adaptive_k"], 64)
    assert clear in (4, 8) and ambiguous == 32
    assert confidence["confidence"] > uncertain["confidence"]
    bank = np.concatenate([np.array([[1., 0.]]), np.tile([0., 1.], (31, 1))])
    alpha, _ = adaptive_alpha(np.array([1., 0.]), np.array([0., 1.]), bank, config["adaptive_fusion"])
    assert 0.1 <= alpha <= 0.9 and alpha > 0.5

