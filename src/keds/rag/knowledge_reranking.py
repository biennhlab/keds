import numpy as np

from .fusion import mmr_select, normalize_scores


class KnowledgeReranker:
    def __init__(self, config):
        self.config = config

    def rank(self, initial, image_features, text_features, visual, modification, k, modality, ids):
        c = self.config
        normalization = c.get("normalization", "minmax")
        components = {"initial": normalize_scores(initial, normalization),
                      "consistency": normalize_scores(np.sum(image_features * text_features, -1), normalization),
                      "modification": normalize_scores((image_features @ modification +
                                                         text_features @ modification) / 2, normalization),
                      "visual": normalize_scores(image_features @ visual, normalization)}
        weights = c.get("weights", {"initial": 0.5, "consistency": 0.15, "modification": 0.25, "visual": 0.1})
        if set(weights) != set(components) or any(v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Invalid knowledge reranking weights")
        scores = sum(weights[n] * values for n, values in components.items()) / sum(weights.values())
        if c.get("mmr", False):
            features = image_features if modality == "image" else text_features
            selected = mmr_select(scores, features, k, c.get("diversity_lambda", 0.8), ids)
        else:
            selected = np.lexsort((ids, -scores))[:k]
        return selected, scores, components

