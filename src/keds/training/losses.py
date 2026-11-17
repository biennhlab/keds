import torch
from torch.nn import functional as F


def symmetric_infonce(images, composed, temperature=0.07):
    if images.shape != composed.shape or len(images) < 2 or temperature <= 0:
        raise ValueError("InfoNCE needs aligned batches of >=2 features and positive temperature")
    logits = F.normalize(images, dim=-1) @ F.normalize(composed, dim=-1).T / temperature
    labels = torch.arange(len(images), device=images.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

