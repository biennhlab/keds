"""Author BKP modules. Parameter names and token order match suoych/KEDs.

CrossFormer has no residual or layer norm; adding either invalidates existing
weights. See THIRD_PARTY_NOTICES.md for attribution and the pinned source.
"""
from __future__ import annotations

import torch
from torch import nn


class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=768, middle_dim=512, output_dim=768, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(embed_dim if i == 0 else middle_dim, middle_dim),
                          nn.Dropout(dropout), nn.ReLU()) for i in range(n_layer)
        ])

    def forward(self, x):
        return self.fc_out(self.layers(x))


class CrossAttention(nn.Module):
    def __init__(self, dim=768, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=True)
        self.to_k = nn.Linear(dim, inner, bias=True)
        self.to_v = nn.Linear(dim, inner, bias=True)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, q, k, v):
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        q, k, v = split(self.to_q(q)), split(self.to_k(k)), split(self.to_v(v))
        weights = ((q @ k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous()
        return self.to_out(out.flatten(2))


class CrossFormer(nn.Module):
    def __init__(self, dim=768, num_layers=3, heads=8, dim_head=64):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossAttention(dim, heads, dim_head) for _ in range(num_layers)
        ])

    def forward(self, q, k, v):
        for layer in self.cross_layers:
            q = layer(q, k, v)
        return q


class BKP(nn.Module):
    architecture = "author_bkp_v1"

    def __init__(self, middle_dim=512, n_layer=2, cross_layers=3, heads=8, dim_head=64,
                 dropout=0.1, dim=768):
        super().__init__()
        self.img2text = IM2TEXT(dim, middle_dim, dim, n_layer, dropout)
        self.retrieval_fuse = CrossFormer(dim, cross_layers, heads, dim_head)
        self.text_condition = CrossFormer(dim, cross_layers, heads, dim_head)

    def forward(self, image_features, image_neighbors, text_neighbors):
        if (image_features.ndim != 2 or image_neighbors.ndim != 3 or
                image_neighbors.shape != text_neighbors.shape or
                image_features.shape[0] != image_neighbors.shape[0] or
                image_features.shape[-1] != image_neighbors.shape[-1]):
            raise ValueError("BKP expects [B,D] and paired shape [B,K,D] neighbor tensors")
        query = self.img2text(image_features).unsqueeze(1)
        images, texts = self.img2text(image_neighbors), self.img2text(text_neighbors)
        # Official order: visual knowledge, textual knowledge, mapped reference.
        return torch.cat([self.retrieval_fuse(query, images, images),
                          self.text_condition(query, texts, texts), query], dim=1)


def build_bkp(config: dict) -> BKP:
    if config.get("architecture") != BKP.architecture:
        raise ValueError("Unsupported architecture; no automatic checkpoint conversion is allowed")
    for key, expected in {"heads": 8, "dim_head": 64, "cross_layers": 3,
                          "middle_dim": 512, "n_layer": 2}.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"author_bkp_v1 requires {key}={expected}; checkpoint head layout is fixed")
    return BKP(**{k: v for k, v in config.items() if k not in ("architecture", "input_normalization")})

