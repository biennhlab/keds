from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.nn import functional as F

from keds.gallery.repository import Gallery, image_digest
from keds.knowledge.database import KnowledgeDatabase
from keds.knowledge.retriever import KnowledgeRetriever
from keds.models.clip_encoder import CLIPEncoder, resolve_device, rgb_image
from keds.models.dual_stream import DualStream
from keds.utils.config import load_config

MODES = ("keds", "phi_m", "phi_a", "image", "text", "fusion")


class KEDsEngine:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        c = self.config
        self.device = resolve_device(c["model"]["device"])
        # Preflight first; never serve random initialized weights if artifacts are absent.
        self.streams = DualStream(c["bkp"], c["checkpoints"], self.device)
        self.database = KnowledgeDatabase(c["knowledge"]["database_path"],
                                          expected_count=c["knowledge"]["expected_count"])
        self.knowledge = KnowledgeRetriever(self.database, c["knowledge"]["index_dir"])
        self.clip = CLIPEncoder(**c["model"])
        self.gallery = None
        hashes = "-".join(self.streams.metadata[x]["sha256"][:8] for x in ("phi_m", "phi_a"))
        self.version = f"author_bkp_v1-{hashes}"

    def load_gallery(self, index_path: str | Path):
        gallery = Gallery(index_path)
        if (gallery.manifest["backbone"] != self.clip.backbone or
                gallery.manifest["clip_revision"] != self.clip.revision):
            raise ValueError("Gallery CLIP version differs from encoder; rebuild the gallery")
        self.gallery = gallery
        return gallery

    def _validate_text(self, modification):
        if not isinstance(modification, str) or not modification.strip():
            raise ValueError("Modification text is required")
        if "*" in modification:
            raise ValueError("Modification cannot contain reserved pseudo-word marker '*'")
        if len(modification) > self.config["inference"]["max_text_length"]:
            raise ValueError("Modification text too long")

    @torch.inference_mode()
    def encode_query(self, image, modification, mode="keds", alpha=None, knowledge_k=None,
                     return_details=False):
        self._validate_text(modification)
        image = rgb_image(image)
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}; choose {MODES}")
        c = self.config
        alpha = c["inference"]["alpha"] if alpha is None else alpha
        knowledge_k = c["knowledge"]["top_k"] if knowledge_k is None else knowledge_k
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        start = time.perf_counter()
        raw = self.clip.encode_images([image], normalize=False)
        visual = F.normalize(raw, dim=-1)
        info = {"mode": mode, "alpha": alpha, "knowledge_k": knowledge_k}
        if mode in ("image", "text", "fusion"):
            if mode == "image":
                query = visual
            else:
                text = self.clip.encode_texts([modification])
                query = text if mode == "text" else F.normalize(0.5 * visual + 0.5 * text, dim=-1)
        else:
            neighbors_i, neighbors_t, neighbor_info = self.knowledge.retrieve(visual, knowledge_k)
            reference = visual if c["bkp"]["input_normalization"] == "l2" else raw
            prompt = c["inference"]["prompt_template"].format(modification=modification.strip())
            features = {}
            if mode in ("keds", "phi_m"):
                pseudo = self.streams.phi_m(reference, neighbors_i, neighbors_t)
                features["phi_m"] = self.clip.compose([prompt], pseudo)
            if mode in ("keds", "phi_a"):
                pseudo = self.streams.phi_a(reference, neighbors_i, neighbors_t)
                features["phi_a"] = self.clip.compose([prompt], pseudo)
            query = (F.normalize(alpha * features["phi_m"] + (1-alpha) * features["phi_a"], dim=-1)
                     if mode == "keds" else features[mode])
            info["neighbors"] = neighbor_info
        if query.shape != (1, 768) or not torch.isfinite(query).all() or query.norm() <= 1e-12:
            raise RuntimeError("Invalid composed query embedding")
        info["inference_ms"] = (time.perf_counter() - start) * 1000
        embedding = F.normalize(query.float(), dim=-1)[0].cpu()
        return (embedding, info) if return_details else embedding

    @torch.inference_mode()
    def retrieve(self, image, modification, top_k=10, gallery=None, exclude_ids=(),
                 exclude_reference=True, **options):
        gallery = gallery or self.gallery
        if gallery is None:
            raise ValueError("Select/load a target gallery; CC3M knowledge is not the result gallery")
        if (gallery.manifest["backbone"] != self.clip.backbone or
                gallery.manifest["clip_revision"] != self.clip.revision):
            raise ValueError("Gallery CLIP version mismatch")
        query, info = self.encode_query(image, modification, return_details=True, **options)
        start = time.perf_counter()
        results = gallery.search(query.unsqueeze(0), top_k, exclude_ids,
                                 image_digest(image) if exclude_reference else None)
        info["search_ms"] = (time.perf_counter() - start) * 1000
        return {"model_version": self.version, "gallery_id": gallery.gallery_id,
                "gallery_version": gallery.version, "knowledge_version": self.knowledge.version,
                "results": results, "details": info}

    def info(self):
        return {"model_version": self.version, "architecture": "author_bkp_v1",
                "mode": self.config["project"]["mode"], "device": str(self.device),
                "backbone": self.clip.backbone, "clip_revision": self.clip.revision,
                "knowledge_count": self.database.size, "knowledge_version": self.knowledge.version,
                "alpha": self.config["inference"]["alpha"],
                "knowledge_k": self.config["knowledge"]["top_k"],
                "checkpoints": {name: {k: v for k, v in meta.items() if k != "path"}
                                for name, meta in self.streams.metadata.items()}}

