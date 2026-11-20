from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from keds.gallery.repository import Gallery, image_digest
from keds.knowledge.database import KnowledgeDatabase
from keds.knowledge.retriever import KnowledgeRetriever
from keds.models.clip_encoder import CLIPEncoder, resolve_device, rgb_image
from keds.models.dual_stream import DualStream
from keds.rag.adaptive import adaptive_alpha
from keds.rag.cache import SemanticCache
from keds.rag.config import fingerprint, rag_config
from keds.rag.gallery_reranking import GalleryReranker
from keds.rag.query_aware import QueryAwareKnowledge
from keds.utils.config import load_config

MODES = ("keds", "phi_m", "phi_a", "image", "text", "fusion")


class KEDsEngine:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        c = self.config
        self.device = resolve_device(c["model"]["device"])
        self.streams = DualStream(c["bkp"], c["checkpoints"], self.device)
        self.database = KnowledgeDatabase(c["knowledge"]["database_path"], c["knowledge"]["expected_count"])
        self.knowledge = KnowledgeRetriever(self.database, c["knowledge"]["index_dir"],
                                            resident_indexes=c["knowledge"].get("resident_indexes", False))
        self.clip = CLIPEncoder(**c["model"])
        self.gallery, self.caches, self._gallery_cache = None, {}, {}
        hashes = "-".join(self.streams.metadata[x]["sha256"][:8] for x in ("phi_m", "phi_a"))
        self.version = f"author_bkp_v1-{hashes}"
        self.rag = rag_config(c.get("rag"))
        self.clip_namespace = fingerprint({"backbone": self.clip.backbone, "revision": self.clip.revision,
                                            "processor": self.clip.processor.to_dict()})
        self.artifact_namespace = fingerprint({"clip": self.clip_namespace, "checkpoints": self.version,
            "database": self.database.sha256, "index": fingerprint(self.knowledge.manifest),
            "prompt": c["inference"]["prompt_template"], "bkp": c["bkp"]})
        self.reference_cache = SemanticCache({"enabled": c["inference"].get("reference_cache", False),
            "semantic": False, "semantic_threshold": 1.0, "max_entries": 128,
            "max_mb": 16, "ttl_seconds": 900})

    def configure_rag(self, settings):
        """Change experiment settings only; never reload or alter model weights."""
        self.rag = rag_config(settings)

    def settings(self, name=None):
        if name is None:
            return self.rag
        if name == "E0":
            return rag_config()
        profiles = self.config.get("rag_profiles", {})
        if name not in profiles:
            raise ValueError(f"Unknown RAG profile: {name}")
        return rag_config({**profiles[name], "profile": name})

    def _cache(self, settings):
        policy = settings["cache"]
        key = fingerprint(policy)
        if key not in self.caches:
            self.caches[key] = SemanticCache(policy)
        return self.caches[key]

    def load_gallery(self, index_path: str | Path):
        from keds.utils.io import read_json
        path = Path(index_path).resolve()
        generation = read_json(path / "CURRENT.json")["generation"]
        cached = self._gallery_cache.get((str(path), generation))
        gallery = Gallery(path, version=generation) if cached is None else cached
        if (gallery.manifest["backbone"] != self.clip.backbone or
                gallery.manifest["clip_revision"] != self.clip.revision):
            raise ValueError("Gallery CLIP version differs from encoder; rebuild it")
        self._gallery_cache = {(str(path), generation): gallery}
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
    def _encode(self, image, modification, mode, alpha, knowledge_k, settings, prefetched=None):
        self._validate_text(modification)
        image = rgb_image(image)
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        c = self.config
        alpha = c["inference"]["alpha"] if alpha is None else alpha
        knowledge_k = c["knowledge"]["top_k"] if knowledge_k is None else knowledge_k
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        started = time.perf_counter()
        cache, profile_hash = self._cache(settings), fingerprint(settings)
        namespace = fingerprint({"artifacts": self.artifact_namespace, "rag": profile_hash,
                                 "mode": mode, "alpha": alpha, "k": knowledge_k})
        image_key, text_key = image_digest(image), hashlib.sha256(modification.encode()).hexdigest()
        query_key = image_key + text_key
        cached = cache.get(namespace + ":composed", query_key)
        if cached is not None:
            embedding, details, context = cached
            details.update(inference_ms=(time.perf_counter()-started)*1000, composed_cache_hit=True)
            details["cache"] = cache.info()
            return embedding, details, context
        reference_cache = self.reference_cache if settings["profile"] == "E0" else cache
        raw = reference_cache.get(self.clip_namespace + ":image", image_key)
        if raw is None:
            raw = prefetched[0] if prefetched is not None else self.clip.encode_images([image], normalize=False)
            reference_cache.put(self.clip_namespace + ":image", image_key, raw)
        raw = raw.to(self.device)
        visual = F.normalize(raw, dim=-1)
        advanced = any(settings[x]["enabled"] for x in ("query_aware", "knowledge_reranking", "adaptive_k"))
        need_text = (advanced or mode in ("text", "fusion") or settings["gallery_reranking"]["enabled"] or
                     settings["cache"]["semantic"])
        text = None
        if need_text:
            text = cache.get(self.clip_namespace + ":text", text_key)
            if text is None:
                text = prefetched[1] if prefetched is not None else self.clip.encode_texts([modification])
                cache.put(self.clip_namespace + ":text", text_key, text)
            text = text.to(self.device)
        descriptor = np.concatenate([visual.cpu().numpy().ravel(), text.cpu().numpy().ravel()]) if text is not None else None
        if settings["cache"]["semantic"]:
            semantic = cache.get(namespace + ":composed", query_key, descriptor)
            if semantic is not None:
                embedding, details, context = semantic
                context["image_key"] = image_key
                details.update(inference_ms=(time.perf_counter()-started)*1000, semantic_cache_hit=True)
                details["cache"] = cache.info()
                return embedding, details, context
        details = {"mode": mode, "alpha": alpha, "knowledge_k": knowledge_k,
                   "rag_profile": settings["profile"], "rag_fingerprint": profile_hash,
                   "composed_cache_hit": False, "semantic_cache_hit": False}
        if mode in ("image", "text", "fusion"):
            query = visual if mode == "image" else text if mode == "text" else F.normalize(0.5*visual + 0.5*text, dim=-1)
        else:
            knowledge_key = fingerprint({"image": image_key, "text": text_key if advanced else None,
                                         "k": knowledge_k, "rag": profile_hash})
            neighbors = reference_cache.get(self.artifact_namespace + ":knowledge", knowledge_key)
            knowledge_start = time.perf_counter()
            if neighbors is None:
                if advanced:
                    selection = QueryAwareKnowledge(self.knowledge, settings).retrieve(visual, text, knowledge_k)[0]
                    neighbors = selection.images, selection.texts, selection.details
                else:
                    neighbors = self.knowledge.retrieve(visual, knowledge_k)
                reference_cache.put(self.artifact_namespace + ":knowledge", knowledge_key, neighbors, descriptor)
                details["knowledge_cache_hit"] = False
            else:
                details["knowledge_cache_hit"] = True
            ni, nt, neighbor_info = neighbors
            ni, nt = ni.to(self.device), nt.to(self.device)
            details["knowledge_ms"] = (time.perf_counter()-knowledge_start)*1000
            details["knowledge_k_used"] = ni.shape[1]
            reference = visual if c["bkp"]["input_normalization"] == "l2" else raw
            prompt = c["inference"]["prompt_template"].format(modification=modification.strip())
            features = {}
            if mode in ("keds", "phi_m"):
                features["phi_m"] = self.clip.compose([prompt], self.streams.phi_m(reference, ni, nt))
            if mode in ("keds", "phi_a"):
                features["phi_a"] = self.clip.compose([prompt], self.streams.phi_a(reference, ni, nt))
            if mode == "keds" and settings["adaptive_fusion"]["enabled"]:
                alpha, confidence = adaptive_alpha(features["phi_m"][0].cpu().numpy(),
                    features["phi_a"][0].cpu().numpy(), torch.cat([ni, nt], 1)[0].cpu().numpy(),
                    settings["adaptive_fusion"], alpha)
                details["fusion_confidence"] = confidence
            details["alpha_used"] = alpha
            query = (F.normalize(alpha*features["phi_m"] + (1-alpha)*features["phi_a"], dim=-1)
                     if mode == "keds" else features[mode])
            details["neighbors"] = neighbor_info
        if query.shape != (1, 768) or not torch.isfinite(query).all() or query.norm() <= 1e-12:
            raise RuntimeError("Invalid composed query embedding")
        embedding = F.normalize(query.float(), dim=-1)[0].cpu()
        context = {"visual": visual[0].cpu(), "text": text[0].cpu() if text is not None else None,
                   "image_key": image_key}
        details["inference_ms"] = (time.perf_counter()-started)*1000
        details["cache"] = cache.info()
        cache.put(namespace + ":composed", query_key, (embedding, details, context), descriptor)
        return embedding, details, context

    def encode_query(self, image, modification, mode="keds", alpha=None, knowledge_k=None,
                     return_details=False, rag_profile=None):
        embedding, details, _ = self._encode(image, modification, mode, alpha, knowledge_k, self.settings(rag_profile))
        return (embedding, details) if return_details else embedding

    @torch.inference_mode()
    def encode_queries(self, images, modifications, mode="keds", rag_profile=None, **options):
        if not images or len(images) != len(modifications):
            raise ValueError("Provide aligned non-empty image/text batches")
        for text in modifications:
            self._validate_text(text)
        settings = self.settings(rag_profile)
        if settings == rag_config():
            return self._encode_e0_batch(images, modifications, mode, options).cpu()
        raw = self.clip.encode_images(images, normalize=False)
        texts = self.clip.encode_texts(modifications)
        # CLIP is batched; variable K samples go through BKP separately without
        # adding unmasked padding or changing the trained CrossFormer interface.
        return torch.stack([self._encode(image, text, mode, options.get("alpha"), options.get("knowledge_k"),
                            settings, (raw[i:i+1], texts[i:i+1]))[0]
                            for i, (image, text) in enumerate(zip(images, modifications))])

    def _encode_e0_batch(self, images, modifications, mode, options):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "text":
            return self.clip.encode_texts(modifications)
        raw = self.clip.encode_images(images, normalize=False)
        visual = F.normalize(raw, dim=-1)
        if mode == "image":
            return visual
        if mode == "fusion":
            return F.normalize(0.5 * visual + 0.5 * self.clip.encode_texts(modifications), dim=-1)
        config = self.config
        alpha = options.get("alpha")
        alpha = config["inference"]["alpha"] if alpha is None else alpha
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        size = options.get("knowledge_k")
        size = config["knowledge"]["top_k"] if size is None else size
        ni, nt, _ = self.knowledge.retrieve(visual, size)
        ni, nt = ni.to(self.device), nt.to(self.device)
        reference = visual if config["bkp"]["input_normalization"] == "l2" else raw
        template = config["inference"]["prompt_template"]
        prompts = [template.format(modification=text.strip()) for text in modifications]
        if mode == "phi_m":
            return self.clip.compose(prompts, self.streams.phi_m(reference, ni, nt))
        if mode == "phi_a":
            return self.clip.compose(prompts, self.streams.phi_a(reference, ni, nt))
        mapped = self.clip.compose(prompts, self.streams.phi_m(reference, ni, nt))
        augmented = self.clip.compose(prompts, self.streams.phi_a(reference, ni, nt))
        return F.normalize(alpha * mapped + (1 - alpha) * augmented, dim=-1)

    @torch.inference_mode()
    def retrieve(self, image, modification, top_k=10, gallery=None, exclude_ids=(),
                 exclude_reference=True, mode="keds", alpha=None, knowledge_k=None,
                 rag_profile=None, candidate_ids=None, include_image_paths=True):
        gallery = self.gallery if gallery is None else gallery
        if gallery is None:
            raise ValueError("Select/load a target gallery; CC3M knowledge is not the result gallery")
        if (gallery.manifest["backbone"] != self.clip.backbone or
                gallery.manifest["clip_revision"] != self.clip.revision):
            raise ValueError("Gallery CLIP version mismatch")
        settings = self.settings(rag_profile)
        query, details, context = self._encode(image, modification, mode, alpha, knowledge_k, settings)
        started = time.perf_counter()
        excluded_hash = context["image_key"] if exclude_reference else None
        if settings["gallery_reranking"]["enabled"]:
            results = GalleryReranker(settings["gallery_reranking"]).rank(gallery, query.unsqueeze(0),
                context["visual"].numpy(), context["text"].numpy(), top_k, exclude_ids, excluded_hash, candidate_ids,
                include_image_paths=include_image_paths)
        else:
            results = gallery.search(query.unsqueeze(0), top_k, exclude_ids, excluded_hash, candidate_ids,
                                     include_image_paths=include_image_paths)
        details["search_ms"] = (time.perf_counter()-started)*1000
        return {"model_version": self.version, "gallery_id": gallery.gallery_id,
                "gallery_version": gallery.version, "knowledge_version": self.knowledge.version,
                "results": results, "details": details}

    def info(self):
        return {"model_version": self.version, "architecture": "author_bkp_v1",
                "mode": self.config["project"]["mode"], "device": str(self.device),
                "backbone": self.clip.backbone, "clip_revision": self.clip.revision,
                "knowledge_count": self.database.size, "knowledge_version": self.knowledge.version,
                "alpha": self.config["inference"]["alpha"], "knowledge_k": self.config["knowledge"]["top_k"],
                "rag": self.rag, "available_rag_profiles": ["E0"] + [n for n in self.config.get("rag_profiles", {}) if n != "E0"],
                "cache_stats": {key: cache.info() for key, cache in self.caches.items()},
                "reference_cache_stats": self.reference_cache.info(),
                "checkpoints": {n: {k: v for k, v in m.items() if k != "path"} for n, m in self.streams.metadata.items()}}

