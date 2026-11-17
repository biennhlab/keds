"""Versioned gallery storage. An atomic CURRENT pointer commits each update.

Readers see complete immutable generations. Updates never delete original images
or old generations; only the current index/metadata changes.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path

import numpy as np
from filelock import FileLock
from PIL import Image

from keds.models.clip_encoder import rgb_image
from keds.utils.io import read_json, sha256, write_json
from keds.utils.faiss_backend import faiss
from keds.utils.tensors import normalized_numpy


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("ID must be 1-128 letters, digits, dots, underscores or hyphens")
    return value


def image_digest(image):
    image = rgb_image(image)
    h = hashlib.sha256()
    h.update(str(image.size).encode())
    h.update(image.tobytes())
    return h.hexdigest()


class Gallery:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.gallery_id = safe_id(self.path.name)
        pointer = read_json(self.path / "CURRENT.json")
        generation = safe_id(pointer["generation"])
        self.generation = self.path / "generations" / generation
        self.manifest = read_json(self.generation / "gallery_manifest.json")
        self.version = generation
        if self.manifest["gallery_id"] != self.gallery_id:
            raise ValueError("Gallery manifest ID mismatch")
        for name, expected in self.manifest["checksums"].items():
            if name not in {"gallery.faiss", "gallery_metadata.jsonl", "gallery_embeddings.npy"}:
                raise ValueError("Unknown gallery artifact")
            if sha256(self.generation / name) != expected:
                raise ValueError(f"Gallery checksum mismatch: {name}")
        self.records = [json.loads(line) for line in
                        (self.generation / "gallery_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        self.by_id = {r["image_id"]: r for r in self.records}
        self.row_by_id = {r["image_id"]: i for i, r in enumerate(self.records)}
        self.index = faiss.read_index(str(self.generation / "gallery.faiss"))
        self.embeddings = np.load(self.generation / "gallery_embeddings.npy", mmap_mode="r")
        n, dim = self.manifest["count"], self.manifest["dim"]
        if (len(self.records) != n or len(self.by_id) != n or self.index.ntotal != n or
                self.index.d != dim or self.embeddings.shape != (n, dim) or
                not isinstance(self.index, faiss.IndexFlatIP)):
            raise ValueError("Gallery count/dimension/ID/index mismatch")
        for record in self.records:
            safe_id(record["image_id"])
            self.image_path(record["image_id"])

    def image_path(self, image_id):
        record = self.by_id[image_id]
        path = (self.path / record["image_path"]).resolve()
        if not path.is_relative_to(self.path / "images") or not path.is_file():
            raise ValueError("Gallery image path missing or outside managed image storage")
        return path

    def validate(self):
        for offset in range(0, len(self.records), 4096):
            block = self.embeddings[offset:offset+4096]
            expected = normalized_numpy(block, self.index.d)
            if not np.allclose(block, expected, atol=1e-6):
                raise ValueError("Gallery vectors not normalized")
            if not np.allclose(self.index.reconstruct_n(offset, len(block)), expected, atol=1e-6):
                raise ValueError("Gallery FAISS vector order differs from metadata/embedding order")
        for row in self.records:
            with Image.open(self.image_path(row["image_id"])) as image:
                image.verify()
        return {"gallery_id": self.gallery_id, "version": self.version,
                "count": len(self.records), "valid": True}

    def search(self, query, top_k=10, exclude_ids=(), exclude_hash=None, candidate_ids=None):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        excluded = set(exclude_ids)
        if exclude_hash:
            excluded.update(r["image_id"] for r in self.records if r["pixel_sha256"] == exclude_hash)
        q = normalized_numpy(query, self.index.d)
        if len(q) != 1:
            raise ValueError("Gallery.search expects one query")
        if candidate_ids is None:
            scores, indices = self.index.search(q, min(len(self.records), top_k + len(excluded)))
        else:
            candidates = list(candidate_ids)
            if not candidates or len(candidates) != len(set(candidates)) or not set(candidates) <= self.by_id.keys():
                raise ValueError("Candidate IDs must be unique members of the fixed gallery")
            rows = np.asarray([self.row_by_id[i] for i in candidates], dtype=np.int64)
            index = faiss.IndexFlatIP(self.index.d)
            index.add(np.ascontiguousarray(self.embeddings[rows]))
            scores, local = index.search(q, min(len(rows), top_k + len(excluded)))
            indices = rows[local]
        output = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            row = self.records[index]
            if row["image_id"] not in excluded:
                output.append({"rank": len(output) + 1, "image_id": row["image_id"],
                               "score": float(score), "image_path": str(self.image_path(row["image_id"]))})
            if len(output) == top_k:
                break
        return output


def commit_gallery(path, records, embeddings, backbone, revision):
    """Caller holds write lock. Commit pointer only after complete validation."""
    path = Path(path)
    if not records:
        raise ValueError("Gallery must contain at least one image")
    if len({r["image_id"] for r in records}) != len(records):
        raise ValueError("Duplicate gallery IDs")
    embeddings = normalized_numpy(embeddings)
    if len(records) != len(embeddings):
        raise ValueError("Metadata/embedding count mismatch")
    generation_id = uuid.uuid4().hex
    target = path / "generations" / generation_id
    target.mkdir(parents=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(target / "gallery.faiss"))
    np.save(target / "gallery_embeddings.npy", embeddings, allow_pickle=False)
    (target / "gallery_metadata.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    names = ("gallery.faiss", "gallery_embeddings.npy", "gallery_metadata.jsonl")
    write_json(target / "gallery_manifest.json", {
        "gallery_id": safe_id(path.name), "version": generation_id, "count": len(records),
        "dim": embeddings.shape[1], "backbone": backbone, "clip_revision": revision,
        "metric": "cosine", "index_type": "IndexFlatIP",
        "checksums": {name: sha256(target / name) for name in names},
    })
    # Validate vector/record correspondence before publishing.
    if index.ntotal != len(records) or not np.allclose(index.reconstruct_n(0, len(records)), embeddings):
        raise ValueError("Gallery commit validation failed")
    write_json(path / "CURRENT.json", {"generation": generation_id})
    return Gallery(path)


def add_images(path, images, encoder, image_ids=None, batch_size=16):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    path = Path(path).resolve()
    safe_id(path.name)
    path.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path / ".write.lock")):
        previous = Gallery(path) if (path / "CURRENT.json").exists() else None
        if previous and (previous.manifest["backbone"] != encoder.backbone or
                         previous.manifest["clip_revision"] != encoder.revision):
            raise ValueError("Cannot append embeddings from a different CLIP version")
        records = list(previous.records) if previous else []
        known_ids = {r["image_id"] for r in records}
        known_hashes = {r["pixel_sha256"] for r in records}
        (path / "images").mkdir(exist_ok=True)
        new_vectors, batch, batch_records = [], [], []
        image_ids = list(image_ids) if image_ids is not None else None
        images = list(images)
        if image_ids is not None and len(images) != len(image_ids):
            raise ValueError("One explicit image ID required per image")
        for i, source in enumerate(images):
            image = rgb_image(source)
            digest = image_digest(image)
            image_id = safe_id(str(image_ids[i]) if image_ids is not None else digest[:24])
            if image_id in known_ids:
                old = next(r for r in records + batch_records if r["image_id"] == image_id)
                if old["pixel_sha256"] != digest:
                    raise ValueError(f"Existing image ID has different content: {image_id}")
                continue
            # Explicit benchmark IDs are preserved, even when bytes coincide.
            if image_ids is None and digest in known_hashes:
                continue
            image_path = path / "images" / f"{digest}.png"
            if not image_path.exists():
                image.save(image_path)
            record = {"image_id": image_id, "image_path": f"images/{digest}.png",
                      "pixel_sha256": digest, "width": image.width, "height": image.height}
            batch.append(image)
            batch_records.append(record)
            known_ids.add(image_id)
            known_hashes.add(digest)
            if len(batch) >= batch_size:
                new_vectors.append(normalized_numpy(encoder.encode_images(batch)))
                records.extend(batch_records)
                batch, batch_records = [], []
        if batch:
            new_vectors.append(normalized_numpy(encoder.encode_images(batch)))
            records.extend(batch_records)
        if not new_vectors:
            if previous:
                return previous
            raise ValueError("No valid images to index")
        values = ([previous.embeddings] if previous else []) + new_vectors
        return commit_gallery(path, records, np.concatenate(values), encoder.backbone, encoder.revision)


def remove_images(path, image_ids):
    path = Path(path)
    image_ids = set(image_ids)
    with FileLock(str(path / ".write.lock")):
        previous = Gallery(path)
        missing = image_ids - previous.by_id.keys()
        if missing:
            raise KeyError(f"Unknown image IDs: {sorted(missing)}")
        keep = [i for i, r in enumerate(previous.records) if r["image_id"] not in image_ids]
        return commit_gallery(path, [previous.records[i] for i in keep], previous.embeddings[keep],
                              previous.manifest["backbone"], previous.manifest["clip_revision"])


def list_galleries(root):
    root = Path(root)
    if not root.exists():
        return []
    output = []
    for pointer in sorted(root.glob("*/CURRENT.json")):
        generation = safe_id(read_json(pointer)["generation"])
        manifest = read_json(pointer.parent / "generations" / generation / "gallery_manifest.json")
        if manifest["gallery_id"] != safe_id(pointer.parent.name):
            raise ValueError("Gallery manifest ID mismatch")
        output.append(manifest)
    return output

