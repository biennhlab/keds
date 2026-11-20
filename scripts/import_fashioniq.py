"""Import official Fashion-IQ validation galleries, preserving IDs and full splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keds.evaluation.datasets import fashioniq, require_full_gallery
from keds.utils.config import load_config
from keds.utils.io import read_json, sha256, write_json

COUNTS = {"dress": 3817, "shirt": 6346, "toptee": 5373}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def verify_upload_manifest(data_root):
    root = Path(data_root).resolve()
    manifest_path = root / "upload_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    records = manifest["files"]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest["fingerprint"]:
        raise ValueError("Upload manifest fingerprint mismatch")
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Upload manifest path outside dataset root")
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Uploaded file checksum mismatch: {record['path']}")
    return fingerprint


def resolve_images(image_root, ids):
    """Resolve original filename stems without dropping or reordering IDs."""
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Split image IDs must be nonempty and unique")
    required = set(ids)
    matches = {}
    for path in Path(image_root).rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS and path.stem in required:
            if path.stem in matches:
                raise ValueError(f"Ambiguous image ID {path.stem}: {matches[path.stem]} and {path}")
            matches[path.stem] = path
    missing = [image_id for image_id in ids if image_id not in matches]
    if missing:
        raise ValueError(f"Missing {len(missing)}/{len(ids)} Fashion-IQ images in {image_root}; "
                         f"examples: {missing[:8]}. Original image files are required.")
    return [matches[image_id] for image_id in ids]


def import_fashioniq(config_path, data_root, category="all", batch_size=8, check_only=False):
    if category not in ("all", *COUNTS):
        raise ValueError("Choose all, dress, shirt or toptee")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = load_config(config_path)
    upload_fingerprint = verify_upload_manifest(data_root)
    categories = list(COUNTS) if category == "all" else [category]
    plans = {}
    # Validate every requested category before loading CLIP or publishing galleries.
    for name in categories:
        queries, ids, provenance = fashioniq(data_root, name, "val")
        if len(ids) != COUNTS[name]:
            raise ValueError(f"Official full {name} val gallery requires {COUNTS[name]} IDs, found {len(ids)}")
        allowed = set(ids)
        if any(q.reference_id not in allowed or q.target_id not in allowed for q in queries):
            raise ValueError(f"Fashion-IQ {name} query references an image outside its full split")
        paths = resolve_images(Path(data_root) / "images", ids)
        plans[name] = (ids, paths, {**provenance, "query_count": len(queries)})
    if check_only:
        return {"status": "files_present", "split": "val", "categories":
                {name: {"count": len(ids), **provenance} for name, (ids, _, provenance) in plans.items()},
                "encoded": False}

    from keds.gallery.repository import Gallery, add_images
    from keds.models.clip_encoder import CLIPEncoder

    existing = {}
    for name, (ids, _, _) in plans.items():
        output = Path(config["gallery"]["root"]) / name
        if (output / "CURRENT.json").exists():
            gallery = Gallery(output)
            require_full_gallery(gallery, ids)
            if (gallery.manifest["backbone"] != config["model"]["backbone"] or
                    gallery.manifest["clip_revision"] != config["model"]["revision"]):
                raise ValueError(f"Existing {name} gallery uses another CLIP version; preserving it")
            existing[name] = gallery
    encoder = CLIPEncoder(**config["model"]) if len(existing) != len(plans) else None
    report = {"dataset": "fashioniq", "split": "val", "protocol": "full", "categories": {},
              "upload_fingerprint": upload_fingerprint,
              "evaluation_run": False}
    for name, (ids, paths, provenance) in plans.items():
        output = Path(config["gallery"]["root"]) / name
        print(f"Fashion-IQ {name}: {len(ids)} images ({'reuse' if name in existing else 'encode'})", flush=True)
        gallery = existing.get(name)
        if gallery is None:
            gallery = add_images(output, paths, encoder, image_ids=ids, batch_size=batch_size)
        require_full_gallery(gallery, ids)
        report["categories"][name] = {**gallery.validate(), **provenance,
                                       "reused": name in existing}
    write_json(Path(config["gallery"]["root"]) / "fashioniq_import.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-root", required=True, help="Contains images/, captions/, image_splits/")
    parser.add_argument("--category", choices=["all", *COUNTS], default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-only", action="store_true", help="Validate metadata/filenames without loading CLIP")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    print(import_fashioniq(args.config, args.data_root, args.category, args.batch_size, args.check_only))
