"""Upload only full-validation Fashion-IQ images and metadata to Modal Volume.

Run with the Modal client Python. Files outside these splits are not uploaded.
Existing data is preserved; a matching upload manifest permits safe reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

COUNTS = {"dress": 3817, "shirt": 6346, "toptee": 5373}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_upload(data_root):
    root = Path(data_root).resolve()
    required, jobs = set(), []
    for category, expected in COUNTS.items():
        split = root / "image_splits" / f"split.{category}.val.json"
        ids = json.loads(split.read_text(encoding="utf-8"))
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"Invalid full {category} validation split")
        required.update(ids)
        jobs.extend((root / folder / name, f"{folder}/{name}") for folder, name in
                    [("image_splits", split.name), ("captions", f"cap.{category}.val.json")])
    images = {}
    for path in (root / "images").rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS and path.stem in required:
            if path.stem in images:
                raise ValueError(f"Duplicate image ID: {path.stem}")
            images[path.stem] = path
    missing = required - images.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} images: {sorted(missing)[:8]}")
    jobs += [(images[image_id], f"images/{images[image_id].name}") for image_id in sorted(required)]
    records = [{"path": relative, "size": path.stat().st_size, "sha256": file_hash(path)}
               for path, relative in jobs]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return jobs, {"dataset": "fashioniq", "split": "val", "gallery_entries": sum(COUNTS.values()),
                  "unique_images": len(images), "fingerprint": fingerprint, "files": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/fashioniq")
    parser.add_argument("--volume", default="keds-artifacts")
    parser.add_argument("--remote-root", default="/keds/datasets/fashioniq")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    jobs, manifest = plan_upload(args.data_root)
    print({key: value for key, value in manifest.items() if key != "files"}, flush=True)
    print(f"Selected {len(jobs)} files, {sum(row['size'] for row in manifest['files']) / 1024**2:.1f} MiB", flush=True)
    if args.check_only:
        return
    import modal

    volume = modal.Volume.from_name(args.volume)
    remote = args.remote_root.rstrip("/")
    manifest_path = remote + "/upload_manifest.json"
    try:
        old = json.loads(b"".join(volume.read_file(manifest_path)))
    except FileNotFoundError:
        old = None
    if old is not None:
        if old != manifest:
            raise ValueError("Existing dataset upload manifest differs; preserving remote files")
        print("Matching upload manifest found; reusing existing dataset.", flush=True)
        return
    with modal.enable_output():
        with volume.batch_upload(force=False) as batch:
            for path, relative in jobs:
                batch.put_file(path, remote + "/" + relative)
            batch.put_file(io.BytesIO(json.dumps(manifest, indent=2).encode()), manifest_path)
    print("Fashion-IQ validation images and metadata uploaded successfully.", flush=True)


if __name__ == "__main__":
    main()
