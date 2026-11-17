from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from keds.utils.io import read_json, sha256


@dataclass(frozen=True)
class Query:
    query_id: str
    reference_id: str
    modification: str
    target_id: str | None = None
    group_members: tuple[str, ...] = ()
    exclude_reference: bool = False


def require_full_gallery(gallery, ids):
    if len(ids) != len(set(ids)) or set(gallery.by_id) != set(ids):
        raise ValueError("Evaluation requires exactly the official full split gallery, with original IDs")


def fashioniq(root, category, split="val"):
    if split != "val" or category not in ("dress", "shirt", "toptee"):
        raise ValueError("Labeled Fashion-IQ evaluation supports official val categories only")
    root = Path(root)
    annotations = root / "captions" / f"cap.{category}.{split}.json"
    split_path = root / "image_splits" / f"split.{category}.{split}.json"
    rows, ids = read_json(annotations), read_json(split_path)
    queries = []
    for i, row in enumerate(rows):
        captions = row["captions"]
        if len(captions) != 2 or row["candidate"] == row["target"]:
            raise ValueError("Malformed Fashion-IQ annotation")
        # Match suoych/KEDs data.py: cap2 and cap1, one composed feature/query.
        queries.append(Query(f"{category}-{i}", row["candidate"],
                             f"{captions[1]} and {captions[0]}", row["target"]))
    return queries, ids, {"annotations_sha256": sha256(annotations), "split_sha256": sha256(split_path),
                           "caption_protocol": "official_keds_cap2_and_cap1"}


def cirr(root, split="val"):
    if split not in ("val", "test1"):
        raise ValueError("Use CIRR val for labeled metrics, test1 for submission only")
    root = Path(root)
    annotations = root / "captions" / f"cap.rc2.{split}.json"
    split_path = root / "image_splits" / f"split.rc2.{split}.json"
    rows, split_info = read_json(annotations), read_json(split_path)
    ids = list(split_info) if isinstance(split_info, dict) else split_info
    queries = []
    for row in rows:
        group = tuple(row["img_set"]["members"])
        target = row.get("target_hard") if split == "val" else None
        if row["reference"] not in group or len(set(group)) != len(group):
            raise ValueError("Invalid CIRR group/reference")
        if split == "val" and (target not in group or target == row["reference"]):
            raise ValueError("Invalid labeled CIRR target")
        queries.append(Query(str(row["pairid"]), row["reference"], row["caption"], target, group, True))
    return queries, ids, {"annotations_sha256": sha256(annotations), "split_sha256": sha256(split_path)}

