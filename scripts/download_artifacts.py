"""Download pinned existing artifacts, verifying Hub LFS SHA256. Never encode/train.

Uses HF_TOKEN from .env or environment. No token is written to reports.
Run with the Python environment containing huggingface-hub and python-dotenv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ("nightfurry/keds-500k-training-resume", "f14b9e3568e8e57fe522c7cf3397dc973840d632")
RESUME = ("nightfurry/keds-500k-resume-training-v21", "42dcde4c9b437e1eac8978c7bed9de117178b12b")
BEST = (RESUME[0], "359deb00753a732d0c18342e843678eb49cb7919")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Also download LAST optimizer checkpoints")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--skip-knowledge", action="store_true")
    parser.add_argument("--clip", action="store_true", help="Also download pinned CLIP/tokenizer")
    args = parser.parse_args()
    token = dotenv_values(ROOT / ".env").get("HF_TOKEN") or os.environ.get("HF_TOKEN") or False
    report = ROOT / "reports" / "artifact_download_manifest.json"
    records = json.loads(report.read_text()) if report.exists() else []
    downloads = [
        (SOURCE, "source", ([] if args.skip_knowledge else ["keds_knowledge_db.pt"])),
        (BEST, f"checkpoints/{BEST[1]}", ["phi_m_best_author_modules.pt", "phi_a_best_author_modules.pt",
                                           "best_metrics.json", "resume_report.json"]),
        (RESUME, "metadata", ["resume_report.json", "best_metrics.json", "retrieval_cache_metadata_v21.json"]),
        (SOURCE, "metadata", ["official_parity_preflight.json", "cc3m_shard_manifest.json"]),
    ]
    if args.resume:
        downloads.append((RESUME, "checkpoints", ["phi_m_last.pt", "phi_a_last.pt"]))
    if args.clip:
        downloads.append((("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
                          "clip", ["config.json", "preprocessor_config.json", "tokenizer.json",
                                   "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                                   "merges.txt", "model.safetensors"]))
    for (repo, revision), folder, names in downloads:
        if not names:
            continue
        info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
        siblings = {x.rfilename: x for x in info.siblings}
        for name in names:
            item = siblings[name]
            record = {"repo": repo, "revision": revision, "file": name, "size": item.size,
                      "sha256": item.lfs.sha256 if item.lfs else None}
            if not args.metadata_only or name.endswith(".json"):
                target_dir = ROOT / "artifacts" / folder
                target = target_dir / name
                if target.exists() and record["sha256"] and digest(target) != record["sha256"]:
                    raise ValueError(f"Existing artifact differs; preserving it: {target}")
                print(f"Downloading {repo}/{name}", flush=True)
                # A verified file is immutable. Avoid invoking Hub download on
                # active mmap weights: replacing them can crash Windows readers.
                result = (str(target) if target.exists() and record["sha256"] else
                          hf_hub_download(repo, name, revision=revision, token=token,
                                          local_dir=target_dir))
                actual = digest(result)
                if record["sha256"] and actual != record["sha256"]:
                    raise ValueError(f"SHA256 mismatch: {name}")
                record.update({"local_path": str(Path(result).relative_to(ROOT)),
                               "sha256": actual, "download_verified": True})
                print(f"Verified {name}: {actual}", flush=True)
            records = [x for x in records if (x["repo"], x["revision"], x["file"]) !=
                       (repo, revision, name)] + [record]
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

