"""Prepare immutable BEST checkpoints and existing CC3M features for Modal.

This also works locally. It does not change configs/model.yaml or old artifacts.
Credentials come from HF_TOKEN or the repository .env, never from CLI arguments.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

SOURCE = ("nightfurry/keds-500k-training-resume", "f14b9e3568e8e57fe522c7cf3397dc973840d632")
BEST = ("nightfurry/keds-500k-resume-training-v21", "359deb00753a732d0c18342e843678eb49cb7919")
CLIP = ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41")
BEST_FOLDER = f"checkpoints/{BEST[1]}"
CLIP_FILES = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors"]


def runtime_config(artifact_root, device="cuda"):
    """Keep parameter architecture and research profiles from the repository."""
    root = Path(artifact_root).resolve()
    values = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
    values["model"].update(device=device, weights_path=str(root / "clip"), local_files_only=True)
    values["knowledge"].update(database_path=str(root / "source/keds_knowledge_db.pt"),
                               index_dir=str(root / "knowledge"), resident_indexes=device == "cuda")
    values["checkpoints"] = {phase: str(root / BEST_FOLDER / f"{phase}_best_author_modules.pt")
                             for phase in ("phi_m", "phi_a")}
    values["gallery"]["root"] = str(root / "gallery")
    return values


def write_config(artifact_root, destination, device="cuda"):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(runtime_config(artifact_root, device), sort_keys=False), encoding="utf-8")
    return path


def download_verified(repo, revision, name, target, token, api):
    """Reuse verified LFS bytes; preserve and reject conflicting existing files."""
    from huggingface_hub import hf_hub_download
    from keds.utils.io import sha256

    info = api.model_info(repo, revision=revision, files_metadata=True)
    item = next((entry for entry in info.siblings if entry.rfilename == name), None)
    if item is None:
        raise ValueError(f"Missing required artifact {repo}@{revision}/{name}")
    expected = item.lfs.sha256 if item.lfs else None
    target = Path(target)
    if target.exists() and expected:
        actual = sha256(target)
        if actual != expected:
            raise ValueError(f"Existing artifact differs; preserving it: {target}")
    else:
        # Download into the Hub cache first: an existing target is never replaced.
        cached = Path(hf_hub_download(repo, name, revision=revision, token=token))
        actual = sha256(cached)
        if expected and actual != expected:
            raise ValueError(f"Remote checksum mismatch: {name}")
        if target.exists():
            if sha256(target) != actual:
                raise ValueError(f"Existing artifact differs; preserving it: {target}")
        else:
            import shutil
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(cached, temporary)
            temporary.replace(target)
    print(f"Verified {name}", flush=True)
    return {"repo": repo, "revision": revision, "file": name, "sha256": actual,
            "path": str(target), "verified": True}


def prepare(artifact_root):
    from dotenv import dotenv_values
    from filelock import FileLock
    from huggingface_hub import HfApi
    from keds.knowledge.database import KnowledgeDatabase
    from keds.knowledge.retriever import KnowledgeRetriever
    from keds.models.bkp import build_bkp
    from keds.models.checkpoint_loader import load_stream
    from keds.utils.io import write_json

    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or dotenv_values(ROOT / ".env").get("HF_TOKEN") or False
    api = HfApi(token=token)
    with FileLock(str(root / ".prepare.lock")):
        jobs = [(SOURCE, "keds_knowledge_db.pt", "source/keds_knowledge_db.pt")]
        jobs += [(BEST, name, f"{BEST_FOLDER}/{name}") for name in
                 ("phi_m_best_author_modules.pt", "phi_a_best_author_modules.pt",
                  "best_metrics.json", "resume_report.json")]
        jobs += [(CLIP, name, f"clip/{name}") for name in CLIP_FILES]
        records = [download_verified(*repo, name, root / relative, token, api)
                   for repo, name, relative in jobs]
        config = runtime_config(root, "cpu")
        checkpoints = {phase: load_stream(build_bkp(config["bkp"]), path, phase)
                       for phase, path in config["checkpoints"].items()}
        print("Validating existing CC3M features and exact FAISS shards", flush=True)
        db = KnowledgeDatabase(config["knowledge"]["database_path"], expected_count=500000)
        bank = db.validate()
        retriever = KnowledgeRetriever(db, config["knowledge"]["index_dir"], build=True)
        retriever.validate_order()
        report = {"artifacts": records, "checkpoints": checkpoints, "knowledge": bank,
                  "index_order_verified": True, "benchmark_run": False}
        write_json(root / "manifests/prepared.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    prepare(args.root)
