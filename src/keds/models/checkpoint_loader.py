from __future__ import annotations

from pathlib import Path

import torch

from keds.utils.io import sha256


class IncompatibleCheckpoint(ValueError):
    pass


def read_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run scripts/download_artifacts.py or configure it.")
    # Never enable arbitrary pickle execution for downloaded artifacts.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise IncompatibleCheckpoint("Checkpoint must be a dictionary")
    return payload


def extract_state(payload: dict) -> dict:
    if "state_dict" in payload:
        state = payload["state_dict"]
    elif all(k in payload for k in ("state_dict_img2text", "state_dict_retrieval_fuse",
                                    "state_dict_text_condition")):
        state = {}
        for name in ("img2text", "retrieval_fuse", "text_condition"):
            for key, value in payload[f"state_dict_{name}"].items():
                state[f"{name}.{key.removeprefix('module.')}"] = value
    else:
        raise IncompatibleCheckpoint("Unknown checkpoint schema. Expected author BKP state_dict/modules.")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise IncompatibleCheckpoint("State dict must contain only tensors")
    if not all(torch.isfinite(v).all().item() for v in state.values()):
        raise IncompatibleCheckpoint("Non-finite checkpoint weights")
    return state


def load_stream(model, path: str | Path, phase: str | None = None) -> dict:
    payload = read_checkpoint(path)
    meta = payload.get("meta", {})
    if payload.get("phase") and phase and payload["phase"] != phase:
        raise IncompatibleCheckpoint(f"Expected {phase}, checkpoint is {payload['phase']}")
    if meta.get("architecture", model.architecture) not in (model.architecture, "suoych/KEDs model.py"):
        raise IncompatibleCheckpoint("Checkpoint architecture version mismatch")
    if meta.get("branch") and phase and meta["branch"].lower().replace("-", "_") != phase:
        raise IncompatibleCheckpoint("Checkpoint branch metadata mismatch")
    if meta.get("dim", 768) != 768 or meta.get("backbone", "openai/clip-vit-large-patch14") not in (
        "openai/clip-vit-large-patch14", "ViT-L/14"
    ):
        raise IncompatibleCheckpoint("Checkpoint CLIP configuration mismatch")
    if meta.get("model", "AuthorBKP_IM2TEXT_CrossFormer3") not in (
        "AuthorBKP_IM2TEXT_CrossFormer3", "openai/clip-vit-large-patch14", "ViT-L/14"
    ):
        raise IncompatibleCheckpoint("Checkpoint belongs to an unsupported model family")
    try:
        model.load_state_dict(extract_state(payload), strict=True)
    except RuntimeError as e:
        raise IncompatibleCheckpoint(f"Strict BKP load failed: {e}") from e
    return {"path": str(Path(path).resolve()), "sha256": sha256(path),
            "epoch": payload.get("epoch"), "phase": payload.get("phase"),
            "meta": meta, "has_optimizer": bool(payload.get("optimizer"))}

