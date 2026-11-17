from __future__ import annotations

import random

import numpy as np
import torch

from keds.models.checkpoint_loader import IncompatibleCheckpoint, load_stream, read_checkpoint


def resume(model, optimizer, checkpoint_path, phase, scheduler=None, exact=False, objective_id=None):
    details = load_stream(model, checkpoint_path, phase)
    payload = read_checkpoint(checkpoint_path)
    if not payload.get("optimizer"):
        raise IncompatibleCheckpoint("Optimizer absent; use a separately identified weights-only fine-tuning run")
    if exact and ("rng" not in payload or "global_step" not in payload or
                  (scheduler is not None and "scheduler" not in payload) or
                  payload.get("meta", {}).get("objective_id") != objective_id):
        raise IncompatibleCheckpoint("Exact resume requires matching objective, RNG, global step and scheduler")
    optimizer.load_state_dict(payload["optimizer"])
    for param, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in state or state[key].shape != param.shape or not torch.isfinite(state[key]).all():
                raise IncompatibleCheckpoint(f"Optimizer tensor incompatible: {key}")
        for key, tensor in state.items():
            if isinstance(tensor, torch.Tensor) and key != "step":
                state[key] = tensor.to(param.device)
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if exact:
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state((rng["numpy_name"], np.array(rng["numpy_keys"], dtype=np.uint32),
                             rng["numpy_pos"], rng["numpy_has_gauss"], rng["numpy_cached_gaussian"]))
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
    return {"next_epoch": int(payload["epoch"]) + 1, "global_step": payload.get("global_step", 0),
            "resume_mode": "exact" if exact else "optimizer_continuation",
            "previous_objective_verified": payload.get("meta", {}).get("objective_id") == objective_id,
            "source": details, "losses": payload.get("losses", [])}

