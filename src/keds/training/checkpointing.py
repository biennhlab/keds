from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, epoch, global_step, phase, meta, scheduler=None, losses=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step,
               "phase": phase, "meta": meta, "losses": losses or [],
               "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       "numpy_name": np_state[0], "numpy_keys": np_state[1].tolist(),
                       "numpy_pos": np_state[2], "numpy_has_gauss": np_state[3],
                       "numpy_cached_gaussian": np_state[4]}}
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

