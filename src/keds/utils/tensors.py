import numpy as np
import torch


def normalized_numpy(values, dim: int | None = None) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float().numpy()
    array = np.ascontiguousarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or (dim is not None and array.shape[1] != dim):
        raise ValueError(f"Expected non-empty [N,{dim or 'D'}] vectors")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Embeddings must be finite and nonzero")
    return np.ascontiguousarray(array / norms, dtype=np.float32)

