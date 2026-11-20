"""Load the pinned single-file CLIP checkpoint without a copy-on-write mapping."""
import json
import math
import mmap
import warnings
from pathlib import Path

import torch


def load_cpu_clip(source, model_class, config):
    path = Path(source) / "model.safetensors"
    with torch.device("meta"):
        model = model_class(config)
    expected = model.state_dict()
    buffers = dict(model.named_buffers())
    dtypes = {torch.float32: "F32", torch.int64: "I64"}
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        if not 2 <= size <= 1024**2:
            raise ValueError("Invalid CLIP safetensors header length")
        header = json.loads(handle.read(size))
        if header.pop("__metadata__", {}).get("format") != "pt":
            raise ValueError("Expected a PyTorch CLIP safetensors checkpoint")
        if not expected.keys() <= header.keys() or header.keys() - expected.keys() - buffers.keys():
            raise ValueError("CLIP safetensors keys differ from the pinned architecture")
        data_start = 8 + size
        cursor = 0
        metadata = []
        file_size = path.stat().st_size
        for name, item in sorted(header.items(), key=lambda entry: entry[1]["data_offsets"][0]):
            reference = expected[name] if name in expected else buffers[name]
            start, end = item["data_offsets"]
            shape = tuple(item["shape"])
            if (shape != tuple(reference.shape) or item["dtype"] != dtypes.get(reference.dtype) or
                    start != cursor or end - start != math.prod(shape) * reference.element_size() or
                    not start < end or data_start + end > file_size):
                raise ValueError(f"Invalid CLIP tensor metadata: {name}")
            metadata.append((name, reference.dtype, shape, data_start + start))
            cursor = end
        if data_start + cursor != file_size:
            raise ValueError("Unexpected trailing CLIP safetensors data")
        backing = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    model.requires_grad_(False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable.*", category=UserWarning)
        state = {name: torch.frombuffer(backing, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape)
                 for name, dtype, shape, offset in metadata}
    for name, reference in buffers.items():
        if name in expected:
            continue
        if not name.endswith(".position_ids"):
            raise ValueError(f"Unsupported CLIP buffer: {name}")
        owner, _, key = name.rpartition(".")
        position_ids = torch.arange(reference.numel()).reshape(reference.shape)
        if name in state and not torch.equal(state.pop(name), position_ids):
            raise ValueError(f"Invalid CLIP position IDs: {name}")
        setattr(model.get_submodule(owner), key, position_ids)
    model.load_state_dict(state, strict=True, assign=True)
    return model
