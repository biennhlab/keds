from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        buffer = bytearray(1024 * 1024)
        while (count := f.readinto(buffer)):
            h.update(memoryview(buffer)[:count])
    return h.hexdigest()


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

