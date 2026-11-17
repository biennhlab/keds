"""Use native FAISS on Linux and isolate the LLVM runtime on Windows.

The PyTorch Windows wheel uses Intel OpenMP; FAISS's wheel uses LLVM OpenMP.
Loading both into one process causes OMP Error #15 and process abort. A small
persistent FAISS worker solves the runtime conflict without unsafe environment
flags, altered DLLs, or approximate/numpy fallback search.
"""
from __future__ import annotations

import atexit
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

_process = None
_mutex = threading.RLock()


def _stop():
    global _process
    if _process is not None:
        _process.stdin.close()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.terminate()
            _process.wait(timeout=5)
        _process = None


def _rpc(**request):
    global _process
    with _mutex:
        if _process is None:
            environment = dict(os.environ)
            source = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
            _process = subprocess.Popen([sys.executable, "-m", "keds.utils.faiss_worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env=environment,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if _process.poll() is not None:
            raise RuntimeError("FAISS worker exited unexpectedly; restart the service")
        pickle.dump(request, _process.stdin, protocol=5)
        _process.stdin.flush()
        try:
            response = pickle.load(_process.stdout)
        except EOFError as e:
            raise RuntimeError("FAISS worker failed") from e
        if not response["ok"]:
            raise ValueError(response["error"])
        return response["result"]


class _FlatIP:
    def __init__(self, dim, _info=None):
        info = _info or _rpc(op="create", dim=dim)
        self._id, self.d, self.ntotal = info["id"], info["d"], info["ntotal"]

    def add(self, values):
        _rpc(op="add", id=self._id, args=[values])
        self.ntotal += len(values)

    def search(self, query, k):
        return _rpc(op="search", id=self._id, args=[query, k])

    def reconstruct_n(self, offset, count):
        return _rpc(op="reconstruct_n", id=self._id, args=[offset, count])

    def __del__(self):
        if _process is not None and _process.poll() is None and hasattr(self, "_id"):
            try:
                _rpc(op="delete", id=self._id)
            except (BrokenPipeError, OSError, RuntimeError, ValueError):
                pass


class _IsolatedBackend:
    IndexFlatIP = _FlatIP

    @staticmethod
    def read_index(path):
        info = _rpc(op="read", path=path)
        return _FlatIP(info["d"], _info=info)

    @staticmethod
    def write_index(index, path):
        return _rpc(op="write", id=index._id, path=path)


if os.name == "nt":
    faiss = _IsolatedBackend()
    atexit.register(_stop)
else:
    import faiss as faiss

