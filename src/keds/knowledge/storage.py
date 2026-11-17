"""Read-only, zero-copy reader for contiguous tensor banks in torch ZIP files.

Windows torch mmap uses copy-on-write and can reserve commit for the entire 3 GB
archive. Read-only NumPy mappings avoid that reservation without re-encoding or
rewriting any artifact. Only a small explicit pickle whitelist is interpreted;
no functions from the artifact are imported or executed.
"""
import collections
import pickle
import struct
import sys
import zipfile

import numpy as np

DTYPES = {"FloatStorage": np.dtype("float32"), "HalfStorage": np.dtype("float16"),
          "DoubleStorage": np.dtype("float64"), "LongStorage": np.dtype("int64"),
          "IntStorage": np.dtype("int32"), "BoolStorage": np.dtype("bool")}


def read_bank_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        name = next(n for n in names if n.endswith("/data.pkl"))
        prefix = name.rsplit("/", 1)[0]
        order_file = f"{prefix}/byteorder"
        if order_file in names and archive.read(order_file).decode() != sys.byteorder:
            raise ValueError("Knowledge archive byte order differs from host")

        def storage_offset(key):
            info = archive.getinfo(f"{prefix}/data/{key}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("Knowledge tensor storage must be uncompressed")
            with open(path, "rb") as raw:
                raw.seek(info.header_offset)
                header = raw.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP tensor local header")
            filename_length, extra_length = struct.unpack_from("<HH", header, 26)
            return info.header_offset + 30 + filename_length + extra_length, info.file_size

        def rebuild(storage, offset, shape, strides, requires_grad, backward_hooks, *extra):
            dtype, key, count = storage
            expected = tuple(np.cumprod((1,) + tuple(shape[:0:-1]))[::-1])
            if tuple(strides) != expected or offset < 0 or np.prod(shape) + offset > count:
                raise ValueError("Knowledge tensors must be contiguous with valid storage bounds")
            start, size = storage_offset(key)
            if count * dtype.itemsize > size:
                raise ValueError("Knowledge storage is truncated")
            return np.memmap(path, mode="r", dtype=dtype, offset=start + offset * dtype.itemsize,
                             shape=tuple(shape), order="C")

        class Reader(pickle.Unpickler):
            def find_class(self, module, symbol):
                if module == "torch._utils" and symbol == "_rebuild_tensor_v2":
                    return rebuild
                if module == "torch" and symbol in DTYPES:
                    return DTYPES[symbol]
                if module == "collections" and symbol == "OrderedDict":
                    return collections.OrderedDict
                raise pickle.UnpicklingError(f"Unsupported knowledge archive type: {module}.{symbol}")

            def persistent_load(self, identifier):
                if not isinstance(identifier, tuple) or len(identifier) != 5 or identifier[0] != "storage":
                    raise pickle.UnpicklingError("Unsupported storage reference")
                _, dtype, key, location, count = identifier
                if dtype not in DTYPES.values() or not isinstance(key, str) or "/" in key:
                    raise pickle.UnpicklingError("Invalid storage descriptor")
                return dtype, key, count

        with archive.open(name) as stream:
            return Reader(stream).load()

