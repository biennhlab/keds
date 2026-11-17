"""Private subprocess protocol. This process MUST NOT import torch.

Only its parent can write this anonymous pipe. Pickle is used exclusively for
trusted in-process numpy RPC, never for uploaded files or model artifacts.
"""
import pickle
import sys

import faiss


def main():
    faiss.omp_set_num_threads(2)
    objects, next_id = {}, 0
    source, target = sys.stdin.buffer, sys.stdout.buffer
    while True:
        try:
            request = pickle.load(source)
        except EOFError:
            break
        try:
            operation = request["op"]
            if operation in ("create", "read"):
                obj = (faiss.IndexFlatIP(request["dim"]) if operation == "create" else
                       faiss.read_index(request["path"]))
                if not isinstance(obj, faiss.IndexFlatIP):
                    raise ValueError("Only exact IndexFlatIP is supported")
                objects[next_id] = obj
                result = {"id": next_id, "d": obj.d, "ntotal": obj.ntotal}
                next_id += 1
            elif operation == "delete":
                objects.pop(request["id"], None)
                result = None
            elif operation == "write":
                result = faiss.write_index(objects[request["id"]], request["path"])
            else:
                obj = objects[request["id"]]
                result = getattr(obj, operation)(*request.get("args", []))
            reply = {"ok": True, "result": result}
        except Exception as e:
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        pickle.dump(reply, target, protocol=5)
        target.flush()


if __name__ == "__main__":
    main()

