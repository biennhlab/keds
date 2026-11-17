"""Strict Candidate-1500 manifests; candidates are supplied, never label-mined."""
import json

from keds.utils.io import sha256


class CandidateManifest:
    def __init__(self, path, query_ids, gallery_ids_by_query, expected_total=6016):
        self.path, self.offsets = path, {}
        if len(query_ids) != expected_total or len(set(query_ids)) != expected_total:
            raise ValueError(f"Full Candidate-1500 requires exactly {expected_total} fixed query IDs")
        with open(path, "rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                query_id, ids = row["query_id"], row["candidate_ids"]
                if query_id in self.offsets or query_id not in gallery_ids_by_query:
                    raise ValueError("Duplicate/unknown query ID in candidate manifest")
                if len(ids) != 1500 or len(set(ids)) != 1500:
                    raise ValueError("Candidate-1500 requires exactly 1500 distinct candidates per query")
                if not set(ids) <= gallery_ids_by_query[query_id]:
                    raise ValueError("Candidate IDs outside the official category gallery")
                self.offsets[query_id] = offset
        if set(self.offsets) != set(query_ids):
            raise ValueError("Candidate manifest must cover all 6016 queries; no silent query sampling")
        self.sha256 = sha256(path)

    def get(self, query_id):
        with open(self.path, "rb") as stream:
            stream.seek(self.offsets[query_id])
            return json.loads(stream.readline())["candidate_ids"]

