import json

import numpy as np
import pytest

from keds.evaluation.candidates import CandidateManifest
from keds.evaluation.statistics import paired_bootstrap


def test_paired_bootstrap_zero_delta_and_reproducibility():
    ranks = [1, 2, 5, 10, 50]
    same = paired_bootstrap(ranks, ranks, repetitions=200)
    assert all(v["delta"] == 0 and v["ci95"] == [0, 0] for v in same.values())
    better = paired_bootstrap([3]*12, [1]*12, groups=["dress"]*4+["shirt"]*4+["toptee"]*4, repetitions=200)
    assert better["R@1"]["delta"] == 100
    assert better["MRR"]["ci_excludes_zero"]
    assert better == paired_bootstrap([3]*12, [1]*12,
        groups=np.repeat(["dress", "shirt", "toptee"], 4), repetitions=200)
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 2])


def test_candidate_manifest_rejects_wrong_protocol_counts_and_duplicate_ids(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps({"query_id": "one", "candidate_ids": ["same"]*1500}) + "\n")
    with pytest.raises(ValueError, match="6016"):
        CandidateManifest(path, ["one"], {"one": {"same"}})
    with pytest.raises(ValueError, match="distinct"):
        CandidateManifest(path, ["one"], {"one": {"same"}}, expected_total=1)
    ids = [f"i{x}" for x in range(1500)]
    path.write_text(json.dumps({"query_id": "one", "candidate_ids": ids}) + "\n")
    candidates = CandidateManifest(path, ["one"], {"one": set(ids)}, expected_total=1)
    assert candidates.get("one") == ids

