from pathlib import Path

import pytest

from keds.evaluation.datasets import cirr, fashioniq
from keds.utils.io import read_json


def test_official_fashioniq_annotations_and_reference_policy():
    root = Path("references/fashion-iq")
    if not root.exists():
        pytest.skip("Official Fashion-IQ metadata not checked out")
    for category in ("dress", "shirt", "toptee"):
        queries, ids, _ = fashioniq(root, category)
        assert all(q.reference_id in ids and q.target_id in ids and not q.exclude_reference for q in queries)
        assert len(queries) > 1000


def test_official_cirr_annotations_subset_and_hidden_test_targets():
    root = Path("references/cirr-dataset")
    if not root.exists():
        pytest.skip("Official CIRR metadata not checked out")
    queries, ids, _ = cirr(root, "val")
    assert len(queries) == 4181
    assert all(q.target_id in q.group_members and q.exclude_reference for q in queries)
    assert all(len(q.group_members) == 6 for q in queries)
    test_queries, _, _ = cirr(root, "test1")
    assert all(q.target_id is None for q in test_queries)
    examples = Path("references/cirr/demo_files/test1_pred_ranks_recall_subset.json")
    if examples.exists():
        assert read_json(examples)["metric"] == "recall_subset"

