import pytest

from keds.evaluation.metrics import category_macro, ranking_metrics


def test_known_full_ranks_mrr_and_macro():
    result = ranking_metrics([["a", "b", "c"], ["a", "c", "b"]], ["a", "b"], ks=(1, 2, 3))
    assert result["R@1"] == 50 and result["R@3"] == 100
    assert result["MRR"] == pytest.approx((1 + 1/3) / 2)
    with pytest.raises(ValueError):
        ranking_metrics([["a", "b"]], ["c"])
    with pytest.raises(ValueError):
        category_macro({"dress": result})

