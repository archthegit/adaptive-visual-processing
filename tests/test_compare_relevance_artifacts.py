import numpy as np

from scripts.compare_relevance_artifacts import compare_topk_logits, max_abs_diff


def test_max_abs_diff_reports_inf_for_shape_mismatch():
    assert max_abs_diff([1.0], [[1.0]]) == float("inf")


def test_max_abs_diff_compares_nested_scores():
    assert np.isclose(max_abs_diff([[1.0, 2.0]], [[1.0, 2.25]]), 0.25)


def test_compare_topk_logits_reports_token_order_and_logit_diff():
    left = {"metadata": {"prefill_next_token_topk": [{"token_id": 1, "logit": 2.0}]}}
    right = {"metadata": {"prefill_next_token_topk": [{"token_id": 1, "logit": 1.75}]}}
    summary = compare_topk_logits(left, right)
    assert summary["token_ids_match"] is True
    assert np.isclose(summary["max_abs_logit_diff_matching_positions"], 0.25)
