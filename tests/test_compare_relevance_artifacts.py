import numpy as np

from scripts.compare_relevance_artifacts import max_abs_diff


def test_max_abs_diff_reports_inf_for_shape_mismatch():
    assert max_abs_diff([1.0], [[1.0]]) == float("inf")


def test_max_abs_diff_compares_nested_scores():
    assert np.isclose(max_abs_diff([[1.0, 2.0]], [[1.0, 2.25]]), 0.25)

