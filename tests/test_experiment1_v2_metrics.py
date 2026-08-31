import pytest

from src.experiment1.v2_metrics import (
    adjacent_nonadjacent_far_similarity,
    benjamini_hochberg,
    bins_to_mass,
    bootstrap_ci_clustered_by_video,
    effective_rank,
    gini_coefficient,
    jensen_shannon_divergence,
    lag_similarity_profile,
    normalized_entropy,
    paired_differences,
    paired_effect_size,
    paired_permutation_pvalue,
    spearman_from_scores,
    temporal_distribution_metrics,
    top_fraction_jaccard,
    top_fraction_mass,
    uniform_top_fraction_expectation,
)


def test_temporal_distribution_metrics_cover_entropy_gini_top20_and_80pct():
    values = [0.7, 0.1, 0.1, 0.1]
    metrics = temporal_distribution_metrics(values)
    assert metrics["normalized_entropy"] < 0.7
    assert metrics["top_20pct_bin_mass"] == pytest.approx(0.7)
    assert metrics["uniform_top_20pct_expectation"] == pytest.approx(0.25)
    assert metrics["gini"] == pytest.approx(gini_coefficient(values))
    assert metrics["bins_to_80pct_mass"] == 2
    assert metrics["first_bin_mass"] == pytest.approx(0.7)
    assert metrics["last_bin_mass"] == pytest.approx(0.1)
    assert metrics["top_bin_relative_temporal_position"] == pytest.approx(0.0)
    assert metrics["rank_order"] == [0, 1, 2, 3]


def test_uniform_distribution_metrics_match_expectations():
    values = [1, 1, 1, 1, 1]
    assert normalized_entropy(values) == pytest.approx(1.0)
    assert top_fraction_mass(values, 0.2) == pytest.approx(0.2)
    assert uniform_top_fraction_expectation(5, 0.2) == pytest.approx(0.2)
    assert bins_to_mass(values, 0.8) == 4
    assert gini_coefficient(values) == pytest.approx(0.0)


def test_jsd_spearman_and_top_fraction_jaccard():
    left = [0.5, 0.3, 0.2]
    right = [0.2, 0.3, 0.5]
    assert jensen_shannon_divergence(left, left) == pytest.approx(0.0)
    assert jensen_shannon_divergence(left, right) > 0
    assert spearman_from_scores(left, left) == pytest.approx(1.0)
    assert spearman_from_scores(left, right) == pytest.approx(-1.0)
    assert top_fraction_jaccard(left, right, 0.34) == pytest.approx(1 / 3)


def test_temporal_representation_similarity_and_effective_rank():
    reps = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    grouped = adjacent_nonadjacent_far_similarity(reps)
    assert grouped["adjacent_cosine_similarity"] is not None
    assert grouped["far_bin_cosine_similarity"] is not None
    profile = lag_similarity_profile(reps)
    assert [item["lag"] for item in profile] == [1, 2, 3]
    assert effective_rank(reps) > 1.0


def test_paired_differences_and_clustered_bootstrap():
    records = [
        {"source_video_id": "v1", "participant_id": "p1", "condition": "baseline", "score": 1.0},
        {"source_video_id": "v1", "participant_id": "p1", "condition": "top_masked", "score": 0.5},
        {"source_video_id": "v2", "participant_id": "p2", "condition": "baseline", "score": 0.8},
        {"source_video_id": "v2", "participant_id": "p2", "condition": "top_masked", "score": 0.3},
    ]
    paired = paired_differences(records, "score")
    assert paired["left_condition"] == "baseline"
    assert paired["right_condition"] == "top_masked"
    assert paired["mean_difference"] == pytest.approx(-0.5)
    ci = bootstrap_ci_clustered_by_video(records, "score", replicates=100, seed=1)
    assert ci["replicates"] == 100
    assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]


def test_permutation_effect_size_and_bh_correction():
    diffs = [1.0, 2.0, 3.0]
    assert 0.0 < paired_permutation_pvalue(diffs, replicates=100, seed=1) <= 1.0
    assert paired_effect_size(diffs) > 0
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0.0 <= value <= 1.0 for value in adjusted)
