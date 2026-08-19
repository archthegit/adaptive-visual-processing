import numpy as np

from src.experiment1.encoder_temporal import (
    adjacent_cosine_similarity,
    pool_temporal_representations,
    temporal_representation_summary,
)


def test_pool_temporal_representations_uses_grid_temporal_axis():
    reps = np.arange(2 * 2 * 2 * 3, dtype=float).reshape(8, 3)
    pooled = pool_temporal_representations(reps, [2, 2, 2], spatial_merge_size=1)
    np.testing.assert_allclose(pooled[0], reps[:4].mean(axis=0))
    np.testing.assert_allclose(pooled[1], reps[4:].mean(axis=0))


def test_pool_temporal_representations_supports_merged_grid():
    reps = np.arange(2 * 1 * 1 * 3, dtype=float).reshape(2, 3)
    pooled = pool_temporal_representations(reps, [2, 2, 2], spatial_merge_size=2)
    np.testing.assert_allclose(pooled, reps)


def test_adjacent_cosine_similarity_and_summary():
    temporal = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    assert adjacent_cosine_similarity(temporal) == [0.0, 1.0]
    summary = temporal_representation_summary("stage", temporal)
    assert summary["mean_adjacent_bin_cosine_similarity"] == 0.5
    assert summary["num_temporal_bins"] == 3
