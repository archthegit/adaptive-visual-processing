import numpy as np
import sys
from types import SimpleNamespace

from src.experiment1.encoder_temporal import (
    VisionTemporalCapture,
    adjacent_cosine_similarity,
    canonicalize_window_ordered_groups,
    pool_temporal_representations,
    qwen_vision_window_indices,
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


def test_window_reordered_groups_recover_canonical_temporal_bins():
    canonical_groups = np.array(
        [
            [[1.0], [1.0]],
            [[2.0], [2.0]],
            [[10.0], [10.0]],
            [[20.0], [20.0]],
        ]
    )
    window_index = np.array([2, 3, 0, 1])
    reverse_indices = np.argsort(window_index)
    window_ordered = canonical_groups[window_index].reshape(8, 1)

    recovered = canonicalize_window_ordered_groups(window_ordered, reverse_indices)
    temporal = pool_temporal_representations(recovered, [2, 2, 2], spatial_merge_size=1)

    np.testing.assert_allclose(recovered, canonical_groups.reshape(8, 1))
    np.testing.assert_allclose(temporal[:, 0], [1.5, 15.0])


def test_qwen_vision_window_indices_finds_nested_visual_module(monkeypatch):
    captured = {}

    def fake_get_vision_window_index(video_grid_thw, spatial_merge_size, window_size, patch_size, kwargs):
        captured["spatial_merge_size"] = spatial_merge_size
        captured["window_size"] = window_size
        captured["patch_size"] = patch_size
        return np.array([2, 0, 1]), None

    monkeypatch.setitem(
        sys.modules,
        "transformers.vision_utils",
        SimpleNamespace(get_vision_window_index=fake_get_vision_window_index),
    )

    class Visual:
        spatial_merge_size = 2
        window_size = 112
        patch_size = 14

    class WrappedModel:
        def __init__(self):
            self.backbone = SimpleNamespace(visual=Visual())

        def named_modules(self):
            return iter((("", self), ("backbone.visual", self.backbone.visual)))

    window_index, reverse_indices = qwen_vision_window_indices(WrappedModel(), [[3, 2, 2]])

    assert captured == {"spatial_merge_size": 2, "window_size": 112, "patch_size": 14}
    np.testing.assert_array_equal(np.asarray(window_index), [2, 0, 1])
    np.testing.assert_array_equal(np.asarray(reverse_indices), [1, 2, 0])


def test_qwen_vision_window_indices_fails_loudly_for_video_without_reverse_indices(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers.vision_utils", SimpleNamespace())

    class Model:
        def named_modules(self):
            return iter((("", self),))

    import pytest

    with pytest.raises(RuntimeError, match="reverse indices"):
        qwen_vision_window_indices(Model(), [[2, 2, 2]])


def test_encoder_artifact_reports_canonical_recovery_for_window_ordered_stages():
    canonical_block_groups = np.array(
        [
            [[1.0], [1.0], [2.0], [2.0]],
            [[10.0], [10.0], [20.0], [20.0]],
        ]
    )
    window_index = np.array([1, 0])
    reverse_indices = np.argsort(window_index)
    window_block = canonical_block_groups[window_index].reshape(8, 1)
    window_merger = np.array([[15.0], [1.5]])

    capture = VisionTemporalCapture(
        video_grid_thw=[[2, 2, 2]],
        spatial_merge_size=2,
        reverse_indices=reverse_indices,
        raw_outputs={
            "vision_block_early": window_block,
            "vision_block_middle": window_block,
            "vision_block_late": window_block,
            "vision_merger_pre_reverse": window_merger,
            "vision_final": np.array([[1.5], [15.0]]),
        },
        stage_specs={
            "vision_block_early": {"order": "window", "spatial_merge_size": 1},
            "vision_block_middle": {"order": "window", "spatial_merge_size": 1},
            "vision_block_late": {"order": "window", "spatial_merge_size": 1},
            "vision_merger_pre_reverse": {"order": "window", "spatial_merge_size": 2},
            "vision_final": {"order": "canonical", "spatial_merge_size": 2},
        },
    )

    artifact = capture.to_json_dict()
    stages = artifact["stages"]
    for label in (
        "vision_block_early",
        "vision_block_middle",
        "vision_block_late",
        "vision_merger_pre_reverse",
    ):
        assert stages[label]["canonical_order_recovered"] is True
    assert stages["vision_final"]["canonical_order_recovered"] is False
    np.testing.assert_allclose(
        np.asarray(stages["vision_merger_pre_reverse"]["temporal_representations"])[:, 0],
        [1.5, 15.0],
    )
