from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_temporal_route_reuse import (
    LAYER_GAPS,
    RETENTION_RATIOS,
    captured_mass,
    jensen_shannon_divergence,
    route_reuse_rows,
    spearman_correlation,
    topk_indices,
    topk_jaccard,
)


def _artifact(question_id: str, video_id: str, layers: int = 12, shift: int = 0) -> dict:
    rows = []
    for layer in range(layers):
        hot = (layer + shift) % 8
        values = np.full(8, 0.05, dtype=np.float64)
        values[hot] = 0.45
        values[(hot + 1) % 8] = 0.20
        values[(hot + 7) % 8] = 0.10
        values = values / values.sum()
        rows.append(values.tolist())
    return {
        "question_id": question_id,
        "question_type": "fine_grained_action_recognition",
        "category": "fine_grained",
        "video_clip": [{"video_id": video_id, "participant_id": video_id.split("-", 1)[0]}],
        "temporal_relevance": {"normalized_temporal_bin_scores": rows},
    }


def _manifest(records: list[tuple[str, str, str]]) -> dict[str, dict]:
    output = {}
    for index, (question_id, video_id, split) in enumerate(records):
        output[question_id] = {
            "question_id": question_id,
            "source_video_id": video_id,
            "participant_id": video_id.split("-", 1)[0],
            "category": "fine_grained",
            "question_type": "fine_grained_action_recognition",
            "duration_group": "short" if index % 2 == 0 else "long",
            "split": split,
        }
    return output


def _fixtures():
    records = [
        ("q-dev-a", "P00-video-a", "dev"),
        ("q-dev-b", "P01-video-b", "dev"),
        ("q-dev-c", "P02-video-c", "dev"),
        ("q-test-a", "P03-video-d", "test"),
        ("q-test-b", "P04-video-e", "test"),
    ]
    artifacts = {
        question_id: _artifact(question_id, video_id, shift=index)
        for index, (question_id, video_id, _split) in enumerate(records)
    }
    manifest = _manifest(records)
    return artifacts, artifacts.copy(), manifest, {"q-dev-a", "q-dev-b", "q-dev-c"}


def test_route_reuse_metric_primitives():
    left = np.asarray([0.4, 0.3, 0.2, 0.1])
    right = np.asarray([0.1, 0.2, 0.3, 0.4])

    assert spearman_correlation(left, left) == pytest.approx(1.0)
    assert spearman_correlation(left, right) == pytest.approx(-1.0)
    assert jensen_shannon_divergence(left, left) == pytest.approx(0.0)
    assert jensen_shannon_divergence(left, right) > 0.0
    assert topk_indices(left, 2) == (0, 1)
    assert topk_jaccard((0, 1), (1, 2)) == pytest.approx(1 / 3)
    assert captured_mass(right, (2, 3)) == pytest.approx(0.7)


def test_route_reuse_rows_are_development_only_and_record_held_out_ids():
    qwen, vila, manifest, dev_ids = _fixtures()

    rows, diagnostics = route_reuse_rows(qwen, vila, manifest, dev_ids)

    assert diagnostics["development_examples_analyzed"] == ["q-dev-a", "q-dev-b", "q-dev-c"]
    assert diagnostics["non_development_examples_held_out"] == ["q-test-a", "q-test-b"]
    assert diagnostics["non_development_examples_held_out_count"] == 2
    assert rows
    assert {row["question_id"] for row in rows} == dev_ids
    assert all(row["is_static_prior_dev_example"] is True for row in rows)
    assert {row["model"] for row in rows} == {"qwen", "vila"}
    assert {row["retention_ratio"] for row in rows} == set(RETENTION_RATIOS)


def test_lopo_static_prior_excludes_held_out_participant_and_source_video():
    qwen, vila, manifest, dev_ids = _fixtures()

    rows, _diagnostics = route_reuse_rows(qwen, vila, manifest, dev_ids)
    static_rows = [row for row in rows if row["selection_method"] == "static_dev_prior_lopo"]

    assert static_rows
    for row in static_rows:
        assert row["layer_gap"] is None
        assert row["source_layer"] is None
        assert row["participant_id"] not in row["static_prior_training_participant_ids"]
        assert row["source_video_id"] not in row["static_prior_training_source_video_ids"]
        assert row["question_id"] not in row["static_prior_training_question_ids"]
        assert row["static_prior_training_question_ids"]


def test_dynamic_gaps_use_identical_target_layer_sets():
    qwen, vila, manifest, dev_ids = _fixtures()

    rows, _diagnostics = route_reuse_rows(qwen, vila, manifest, dev_ids)
    dynamic_rows = [row for row in rows if row["selection_method"] == "dynamic_source_layer_topk"]
    expected_targets = {8, 9, 10, 11}

    for model in ("qwen", "vila"):
        for question_id in dev_ids:
            for ratio in RETENTION_RATIOS:
                by_gap = {
                    gap: {
                        row["target_layer"]
                        for row in dynamic_rows
                        if row["model"] == model
                        and row["question_id"] == question_id
                        and row["retention_ratio"] == ratio
                        and row["layer_gap"] == gap
                    }
                    for gap in LAYER_GAPS
                }
                assert set(by_gap) == set(LAYER_GAPS)
                assert all(targets == expected_targets for targets in by_gap.values())


def test_static_prior_rows_are_not_duplicated_by_gap():
    qwen, vila, manifest, dev_ids = _fixtures()

    rows, _diagnostics = route_reuse_rows(qwen, vila, manifest, dev_ids)
    static_rows = [row for row in rows if row["selection_method"] == "static_dev_prior_lopo"]

    assert static_rows
    assert {row["layer_gap"] for row in static_rows} == {None}
    for model in ("qwen", "vila"):
        for question_id in dev_ids:
            for ratio in RETENTION_RATIOS:
                subset = [
                    row
                    for row in static_rows
                    if row["model"] == model
                    and row["question_id"] == question_id
                    and row["retention_ratio"] == ratio
                ]
                assert len(subset) == 4
                assert {row["target_layer"] for row in subset} == {8, 9, 10, 11}
