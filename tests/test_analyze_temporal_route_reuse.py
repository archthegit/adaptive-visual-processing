from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_temporal_route_reuse import (
    build_static_priors,
    captured_mass,
    jensen_shannon_divergence,
    route_reuse_rows,
    spearman_correlation,
    topk_indices,
    topk_jaccard,
)


def _artifact(question_id: str, video_id: str, layers: int = 10, shift: int = 0) -> dict:
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


def _manifest(question_ids: list[str]) -> dict[str, dict]:
    output = {}
    for index, question_id in enumerate(question_ids):
        output[question_id] = {
            "question_id": question_id,
            "source_video_id": f"P{index:02d}-video",
            "participant_id": f"P{index:02d}",
            "category": "fine_grained",
            "question_type": "fine_grained_action_recognition",
            "duration_group": "short" if index % 2 == 0 else "long",
            "split": "dev" if index == 0 else "test",
        }
    return output


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


def test_route_reuse_rows_include_dynamic_and_static_without_dev_leakage():
    qwen = {
        "q-dev": _artifact("q-dev", "P00-video", shift=0),
        "q-test-a": _artifact("q-test-a", "P01-video", shift=1),
        "q-test-b": _artifact("q-test-b", "P02-video", shift=2),
    }
    vila = {
        "q-dev": _artifact("q-dev", "P00-video", shift=0),
        "q-test-a": _artifact("q-test-a", "P01-video", shift=1),
        "q-test-b": _artifact("q-test-b", "P02-video", shift=2),
    }
    manifest = _manifest(["q-dev", "q-test-a", "q-test-b"])

    rows, diagnostics = route_reuse_rows(qwen, vila, manifest, {"q-dev"})

    assert diagnostics["dev_examples_used_for_static_prior"] == ["q-dev"]
    assert diagnostics["static_prior_evaluation_examples"] == ["q-test-a", "q-test-b"]
    assert rows
    assert {row["selection_method"] for row in rows} == {
        "dynamic_source_layer_topk",
        "static_dev_prior_topk",
    }
    assert all(row["question_id"] != "q-dev" for row in rows)
    assert all(row["is_static_prior_dev_example"] is False for row in rows)
    assert {row["model"] for row in rows} == {"qwen", "vila"}
    assert {row["layer_gap"] for row in rows} == {1, 2, 4, 8}
    assert {row["retention_ratio"] for row in rows} == {0.25, 0.5, 0.75}
    assert all(0.0 <= row["reused_captured_mass"] <= 1.0 for row in rows)
    assert all(0.0 <= row["oracle_captured_mass"] <= 1.0 for row in rows)
    assert all(row["reuse_efficiency"] <= 1.0 + 1e-12 for row in rows)


def test_static_prior_requires_matched_dev_examples_and_never_uses_test_only():
    qwen = {"q-test": _artifact("q-test", "P01-video")}
    vila = {"q-test": _artifact("q-test", "P01-video")}

    with pytest.raises(ValueError, match="No development examples"):
        build_static_priors({"qwen": qwen, "vila": vila}, {"q-dev"}, {"q-test"})


def test_static_prior_evaluation_requires_non_dev_examples():
    qwen = {"q-dev": _artifact("q-dev", "P00-video")}
    vila = {"q-dev": _artifact("q-dev", "P00-video")}
    manifest = _manifest(["q-dev"])

    with pytest.raises(ValueError, match="No matched non-development examples"):
        route_reuse_rows(qwen, vila, manifest, {"q-dev"})
