from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiment1.adaptive_router import (
    build_router_dataset,
    calibrate_router_policy,
    causal_policy_decision,
    cluster_bootstrap_ci,
    evaluate_router,
    load_router_dataset,
    make_decisions,
    question_frame_weights,
    route_feature_dict,
    select_fixed_policy_training_only,
    summarize_router_decisions,
    train_router_models,
)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _feature_payload(layer: int = 4, frame_count: int = 8):
    return {
        "schema_version": "adaptive_compaction_router_features_v1",
        "layer": layer,
        "frame_count": frame_count,
        "native_temporal_cell_ids": [0, 1, 2, 3],
        "token_count_per_cell": [2, 2, 2, 2],
        "cell_mean_residual": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "question_mean_residual": np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        "cell_residual_norm": [1.0, 1.1, 1.2, 1.3],
        "cell_residual_dispersion": [0.1, 0.2, 0.3, 0.4],
    }


def test_compact_route_features_are_bounded_and_position_aware():
    features = route_feature_dict(_feature_payload(), {"retained_cell_ids": [0, 3], "compaction_layer": 4, "retention_fraction": 0.5})

    assert "question_0" not in features
    assert features["retained_cell_count"] == 2
    assert features["dropped_cell_count"] == 2
    assert features["retained_contains_first_cell"] == 1
    assert features["retained_contains_last_cell"] == 1
    assert features["temporal_coverage_span"] == pytest.approx(1.0)
    assert len(features) < 80


def _make_artifacts(
    root: Path,
    split: str,
    qid: str,
    source: str,
    *,
    frame_counts: tuple[int, ...] = (8, 16),
    layers: tuple[int, ...] = (4, 8),
):
    torch = pytest.importorskip("torch")
    manifest = {
        "question_id": qid,
        "source_video_id": source,
        "split": split,
        "category": "fine_grained",
        "question_type": "fine_grained_action_recognition",
    }
    for frame_count in frame_counts:
        feature_entries = []
        artifact_dir = root / "artifacts" / qid / f"frames_{frame_count}"
        for layer in layers:
            feature_path = root / "features" / qid / f"frames_{frame_count}" / f"layer_{layer}.pt"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_feature_payload(layer=layer, frame_count=frame_count), feature_path)
            feature_entries.append(
                {
                    "layer": layer,
                    "frame_count": frame_count,
                    "feature_file": str(feature_path),
                    "dtype": "torch.float32",
                    "cell_mean_residual_shape": [4, 3],
                    "question_mean_residual_shape": [3],
                }
            )
        dense = {
            "question_id": qid,
            "source_video_id": source,
            "split": split,
            "frame_count": frame_count,
            "condition": "dense_custom",
            "native_temporal_cell_count": 4,
            "router_features": feature_entries,
            "status": "complete",
            "correct": True,
        }
        _write_json(artifact_dir / "dense_custom.json", dense)
        idx = 0
        for layer in layers:
            for cells, family, delta, flip, retention in [
                ([0, 2], "uniform_temporal_coverage", -0.05 if layer == 4 else 0.02, False, 0.5),
                ([1, 3], "deterministic_random_subset", -0.6 if layer == 4 else -0.2, True, 0.5),
                ([0], "prefix_retention", -0.01 if frame_count == 8 else -0.3, False if frame_count == 8 else True, 0.25),
            ]:
                _write_json(
                    artifact_dir / f"act_{idx}.json",
                    {
                        "question_id": qid,
                        "source_video_id": source,
                        "split": split,
                        "frame_count": frame_count,
                        "condition": "hard_evict",
                        "compaction_layer": layer,
                        "native_temporal_cell_count": 4,
                        "retention_fraction": retention,
                        "retained_cell_ids": cells,
                        "route_family": family,
                        "action_id": f"{qid}_f{frame_count}_l{layer}_a{idx}",
                        "delta_correct_choice_log_probability_from_dense": delta,
                        "delta_answer_margin_from_dense": delta / 2.0,
                        "prediction_changed": flip,
                        "correct": not flip,
                        "paired_flop_reduction_from_dense": 0.2 + 0.02 * layer + (0.1 if retention == 0.25 else 0.0),
                        "status": "complete",
                    },
                )
                idx += 1
    return manifest


def _dataset(tmp_path: Path):
    train_a = _make_artifacts(tmp_path / "train_labels", "train", "q-train-a", "v-train-a")
    train_b = _make_artifacts(tmp_path / "train_labels", "train", "q-train-b", "v-train-b")
    dev_a = _make_artifacts(tmp_path / "dev_labels", "development", "q-dev-a", "v-dev-a")
    dev_b = _make_artifacts(tmp_path / "dev_labels", "development", "q-dev-b", "v-dev-b")
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(train_path, [train_a, train_b])
    _write_jsonl(dev_path, [dev_a, dev_b])
    return build_router_dataset(
        train_label_dir=tmp_path / "train_labels",
        development_label_dir=tmp_path / "dev_labels",
        train_manifest=train_path,
        development_manifest=dev_path,
        output_dir=tmp_path / "dataset",
        git_commit="abc",
    )


def test_separate_label_dirs_and_float32_dataset(tmp_path):
    dataset = _dataset(tmp_path)
    loaded = load_router_dataset(tmp_path / "dataset")

    assert dataset.features.dtype == np.float32
    assert loaded.features.shape == dataset.features.shape
    assert loaded.features.shape[1] < 80
    assert {row["split"] for row in loaded.rows} == {"train", "development"}
    assert {row["frame_count"] for row in loaded.rows} == {8, 16}
    assert {row["accuracy_delta_from_dense"] for row in loaded.rows} == {0.0, -1.0}


def test_build_router_dataset_rejects_source_overlap(tmp_path):
    train = _make_artifacts(tmp_path / "train_labels", "train", "q-train", "shared")
    dev = _make_artifacts(tmp_path / "dev_labels", "development", "q-dev", "shared")
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(train_path, [train])
    _write_jsonl(dev_path, [dev])

    with pytest.raises(ValueError, match="source-video overlap"):
        build_router_dataset(
            train_label_dir=tmp_path / "train_labels",
            development_label_dir=tmp_path / "dev_labels",
            train_manifest=train_path,
            development_manifest=dev_path,
            output_dir=tmp_path / "dataset",
            git_commit="abc",
        )


def test_test_artifacts_require_explicit_final_access(tmp_path):
    train = _make_artifacts(tmp_path / "train_labels", "train", "q-train", "v-train")
    dev = _make_artifacts(tmp_path / "dev_labels", "development", "q-dev", "v-dev")
    test = _make_artifacts(tmp_path / "test_labels", "test", "q-test", "v-test")
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, [train])
    _write_jsonl(dev_path, [dev])
    _write_jsonl(test_path, [test])

    with pytest.raises(RuntimeError, match="allow_test"):
        build_router_dataset(
            train_label_dir=tmp_path / "train_labels",
            development_label_dir=tmp_path / "dev_labels",
            test_label_dir=tmp_path / "test_labels",
            train_manifest=train_path,
            development_manifest=dev_path,
            test_manifest=test_path,
            output_dir=tmp_path / "dataset",
            git_commit="abc",
        )


def test_question_frame_weights_equalize_action_counts():
    rows = [
        {"question_id": "q1", "frame_count": 8},
        {"question_id": "q1", "frame_count": 8},
        {"question_id": "q1", "frame_count": 16},
    ]
    weights = question_frame_weights(rows)
    assert weights.tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert weights[:2].sum() == pytest.approx(weights[2])


def test_frame_counts_are_never_mixed_and_decisions_are_per_question_frame(tmp_path):
    dataset = _dataset(tmp_path)
    safety_probability = np.ones(len(dataset.rows), dtype=np.float32)
    predicted = dataset.logp_delta
    decisions = make_decisions(
        dataset,
        split="development",
        safety_probability=safety_probability,
        predicted_logp=predicted,
        threshold=0.5,
        layer_order=[4, 8],
    )

    learned = [row for row in decisions if row["baseline"] == "causal_learned_router"]
    assert {(row["question_id"], row["frame_count"]) for row in learned} == {
        ("q-dev-a", 8),
        ("q-dev-a", 16),
        ("q-dev-b", 8),
        ("q-dev-b", 16),
    }
    assert len(learned) == 4


def test_causal_policy_stops_early_without_future_layer_features():
    rows = [
        {"compaction_layer": 4, "_unsafe_probability": 0.01, "paired_flop_reduction_from_dense": 0.2, "_predicted_logp_delta": 0.0, "action_id": "early"},
        {"compaction_layer": 8, "_unsafe_probability": 0.01, "paired_flop_reduction_from_dense": 0.9, "_predicted_logp_delta": 0.0, "action_id": "future"},
    ]
    chosen = causal_policy_decision(rows, threshold=0.1, layer_order=[4, 8])
    assert chosen["action_id"] == "early"


def test_baselines_are_distinct_and_fixed_selected_from_train_only(tmp_path):
    dataset = _dataset(tmp_path)
    decisions = make_decisions(
        dataset,
        split="development",
        safety_probability=np.ones(len(dataset.rows), dtype=np.float32),
        predicted_logp=dataset.logp_delta,
        threshold=0.5,
        layer_order=[4, 8],
    )
    baselines = {row["baseline"] for row in decisions}
    assert baselines == {
        "dense",
        "fixed_uniform",
        "fixed_random",
        "causal_learned_router",
        "causal_oracle",
        "global_oracle_upper_bound",
    }
    rows = load_router_dataset(tmp_path / "dataset").rows
    assert select_fixed_policy_training_only(rows)[0] in {4, 8}


def test_cluster_bootstrap_resamples_question_clusters():
    rows = [
        {"question_id": "q1", "frame_count": 8, "delta_logp": 1.0},
        {"question_id": "q1", "frame_count": 16, "delta_logp": 1.0},
        {"question_id": "q2", "frame_count": 8, "delta_logp": -1.0},
        {"question_id": "q2", "frame_count": 16, "delta_logp": -1.0},
    ]
    result = cluster_bootstrap_ci(rows, "delta_logp", samples=20, seed=1)
    assert result["n_questions"] == 2
    assert result["n_decisions"] == 4


def test_summarize_reports_selected_layer_and_frame_metrics(tmp_path):
    dataset = _dataset(tmp_path)
    decisions = make_decisions(
        dataset,
        split="development",
        safety_probability=np.ones(len(dataset.rows), dtype=np.float32),
        predicted_logp=dataset.logp_delta,
        threshold=0.5,
        layer_order=[4, 8],
    )
    summary = summarize_router_decisions(decisions, bootstrap_samples=10, seed=1)
    assert "8" in summary["causal_learned_router"]["by_frame_count"]
    assert "selected_layer_distribution" in summary["causal_learned_router"]


def test_train_calibrate_and_frozen_eval_when_sklearn_available(tmp_path):
    pytest.importorskip("sklearn")
    _dataset(tmp_path)
    train_router_models(tmp_path / "dataset", tmp_path / "models", seed=7, git_commit="abc")
    policy = calibrate_router_policy(
        tmp_path / "dataset",
        tmp_path / "models",
        tmp_path / "policy",
        unsafe_rate_bound=0.5,
        bootstrap_samples=10,
        seed=7,
        git_commit="abc",
    )
    assert "unsafe_probability_threshold" in policy
    assert "achieved_development_metrics" in policy
    summary = evaluate_router(
        tmp_path / "dataset",
        tmp_path / "models",
        tmp_path / "eval",
        split="development",
        frozen_policy=tmp_path / "policy" / "frozen_policy.json",
        bootstrap_samples=10,
        seed=7,
    )
    assert "causal_learned_router" in summary
    with pytest.raises(RuntimeError, match="allow-test"):
        evaluate_router(
            tmp_path / "dataset",
            tmp_path / "models",
            tmp_path / "eval_test",
            split="test",
            frozen_policy=tmp_path / "policy" / "frozen_policy.json",
            bootstrap_samples=10,
            seed=7,
        )
