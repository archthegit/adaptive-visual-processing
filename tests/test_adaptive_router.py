from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiment1.adaptive_router import (
    RouterDataset,
    build_router_dataset,
    calibrate_router_policy,
    causal_policy_decision,
    cluster_binary_upper_bound,
    cluster_bootstrap_ci,
    deterministic_group_balanced_undersample_indices,
    evaluate_router,
    load_router_dataset,
    make_decisions,
    question_frame_weights,
    route_feature_dict,
    select_fixed_policy_training_only,
    summarize_router_decisions,
    train_router_models,
    uniform_route_row,
    validate_frozen_policy_integrity,
    group_balanced_undersampling_summary,
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


def _unequal_group_rows(group_sizes):
    rows = []
    for group_idx, size in enumerate(group_sizes):
        for row_idx in range(size):
            rows.append({"question_id": f"q{group_idx}", "frame_count": 8, "action_id": f"g{group_idx}_a{row_idx}"})
    return rows


@pytest.mark.parametrize("group_sizes", [(3, 2), (70, 48), (70, 24)])
def test_group_balanced_undersampling_equalizes_without_expansion(group_sizes):
    rows = _unequal_group_rows(group_sizes)
    indices = deterministic_group_balanced_undersample_indices(rows, seed=7)
    counts = {(rows[idx]["question_id"], rows[idx]["frame_count"]): 0 for idx in indices}
    for idx in indices:
        counts[(rows[idx]["question_id"], rows[idx]["frame_count"])] += 1
    assert set(counts.values()) == {min(group_sizes)}
    assert len(indices) == min(group_sizes) * len(group_sizes)
    assert len(indices) <= len(rows)
    assert deterministic_group_balanced_undersample_indices(rows, seed=7).tolist() == indices.tolist()
    summary = group_balanced_undersampling_summary(rows, seed=7, feature_dim=64)
    assert summary["original_training_rows"] == len(rows)
    assert summary["balanced_training_rows"] == len(indices)
    assert summary["balanced_feature_ram_bytes"] == len(indices) * 64 * 4
    assert summary["equal_total_rows_per_question_frame"]
    assert summary["no_expansion_beyond_original_rows"]


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


def _exhaustive_dataset():
    rows = []
    pairs = ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3])
    for split, qid, source in [("train", "q-train", "v-train"), ("development", "q-dev", "v-dev")]:
        for frame_count in (8, 16):
            for layer in (4, 8):
                for pair in pairs:
                    action_id = f"{qid}_f{frame_count}_l{layer}_{pair[0]}{pair[1]}"
                    safe = pair in ([0, 3], [1, 3])
                    rows.append(
                        {
                            "question_id": qid,
                            "source_video_id": source,
                            "split": split,
                            "category": "fine_grained",
                            "frame_count": frame_count,
                            "compaction_layer": layer,
                            "retention_fraction": 0.5,
                            "native_temporal_cell_count": 4,
                            "retained_cell_ids": list(pair),
                            "route_family": "exhaustive",
                            "action_id": action_id,
                            "safe": safe,
                            "delta_correct_choice_log_probability_from_dense": 0.1 if safe else -0.2,
                            "delta_answer_margin_from_dense": 0.05 if safe else -0.1,
                            "prediction_changed": not safe,
                            "paired_flop_reduction_from_dense": 0.5,
                            "accuracy_delta_from_dense": 0.0 if safe else -1.0,
                        }
                    )
    n = len(rows)
    return RouterDataset(
        rows=rows,
        feature_names=["x"],
        features=np.zeros((n, 1), dtype=np.float32),
        safe=np.asarray([int(row["safe"]) for row in rows], dtype=np.int64),
        logp_delta=np.asarray([row["delta_correct_choice_log_probability_from_dense"] for row in rows], dtype=np.float32),
        margin_delta=np.asarray([row["delta_answer_margin_from_dense"] for row in rows], dtype=np.float32),
        flop_reduction=np.asarray([row["paired_flop_reduction_from_dense"] for row in rows], dtype=np.float32),
    )


def test_exhaustive_action_spaces_get_uniform_and_random_baselines():
    dataset = _exhaustive_dataset()
    decisions = make_decisions(
        dataset,
        split="development",
        safety_probability=np.ones(len(dataset.rows), dtype=np.float32),
        predicted_logp=dataset.logp_delta,
        threshold=0.5,
        layer_order=[4, 8],
        random_seed=123,
    )
    fixed_uniform = [row for row in decisions if row["baseline"] == "fixed_uniform"]
    fixed_random = [row for row in decisions if row["baseline"] == "fixed_random"]
    assert fixed_uniform and fixed_random
    assert all(row["compacted"] for row in fixed_uniform)
    assert all(row["compacted"] for row in fixed_random)
    assert all(row["route_family"] == "exhaustive" for row in fixed_uniform + fixed_random)
    assert {tuple(row["retained_cell_ids"]) for row in fixed_uniform} == {(0, 3)}


def test_uniform_subset_minimizes_even_spacing_without_route_family():
    candidates = [
        {"retained_cell_ids": [0, 1], "native_temporal_cell_count": 4, "action_id": "a"},
        {"retained_cell_ids": [0, 3], "native_temporal_cell_count": 4, "action_id": "b"},
        {"retained_cell_ids": [1, 2], "native_temporal_cell_count": 4, "action_id": "c"},
    ]
    assert uniform_route_row(candidates)["action_id"] == "b"


def test_clustered_unsafe_upper_bound_is_conservative():
    rows = [
        {"question_id": "q1", "unsafe": False},
        {"question_id": "q1", "unsafe": False},
        {"question_id": "q2", "unsafe": True},
        {"question_id": "q2", "unsafe": True},
    ]
    result = cluster_binary_upper_bound(rows, "unsafe", samples=100, seed=3)
    assert result["observed"] == pytest.approx(0.5)
    assert result["upper"] >= result["observed"]
    assert result["question_any_observed"] == pytest.approx(0.5)


def test_wilson_bound_nonzero_for_all_safe_and_decreases_with_more_questions():
    few = [{"question_id": f"q{i}", "unsafe": False} for i in range(5)]
    many = [{"question_id": f"q{i}", "unsafe": False} for i in range(50)]
    few_result = cluster_binary_upper_bound(few, "unsafe")
    many_result = cluster_binary_upper_bound(many, "unsafe")
    assert few_result["observed"] == 0.0
    assert few_result["question_any_upper"] > 0.0
    assert many_result["question_any_upper"] < few_result["question_any_upper"]


def test_any_unsafe_frame_marks_question_unsafe():
    rows = [
        {"question_id": "q1", "frame_count": 8, "unsafe": False},
        {"question_id": "q1", "frame_count": 16, "unsafe": True},
        {"question_id": "q2", "frame_count": 8, "unsafe": False},
        {"question_id": "q2", "frame_count": 16, "unsafe": False},
    ]
    result = cluster_binary_upper_bound(rows, "unsafe")
    assert result["observed"] == pytest.approx(0.25)
    assert result["question_any_observed"] == pytest.approx(0.5)


def test_frozen_policy_integrity_rejects_manifest_and_model_hash_drift(tmp_path):
    safety = tmp_path / "safety.joblib"
    regressor = tmp_path / "regressor.joblib"
    safety.write_text("safety-a")
    regressor.write_text("regressor-a")
    _write_json(
        tmp_path / "model_index.json",
        {
            "safety_models": {"logistic_regression": str(safety)},
            "quality_delta_regressors": {"ridge": str(regressor)},
        },
    )
    metadata = {
        "feature_schema_hash": "features",
        "manifest_hashes": {"train": "train-hash", "development": "dev-hash"},
    }
    from src.experiment1.adaptive_router import sha256_file

    policy = {
        "feature_schema_hash": "features",
        "train_manifest_hash": "train-hash",
        "development_manifest_hash": "dev-hash",
        "dense_fallback_policy": False,
        "selected_safety_model": "logistic_regression",
        "selected_quality_regressor": "ridge",
        "model_hashes": {
            "safety:logistic_regression": sha256_file(safety),
            "regressor:ridge": sha256_file(regressor),
        },
    }
    validate_frozen_policy_integrity(metadata, tmp_path, policy)
    safety.write_text("safety-b")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_policy_integrity(metadata, tmp_path, policy)
    safety.write_text("safety-a")
    bad_metadata = dict(metadata, manifest_hashes={"train": "changed", "development": "dev-hash"})
    with pytest.raises(ValueError, match="Train manifest hash"):
        validate_frozen_policy_integrity(bad_metadata, tmp_path, policy)


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
        "fixed_prefix",
        "fixed_suffix",
        "causal_learned_router",
        "causal_oracle",
        "global_oracle_upper_bound",
    }
    rows = load_router_dataset(tmp_path / "dataset").rows
    assert select_fixed_policy_training_only(rows)[0] in {4, 8}


def test_no_eligible_compacting_policy_produces_dense_fallback():
    dataset = _exhaustive_dataset()
    decisions = make_decisions(
        dataset,
        split="development",
        safety_probability=np.zeros(len(dataset.rows), dtype=np.float32),
        predicted_logp=dataset.logp_delta,
        threshold=0.0,
        layer_order=[4, 8],
    )
    learned = [row for row in decisions if row["baseline"] == "causal_learned_router"]
    assert learned
    assert all(not row["compacted"] for row in learned)
    assert all(row["flop_reduction"] == 0.0 for row in learned)


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
