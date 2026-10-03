from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiment1.adaptive_router import (
    _choose_row,
    bootstrap_ci,
    build_router_dataset,
    load_router_dataset,
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


def test_route_feature_construction_retained_dropped_and_positions():
    features = route_feature_dict(_feature_payload(), {"retained_cell_ids": [0, 3], "compaction_layer": 4, "retention_fraction": 0.5})

    assert features["scalars"]["retained_cell_count"] == 2
    assert features["scalars"]["dropped_cell_count"] == 2
    assert features["scalars"]["retained_contains_first_cell"] == 1
    assert features["scalars"]["retained_contains_last_cell"] == 1
    assert features["scalars"]["temporal_coverage_span"] == pytest.approx(1.0)
    assert "retained_minus_dropped_mean" in features["vectors"]
    assert features["vectors"]["question"].shape == (3,)


def _make_artifacts(root: Path, split: str, qid: str, source: str, safe_delta: float, unsafe_delta: float):
    torch = pytest.importorskip("torch")
    manifest = {
        "question_id": qid,
        "source_video_id": source,
        "split": split,
        "category": "fine_grained",
        "question_type": "fine_grained_action_recognition",
    }
    feature_dir = root / "features" / qid / "frames_8"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "layer_4.pt"
    torch.save(_feature_payload(), feature_path)
    dense = {
        "question_id": qid,
        "source_video_id": source,
        "split": split,
        "frame_count": 8,
        "condition": "dense_custom",
        "native_temporal_cell_count": 4,
        "router_features": [
            {
                "layer": 4,
                "frame_count": 8,
                "feature_file": str(feature_path),
                "dtype": "torch.float32",
                "cell_mean_residual_shape": [4, 3],
                "question_mean_residual_shape": [3],
            }
        ],
        "status": "complete",
        "correct": True,
    }
    artifact_dir = root / "artifacts" / qid / "frames_8"
    _write_json(artifact_dir / "dense_custom.json", dense)
    for idx, (cells, delta, family, flip) in enumerate(
        [
            ([0, 2], safe_delta, "uniform_temporal_coverage", False),
            ([1, 3], unsafe_delta, "deterministic_random_subset", True),
        ]
    ):
        _write_json(
            artifact_dir / f"act_{idx}.json",
            {
                "question_id": qid,
                "source_video_id": source,
                "split": split,
                "frame_count": 8,
                "condition": "hard_evict",
                "compaction_layer": 4,
                "native_temporal_cell_count": 4,
                "retention_fraction": 0.5,
                "retained_cell_ids": cells,
                "route_family": family,
                "action_id": f"act_{idx}",
                "delta_correct_choice_log_probability_from_dense": delta,
                "delta_answer_margin_from_dense": delta / 2.0,
                "prediction_changed": flip,
                "correct": not flip,
                "paired_flop_reduction_from_dense": 0.4 + idx * 0.1,
                "status": "complete",
            },
        )
    return manifest


def test_build_router_dataset_and_safety_labels(tmp_path):
    train_manifest = _make_artifacts(tmp_path / "labels", "train", "q-train", "v-train", -0.05, -0.6)
    dev_manifest = _make_artifacts(tmp_path / "labels", "development", "q-dev", "v-dev", 0.01, -0.2)
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(train_path, [train_manifest])
    _write_jsonl(dev_path, [dev_manifest])

    dataset = build_router_dataset(
        label_dir=tmp_path / "labels",
        train_manifest=train_path,
        development_manifest=dev_path,
        output_dir=tmp_path / "dataset",
        git_commit="abc",
    )
    loaded = load_router_dataset(tmp_path / "dataset")

    assert dataset.features.shape[0] == 4
    assert loaded.features.shape == dataset.features.shape
    assert set(dataset.safe.tolist()) == {0, 1}
    assert all(row["source_video_id"] in {"v-train", "v-dev"} for row in dataset.rows)
    assert {row["accuracy_delta_from_dense"] for row in dataset.rows} == {0.0, -1.0}


def test_build_router_dataset_rejects_train_development_source_overlap(tmp_path):
    train_manifest = _make_artifacts(tmp_path / "labels", "train", "q-train", "shared", -0.05, -0.6)
    dev_manifest = _make_artifacts(tmp_path / "labels", "development", "q-dev", "shared", 0.01, -0.2)
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(train_path, [train_manifest])
    _write_jsonl(dev_path, [dev_manifest])

    with pytest.raises(ValueError, match="source-video overlap"):
        build_router_dataset(
            label_dir=tmp_path / "labels",
            train_manifest=train_path,
            development_manifest=dev_path,
            output_dir=tmp_path / "dataset",
            git_commit="abc",
        )


def test_build_router_dataset_requires_explicit_test_access(tmp_path):
    train_manifest = _make_artifacts(tmp_path / "labels", "train", "q-train", "v-train", -0.05, -0.6)
    dev_manifest = _make_artifacts(tmp_path / "labels", "development", "q-dev", "v-dev", 0.01, -0.2)
    test_manifest = _make_artifacts(tmp_path / "labels", "test", "q-test", "v-test", 0.02, -0.3)
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, [train_manifest])
    _write_jsonl(dev_path, [dev_manifest])
    _write_jsonl(test_path, [test_manifest])

    with pytest.raises(RuntimeError, match="allow_test"):
        build_router_dataset(
            label_dir=tmp_path / "labels",
            train_manifest=train_path,
            development_manifest=dev_path,
            test_manifest=test_path,
            output_dir=tmp_path / "dataset",
            git_commit="abc",
        )

    dataset = build_router_dataset(
        label_dir=tmp_path / "labels",
        train_manifest=train_path,
        development_manifest=dev_path,
        test_manifest=test_path,
        allow_test=True,
        output_dir=tmp_path / "dataset",
        git_commit="abc",
    )
    assert {row["split"] for row in dataset.rows} == {"train", "development", "test"}


def test_dense_fallback_fixed_policy_oracle_and_bootstrap():
    rows = [
        {"split": "train", "compaction_layer": 4, "retention_fraction": 0.5, "safe": True, "paired_flop_reduction_from_dense": 0.4},
        {"split": "train", "compaction_layer": 8, "retention_fraction": 0.25, "safe": False, "paired_flop_reduction_from_dense": 0.8},
        {"split": "development", "compaction_layer": 4, "retention_fraction": 0.5, "safe": True, "paired_flop_reduction_from_dense": 0.4},
    ]
    assert select_fixed_policy_training_only(rows) == (4, 0.5)
    chosen = _choose_row([dict(rows[2], delta_correct_choice_log_probability_from_dense=-0.01)])
    assert chosen is not None
    assert chosen["compaction_layer"] == 4
    assert bootstrap_ci([0.0, 1.0], samples=10, seed=1)["n"] == 2
    summary = summarize_router_decisions(
        [
            {"baseline": "dense", "compacted": False, "safe": True, "prediction_changed": False, "delta_logp": 0.0, "delta_margin": 0.0, "flop_reduction": 0.0, "accuracy_delta_from_dense": 0.0},
            {"baseline": "oracle_safe", "compacted": True, "safe": True, "prediction_changed": False, "delta_logp": 0.1, "delta_margin": 0.1, "flop_reduction": 0.5, "accuracy_delta_from_dense": 0.0},
        ],
        bootstrap_samples=10,
        seed=1,
    )
    assert summary["dense"]["coverage"] == 0.0
    assert summary["oracle_safe"]["coverage"] == 1.0


def test_train_router_models_is_deterministic_when_sklearn_available(tmp_path):
    pytest.importorskip("sklearn")
    train_manifest = _make_artifacts(tmp_path / "labels", "train", "q-train", "v-train", -0.05, -0.6)
    dev_manifest = _make_artifacts(tmp_path / "labels", "development", "q-dev", "v-dev", 0.01, -0.2)
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(train_path, [train_manifest])
    _write_jsonl(dev_path, [dev_manifest])
    build_router_dataset(
        label_dir=tmp_path / "labels",
        train_manifest=train_path,
        development_manifest=dev_path,
        output_dir=tmp_path / "dataset",
        git_commit="abc",
    )

    first = train_router_models(tmp_path / "dataset", tmp_path / "models_a", seed=7, git_commit="abc")
    second = train_router_models(tmp_path / "dataset", tmp_path / "models_b", seed=7, git_commit="abc")

    assert first["primary_safety_model"] == second["primary_safety_model"]
