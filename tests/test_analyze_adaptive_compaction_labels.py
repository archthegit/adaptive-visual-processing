from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest

from scripts.analyze_adaptive_compaction_labels import (
    analyze,
    choose_fixed_route,
    categorize_failure,
    oracle_layer_distribution_plot_rows,
    oracle_tables,
    route_is_safe,
    source_cluster_bootstrap,
)
from src.experiment1.adaptive_compaction import candidate_routes


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _append_record(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _manifest(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _dense(qid: str, source: str, category: str, frame_count: int, *, split: str = "train", native: int = 4):
    return {
        "question_id": qid,
        "source_video_id": source,
        "category": category,
        "split": split,
        "frame_count": frame_count,
        "condition": "dense_custom",
        "native_temporal_cell_count": native,
        "predicted_idx": 0,
        "predicted_answer": "A",
        "correct": True,
        "correct_choice_log_probability": -1.0,
        "answer_margin": 0.2,
        "estimated_attention_flops": 1000,
        "run_config": {
            "compaction_layers": [4],
            "retention_fractions": [0.5],
            "seed": 123,
        },
    }


def _action_from_route(qid: str, source: str, category: str, frame_count: int, route, *, delta: float, flip: bool, split: str = "train"):
    return {
        "question_id": qid,
        "source_video_id": source,
        "category": category,
        "split": split,
        "frame_count": frame_count,
        "condition": "hard_evict",
        "compaction_layer": route.compaction_layer,
        "native_temporal_cell_count": route.native_temporal_cell_count,
        "retention_fraction": route.retention_fraction,
        "retained_cell_ids": list(route.retained_cells),
        "route_family": route.route_family,
        "action_id": route.action_id,
        "delta_correct_choice_log_probability_from_dense": delta,
        "delta_answer_margin_from_dense": delta / 2,
        "prediction_changed": flip,
        "paired_flop_reduction_from_dense": 0.4,
        "predicted_idx": 1 if flip else 0,
        "predicted_answer": "B" if flip else "A",
        "correct": not flip,
    }


def _complete_group(root: Path, records: Path, qid: str, source: str, category: str, frame_count: int, *, split: str = "train", deltas=None):
    dense_path = root / "artifacts" / qid / f"frames_{frame_count}" / "dense_custom.json"
    dense = _dense(qid, source, category, frame_count, split=split)
    _write_json(dense_path, dense)
    _append_record(records, {"question_id": qid, "frame_count": frame_count, "action_id": "dense_custom", "status": "complete", "artifact": str(dense_path)})
    routes = candidate_routes(
        question_id=qid,
        frame_count=frame_count,
        compaction_layer=4,
        native_temporal_cell_count=4,
        retention_fraction=0.5,
        seed=123,
    )
    deltas = deltas or {}
    for idx, route in enumerate(routes):
        delta = deltas.get(route.retained_cells, 0.05 if idx else -0.2)
        flip = delta < -0.15
        action = _action_from_route(qid, source, category, frame_count, route, delta=delta, flip=flip, split=split)
        path = root / "artifacts" / qid / f"frames_{frame_count}" / f"{route.action_id}.json"
        _write_json(path, action)
        _append_record(records, {"question_id": qid, "frame_count": frame_count, "action_id": route.action_id, "status": "complete", "artifact": str(path)})
    return routes


def test_analyzer_accepts_partial_frame_availability_and_writes_outputs(tmp_path):
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "fine_grained"}])
    _complete_group(tmp_path, records, "q1", "v1", "fine_grained", 8)

    summary = analyze(records, manifest, tmp_path / "out", bootstrap_samples=20, seed=1, quality_floor=-0.10)

    assert summary["num_questions"] == 1
    assert summary["num_question_frame_groups"] == 1
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "coverage.csv").exists()
    assert (tmp_path / "out" / "oracle.csv").exists()


def test_oracle_csv_is_separated_by_frame_count(tmp_path):
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "fine_grained"}])
    _complete_group(tmp_path, records, "q1", "v1", "fine_grained", 8)
    _complete_group(tmp_path, records, "q1", "v1", "fine_grained", 16)

    analyze(records, manifest, tmp_path / "out", bootstrap_samples=10, seed=1, quality_floor=-0.10)
    rows = _read_csv(tmp_path / "out" / "oracle.csv")
    frame_counts = {row["frame_count"] for row in rows if row["category"] == "fine_grained"}

    assert {"8", "16", "all"}.issubset(frame_counts)


def test_incomplete_action_group_is_excluded(tmp_path):
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "gaze"}])
    routes = _complete_group(tmp_path, records, "q1", "v1", "gaze", 8)
    path = tmp_path / "artifacts" / "q1" / "frames_8" / f"{routes[0].action_id}.json"
    path.unlink()

    summary = analyze(records, manifest, tmp_path / "out", bootstrap_samples=10, seed=1, quality_floor=-0.10)

    assert summary["num_question_frame_groups"] == 0
    assert summary["exclusion_counts"]["incomplete_or_missing_action_artifacts"] >= 1


def test_oracle_dense_fallback_when_no_safe_action(tmp_path):
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "ingredient"}])
    routes = candidate_routes(question_id="q1", frame_count=8, compaction_layer=4, native_temporal_cell_count=4, retention_fraction=0.5, seed=123)
    _complete_group(tmp_path, records, "q1", "v1", "ingredient", 8, deltas={route.retained_cells: -0.5 for route in routes})

    analyze(records, manifest, tmp_path / "out", bootstrap_samples=10, seed=1, quality_floor=-0.10)
    question_summary = (tmp_path / "out" / "question_summary.csv").read_text()

    assert "dense_fallback" in question_summary
    assert "True" in question_summary


def test_oracle_tie_break_uses_lexicographically_smallest_action_id():
    base = {
        "safe": True,
        "flop_reduction": 0.5,
        "delta_logp": 0.1,
        "question_id": "q1",
        "source_video_id": "v1",
        "category": "gaze",
        "frame_count": 8,
        "delta_margin": 0.0,
        "prediction_changed": False,
        "compaction_layer": 4,
        "retention_fraction": 0.5,
    }
    groups = [{
        "question_id": "q1",
        "source_video_id": "v1",
        "category": "gaze",
        "frame_count": 8,
        "dense_correct": True,
        "actions": [dict(base, action_id="z_action"), dict(base, action_id="a_action")],
    }]
    _oracle, question_summary = oracle_tables(groups, bootstrap_samples=5, seed=1)
    assert question_summary[0]["oracle_selected_action"] == "a_action"


def test_source_video_clustered_bootstrap_uses_source_unit():
    rows = [
        {"source_video_id": "v1", "value": 1.0},
        {"source_video_id": "v1", "value": 1.0},
        {"source_video_id": "v2", "value": -1.0},
    ]
    result = source_cluster_bootstrap(rows, "value", samples=20, seed=1)
    assert result["n_source_videos"] == 2
    assert result["n"] == 3


def test_seeded_random_baseline_is_deterministic():
    routes = candidate_routes(question_id="q1", frame_count=8, compaction_layer=4, native_temporal_cell_count=4, retention_fraction=0.5, seed=123)
    actions = [
        {"question_id": "q1", "frame_count": 8, "compaction_layer": 4, "retention_fraction": 0.5, "retained_cell_ids": list(route.retained_cells), "action_id": route.action_id}
        for route in routes
    ]
    first = choose_fixed_route(actions, "seeded_random", 9)
    second = choose_fixed_route(actions, "seeded_random", 9)
    assert first["action_id"] == second["action_id"]


def test_test_split_is_rejected(tmp_path):
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "fine_grained"}])
    _complete_group(tmp_path, records, "q1", "v1", "fine_grained", 8, split="test")
    with pytest.raises(RuntimeError, match="test artifact"):
        analyze(records, manifest, tmp_path / "out", bootstrap_samples=10, seed=1, quality_floor=-0.10)


def test_test_manifest_row_is_rejected_even_without_artifact(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("")
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "fine_grained", "split": "test"}])
    with pytest.raises(RuntimeError, match="manifest containing test rows"):
        analyze(records, manifest, tmp_path / "out", bootstrap_samples=10, seed=1, quality_floor=-0.10)


def test_safety_rule_requires_quality_floor_and_no_flip():
    assert route_is_safe({"delta_correct_choice_log_probability_from_dense": -0.09, "prediction_changed": False}, -0.10)
    assert not route_is_safe({"delta_correct_choice_log_probability_from_dense": -0.11, "prediction_changed": False}, -0.10)
    assert not route_is_safe({"delta_correct_choice_log_probability_from_dense": 0.1, "prediction_changed": True}, -0.10)


def test_missing_and_failed_records_are_reported(tmp_path):
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "object_motion"}])
    _append_record(records, {"question_id": "q1", "frame_count": 8, "action_id": "dense_custom", "status": "complete", "artifact": str(tmp_path / "missing.json")})
    _append_record(records, {"question_id": "q1", "status": "failed", "error": "insufficient distinct frames"})

    summary = analyze(records, manifest, tmp_path / "out", bootstrap_samples=10, seed=1, quality_floor=-0.10)

    assert summary["num_excluded_groups"] >= 2
    assert "insufficient_distinct_frames" in summary["exclusion_counts"]


def test_failures_join_category_and_unique_exclusion_counts(tmp_path):
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, [{"question_id": "q1", "source_video_id": "v1", "category": "gaze"}])
    _append_record(records, {"question_id": "q1", "frame_count": 8, "status": "failed", "error": "Adaptive fixed-center sampling requires 32 distinct center frames"})
    _append_record(records, {"question_id": "q1", "frame_count": 8, "status": "failed", "error": "Adaptive fixed-center sampling requires 32 distinct center frames"})

    summary = analyze(records, manifest, tmp_path / "out", bootstrap_samples=10, seed=1, quality_floor=-0.10)
    coverage = _read_csv(tmp_path / "out" / "coverage.csv")

    assert summary["num_excluded_groups"] == 1
    assert summary["raw_failure_record_count"] == 2
    assert coverage[0]["category"] == "gaze"
    assert coverage[0]["frame_count"] == "8"
    assert coverage[0]["exclusion_reason"] == "insufficient_distinct_frames"


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("Command ['ffprobe', ...] returned non-zero exit status 1", "corrupted_or_unreadable_video"),
        ("moov atom not found", "corrupted_or_unreadable_video"),
        ("invalid data found when processing input", "corrupted_or_unreadable_video"),
        ("Adaptive fixed-center sampling requires 32 distinct center frames", "insufficient_distinct_frames"),
    ],
)
def test_failure_classification_real_patterns(message, reason):
    assert categorize_failure(message) == reason


def test_oracle_layer_distribution_plot_rows_excludes_all_category():
    rows = [
        {"category": "all", "frame_count": 8, "selected_layer_distribution": json.dumps({"4": 10})},
        {"category": "gaze", "frame_count": 8, "selected_layer_distribution": json.dumps({"4": 2, "None": 1})},
    ]
    plot_rows = oracle_layer_distribution_plot_rows(rows)
    assert {row["category"] for row in plot_rows} == {"gaze"}
    assert {row["selected_layer"] for row in plot_rows} == {"4", "dense_fallback"}
