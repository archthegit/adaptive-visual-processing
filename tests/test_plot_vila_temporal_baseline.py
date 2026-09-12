from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_vila_temporal_baseline import (
    aggregate_diagnostics,
    load_latest_records,
    save_accuracy_by_category,
    save_first_last_bin_mass,
    save_lift_heatmap,
    save_top_bin_position,
    validate_artifacts,
    write_report,
)


def _artifact(question_id: str, video_id: str, question_type: str, correct: bool, shift: int = 0) -> dict:
    base = np.asarray([0.20, 0.16, 0.13, 0.11, 0.10, 0.09, 0.09, 0.12], dtype=np.float64)
    rows = []
    for layer in range(32):
        row = np.roll(base, (layer + shift) % 8)
        rows.append((row / row.sum()).tolist())
    return {
        "question_id": question_id,
        "question_type": question_type,
        "correct": correct,
        "video_clip": [{"video_id": video_id, "participant_id": video_id.split("-", 1)[0]}],
        "sampled_frame_indices": [list(range(8))],
        "sampled_timestamps": [[float(index) for index in range(8)]],
        "temporal_relevance": {
            "normalized_temporal_bin_scores": rows,
            "absolute_question_to_visual_attention_mass": [0.1 + 0.01 * layer for layer in range(32)],
            "metadata": {"num_temporal_bins": 8},
        },
        "metadata": {"model_backend": "vila_llama3", "num_decoder_layers": 32},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def test_load_latest_records_keeps_complete_and_records_failed(tmp_path):
    input_dir = tmp_path / "run"
    input_dir.mkdir()
    artifact1 = input_dir / "q1.json"
    artifact2 = input_dir / "q2.json"
    _write_json(artifact1, _artifact("q1", "P01-video-a", "fine_grained_action_recognition", True))
    _write_json(artifact2, _artifact("q2", "P02-video-b", "gaze_interaction_anticipation", False, shift=1))
    records = [
        {"question_id": "q1", "status": "failed", "error": "old failure"},
        {"question_id": "q1", "status": "complete", "artifact": str(artifact1)},
        {"question_id": "q2", "status": "complete", "artifact": str(artifact2)},
        {"question_id": "q3", "status": "failed", "error": "sampling ineligible"},
    ]
    (input_dir / "records.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")

    artifacts, failed, complete_records = load_latest_records(input_dir)

    assert [artifact["question_id"] for artifact in artifacts] == ["q1", "q2"]
    assert [record["question_id"] for record in complete_records] == ["q1", "q2"]
    assert failed == [{"question_id": "q3", "status": "failed", "error": "sampling ineligible"}]


def test_validate_artifacts_enforces_vila_shape_counts_and_unique_videos():
    artifacts = [
        _artifact("q1", "P01-video-a", "fine_grained_action_recognition", True),
        _artifact("q2", "P02-video-b", "gaze_interaction_anticipation", False, shift=1),
    ]
    failed = [{"question_id": "q3", "status": "failed", "error": "sampling ineligible"}]

    validation = validate_artifacts(
        artifacts,
        failed,
        expected_complete=2,
        expected_failed=1,
        expected_category_counts={"fine_grained": 1, "gaze": 1},
    )

    assert validation["num_complete_artifacts"] == 2
    assert validation["num_failed_records"] == 1
    assert validation["overall_accuracy"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert validation["category_counts"] == {"fine_grained": 1, "gaze": 1}

    duplicate = [_artifact("q4", "P01-video-a", "fine_grained_action_recognition", True)]
    with pytest.raises(RuntimeError, match="Duplicate source video"):
        validate_artifacts(
            artifacts + duplicate,
            failed,
            expected_complete=3,
            expected_failed=1,
            expected_category_counts={"fine_grained": 2, "gaze": 1},
        )


def test_vila_baseline_plots_and_report_are_written(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    artifacts = [
        _artifact("q1", "P01-video-a", "fine_grained_action_recognition", True),
        _artifact("q2", "P02-video-b", "gaze_interaction_anticipation", False, shift=1),
    ]
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    mean_distribution = np.stack(
        [np.asarray(artifact["temporal_relevance"]["normalized_temporal_bin_scores"], dtype=np.float64) for artifact in artifacts]
    ).mean(axis=0)
    heatmap = save_lift_heatmap(mean_distribution, output_dir / "decoder_attention_lift_heatmap.png", len(artifacts))
    first_last = save_first_last_bin_mass(artifacts, output_dir / "decoder_first_last_bin_mass.png", samples=8, seed=1)
    top_positions = save_top_bin_position(artifacts, output_dir / "decoder_top_bin_position.png")
    accuracy = save_accuracy_by_category(artifacts, output_dir / "accuracy_by_category.png")
    stats = aggregate_diagnostics(artifacts, samples=8, seed=2)
    diagnostics = {
        "validation": {
            "num_complete_artifacts": 2,
            "num_unique_source_videos": 2,
            "num_failed_records": 1,
            "overall_accuracy": {"correct": 1, "total": 2, "accuracy": 0.5},
        },
        **stats,
        "decoder_attention_lift_heatmap": heatmap,
        "first_last_bin_mass": first_last,
        "top_bin_position": top_positions,
        "accuracy": accuracy,
    }
    report_path = tmp_path / "report.md"
    write_report(report_path, diagnostics)

    assert heatmap["max_absolute_lift_minus_uniform_deviation"] > 0
    assert len(top_positions["fraction_by_layer_bin"]) == 32
    assert "overall" in accuracy
    assert (output_dir / "decoder_attention_lift_heatmap.png").is_file()
    assert (output_dir / "decoder_first_last_bin_mass.png").is_file()
    assert (output_dir / "decoder_top_bin_position.png").is_file()
    assert (output_dir / "accuracy_by_category.png").is_file()
    assert "Repeated-frame, reversed-video, and matched-Qwen controls are still required" in report_path.read_text()
