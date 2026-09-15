from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.analyze_route_reuse_pilot as pilot
from scripts.analyze_route_reuse_pilot import (
    CONDITIONS,
    EXPECTED_ROUTED_LAYERS,
    analyze,
)


def _route(condition: str, commit: str = "abc123") -> dict:
    layer_routes = {}
    for layer in EXPECTED_ROUTED_LAYERS:
        layer_routes[str(layer)] = {
            "source_anchor_layer": max(anchor for anchor in (8, 12, 16, 20, 24) if anchor < layer),
            "selected_native_unit_ids": [0, 1],
            "omitted_native_unit_ids": [2, 3],
            "selected_analysis_bins": [0, 1, 2, 3],
            "omitted_analysis_bins": [4, 5, 6, 7],
            "allowed_visual_token_indices": [0, 1, 2, 3],
            "blocked_visual_token_indices": [4, 5, 6, 7],
            "actual_retained_visual_token_fraction": 0.5,
        }
    return {
        "type": "baseline_derived_causal_route_replay",
        "condition": condition,
        "routing_unit_type": "qwen_native_temporal_cell",
        "native_routing_units": [
            {"unit_id": index, "visual_token_indices": [index * 2, index * 2 + 1]}
            for index in range(4)
        ],
        "retained_native_units": 2,
        "mean_actual_retained_visual_token_fraction": 0.5,
        "anchor_layers": [8, 12, 16, 20, 24],
        "layer_routes": layer_routes,
        "git_commit": commit,
    }


def _base_artifact(question_id: str, index: int) -> dict:
    category = ("fine_grained", "gaze", "ingredient", "object_motion")[index % 4]
    predicted = index % 5
    correct = index % 3 == 0
    return {
        "question_id": question_id,
        "model_backend": "qwen",
        "model_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "question_type": f"{category}_synthetic",
        "category": category,
        "question": f"What happens in example {index}?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
        "predicted_idx": predicted,
        "correct": correct,
        "video_clip": [{"video_id": f"P{index % 5:02d}-video-{index:02d}", "participant_id": f"P{index % 5:02d}"}],
        "sampled_frame_indices": [list(range(8))],
        "sampled_timestamps": [[float(value) for value in range(8)]],
        "frame_bin_mappings": [[{"sample_position": value, "analysis_bin": value} for value in range(8)]],
        "sampling_metadata": [{"mode": "cross_model_8", "frames_per_bin": 1}],
        "answer_choice_scores": {
            "correct_choice_log_probability": -2.0 - index * 0.01,
            "correct_vs_best_incorrect_margin": -0.5 + index * 0.01,
        },
    }


def _routed_artifact(question_id: str, index: int, condition: str, delta: float) -> dict:
    artifact = _base_artifact(question_id, index)
    artifact["predicted_idx"] = (index + (0 if condition == "route_reuse_gap4_top50" else 1)) % 5
    artifact["correct"] = artifact["predicted_idx"] == artifact["correct_idx"]
    artifact["metadata"] = {
        "route_reuse": _route(condition),
        "prefill_runtime_seconds": 1.0,
        "answer_scoring_runtime_seconds": 0.5,
        "generation_runtime_seconds": 0.25,
    }
    artifact["route_reuse"] = artifact["metadata"]["route_reuse"]
    artifact["answer_choice_scores"] = {
        "correct_choice_log_probability": 99.0,
        "correct_vs_best_incorrect_margin": 99.0,
    }
    artifact["intervention_answer_choice_scores"] = {
        "correct_choice_log_probability": artifact["answer_choice_scores"]["correct_choice_log_probability"] - 99.0 - 2.0 - index * 0.01 + delta,
        "correct_vs_best_incorrect_margin": -0.5 + index * 0.01 + delta,
    }
    return artifact


def _write_run(output_dir: Path, artifacts: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for question_id, artifact in artifacts.items():
        artifact_path = output_dir / f"{question_id}.json"
        artifact_path.write_text(json.dumps(artifact))
        records.append(
            {
                "question_id": question_id,
                "status": "complete",
                "artifact": artifact_path.name,
            }
        )
    (output_dir / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _write_dev_manifest(path: Path, question_ids: list[str]) -> None:
    rows = []
    for index, question_id in enumerate(question_ids):
        category = ("fine_grained", "gaze", "ingredient", "object_motion")[index % 4]
        rows.append(
            {
                "question_id": question_id,
                "source_video_id": f"P{index % 5:02d}-video-{index:02d}",
                "participant_id": f"P{index % 5:02d}",
                "category": category,
                "question_type": f"{category}_synthetic",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _fixtures(tmp_path: Path):
    question_ids = [f"q{index:02d}" for index in range(15)]
    manifest = tmp_path / "dev.jsonl"
    _write_dev_manifest(manifest, question_ids)
    baseline_dir = tmp_path / "baseline"
    _write_run(
        baseline_dir,
        {question_id: _base_artifact(question_id, index) for index, question_id in enumerate(question_ids)},
    )
    dirs = {}
    deltas = {
        "adaptive": 0.20,
        "random": 0.05,
        "uniform": -0.05,
    }
    for label, condition in CONDITIONS.items():
        path = tmp_path / label
        _write_run(
            path,
            {
                question_id: _routed_artifact(question_id, index, condition, deltas[label])
                for index, question_id in enumerate(question_ids)
            },
        )
        dirs[label] = path
    return baseline_dir, dirs, manifest, question_ids


def test_route_reuse_pilot_analyzer_uses_intervention_scores_and_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline_dir, dirs, manifest, _question_ids = _fixtures(tmp_path)
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(pilot, "save_bar_plot", lambda _rows, _field, output, _ylabel, _title: Path(output).write_bytes(b"png"))
    monkeypatch.setattr(pilot, "save_accuracy_flip_plot", lambda _rows, output: Path(output).write_bytes(b"png"))

    summary = analyze(
        baseline_dir=baseline_dir,
        condition_dirs=dirs,
        dev_manifest=manifest,
        output_dir=output_dir,
        bootstrap_samples=25,
        seed=11,
    )

    assert summary["validation"]["code_commits"] == ["abc123"]
    assert summary["validation"]["conditions"]["adaptive"]["num_examples"] == 15
    assert summary["causal_gate"]["status"] == "PASS"
    for filename in (
        "summary.json",
        "per_example.csv",
        "quality_delta.png",
        "margin_delta.png",
        "accuracy_and_flip_rates.png",
        "report.md",
    ):
        assert (output_dir / filename).is_file()

    with (output_dir / "per_example.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 45
    adaptive_q00 = next(row for row in rows if row["question_id"] == "q00" and row["condition_label"] == "adaptive")
    assert float(adaptive_q00["routed_correct_choice_log_probability"]) == pytest.approx(-1.8)
    assert float(adaptive_q00["routed_correct_choice_log_probability"]) != pytest.approx(99.0)


def test_route_reuse_pilot_analyzer_rejects_held_out_artifact_rows(tmp_path: Path):
    baseline_dir, dirs, manifest, question_ids = _fixtures(tmp_path)
    extra_id = "q-heldout"
    extra_artifact = _routed_artifact(extra_id, 99, CONDITIONS["adaptive"], 0.1)
    adaptive_records = {question_id: _routed_artifact(question_id, index, CONDITIONS["adaptive"], 0.2) for index, question_id in enumerate(question_ids)}
    adaptive_records[extra_id] = extra_artifact
    _write_run(dirs["adaptive"], adaptive_records)

    with pytest.raises(RuntimeError, match="expected exactly the 15 dev IDs"):
        analyze(
            baseline_dir=baseline_dir,
            condition_dirs=dirs,
            dev_manifest=manifest,
            output_dir=tmp_path / "analysis",
            bootstrap_samples=5,
            seed=11,
        )


def test_route_reuse_pilot_analyzer_requires_intervention_scores(tmp_path: Path):
    baseline_dir, dirs, manifest, question_ids = _fixtures(tmp_path)
    artifacts = {
        question_id: _routed_artifact(question_id, index, CONDITIONS["uniform"], -0.05)
        for index, question_id in enumerate(question_ids)
    }
    artifacts["q00"].pop("intervention_answer_choice_scores")
    _write_run(dirs["uniform"], artifacts)

    with pytest.raises(RuntimeError, match="missing intervention_answer_choice_scores"):
        analyze(
            baseline_dir=baseline_dir,
            condition_dirs=dirs,
            dev_manifest=manifest,
            output_dir=tmp_path / "analysis",
            bootstrap_samples=5,
            seed=11,
        )
