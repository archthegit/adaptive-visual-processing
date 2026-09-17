from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_route_refresh_pilot import analyze
from src.experiment1.route_reuse import route_spec_from_baseline_artifact


def _scores() -> list[list[float]]:
    rows = []
    for layer in range(28):
        row = [0.02] * 8
        row[layer % 8] = 0.30
        row[(layer + 1) % 8] = 0.24
        total = sum(row)
        rows.append([value / total for value in row])
    return rows


def _artifact(question_id: str, index: int) -> dict:
    cells = []
    for native in range(4):
        for spatial in range(2):
            token = native * 2 + spatial
            cells.append(
                {
                    "token_index": token,
                    "visual_index": token,
                    "modality": "video",
                    "temporal_bin": native,
                    "grid_t": 4,
                    "analysis_bin": None,
                }
            )
    return {
        "question_id": question_id,
        "model_backend": "qwen",
        "model_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "question": f"Question {question_id}?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
        "predicted_idx": index % 5,
        "correct": index % 3 == 0,
        "video_clip": [{"video_id": f"P{index % 4:02d}-video-{index}", "participant_id": f"P{index % 4:02d}"}],
        "sampled_frame_indices": [list(range(8))],
        "sampled_timestamps": [[float(value) for value in range(8)]],
        "frame_bin_mappings": [[{"sample_position": value, "analysis_bin": value} for value in range(8)]],
        "sampling_metadata": [{"mode": "cross_model_8"}],
        "temporal_relevance": {"normalized_temporal_bin_scores": _scores()},
        "token_layout": {"visual_token_cells": cells, "visual_token_indices": list(range(8))},
        "answer_choice_scores": {
            "correct_choice_log_probability": -2.0 - index * 0.01,
            "correct_vs_best_incorrect_margin": -0.4 + index * 0.01,
        },
    }


def _condition_artifact(base: dict, condition: str, delta: float, path: str) -> dict:
    artifact = dict(base)
    artifact["route_reuse"] = route_spec_from_baseline_artifact(
        base,
        model="qwen",
        condition=condition,
        baseline_artifact=path,
        seed=7,
        git_commit="manifest",
    )
    artifact["metadata"] = {"route_reuse": {"type": "baseline_derived_causal_route_replay", "condition": condition}}
    artifact["run_config"] = {"git_commit": "exec"}
    artifact["answer_choice_scores"] = {"correct_choice_log_probability": 99.0, "correct_vs_best_incorrect_margin": 99.0}
    artifact["intervention_answer_choice_scores"] = {
        "correct_choice_log_probability": base["answer_choice_scores"]["correct_choice_log_probability"] + delta,
        "correct_vs_best_incorrect_margin": base["answer_choice_scores"]["correct_vs_best_incorrect_margin"] + delta,
    }
    artifact["predicted_idx"] = base["predicted_idx"]
    artifact["correct"] = base["correct"]
    return artifact


def _write_run(path: Path, artifacts: dict[str, dict]) -> None:
    path.mkdir(parents=True)
    records = []
    for qid, artifact in artifacts.items():
        artifact_path = path / f"{qid}.json"
        artifact_path.write_text(json.dumps(artifact))
        records.append({"question_id": qid, "status": "complete", "artifact": artifact_path.name})
    (path / "records.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")


def _write_manifest(path: Path, question_ids: list[str]) -> None:
    rows = [
        {
            "question_id": qid,
            "source_video_id": f"P{index % 4:02d}-video-{index}",
            "participant_id": f"P{index % 4:02d}",
            "category": "gaze",
            "question_type": "gaze_synthetic",
        }
        for index, qid in enumerate(question_ids)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_route_refresh_analyzer_compares_gap2_gap3_gap4(tmp_path: Path):
    qids = [f"q{index:02d}" for index in range(15)]
    baseline = {qid: _artifact(qid, index) for index, qid in enumerate(qids)}
    baseline_dir = tmp_path / "baseline"
    _write_run(baseline_dir, baseline)
    manifest = tmp_path / "dev.jsonl"
    _write_manifest(manifest, qids)
    dirs = {}
    deltas = {"gap2": 0.3, "gap3": 0.2, "gap4": 0.1}
    conditions = {
        "gap2": "route_reuse_gap2_top50",
        "gap3": "route_reuse_gap3_top50",
        "gap4": "route_reuse_gap4_top50",
    }
    for label, condition in conditions.items():
        path = tmp_path / label
        _write_run(
            path,
            {qid: _condition_artifact(artifact, condition, deltas[label], str(baseline_dir / f"{qid}.json")) for qid, artifact in baseline.items()},
        )
        dirs[label] = path

    summary = analyze(
        baseline_dir=baseline_dir,
        condition_dirs=dirs,
        dev_manifest=manifest,
        output_dir=tmp_path / "analysis",
        bootstrap_samples=10,
        seed=3,
    )

    assert summary["validation"]["conditions"]["gap2"]["theoretical_edge_savings"]["num_routed_layers"] == 10
    assert summary["validation"]["conditions"]["gap3"]["theoretical_edge_savings"]["num_routed_layers"] == 13
    assert summary["validation"]["conditions"]["gap4"]["theoretical_edge_savings"]["num_routed_layers"] == 15
    assert summary["metrics"]["preregistered_ordering_gap2_gt_gap3_gt_gap4"] is True
    assert summary["metrics"]["paired_differences"]["gap2_minus_gap4"]["log_probability_delta"]["mean"] > 0
    assert (tmp_path / "analysis" / "per_example.csv").is_file()
    assert (tmp_path / "analysis" / "summary.json").is_file()
