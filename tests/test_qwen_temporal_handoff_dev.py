from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_qwen_temporal_handoff_dev import (
    CONDITIONS,
    analyze_outputs,
    artifact_path,
    gate_decision,
    latest_artifacts,
    per_example_rows,
    prepare_run_config,
    summarize_analysis,
    validate_resumed_condition_artifact,
    validate_all_outputs,
)


BASE_RUN_CONFIG = {
    "schema_version": "qwen_temporal_handoff_dev_pilot_v1",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "resolution_config": "medium",
    "sampling_mode": "cross_model_8",
    "handoff_layer": 8,
    "retain_regions": 2,
    "memory_tokens_per_region": 2,
    "seed": 20260818,
    "warmup": 3,
    "repeats": 10,
    "git_commit": "abc123",
    "conditions": list(CONDITIONS),
    "timing_scope": "test fixture",
}


def _profile(latency: float, peak: int = 100) -> dict:
    return {
        "prefill_latency_seconds_median": latency,
        "prefill_latency_seconds_mean": latency,
        "prefill_latency_seconds_stddev": 0.0,
        "incremental_peak_allocated_bytes": peak,
        "absolute_peak_allocated_bytes": peak + 10,
    }


def _artifact(
    qid: str,
    condition: str,
    *,
    logp: float,
    margin: float,
    correct: bool = True,
    pred: int = 0,
    seq: int = 100,
    original_seq: int = 100,
    flops: float = 1000.0,
    latency: float = 1.0,
    memory_tokens: int = 0,
    handed_off_regions: list[int] | None = None,
) -> dict:
    layers = [
        {
            "layer": idx,
            "layer_type": "full_attention",
            "sequence_length_in": original_seq if idx == 0 else seq,
            "sequence_length_out": seq,
            "visual_token_count_in": 40,
            "visual_token_count_out": 20 if condition != "dense_custom" else 40,
            "memory_token_count_in": memory_tokens,
            "memory_token_count_out": memory_tokens,
            "text_token_count_in": 60,
            "text_token_count_out": 60,
            "attention_q_len": seq,
            "attention_k_len": seq,
            "estimated_qk_flops": int(flops / 2),
            "estimated_av_flops": int(flops / 2),
            "causal_path": "native_sdpa_is_causal",
            "explicit_mask_materialized": False,
            "mask_shape": None,
            "native_sdpa_is_causal_used": True,
            "compaction_applied_after_layer": False,
        }
        for idx in range(2)
    ]
    compaction_plan = None
    if condition != "dense_custom":
        compaction_plan = {
            "handed_off_temporal_regions": handed_off_regions or [0, 2],
            "memory_tokens_per_region": 0 if condition == "hard_evict" else 2,
        }
    return {
        "question_id": qid,
        "condition": condition,
        "status": "complete",
        "model_backend": "qwen",
        "model_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "generation_supported": False,
        "category": "gaze",
        "question_type": "gaze_x",
        "participant_id": "P01",
        "source_video_id": f"video_{qid}",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
        "predicted_idx": pred,
        "correct": correct,
        "answer_choice_scores": {
            "choice_logits": [5.0 - pred, 1.0 + pred, 0.0, -1.0, -2.0],
            "correct_choice_log_probability": logp,
            "correct_vs_best_incorrect_margin": margin,
        },
        "sampled_frame_indices": [[1, 2, 3, 4, 5, 6, 7, 8]],
        "sampled_timestamps": [[float(i) for i in range(8)]],
        "frame_bin_mappings": [[]],
        "temporal_handoff": {
            "schema_version": "qwen_temporal_handoff_prefill_v1",
            "handoff_layer": 8,
            "retained_temporal_regions": [1, 3] if condition != "dense_custom" else [],
            "memory_tokens_per_region": 0 if condition in {"dense_custom", "hard_evict"} else 2,
            "condition": condition,
            "selected_retained_native_cells": [1, 3] if condition != "dense_custom" else [],
            "compaction_plan": compaction_plan,
            "instrumentation": {
                "layers": layers,
                "total_estimated_attention_flops": flops,
                "final_question_token_indices": [90, 91],
                "final_visual_token_indices": list(range(20 if condition != "dense_custom" else 40)),
                "final_memory_token_indices": list(range(memory_tokens)),
            },
        },
        "metadata": {
            "git_commit": "abc123",
            "run_config": dict(BASE_RUN_CONFIG),
            "resolution": {"name": "medium"},
            "sampling_mode": "cross_model_8",
            "query_scope": "question",
            "original_sequence_length": original_seq,
            "final_sequence_length": seq,
            "cuda_profile": {
                "combined_prefill": _profile(latency),
                "decoder_stack_excluding_lm_head": _profile(latency),
                "final_token_lm_head": _profile(0.01),
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_run_config(root: Path, config: dict | None = None) -> None:
    _write_json(root / "run_config.json", dict(config or BASE_RUN_CONFIG))


def _write_complete_record(root: Path, artifact: dict) -> None:
    path = artifact_path(root, artifact["question_id"], artifact["condition"])
    _write_json(path, artifact)
    with (root / "records.jsonl").open("a") as handle:
        handle.write(json.dumps({"question_id": artifact["question_id"], "condition": artifact["condition"], "status": "complete", "artifact": str(path)}) + "\n")


def _write_fixture(root: Path, qids: list[str], *, handoff_logp: float = -0.95, hard_logp: float = -1.20, random_logp: float = -1.10) -> Path:
    manifest = root / "manifest.jsonl"
    with manifest.open("w") as handle:
        for qid in qids:
            handle.write(json.dumps({"question_id": qid, "category": "gaze", "question_type": "gaze_x", "participant_id": f"P{qid}", "source_video_id": f"video_{qid}"}) + "\n")
            dense = _artifact(qid, "dense_custom", logp=-1.0, margin=0.0, seq=100, original_seq=100, flops=1000, latency=1.0)
            handoff = _artifact(qid, "handoff_mean", logp=handoff_logp, margin=0.10, seq=60, original_seq=100, flops=500, latency=0.70, memory_tokens=4)
            hard = _artifact(qid, "hard_evict", logp=hard_logp, margin=-0.10, seq=56, original_seq=100, flops=450, latency=0.65)
            random = _artifact(qid, "random_handoff", logp=random_logp, margin=0.00, seq=60, original_seq=100, flops=500, latency=0.75, memory_tokens=4)
            for artifact in (dense, handoff, hard, random):
                _write_complete_record(root, artifact)
            _write_json(
                root / "artifacts" / qid / "dense_equivalence_report.json",
                {"question_id": qid, "passed": True, "stock_sequence_length": 100},
            )
    return manifest


def test_latest_artifacts_supports_condition_level_resume_records(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"], handoff_logp=-0.9)
    artifacts = latest_artifacts(root)
    assert set(artifacts["q00"]) == set(CONDITIONS)
    assert artifacts["q00"]["handoff_mean"]["answer_choice_scores"]["correct_choice_log_probability"] == -0.9


def test_same_configuration_resume_succeeds_and_validates_artifact(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"])
    _write_run_config(root)
    saved = prepare_run_config(root, dict(BASE_RUN_CONFIG), overwrite=False)
    assert saved["handoff_layer"] == 8
    artifact = validate_resumed_condition_artifact(
        artifact_path(root, "q00", "dense_custom"),
        question_id="q00",
        condition="dense_custom",
        saved_run_config=saved,
    )
    assert artifact["status"] == "complete"


def test_changed_handoff_layer_is_rejected_without_overwriting_config(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"])
    _write_run_config(root)
    requested = dict(BASE_RUN_CONFIG)
    requested["handoff_layer"] = 12
    before = (root / "run_config.json").read_text()
    with pytest.raises(RuntimeError, match="handoff_layer"):
        prepare_run_config(root, requested, overwrite=False)
    assert (root / "run_config.json").read_text() == before


def test_changed_memory_token_count_is_rejected(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"])
    _write_run_config(root)
    requested = dict(BASE_RUN_CONFIG)
    requested["memory_tokens_per_region"] = 4
    with pytest.raises(RuntimeError, match="memory_tokens_per_region"):
        prepare_run_config(root, requested, overwrite=False)


def test_changed_git_commit_is_rejected(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"])
    _write_run_config(root)
    requested = dict(BASE_RUN_CONFIG)
    requested["git_commit"] = "different"
    with pytest.raises(RuntimeError, match="git_commit"):
        prepare_run_config(root, requested, overwrite=False)


def test_corrupted_or_missing_resumed_artifact_is_rejected(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"])
    _write_run_config(root)
    path = artifact_path(root, "q00", "handoff_mean")
    path.write_text("")
    with pytest.raises(RuntimeError, match="missing or empty"):
        validate_resumed_condition_artifact(
            path,
            question_id="q00",
            condition="handoff_mean",
            saved_run_config=BASE_RUN_CONFIG,
        )
    path.unlink()
    with pytest.raises(RuntimeError, match="missing or empty"):
        validate_resumed_condition_artifact(
            path,
            question_id="q00",
            condition="handoff_mean",
            saved_run_config=BASE_RUN_CONFIG,
        )


def test_overwrite_clears_stale_experiment_outputs_only(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"])
    _write_run_config(root)
    for name in (
        "run_summary.json",
        "per_example.csv",
        "analysis_summary.json",
        "report.md",
        "latency_speedup.png",
        "quality_delta.png",
        "margin_delta.png",
        "accuracy_and_flip_rate.png",
        "sequence_and_flops_reduction.png",
    ):
        (root / name).write_text("stale")
    unrelated = root / "notes.txt"
    unrelated.write_text("keep")
    requested = dict(BASE_RUN_CONFIG)
    requested["git_commit"] = "newcommit"
    saved = prepare_run_config(root, requested, overwrite=True)
    assert saved["git_commit"] == "newcommit"
    assert not (root / "records.jsonl").exists()
    assert not (root / "artifacts").exists()
    assert not (root / "run_summary.json").exists()
    assert unrelated.read_text() == "keep"
    assert json.loads((root / "run_config.json").read_text())["git_commit"] == "newcommit"


def test_pairing_validation_and_analysis_outputs(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)])
    summary = analyze_outputs(output_dir=root, manifest=manifest, bootstrap_samples=100, seed=1, write_outputs=False)
    assert summary["dense_equivalence"]["all_passed"] is True
    assert summary["conditions"]["handoff_mean"]["median_stack_latency_speedup"] == pytest.approx(1.0 / 0.70)
    assert summary["conditions"]["handoff_mean"]["median_sequence_reduction_fraction"] == pytest.approx(0.4)
    assert summary["conditions"]["handoff_mean"]["median_attention_flop_reduction_fraction"] == pytest.approx(0.5)


def test_validation_rejects_missing_condition(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)])
    (root / "artifacts" / "q00" / "random_handoff.json").unlink()
    records_path = root / "records.jsonl"
    records = [
        line
        for line in records_path.read_text().splitlines()
        if not (json.loads(line)["question_id"] == "q00" and json.loads(line)["condition"] == "random_handoff")
    ]
    records_path.write_text("\n".join(records) + "\n")
    artifacts = latest_artifacts(root)
    reports = {
        path.parent.name: json.loads(path.read_text())
        for path in root.glob("artifacts/*/dense_equivalence_report.json")
    }
    records = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    }
    with pytest.raises(RuntimeError, match="missing conditions"):
        validate_all_outputs(artifacts, reports, records)


def test_validation_rejects_failed_dense_equivalence(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)])
    _write_json(root / "artifacts" / "q00" / "dense_equivalence_report.json", {"question_id": "q00", "passed": False, "stock_sequence_length": 100})
    with pytest.raises(RuntimeError, match="dense equivalence gate failed"):
        analyze_outputs(output_dir=root, manifest=manifest, bootstrap_samples=10, seed=1, write_outputs=False)


def test_gate_promising_reject_and_inconclusive():
    base = {
        "conditions": {
            "handoff_mean": {
                "median_stack_latency_speedup": 1.5,
                "condition_accuracy": 1.0,
                "dense_accuracy": 1.0,
                "correct_choice_log_probability_delta": {"mean": 0.1},
                "answer_margin_delta": {"mean": 0.1},
            },
            "hard_evict": {
                "correct_choice_log_probability_delta": {"mean": -0.2},
                "answer_margin_delta": {"mean": -0.2},
            },
            "random_handoff": {
                "correct_choice_log_probability_delta": {"mean": 0.0},
                "answer_margin_delta": {"mean": 0.0},
            },
        }
    }
    assert gate_decision(base, {"q": {"passed": True}})["status"] == "PROMISING"
    rejected = json.loads(json.dumps(base))
    rejected["conditions"]["handoff_mean"]["median_stack_latency_speedup"] = 1.05
    assert gate_decision(rejected, {"q": {"passed": True}})["status"] == "REJECT"
    inconclusive = json.loads(json.dumps(base))
    inconclusive["conditions"]["handoff_mean"]["correct_choice_log_probability_delta"]["mean"] = -0.1
    inconclusive["conditions"]["handoff_mean"]["answer_margin_delta"]["mean"] = 0.1
    assert gate_decision(inconclusive, {"q": {"passed": True}})["status"] == "INCONCLUSIVE"


def test_per_example_rows_pair_against_dense_control(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)])
    artifacts = latest_artifacts(root)
    records = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    }
    rows = per_example_rows(artifacts, records)
    handoff = [row for row in rows if row["question_id"] == "q00" and row["condition"] == "handoff_mean"][0]
    assert handoff["delta_correct_choice_log_probability"] == pytest.approx(0.05)
    assert handoff["stack_latency_speedup"] == pytest.approx(1.0 / 0.70)
    assert handoff["attention_flop_reduction_fraction"] == pytest.approx(0.5)
