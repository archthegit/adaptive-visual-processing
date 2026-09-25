from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_qwen_temporal_compaction_heldout import (
    CONDITIONS,
    FROZEN_RETAINED_CELLS,
    analyze_outputs,
    artifact_path,
    config_mismatches,
    manifest_by_id,
    prepare_run_config,
    primary_gate,
    requested_run_config,
    select_profiling_subset,
    summarize_analysis,
    validate_all_outputs,
    validate_condition_artifact,
)


BASE_CONFIG = {
    "schema_version": "qwen_temporal_compaction_heldout_pair_1_3_v1",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "resolution_config": "medium",
    "sampling_mode": "cross_model_8",
    "handoff_layer": 8,
    "retained_native_temporal_cells": [1, 3],
    "retention_ratio": 0.5,
    "primary_condition": "hard_evict",
    "secondary_condition": "handoff_mean",
    "memory_tokens_per_region": 2,
    "seed": 20260818,
    "conditions": list(CONDITIONS),
    "quality_forward_repetitions": 1,
    "profiling_subset_question_ids": ["q00", "q01"],
    "warmup": 3,
    "repeats": 10,
    "git_commit": "abc123",
}


def _scores(logp: float, margin: float, pred: int = 0) -> dict:
    logits = [0.0, -1.0, -2.0, -3.0, -4.0]
    logits[pred] = 5.0
    return {
        "choice_logits": logits,
        "correct_choice_log_probability": logp,
        "correct_vs_best_incorrect_margin": margin,
    }


def _profile(latency: float | None, *, profiled: bool = True) -> dict:
    if latency is None:
        return {"profiled": False}
    return {
        "profiled": profiled,
        "prefill_latency_seconds_median": latency,
        "prefill_latency_seconds_mean": latency,
        "prefill_latency_seconds_stddev": 0.0,
        "incremental_peak_allocated_bytes": 100,
        "absolute_peak_allocated_bytes": 120,
    }


def _artifact(
    qid: str,
    condition: str,
    *,
    logp: float,
    margin: float,
    correct: bool = True,
    retained: list[int] | None = None,
    memory: int | None = None,
    latency: float | None = 1.0,
    flops: float = 1000.0,
) -> dict:
    if retained is None:
        retained = [] if condition == "dense_custom" else list(FROZEN_RETAINED_CELLS)
    if memory is None:
        memory = 2 if condition == "handoff_mean" else 0
    seq = 100 if condition == "dense_custom" else 50 if condition == "hard_evict" else 54
    return {
        "question_id": qid,
        "condition": condition,
        "status": "complete",
        "model_backend": "qwen",
        "model_checkpoint": BASE_CONFIG["model_id"],
        "generation_supported": False,
        "correct_idx": 0,
        "predicted_idx": 0 if correct else 1,
        "correct": correct,
        "answer_choice_scores": _scores(logp, margin, pred=0 if correct else 1),
        "temporal_handoff": {
            "schema_version": "qwen_temporal_handoff_prefill_v1",
            "handoff_layer": 8,
            "retained_temporal_regions": retained,
            "selected_retained_native_cells": retained,
            "memory_tokens_per_region": memory,
            "condition": condition,
            "instrumentation": {
                "total_estimated_attention_flops": flops,
                "final_visual_token_indices": list(range(20)),
                "final_memory_token_indices": list(range(memory * 2)),
                "layers": [
                    {
                        "layer": 0,
                        "layer_type": "full_attention",
                        "native_sdpa_is_causal_used": True,
                        "explicit_mask_materialized": False,
                    }
                ],
            },
        },
        "metadata": {
            "git_commit": BASE_CONFIG["git_commit"],
            "run_config": dict(BASE_CONFIG),
            "resolution": {"name": "medium"},
            "sampling_mode": "cross_model_8",
            "original_sequence_length": 100,
            "final_sequence_length": seq,
            "cuda_profile": {
                "decoder_stack_excluding_lm_head": _profile(latency),
                "combined_prefill": _profile(latency),
                "final_token_lm_head": _profile(0.01 if latency is not None else None),
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_fixture(root: Path, qids: list[str], *, hard_delta: float = -0.05, handoff_delta: float = -0.02) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "run_config.json", BASE_CONFIG)
    manifest = root / "manifest.jsonl"
    with manifest.open("w") as handle:
        for idx, qid in enumerate(qids):
            handle.write(json.dumps({"question_id": qid, "category": "gaze", "participant_id": f"P{idx}", "source_video_id": f"v{idx}"}) + "\n")
            dense = _artifact(qid, "dense_custom", logp=-1.0, margin=0.0, latency=1.0, flops=1000.0)
            hard = _artifact(qid, "hard_evict", logp=-1.0 + hard_delta, margin=hard_delta, latency=0.70, flops=500.0)
            handoff = _artifact(qid, "handoff_mean", logp=-1.0 + handoff_delta, margin=handoff_delta, latency=0.72, flops=540.0)
            for artifact in (dense, hard, handoff):
                path = artifact_path(root, qid, artifact["condition"])
                _write_json(path, artifact)
                with (root / "records.jsonl").open("a") as records:
                    records.write(json.dumps({"question_id": qid, "condition": artifact["condition"], "status": "complete", "artifact": str(path)}) + "\n")
            _write_json(root / "artifacts" / qid / "dense_equivalence_report.json", {"question_id": qid, "passed": True})
    return manifest


def test_frozen_pair_enforced_by_validation():
    artifact = _artifact("q00", "hard_evict", logp=-1.0, margin=0.0, retained=[0, 1])
    with pytest.raises(RuntimeError, match="retained cells"):
        validate_condition_artifact(artifact, question_id="q00", condition="hard_evict", run_config=BASE_CONFIG)


def test_resume_config_mismatch_rejects_changed_frozen_pair(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    _write_json(root / "run_config.json", BASE_CONFIG)
    (root / "records.jsonl").write_text("")
    requested = dict(BASE_CONFIG)
    requested["retained_native_temporal_cells"] = [0, 2]
    assert "retained_native_temporal_cells" in config_mismatches(BASE_CONFIG, requested)
    with pytest.raises(RuntimeError, match="retained_native_temporal_cells"):
        prepare_run_config(root, requested, overwrite=False)


def test_dense_equivalence_rejection(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, ["q00"])
    _write_json(root / "artifacts" / "q00" / "dense_equivalence_report.json", {"question_id": "q00", "passed": False})
    artifacts = {
        "q00": {
            condition: json.loads(artifact_path(root, "q00", condition).read_text())
            for condition in CONDITIONS
        }
    }
    equivalence = {"q00": {"question_id": "q00", "passed": False}}
    with pytest.raises(RuntimeError, match="dense equivalence"):
        validate_all_outputs(artifacts, equivalence, manifest_by_id(manifest, expected_examples=1), BASE_CONFIG)


def test_exact_example_pairing_rejects_missing_condition(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, ["q00"])
    artifact_path(root, "q00", "handoff_mean").unlink()
    (root / "records.jsonl").unlink()
    with pytest.raises(RuntimeError, match="missing conditions"):
        analyze_outputs(root, manifest, bootstrap_samples=100, seed=1, write_outputs=False, expected_examples=1)


def test_example_level_bootstrap_and_primary_gate_pass(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(4)], hard_delta=-0.03)
    summary = analyze_outputs(root, manifest, bootstrap_samples=100, seed=1, write_outputs=False, expected_examples=4)
    assert summary["conditions"]["hard_evict"]["correct_choice_log_probability_delta"]["mean"] == pytest.approx(-0.03)
    assert summary["conditions"]["hard_evict"]["median_stack_latency_reduction_fraction"] == pytest.approx(0.30)
    assert primary_gate(summary)["status"] == "PASS"


def test_primary_gate_fails_quality_noninferiority():
    rows = []
    for idx in range(4):
        qid = f"q{idx}"
        for condition, delta in (("dense_custom", 0.0), ("hard_evict", -0.3), ("handoff_mean", -0.2)):
            rows.append(
                {
                    "question_id": qid,
                    "condition": condition,
                    "delta_correct_choice_log_probability": delta,
                    "delta_answer_margin": delta,
                    "dense_correct": True,
                    "condition_correct": True,
                    "prediction_changed": False,
                    "stack_latency_reduction_fraction": 0.3 if condition != "dense_custom" else 0.0,
                    "stack_latency_speedup": 1.4 if condition != "dense_custom" else 1.0,
                    "attention_flop_reduction_fraction": 0.5 if condition != "dense_custom" else 0.0,
                    "sequence_reduction_fraction": 0.5 if condition != "dense_custom" else 0.0,
                    "timing_profiled": True,
                    "condition_incremental_peak_allocated_bytes": 10,
                    "dense_incremental_peak_allocated_bytes": 10,
                }
            )
    summary = summarize_analysis(rows, bootstrap_samples=100, seed=1)
    assert summary["primary_heldout_gate"]["status"] == "FAIL"
    assert summary["primary_heldout_gate"]["quality_noninferiority_pass"] is False


def test_handoff_versus_eviction_comparison_requires_ci_above_zero(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(5)], hard_delta=-0.05, handoff_delta=-0.01)
    summary = analyze_outputs(root, manifest, bootstrap_samples=100, seed=1, write_outputs=False, expected_examples=5)
    comparison = summary["paired_condition_differences"]["handoff_mean_minus_hard_evict"]["correct_choice_log_probability_delta"]
    assert comparison["mean"] == pytest.approx(0.04)
    assert summary["secondary_memory_comparison"]["memory_benefit_supported_logp"] is True


def test_profiling_subset_selected_without_outcomes_and_recorded_in_config():
    qids = [f"q{i:02d}" for i in range(10)]
    first = select_profiling_subset(qids, count=3, seed=7)
    second = select_profiling_subset(qids, count=3, seed=7)
    assert first == second
    assert len(first) == 3

    class Args:
        model_id = BASE_CONFIG["model_id"]
        resolution_config = "medium"
        seed = 7
        profile_count = 3
        warmup = 3
        repeats = 10
        baseline_dir = "baseline"
        manifest = "manifest"
        output_dir = "output"

    config = requested_run_config(
        args=Args(),
        git_commit="abc123",
        manifest_records={qid: {"question_id": qid} for qid in qids},
    )
    assert config["profiling_subset_question_ids"] == first
