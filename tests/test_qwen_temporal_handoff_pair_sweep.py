from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_qwen_temporal_handoff_pair_sweep import (
    RETAINED_PAIRS,
    analyze_outputs,
    artifact_path,
    average_ranks,
    config_mismatches,
    pair_key,
    per_example_summary,
    prepare_run_config,
    spearman,
)


BASE_CONFIG = {
    "schema_version": "qwen_temporal_handoff_pair_sweep_v1",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "resolution_config": "medium",
    "sampling_mode": "cross_model_8",
    "handoff_layer": 8,
    "memory_tokens_per_region": 2,
    "seed": 20260818,
    "conditions": ["handoff_mean", "hard_evict"],
    "retained_pairs": [list(pair) for pair in RETAINED_PAIRS],
    "git_commit": "abc123",
}


def _scores(logp: float, margin: float) -> dict:
    return {
        "choice_logits": [logp + 5.0, 1.0, 0.0, -1.0, -2.0],
        "correct_choice_log_probability": logp,
        "correct_vs_best_incorrect_margin": margin,
    }


def _artifact(qid: str, condition: str, pair: tuple[int, int] | None, *, logp: float, margin: float, mass: float | None = None) -> dict:
    return {
        "question_id": qid,
        "condition": condition,
        "retained_pair": list(pair) if pair is not None else None,
        "status": "complete",
        "model_backend": "qwen",
        "model_checkpoint": BASE_CONFIG["model_id"],
        "correct_idx": 0,
        "predicted_idx": 0,
        "correct": True,
        "answer_choice_scores": _scores(logp, margin),
        "retained_baseline_attention_mass": mass,
        "temporal_handoff": {
            "instrumentation": {
                "total_estimated_attention_flops": 500.0 if condition != "dense_custom" else 1000.0,
                "final_visual_token_indices": list(range(20 if condition != "dense_custom" else 40)),
                "final_memory_token_indices": [0, 1, 2, 3] if condition == "handoff_mean" else [],
                "layers": [
                    {
                        "layer": 0,
                        "layer_type": "full_attention",
                        "native_sdpa_is_causal_used": True,
                        "explicit_mask_materialized": False,
                        "mask_shape": None,
                    }
                ],
            }
        },
        "metadata": {
            "git_commit": BASE_CONFIG["git_commit"],
            "run_config": dict(BASE_CONFIG),
            "original_sequence_length": 100,
            "final_sequence_length": 60 if condition == "handoff_mean" else 56 if condition == "hard_evict" else 100,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_fixture(root: Path, qids: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "run_config.json", BASE_CONFIG)
    manifest = root / "manifest.jsonl"
    with manifest.open("w") as handle:
        for q_index, qid in enumerate(qids):
            handle.write(json.dumps({"question_id": qid, "category": "gaze", "participant_id": f"P{q_index}", "source_video_id": f"v{q_index}"}) + "\n")
            dense = _artifact(qid, "dense_custom", None, logp=-1.0, margin=0.0)
            _write_json(artifact_path(root, qid, None, "dense_custom"), dense)
            _write_json(root / "artifacts" / qid / "dense_equivalence_report.json", {"question_id": qid, "passed": True})
            for pair_index, pair in enumerate(RETAINED_PAIRS):
                mass = 0.1 + 0.1 * pair_index
                handoff_logp = -1.2 + 0.1 * pair_index
                hard_logp = handoff_logp - 0.05
                _write_json(artifact_path(root, qid, pair, "handoff_mean"), _artifact(qid, "handoff_mean", pair, logp=handoff_logp, margin=handoff_logp + 1.0, mass=mass))
                _write_json(artifact_path(root, qid, pair, "hard_evict"), _artifact(qid, "hard_evict", pair, logp=hard_logp, margin=hard_logp + 1.0, mass=mass))
    return manifest


def test_retained_pairs_are_exhaustive_six_pairs():
    assert RETAINED_PAIRS == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert [pair_key(pair) for pair in RETAINED_PAIRS] == ["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]


def test_average_rank_spearman_handles_ties():
    assert average_ranks([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_per_example_summary_uses_examples_not_pair_rows():
    rows = []
    qid = "q00"
    for pair_index, pair in enumerate(RETAINED_PAIRS):
        mass = 0.1 + 0.1 * pair_index
        rows.append(
            {
                "question_id": qid,
                "condition": "handoff_mean",
                "retained_pair": pair_key(pair),
                "retained_attention_mass": mass,
                "correct_choice_log_probability": -1.2 + 0.1 * pair_index,
                "answer_margin": -0.2 + 0.1 * pair_index,
                "delta_logp_vs_dense": -0.2 + 0.1 * pair_index,
            }
        )
        rows.append(
            {
                "question_id": qid,
                "condition": "hard_evict",
                "retained_pair": pair_key(pair),
                "retained_attention_mass": mass,
                "correct_choice_log_probability": -1.25 + 0.1 * pair_index,
                "answer_margin": -0.25 + 0.1 * pair_index,
                "delta_logp_vs_dense": -0.25 + 0.1 * pair_index,
            }
        )
    summary = per_example_summary(rows)
    assert len(summary) == 1
    row = summary[0]
    assert row["attention_selected_pair"] == "2_3"
    assert row["attention_selected_rank_logp"] == 1
    assert row["attention_selected_is_best"] is True
    assert row["mean_handoff_minus_hard_evict_logp"] == pytest.approx(0.05)
    assert row["spearman_attention_mass_vs_logp"] == pytest.approx(1.0)


def test_analyze_outputs_writes_example_level_summary(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)])
    summary = analyze_outputs(root, manifest, bootstrap_samples=100, seed=1, write_outputs=False)
    assert summary["num_examples"] == 15
    assert summary["num_pair_condition_rows"] == 180
    assert summary["memory_benefit"]["mean_handoff_minus_hard_evict_logp"]["mean"] == pytest.approx(0.05)
    assert summary["selection_benefit"]["fraction_best"] == pytest.approx(1.0)


def test_config_resume_mismatch_rejects_changed_memory_tokens(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    _write_json(root / "run_config.json", BASE_CONFIG)
    (root / "records.jsonl").write_text("")
    requested = dict(BASE_CONFIG)
    requested["memory_tokens_per_region"] = 4
    mismatches = config_mismatches(BASE_CONFIG, requested)
    assert "memory_tokens_per_region" in mismatches
    with pytest.raises(RuntimeError, match="memory_tokens_per_region"):
        prepare_run_config(root, requested, overwrite=False)


def test_overwrite_cleans_pair_sweep_outputs_only(tmp_path: Path):
    root = tmp_path / "run"
    _write_fixture(root, ["q00"])
    (root / "per_pair.csv").write_text("stale")
    keep = root / "notes.txt"
    keep.write_text("keep")
    requested = dict(BASE_CONFIG)
    requested["git_commit"] = "new"
    saved = prepare_run_config(root, requested, overwrite=True)
    assert saved["git_commit"] == "new"
    assert not (root / "artifacts").exists()
    assert not (root / "per_pair.csv").exists()
    assert keep.read_text() == "keep"
