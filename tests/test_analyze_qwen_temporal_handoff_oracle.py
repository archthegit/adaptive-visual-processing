from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_qwen_temporal_handoff_oracle import analyze, oracle_per_example
from scripts.run_qwen_temporal_handoff_pair_sweep import RETAINED_PAIRS, artifact_path, pair_key


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
        "answer_choice_scores": _scores(logp, margin),
        "retained_baseline_attention_mass": mass,
        "correct": logp > -10,
        "metadata": {"original_sequence_length": 100, "final_sequence_length": 60 if condition != "dense_custom" else 100},
        "temporal_handoff": {
            "instrumentation": {
                "total_estimated_attention_flops": 500.0 if condition != "dense_custom" else 1000.0,
                "final_visual_token_indices": list(range(20)),
                "final_memory_token_indices": [0, 1, 2, 3] if condition == "handoff_mean" else [],
            }
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_fixture(root: Path, qids: list[str], *, selected_bad: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    with manifest.open("w") as handle:
        for qindex, qid in enumerate(qids):
            handle.write(json.dumps({"question_id": qid, "category": "gaze", "participant_id": f"P{qindex}", "source_video_id": f"v{qindex}"}) + "\n")
            _write_json(artifact_path(root, qid, None, "dense_custom"), _artifact(qid, "dense_custom", None, logp=-1.0, margin=0.0))
            for pair_index, pair in enumerate(RETAINED_PAIRS):
                mass = 0.6 - pair_index * 0.05 if selected_bad else 0.1 + pair_index * 0.1
                handoff_logp = -1.2 + pair_index * 0.1
                hard_logp = handoff_logp - 0.05
                _write_json(artifact_path(root, qid, pair, "handoff_mean"), _artifact(qid, "handoff_mean", pair, logp=handoff_logp, margin=handoff_logp + 1.0, mass=mass))
                _write_json(artifact_path(root, qid, pair, "hard_evict"), _artifact(qid, "hard_evict", pair, logp=hard_logp, margin=hard_logp + 1.0, mass=mass))
    return manifest


def test_oracle_per_example_computes_best_worst_regret_and_threshold_counts():
    rows = []
    qid = "q00"
    for pair_index, pair in enumerate(RETAINED_PAIRS):
        for condition in ("handoff_mean", "hard_evict"):
            rows.append(
                {
                    "question_id": qid,
                    "condition": condition,
                    "retained_pair": pair_key(pair),
                    "retained_attention_mass": 0.1 + 0.1 * pair_index,
                    "correct_choice_log_probability": -1.2 + 0.1 * pair_index - (0.05 if condition == "hard_evict" else 0.0),
                    "answer_margin": -0.2 + 0.1 * pair_index - (0.05 if condition == "hard_evict" else 0.0),
                    "dense_correct_choice_log_probability": -1.0,
                    "dense_answer_margin": 0.0,
                }
            )
    per_example = oracle_per_example(rows)
    assert len(per_example) == 2
    item = next(row for row in per_example if row["condition"] == "handoff_mean")
    assert item["best_pair"] == "2_3"
    assert item["worst_pair"] == "0_1"
    assert item["attention_selected_pair"] == "2_3"
    assert item["best_pair_delta_logp_vs_dense"] == pytest.approx(0.3)
    assert item["worst_pair_delta_logp_vs_dense"] == pytest.approx(-0.2)
    assert item["attention_selected_regret_logp"] == pytest.approx(0.0)
    assert item["any_pair_preserves_or_improves_dense_logp"] is True
    assert item["num_pairs_within_0_1_logp_of_dense"] == 5


def test_analyze_outputs_writes_oracle_files_and_uses_examples_as_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)])
    monkeypatch.setattr("scripts.analyze_qwen_temporal_handoff_oracle.write_plots", lambda *_args, **_kwargs: None)
    summary = analyze(root, manifest, bootstrap_samples=100, seed=1, write_outputs=True)
    assert summary["conditions"]["handoff_mean"]["num_examples"] == 15
    assert summary["conditions"]["handoff_mean"]["best_pair_delta_logp_vs_dense"]["mean"] == pytest.approx(0.3)
    assert summary["conditions"]["handoff_mean"]["fraction_any_pair_preserves_or_improves_dense_logp"]["mean"] == pytest.approx(1.0)
    assert summary["answers"]["does_compact_memory_improve_oracle_ceiling_over_hard_deletion"] is True
    assert (root / "oracle_per_example.csv").exists()
    assert (root / "fixed_pair_summary.csv").exists()
    assert (root / "oracle_summary.json").exists()
    assert (root / "oracle_report.md").exists()


def test_attention_selected_regret_detects_bad_attention_selection(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)], selected_bad=True)
    summary = analyze(root, manifest, bootstrap_samples=100, seed=1, write_outputs=False)
    regret = summary["conditions"]["handoff_mean"]["attention_selected_regret_logp"]["mean"]
    assert regret > 0.0
    assert summary["answers"]["is_there_oracle_headroom_for_better_selector"] is True


def test_fixed_pair_summary_reports_all_six_pairs_per_condition(tmp_path: Path):
    root = tmp_path / "run"
    manifest = _write_fixture(root, [f"q{i:02d}" for i in range(15)])
    summary = analyze(root, manifest, bootstrap_samples=100, seed=1, write_outputs=False)
    rows = summary["fixed_pair_summary"]
    assert len(rows) == 12
    handoff_pairs = {row["retained_pair"] for row in rows if row["condition"] == "handoff_mean"}
    assert handoff_pairs == {pair_key(pair) for pair in RETAINED_PAIRS}
    assert min(row["rank_across_fixed_pairs"] for row in rows if row["condition"] == "handoff_mean") == 1
