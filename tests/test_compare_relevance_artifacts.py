import numpy as np

from scripts.compare_relevance_artifacts import compare_answer_choice_scores, compare_dirs, compare_topk_logits, max_abs_diff
from src.io import append_jsonl, write_json


def test_max_abs_diff_reports_inf_for_shape_mismatch():
    assert max_abs_diff([1.0], [[1.0]]) == float("inf")


def test_max_abs_diff_compares_nested_scores():
    assert np.isclose(max_abs_diff([[1.0, 2.0]], [[1.0, 2.25]]), 0.25)


def test_compare_topk_logits_reports_token_order_and_logit_diff():
    left = {"metadata": {"prefill_next_token_topk": [{"token_id": 1, "logit": 2.0}]}}
    right = {"metadata": {"prefill_next_token_topk": [{"token_id": 1, "logit": 1.75}]}}
    summary = compare_topk_logits(left, right)
    assert summary["token_ids_match"] is True
    assert np.isclose(summary["max_abs_logit_diff_matching_positions"], 0.25)


def test_compare_answer_choice_scores_reports_baseline_vs_masked_deltas():
    left = {
        "answer_choice_scores": {
            "choice_logits": [1.0, 0.0],
            "choice_log_probabilities": [-0.5, -1.5],
            "normalized_choice_probabilities": [0.73, 0.27],
            "correct_choice_log_probability": -0.5,
            "correct_vs_strongest_incorrect_margin": 1.0,
        }
    }
    right = {
        "answer_choice_scores": {
            "choice_logits": [0.25, 0.75],
            "choice_log_probabilities": [-1.2, -0.7],
            "normalized_choice_probabilities": [0.38, 0.62],
            "correct_choice_log_probability": -1.2,
            "correct_vs_strongest_incorrect_margin": -0.5,
        }
    }

    summary = compare_answer_choice_scores(left, right)

    assert np.isclose(summary["max_abs_diff_by_field"]["normalized_choice_probabilities"], 0.35)
    assert np.isclose(summary["correct_choice_log_probability_delta"], -0.7)
    assert np.isclose(summary["correct_margin_delta"], -1.5)
    assert "pre-encoder" in summary["comparison_scope"]


def test_compare_dirs_includes_answer_choice_scores_for_pre_encoder_artifacts(tmp_path):
    left_dir = tmp_path / "baseline"
    right_dir = tmp_path / "masked"
    left_dir.mkdir()
    right_dir.mkdir()
    base_artifact = left_dir / "q1.json"
    masked_artifact = right_dir / "q1.json"
    common = {
        "question_id": "q1",
        "correct": True,
        "metadata": {"attention_extraction": "reduced_sdpa"},
        "temporal_relevance": {
            "raw_temporal_bin_scores": [[0.1, 0.9]],
            "normalized_temporal_bin_scores": [[0.1, 0.9]],
            "absolute_question_to_visual_attention_mass": [0.5],
        },
    }
    write_json(
        base_artifact,
        {
            **common,
            "answer_choice_scores": {
                "choice_logits": [1.0, 0.0],
                "choice_log_probabilities": [-0.5, -1.5],
                "normalized_choice_probabilities": [0.73, 0.27],
                "correct_choice_log_probability": -0.5,
                "correct_vs_strongest_incorrect_margin": 1.0,
            },
        },
    )
    write_json(
        masked_artifact,
        {
            **common,
            "answer_choice_scores": {
                "choice_logits": [0.25, 0.75],
                "choice_log_probabilities": [-1.2, -0.7],
                "normalized_choice_probabilities": [0.38, 0.62],
                "correct_choice_log_probability": -1.2,
                "correct_vs_strongest_incorrect_margin": -0.5,
            },
            "metadata": {
                "attention_extraction": "reduced_sdpa",
                "pre_encoder_removed_temporal_bins": [1],
                "answer_choice_comparison_scope": "compare_answer_choice_scores_with_matching_unmasked_baseline_artifact",
            },
        },
    )
    append_jsonl(left_dir / "records.jsonl", {"question_id": "q1", "status": "complete", "artifact": str(base_artifact)})
    append_jsonl(
        right_dir / "records.jsonl",
        {"question_id": "q1", "status": "complete", "artifact": str(masked_artifact)},
    )

    summary = compare_dirs(left_dir, right_dir)

    assert np.isclose(
        summary["comparisons"][0]["answer_choice_scores"]["correct_choice_log_probability_delta"],
        -0.7,
    )
