import json

from src.experiment1.v2_analysis import (
    average_decoder_heatmap,
    condition_table,
    flatten_completed_artifacts,
    expected_conditions,
    expected_run_matrix,
    reversed_distribution_to_original_bins,
    validate_completeness,
    write_v2_analysis_outputs,
)
from src.io import write_jsonl


def _primary_records():
    return [
        {
            "split": "dev",
            "condition": "unused",
            "question_id": "q1",
            "source_video_id": "v1",
            "participant_id": "p1",
            "category": "gaze",
            "duration_group": "short",
        },
        {
            "split": "test",
            "condition": "unused",
            "question_id": "q2",
            "source_video_id": "v2",
            "participant_id": "p2",
            "category": "ingredient",
            "duration_group": "long",
        },
    ]


def test_expected_run_matrix_expands_conditions_per_source_video():
    records = _primary_records()
    matrix = expected_run_matrix(records, include_fusion_depth=False)
    expected = [
        row
        for record in records
        for row in expected_run_matrix([record], include_fusion_depth=False)
    ]
    assert len(matrix) == len(expected)
    assert {row["source_video_id"] for row in matrix} == {"v1", "v2"}
    assert {row["condition"] for row in matrix} == set(expected_conditions(include_fusion_depth=False))
    assert all(row["split"] == "test" for row in matrix if row["condition"].startswith("mask_"))
    assert [row for row in matrix if row["condition"] == "baseline_fixed_budget"][0]["sampling_mode"] == "fixed_budget"


def test_expected_run_matrix_only_requires_existing_same_video_controls():
    matrix = expected_run_matrix(_primary_records(), include_fusion_depth=False, same_video_question_ids={"q2"})
    same_video = [row for row in matrix if row["condition"] == "same_video_different_query"]
    assert [row["question_id"] for row in same_video] == ["q2"]


def test_validate_completeness_counts_complete_missing_and_failed(tmp_path):
    output_root = tmp_path / "runs"
    condition_dir = output_root / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "q1.json").write_text(json.dumps({"question_id": "q1", "correct": True}))
    (condition_dir / "q2.json").write_text(json.dumps({"question_id": "q2", "status": "failed", "error": "boom"}))
    report = validate_completeness(_primary_records(), output_root, include_fusion_depth=False)
    assert report["complete"] == 1
    assert report["failed"] == 1
    assert report["missing"] == len(expected_run_matrix(_primary_records(), False)) - 2
    assert report["failed_by_condition"] == {"baseline": 1}


def test_write_v2_analysis_outputs_creates_final_artifacts(tmp_path):
    manifest = tmp_path / "primary.jsonl"
    write_jsonl(manifest, _primary_records())
    output_root = tmp_path / "runs"
    final_dir = tmp_path / "final"
    outputs = write_v2_analysis_outputs(
        manifest,
        output_root,
        final_dir,
        bootstrap_replicates=10,
        include_fusion_depth=False,
    )
    assert outputs["completeness"]["expected"] == len(expected_run_matrix(_primary_records(), False))
    assert (final_dir / "figures").is_dir()
    assert (final_dir / "tables" / "expected_run_matrix.jsonl").exists()
    assert (final_dir / "statistical_results.json").exists()
    assert (final_dir / "completeness_report.json").exists()
    assert (final_dir / "experiment_report.md").exists()
    assert (final_dir / "paper_artifacts_manifest.json").exists()


def test_condition_table_and_average_decoder_heatmap_from_completed_artifacts(tmp_path):
    output_root = tmp_path / "runs"
    condition_dir = output_root / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "q1.json").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "correct": True,
                "video_clip": [{"video_id": "v1", "participant_id": "p1"}],
                "answer_choice_scores": {
                    "correct_choice_log_probability": -0.1,
                    "correct_vs_best_incorrect_margin": 0.4,
                },
                "token_layout": {"num_visual_tokens": 100},
                "metadata": {"generation_runtime_seconds": 2.0},
                "temporal_relevance": {
                    "normalized_temporal_bin_scores": [[0.2, 0.8], [0.7, 0.3]],
                    "layer_metrics": [
                        {"normalized_temporal_entropy": 0.5, "top1_temporal_bin_mass": 0.8, "bins_to_80pct_mass": 1},
                        {"normalized_temporal_entropy": 0.6, "top1_temporal_bin_mass": 0.7, "bins_to_80pct_mass": 2},
                    ],
                },
            }
        )
    )
    rows = [
        {
            "condition": "baseline",
            "correct": True,
            "correct_choice_log_probability": -0.1,
            "correct_vs_best_incorrect_margin": 0.4,
            "latency_seconds": 2.0,
            "visual_token_count": 100,
        }
    ]
    table = condition_table(rows)
    assert table[0]["accuracy"] == 1.0
    assert table[0]["mean_correct_choice_log_probability"] == -0.1
    heatmap = average_decoder_heatmap(output_root, bins=4)
    assert heatmap.shape == (2, 4)


def test_decoder_intervention_rows_use_intervention_answer_scores(tmp_path):
    output_root = tmp_path / "runs"
    condition_dir = output_root / "fusion_block_top20_after_layer_8"
    condition_dir.mkdir(parents=True)
    (condition_dir / "q1.json").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "video_clip": [{"video_id": "v1", "participant_id": "p1"}],
                "metadata": {"answer_choice_comparison_scope": "same_artifact_intervention_answer_choice_scores"},
                "answer_choice_scores": {
                    "correct_choice_log_probability": -0.1,
                    "correct_vs_best_incorrect_margin": 2.0,
                },
                "intervention_answer_choice_scores": {
                    "correct_choice_log_probability": -3.0,
                    "correct_vs_best_incorrect_margin": -1.0,
                },
            }
        )
    )

    rows = flatten_completed_artifacts(output_root, ["fusion_block_top20_after_layer_8"])

    assert rows[0]["answer_score_source"] == "intervention_answer_choice_scores"
    assert rows[0]["correct_choice_log_probability"] == -3.0
    assert rows[0]["correct_vs_best_incorrect_margin"] == -1.0


def test_reversed_distribution_remaps_presented_scores_to_original_bins():
    distribution = [0.1, 0.2, 0.7]
    mapping = [
        {"presented_analysis_bin": 0, "original_analysis_bin": 2},
        {"presented_analysis_bin": 1, "original_analysis_bin": 1},
        {"presented_analysis_bin": 2, "original_analysis_bin": 0},
    ]
    assert reversed_distribution_to_original_bins(distribution, mapping) == [0.7, 0.2, 0.1]
