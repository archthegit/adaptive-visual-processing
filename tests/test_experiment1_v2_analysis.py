import json

from src.experiment1.v2_analysis import (
    average_decoder_heatmap,
    condition_table,
    encoder_layer_statistics,
    encoder_representation_statistics,
    flatten_completed_artifacts,
    expected_conditions,
    expected_run_matrix,
    reversed_distribution_to_original_bins,
    reversed_video_layer_statistics,
    temporal_control_layer_statistics,
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


def _write_temporal_artifact(path, distribution, *, condition="baseline", category="gaze", duration_group="short"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "question_id": path.stem,
                "category": category,
                "duration_group": duration_group,
                "video_clip": [{"video_id": f"v-{path.stem}", "participant_id": "p1"}],
                "metadata": {"condition": condition},
                "temporal_relevance": {
                    "normalized_temporal_bin_scores": [distribution, list(reversed(distribution))],
                },
            }
        )
    )


def test_temporal_control_layer_statistics_outputs_metric_deltas_and_inference_fields(tmp_path):
    output_root = tmp_path / "runs"
    _write_temporal_artifact(output_root / "baseline" / "q1.json", [0.7, 0.2, 0.1])
    _write_temporal_artifact(output_root / "repeated_frame" / "q1.json", [0.34, 0.33, 0.33], condition="repeated_frame")
    _write_temporal_artifact(output_root / "mismatched_query" / "q1.json", [0.1, 0.2, 0.7], condition="mismatched_query")

    stats = temporal_control_layer_statistics(output_root, bootstrap_replicates=8, seed=1)

    entropy_delta = stats["repeated_frame"]["metric_deltas"]["layer_0:normalized_entropy"]
    assert entropy_delta["num_pairs"] == 1
    assert entropy_delta["metric"] == "normalized_entropy"
    assert "delta_bootstrap_ci" in entropy_delta
    assert "paired_permutation_p" in entropy_delta
    assert "benjamini_hochberg_q" in entropy_delta
    assert "top1_mass" in {item["metric"] for item in stats["repeated_frame"]["metric_deltas"].values()}
    mismatch_jsd = stats["mismatched_query"]["temporal_jsd"]["0"]
    assert mismatch_jsd["metric"] == "temporal_jsd"
    assert mismatch_jsd["by_duration_group"]["short"]["num_pairs"] == 1


def test_reversed_video_layer_statistics_reports_content_vs_position_deltas(tmp_path):
    output_root = tmp_path / "runs"
    _write_temporal_artifact(output_root / "baseline" / "q1.json", [0.7, 0.2, 0.1])
    reversed_path = output_root / "reversed_video" / "q1.json"
    _write_temporal_artifact(reversed_path, [0.1, 0.2, 0.7], condition="reversed_video")
    data = json.loads(reversed_path.read_text())
    data["presented_to_original_frame_bin_mappings"] = [
        [
            {"presented_analysis_bin": 0, "original_analysis_bin": 2},
            {"presented_analysis_bin": 1, "original_analysis_bin": 1},
            {"presented_analysis_bin": 2, "original_analysis_bin": 0},
        ]
    ]
    reversed_path.write_text(json.dumps(data))

    stats = reversed_video_layer_statistics(output_root, bootstrap_replicates=8, seed=2)

    layer = stats["paired_layer_deltas"]["0"]
    assert layer["metric"] == "content_following_minus_position_following_spearman"
    assert layer["num_pairs"] == 1
    assert layer["mean_delta"] > 0
    assert "benjamini_hochberg_q" in layer


def test_encoder_layer_and_representation_statistics_are_reported(tmp_path):
    output_root = tmp_path / "runs"
    path = output_root / "baseline" / "q1.json"
    _write_temporal_artifact(path, [0.7, 0.2, 0.1])
    data = json.loads(path.read_text())
    data["encoder_attention_temporal"] = {
        "available": True,
        "normalized_incoming_temporal_attention": [
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
            [[0.2, 0.3, 0.5], [0.1, 0.3, 0.6]],
        ],
    }
    data["encoder_temporal"] = {
        "available": True,
        "stages": {
            "vision_final": {
                "analysis_bin_remap_active": True,
                "num_temporal_bins": 3,
                "temporal_representations": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            }
        },
    }
    path.write_text(json.dumps(data))

    layer_stats = encoder_layer_statistics(output_root, bootstrap_replicates=8, seed=3)
    rep_stats = encoder_representation_statistics(output_root)

    assert "layer_0:entropy" in layer_stats
    assert "layer_1:consecutive_layer_jsd" in layer_stats
    assert "layer_0:head_agreement" in layer_stats
    assert layer_stats["layer_0:entropy"]["by_category"]["gaze"]["num_records"] == 1
    assert rep_stats["by_stage"]["vision_final"]["num_records"] == 1
    assert rep_stats["records"][0]["analysis_bin_remap_active"] is True
