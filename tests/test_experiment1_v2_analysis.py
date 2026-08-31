import json

from src.experiment1.v2_analysis import (
    expected_conditions,
    expected_run_matrix,
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
    assert len(matrix) == len(records) * len(expected_conditions(include_fusion_depth=False))
    assert {row["source_video_id"] for row in matrix} == {"v1", "v2"}
    assert {row["condition"] for row in matrix} == set(expected_conditions(include_fusion_depth=False))


def test_validate_completeness_counts_complete_missing_and_failed(tmp_path):
    output_root = tmp_path / "runs"
    condition_dir = output_root / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "q1.json").write_text(json.dumps({"question_id": "q1", "correct": True}))
    (condition_dir / "q2.json").write_text(json.dumps({"question_id": "q2", "status": "failed", "error": "boom"}))
    report = validate_completeness(_primary_records(), output_root, include_fusion_depth=False)
    assert report["complete"] == 1
    assert report["failed"] == 1
    assert report["missing"] == len(_primary_records()) * len(expected_conditions(False)) - 2
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
    assert outputs["completeness"]["expected"] == len(_primary_records()) * len(expected_conditions(False))
    assert (final_dir / "figures").is_dir()
    assert (final_dir / "tables" / "expected_run_matrix.jsonl").exists()
    assert (final_dir / "statistical_results.json").exists()
    assert (final_dir / "completeness_report.json").exists()
    assert (final_dir / "experiment_report.md").exists()
