from pathlib import Path
from types import SimpleNamespace

from scripts.prepare_experiment1_v3_sampling import audit_sampling, rewrite_records_for_v3, summarize_records
from src.experiment1.v2_manifest import duration_tertiles
from src.experiment1.v2_sampling import primary_policy_from_development_durations


def _record(question_id, category, duration, group="short"):
    return {
        "question_id": question_id,
        "question_type": f"{category}_type",
        "category": category,
        "question": f"{question_id}?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
        "correct_answer": "A",
        "source_video_id": question_id,
        "participant_id": "P01",
        "input_key": "video 1",
        "start_seconds": 0.0,
        "end_seconds": duration,
        "analyzed_start_seconds": 0.0,
        "analyzed_end_seconds": duration,
        "analyzed_duration_seconds": duration,
        "source_video_duration_seconds": 1000.0,
        "duration_group": group,
        "duration_group_basis": "source_mp4_duration_seconds",
        "is_unbounded_full_video": False,
        "video_clip": [],
        "raw_metadata": {},
        "split": "test",
    }


def test_v3_rewrite_preserves_question_video_ids_and_uses_analyzed_duration_groups():
    records = [
        _record("q_short", "gaze", 4.0),
        _record("q_medium", "gaze", 40.0),
        _record("q_long", "gaze", 400.0),
    ]
    thresholds = duration_tertiles(record["analyzed_duration_seconds"] for record in records)
    rewritten = rewrite_records_for_v3(records, thresholds)

    assert [record["question_id"] for record in rewritten] == ["q_short", "q_medium", "q_long"]
    assert [record["source_video_id"] for record in rewritten] == ["q_short", "q_medium", "q_long"]
    assert [record["duration_group"] for record in rewritten] == ["short", "medium", "long"]
    assert all(record["duration_group_basis"] == "analyzed_duration_seconds" for record in rewritten)


def test_v3_audit_reports_full_coverage_and_no_category_collapses_to_one_bin(tmp_path, monkeypatch):
    records = [
        _record("gaze_short", "gaze", 4.0, "short"),
        _record("fine_medium", "fine_grained", 64.0, "medium"),
        _record("ingredient_long", "ingredient", 640.0, "long"),
    ]
    for record in records:
        path = tmp_path / "P01" / f"{record['source_video_id']}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    inventory = {
        record["source_video_id"]: SimpleNamespace(fps=10.0, num_frames=10000)
        for record in records
    }
    monkeypatch.setattr("scripts.prepare_experiment1_v3_sampling.decord_length", lambda path: 10000)
    policy = primary_policy_from_development_durations([64.0])

    audit = audit_sampling(records, inventory, tmp_path, policy)

    assert audit["num_audited"] == 3
    assert audit["failures"] == []
    assert audit["full_coverage_failures"] == []
    assert audit["out_of_bounds_failures"] == []
    assert audit["collapsed_one_bin_categories"] == []
    assert audit["min_median_max_bins"]["min"] >= 8
    assert audit["min_median_max_bins"]["max"] <= 64
    assert audit["min_median_max_frames"]["max"] <= 128
    assert audit["shortest_gaze_sampling_summary"]["question_id"] == "gaze_short"
    assert summarize_records(records)["num_examples"] == 3
