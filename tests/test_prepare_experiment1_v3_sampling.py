from pathlib import Path
import statistics
from types import SimpleNamespace

from scripts.prepare_experiment1_v3_sampling import (
    assert_v3_protocol_invariants,
    audit_sampling,
    recompute_v3_mismatched_queries,
    recompute_v3_split,
    rewrite_records_for_v3,
    summarize_records,
)
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


def test_v3_split_preserves_exact_77_ids_and_recomputes_stratified_dev_test():
    records = []
    categories = ["fine_grained", "gaze", "ingredient", "object_motion"]
    groups = ["short", "medium", "long"]
    for idx in range(77):
        category = categories[idx % len(categories)]
        group = groups[idx % len(groups)]
        duration = {"short": 5.0, "medium": 50.0, "long": 500.0}[group]
        records.append(_record(f"q_{idx:03d}", category, duration, group))
    frozen_question_ids = {record["question_id"] for record in records}
    frozen_video_ids = {record["source_video_id"] for record in records}

    split_records, split_strata = recompute_v3_split(records, seed=20260830, dev_fraction=0.2)

    assert {record["question_id"] for record in split_records} == frozen_question_ids
    assert {record["source_video_id"] for record in split_records} == frozen_video_ids
    assert sum(1 for record in split_records if record["split"] == "dev") > 0
    assert sum(1 for record in split_records if record["split"] == "test") > 0
    assert all(
        not item["dev_test_coverage_feasible"] or item["has_dev_and_test_when_feasible"]
        for item in split_strata.values()
    )
    split_by_video = {}
    for record in split_records:
        previous = split_by_video.setdefault(record["source_video_id"], record["split"])
        assert previous == record["split"]


def test_v3_split_same_seed_identical_different_seed_changes_feasible_stratum():
    records = [_record(f"q_{idx:02d}", "gaze", 20.0, "medium") for idx in range(20)]

    split_a, strata_a = recompute_v3_split(records, seed=20260830, dev_fraction=0.2)
    split_b, strata_b = recompute_v3_split(records, seed=20260830, dev_fraction=0.2)
    split_c, strata_c = recompute_v3_split(records, seed=20260831, dev_fraction=0.2)

    assignment_a = {record["question_id"]: record["split"] for record in split_a}
    assignment_b = {record["question_id"]: record["split"] for record in split_b}
    assignment_c = {record["question_id"]: record["split"] for record in split_c}
    assert assignment_a == assignment_b
    assert assignment_a != assignment_c
    assert strata_a == strata_b
    assert strata_a == strata_c
    assert all(item["has_dev_and_test_when_feasible"] for item in strata_a.values())


def test_v3_split_counts_match_summary():
    records = []
    for idx in range(24):
        category = "gaze" if idx < 12 else "ingredient"
        group = "short" if idx % 2 == 0 else "long"
        records.append(_record(f"q_{idx:02d}", category, 20.0, group))

    split_records, _split_strata = recompute_v3_split(records, seed=20260830, dev_fraction=0.25)
    summary = summarize_records(split_records)

    assert summary["num_examples"] == len(split_records)
    assert sum(summary["by_split"].values()) == len(split_records)
    assert sum(summary["by_category"].values()) == len(split_records)
    assert sum(summary["by_duration_group"].values()) == len(split_records)
    assert sum(summary["by_category_and_duration"].values()) == len(split_records)


def test_v3_mismatches_are_recomputed_for_new_category_duration_groups():
    records = []
    for idx, duration in enumerate([5.0, 6.0, 50.0, 55.0]):
        group = "short" if idx < 2 else "medium"
        record = _record(f"q_{idx}", "gaze", duration, group)
        record["question"] = " ".join(["word"] * (idx + 3))
        records.append(record)
    split_records, split_strata = recompute_v3_split(records, seed=20260830, dev_fraction=0.5)

    mismatches = recompute_v3_mismatched_queries(split_records, seed=20260830)
    summary = {"primary_summary": summarize_records(split_records)}
    assert_v3_protocol_invariants(records, split_records, mismatches, split_strata, summary)

    by_question = {record["question_id"]: record for record in split_records}
    donor_ids = [mismatch["mismatched_question_id"] for mismatch in mismatches["mismatches"].values()]
    assert sorted(donor_ids) == sorted(by_question)
    for question_id, mismatch in mismatches["mismatches"].items():
        record = by_question[question_id]
        donor = by_question[mismatch["mismatched_question_id"]]
        assert mismatch["category"] == record["category"] == donor["category"]
        assert mismatch["duration_group"] == record["duration_group"] == donor["duration_group"]
        assert mismatch["mismatched_source_video_id"] != record["source_video_id"]


def test_v3_mismatch_invariants_validate_donor_record_not_only_copied_metadata():
    records = [_record("q0", "gaze", 5.0, "short"), _record("q1", "ingredient", 6.0, "short")]
    split_records, split_strata = recompute_v3_split(records, seed=20260830, dev_fraction=0.5)
    mismatches = {
        "seed": 20260830,
        "mismatches": {
            "q0": {
                "mismatched_question_id": "q1",
                "mismatched_source_video_id": "q1",
                "category": "gaze",
                "duration_group": "short",
                "question": "bad donor",
                "choices": ["A", "B", "C", "D", "E"],
                "correct_idx": 0,
            },
            "q1": {
                "mismatched_question_id": "q0",
                "mismatched_source_video_id": "q0",
                "category": "ingredient",
                "duration_group": "short",
                "question": "bad donor",
                "choices": ["A", "B", "C", "D", "E"],
                "correct_idx": 0,
            },
        },
    }

    import pytest

    with pytest.raises(ValueError, match="donor category"):
        assert_v3_protocol_invariants(records, split_records, mismatches, split_strata)


def test_v3_target_delta_t_uses_corrected_dev_split_after_regrouping():
    records = []
    for idx, duration in enumerate([5.0, 6.0, 50.0, 55.0, 500.0, 550.0]):
        group = "short" if idx < 2 else "medium" if idx < 4 else "long"
        records.append(_record(f"q_{idx}", "ingredient", duration, group))

    split_records, _split_strata = recompute_v3_split(records, seed=20260830, dev_fraction=0.5)
    corrected_dev_durations = [
        record["analyzed_duration_seconds"]
        for record in split_records
        if record["split"] == "dev"
    ]
    policy = primary_policy_from_development_durations(corrected_dev_durations)

    assert corrected_dev_durations
    assert policy.delta_t_seconds == statistics.median(corrected_dev_durations) / 16.0
