import json
from pathlib import Path

import pytest

from src.experiment1.v2_manifest import (
    Experiment1V2Config,
    build_experiment1_v2_manifests,
    choose_primary_per_video,
    duration_tertiles,
    parse_ffprobe_stream,
    PrimaryCandidate,
)
from src.dataset import parse_vqa_example


QUESTION_TYPES = {
    "fine_grained": "fine_grained_action_localization",
    "gaze": "gaze_interaction_anticipation",
    "ingredient": "ingredient_ingredient_retrieval",
    "object_motion": "object_motion_object_movement_counting",
}


def _time(seconds):
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"00:{minutes:02d}:{secs:06.3f}"


def _write_dataset(root: Path, mp4_dir: Path):
    records_by_type = {question_type: {} for question_type in QUESTION_TYPES.values()}
    durations = [10.0, 100.0, 300.0]
    idx = 0
    for category, question_type in QUESTION_TYPES.items():
        for duration in durations:
            for rep in range(2):
                idx += 1
                video_id = f"P{idx % 5:02d}-v2-{idx:03d}"
                participant = video_id.split("-")[0]
                path = mp4_dir / participant / f"{video_id}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not a real mp4 but ffprobe is injected")
                question_id = f"{question_type}_{idx}"
                records_by_type[question_type][question_id] = {
                    "inputs": {
                        "video 1": {
                            "id": video_id,
                            "start_time": _time(0.0),
                            "end_time": _time(duration),
                        }
                    },
                    "question": f"What happens in {category} example {idx}?",
                    "choices": ["A", "B", "C", "D", "E"],
                    "correct_idx": idx % 5,
                }

        extra_video_id = f"P99-extra-{category}"
        extra_path = mp4_dir / "P99" / f"{extra_video_id}.mp4"
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        extra_path.write_bytes(b"extra")
        first_qid = next(iter(records_by_type[question_type]))
        records_by_type[question_type][f"{question_type}_extra_{category}"] = {
            "inputs": {"video 1": {"id": records_by_type[question_type][first_qid]["inputs"]["video 1"]["id"], "start_time": "00:00:00.000", "end_time": "00:00:10.000"}},
            "question": f"Additional question for {category}?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": 0,
        }

    root.mkdir(parents=True, exist_ok=True)
    for question_type, records in records_by_type.items():
        (root / f"{question_type}.json").write_text(json.dumps(records))


def _fake_ffprobe(command):
    path = Path(command[-1])
    duration = 600.0
    return json.dumps(
        {
            "streams": [
                {
                    "width": 320,
                    "height": 240,
                    "avg_frame_rate": "30/1",
                    "nb_frames": str(int(duration * 30)),
                    "duration": str(duration),
                }
            ],
            "format": {"duration": str(duration)},
        }
    )


def test_parse_ffprobe_stream_records_complete_mp4_metadata(tmp_path):
    path = tmp_path / "P01-test.mp4"
    path.write_bytes(b"abc")
    record = parse_ffprobe_stream("P01-test", path, _fake_ffprobe(["ffprobe", str(path)]))
    assert record.video_id == "P01-test"
    assert record.participant_id == "P01"
    assert record.fps == 30.0
    assert record.num_frames == 18000
    assert record.complete is True


def test_duration_tertiles_are_deterministic():
    thresholds = duration_tertiles([10, 20, 30, 40, 50, 60])
    assert thresholds == pytest.approx({"short_medium": 26.6666667, "medium_long": 43.3333333})


def test_build_experiment1_v2_manifests_are_source_disjoint_and_deranged(tmp_path):
    questions_dir = tmp_path / "questions"
    mp4_dir = tmp_path / "mp4"
    _write_dataset(questions_dir, mp4_dir)
    outputs_a = build_experiment1_v2_manifests(
        questions_dir,
        mp4_dir,
        Experiment1V2Config(seed=123, dev_fraction=0.5),
        ffprobe_runner=_fake_ffprobe,
    )
    outputs_b = build_experiment1_v2_manifests(
        questions_dir,
        mp4_dir,
        Experiment1V2Config(seed=123, dev_fraction=0.5),
        ffprobe_runner=_fake_ffprobe,
    )
    assert outputs_a["primary_manifest"] == outputs_b["primary_manifest"]
    primary = outputs_a["primary_manifest"]
    assert len(primary) == len({record["source_video_id"] for record in primary})
    assert len(primary) == len({record["question_id"] for record in primary})
    by_video = {}
    for record in primary:
        previous = by_video.setdefault(record["source_video_id"], record["split"])
        assert previous == record["split"]
    mismatches = outputs_a["mismatched_queries"]["mismatches"]
    for record in primary:
        mismatch = mismatches[record["question_id"]]
        assert mismatch["mismatched_source_video_id"] != record["source_video_id"]
        assert mismatch["category"] == record["category"]
        assert mismatch["duration_group"] == record["duration_group"]
    assert outputs_a["additional_questions"]
    assert outputs_a["split_summary"]["primary_manifest_count"] == len(primary)
    assert outputs_a["split_summary"]["duration_tertile_basis"] == "unique eligible source MP4 durations from ffprobe"
    assert "realtime_sampling_policy" in outputs_a["split_summary"]
    assert outputs_a["split_summary"]["realtime_sampling_policy"]["frames_per_bin"] == 2
    assert all(record["duration_group_basis"] == "source_mp4_duration_seconds" for record in primary)
    assert all(record["source_video_duration_seconds"] == 600.0 for record in primary)


def test_duration_groups_use_source_mp4_duration_not_short_question_duration(tmp_path):
    questions_dir = tmp_path / "questions"
    mp4_dir = tmp_path / "mp4"
    question_type = "gaze_interaction_anticipation"
    records = {}
    source_durations = {
        "P01-short-a": 10.0,
        "P02-short-b": 20.0,
        "P03-medium-a": 100.0,
        "P04-medium-b": 120.0,
        "P05-long-a": 1000.0,
        "P06-long-b": 1200.0,
    }
    for idx, (video_id, _source_duration) in enumerate(source_durations.items()):
        participant = video_id.split("-")[0]
        path = mp4_dir / participant / f"{video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        records[f"{question_type}_{idx}"] = {
            "inputs": {
                "video 1": {
                    "id": video_id,
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:05.000",
                }
            },
            "question": f"Question {idx}?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": 0,
        }
    questions_dir.mkdir(parents=True)
    (questions_dir / f"{question_type}.json").write_text(json.dumps(records))

    def runner(command):
        path = Path(command[-1])
        duration = source_durations[path.stem]
        return json.dumps(
            {
                "streams": [
                    {
                        "width": 320,
                        "height": 240,
                        "avg_frame_rate": "30/1",
                        "nb_frames": str(int(duration * 30)),
                        "duration": str(duration),
                    }
                ],
                "format": {"duration": str(duration)},
            }
        )

    outputs = build_experiment1_v2_manifests(
        questions_dir,
        mp4_dir,
        Experiment1V2Config(seed=123, dev_fraction=0.34, min_test_per_category=0),
        ffprobe_runner=runner,
    )

    by_video = {record["source_video_id"]: record for record in outputs["primary_manifest"]}
    assert by_video["P01-short-a"]["duration_group"] == "short"
    assert by_video["P03-medium-a"]["duration_group"] == "medium"
    assert by_video["P05-long-a"]["duration_group"] == "long"
    assert {record["analyzed_duration_seconds"] for record in outputs["primary_manifest"]} == {5.0}


def _candidate(video_id: str, question_id: str, question_type: str, category: str, duration_group: str):
    example = parse_vqa_example(
        question_id,
        {
            "inputs": {"video 1": {"id": video_id, "start_time": "00:00:00.000", "end_time": "00:00:10.000"}},
            "question": f"{question_id}?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": 0,
        },
        annotation_file=Path(f"{question_type}.json"),
    )
    return PrimaryCandidate(
        example=example,
        category=category,
        source_video_id=video_id,
        participant_id=video_id.split("-")[0],
        source_video_duration_seconds=10.0,
        analyzed_start_seconds=0.0,
        analyzed_end_seconds=10.0,
        analyzed_duration_seconds=10.0,
        is_unbounded_full_video=False,
        duration_group=duration_group,
    )


def test_primary_selection_considers_all_candidates_for_balancing_not_alphabetical_first():
    candidates = [
        _candidate("P01-v1", "q_a_fine", "fine_grained_action_localization", "fine_grained", "short"),
        _candidate("P01-v1", "q_z_gaze", "gaze_interaction_anticipation", "gaze", "short"),
        _candidate("P02-v2", "q_fine2", "fine_grained_action_localization", "fine_grained", "short"),
    ]

    selected, _additional = choose_primary_per_video(candidates, seed=1)

    by_video = {item.source_video_id: item for item in selected}
    assert by_video["P01-v1"].category == "gaze"
    assert by_video["P02-v2"].category == "fine_grained"
