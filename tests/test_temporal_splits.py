from src.dataset import parse_vqa_example
import pytest

from src.experiment1.temporal_splits import (
    TemporalSplitConfig,
    create_temporal_splits,
    single_video_temporal_examples,
    temporal_manifest_record,
)


QUESTION_TYPES = {
    "gaze": "gaze_interaction_anticipation",
    "ingredient": "ingredient_ingredient_retrieval",
    "fine_grained": "fine_grained_action_localization",
    "object_motion": "object_motion_object_movement_counting",
}


def _example(category, idx, video_suffix=None, image=False, extra_video=False, video_id=None):
    qtype = QUESTION_TYPES[category]
    video_id = video_id or f"P{idx % 4:02d}-video-{video_suffix or idx}"
    inputs = {
        "video 1": {
            "id": video_id,
            "start_time": f"00:00:{idx % 50:02d}.000",
            "end_time": f"00:01:{idx % 50:02d}.000",
        }
    }
    if image:
        inputs = {"image 1": {"id": video_id, "time": "00:00:01.000"}}
    if extra_video:
        inputs["video 2"] = {"id": f"P{idx % 4:02d}-extra-{idx}"}
    return parse_vqa_example(
        f"{qtype}_{idx}",
        {
            "inputs": inputs,
            "question": "What happened?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": idx % 5,
        },
    )


def test_single_video_temporal_filter_rejects_images_and_multi_video():
    valid = _example("gaze", 1)
    image = _example("gaze", 2, image=True)
    multi = _example("gaze", 3, extra_video=True)
    assert single_video_temporal_examples([valid, image, multi]) == [valid]


def test_temporal_splits_are_deterministic_and_disjoint():
    examples = []
    for category_index, category in enumerate(QUESTION_TYPES):
        examples.extend(
            _example(category, idx + category_index * 100, video_suffix=f"{category}-{idx}")
            for idx in range(12)
        )
    config = TemporalSplitConfig(
        engineering_per_category=1,
        pilot_per_category=2,
        confirmatory_per_category=2,
        seed=7,
        max_per_source_video=2,
    )
    splits_a, summary_a = create_temporal_splits(examples, config)
    splits_b, summary_b = create_temporal_splits(examples, config)
    assert {
        name: [item.question_id for item in items]
        for name, items in splits_a.items()
    } == {
        name: [item.question_id for item in items]
        for name, items in splits_b.items()
    }
    split_ids = [set(item.question_id for item in items) for items in splits_a.values()]
    assert not (split_ids[0] & split_ids[1])
    assert not (split_ids[0] & split_ids[2])
    assert not (split_ids[1] & split_ids[2])
    pilot_videos = {item.inputs[0].video_id for item in splits_a["pilot"]}
    confirmatory_videos = {item.inputs[0].video_id for item in splits_a["confirmatory"]}
    assert not (pilot_videos & confirmatory_videos)
    assert summary_a == summary_b


def test_temporal_splits_fail_when_confirmatory_video_isolation_is_impossible():
    examples = []
    for category_index, category in enumerate(QUESTION_TYPES):
        examples.extend(
            _example(category, idx + category_index * 100, video_id=f"P{category_index:02d}-shared-{idx % 2}")
            for idx in range(8)
        )
    with pytest.raises(ValueError, match="excluding source videos"):
        create_temporal_splits(
            examples,
            TemporalSplitConfig(
                engineering_per_category=1,
                pilot_per_category=2,
                confirmatory_per_category=2,
                seed=7,
                max_per_source_video=10,
            ),
        )


def test_temporal_manifest_record_exposes_temporal_source_metadata():
    record = temporal_manifest_record(_example("ingredient", 42))
    assert record["source_video_id"].startswith("P")
    assert record["bounded_duration_seconds"] == 60.0
    assert record["is_unbounded_full_video"] is False
    assert record["experiment"] == "experiment1_temporal_query_relevance"
