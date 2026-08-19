from src.dataset import parse_vqa_example
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


def _example(category, idx, video_suffix=None, image=False, extra_video=False):
    qtype = QUESTION_TYPES[category]
    video_id = f"P{idx % 4:02d}-video-{video_suffix or idx}"
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
    for category in QUESTION_TYPES:
        examples.extend(_example(category, idx + category_index * 100) for idx in range(8) for category_index in [list(QUESTION_TYPES).index(category)])
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
    assert summary_a == summary_b


def test_temporal_manifest_record_exposes_temporal_source_metadata():
    record = temporal_manifest_record(_example("ingredient", 42))
    assert record["source_video_id"].startswith("P")
    assert record["bounded_duration_seconds"] == 60.0
    assert record["is_unbounded_full_video"] is False
    assert record["experiment"] == "experiment1_temporal_query_relevance"
