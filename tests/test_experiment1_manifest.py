from src.dataset import parse_vqa_example
from src.experiment1.manifest import (
    available_experiment1_types,
    count_video_inputs,
    experiment1_manifest_record,
    select_experiment1_examples,
    select_video_aware_experiment1_examples,
)


def _example(question_type: str, idx: int, video_id: str = "P00-synthetic"):
    return parse_vqa_example(
        f"{question_type}_{idx}",
        {
            "inputs": {"video 1": {"id": video_id}},
            "question": "Where is the relevant object?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": idx % 5,
        },
    )


def _multi_input_example(question_type: str, idx: int, inputs):
    return parse_vqa_example(
        f"{question_type}_{idx}",
        {
            "inputs": inputs,
            "question": "Where is the relevant object?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": idx % 5,
        },
    )


def test_experiment1_manifest_balances_four_categories():
    examples = []
    for question_type in [
        "gaze_gaze_estimation",
        "ingredient_ingredient_recognition",
        "fine_grained_action_recognition",
        "object_motion_object_movement_counting",
        "recipe_step_recognition",
    ]:
        examples.extend(_example(question_type, idx) for idx in range(4))

    selected = select_experiment1_examples(examples, examples_per_category=2, seed=3)
    assert len(selected) == 8
    assert available_experiment1_types(examples) == {
        "fine_grained": ["fine_grained_action_recognition"],
        "gaze": ["gaze_gaze_estimation"],
        "ingredient": ["ingredient_ingredient_recognition"],
        "object_motion": ["object_motion_object_movement_counting"],
    }
    assert {experiment1_manifest_record(example)["category"] for example in selected} == {
        "gaze",
        "ingredient",
        "fine_grained",
        "object_motion",
    }


def test_video_aware_experiment1_manifest_prefers_available_videos():
    examples = [
        _example("gaze_gaze_estimation", 1, "P01-have"),
        _example("ingredient_ingredient_recognition", 2, "P01-have"),
        _example("fine_grained_action_recognition", 3, "P02-new"),
        _example("object_motion_object_movement_counting", 4, "P03-new"),
        _example("gaze_gaze_estimation", 5, "P04-too-many"),
    ]
    selected = select_video_aware_experiment1_examples(
        examples,
        target_size=4,
        max_new_videos=2,
        seed=11,
        preferred_video_ids={"P01-have"},
    )
    assert len(selected) == 4
    used_videos = {video_id for example in selected for video_id in example.video_ids}
    assert "P01-have" in used_videos
    assert len(used_videos - {"P01-have"}) <= 2


def test_video_input_limit_counts_images_separately():
    video_plus_image = _multi_input_example(
        "object_motion_object_movement_counting",
        10,
        {
            "video 1": {"id": "P01-video"},
            "image 1": {"id": "P01-video", "time": "00:00:01.000"},
        },
    )
    two_videos = _multi_input_example(
        "object_motion_object_movement_counting",
        11,
        {
            "video 1": {"id": "P01-video-a"},
            "video 2": {"id": "P02-video-b"},
        },
    )
    selected = select_video_aware_experiment1_examples(
        [video_plus_image, two_videos],
        target_size=2,
        max_new_videos=10,
        seed=5,
        max_video_inputs=1,
    )
    assert selected == [video_plus_image]
    assert count_video_inputs(video_plus_image) == 1
    assert count_video_inputs(two_videos) == 2
