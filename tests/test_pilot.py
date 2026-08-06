from src.dataset import parse_vqa_example
from src.pilot import broader_category, manifest_record, select_pilot_examples


def _example(question_type: str, idx: int):
    question_id = f"{question_type}_{idx}"
    return parse_vqa_example(
        question_id,
        {
            "inputs": {
                "video 1": {
                    "id": "P00-synthetic",
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:01.000",
                }
            },
            "question": f"Where is the object in {question_type}?",
            "choices": ["A", "B", "C", "D", "E"],
            "correct_idx": idx % 5,
        },
    )


def test_broader_category():
    assert broader_category("object_motion_object_movement_counting") == "object_motion"
    assert broader_category("fine_grained_action_recognition") == "fine_grained"
    assert broader_category("gaze_gaze_estimation") == "gaze"


def test_select_pilot_examples_is_deterministic():
    examples = []
    for question_type in [
        "gaze_gaze_estimation",
        "fine_grained_action_recognition",
        "ingredient_ingredient_recognition",
        "object_motion_object_movement_counting",
        "3d_perception_object_location",
    ]:
        examples.extend(_example(question_type, idx) for idx in range(5))

    first = select_pilot_examples(examples, pilot_size=10, seed=7)
    second = select_pilot_examples(examples, pilot_size=10, seed=7)
    assert [example.question_id for example in first] == [example.question_id for example in second]


def test_manifest_record_contains_required_fields():
    record = manifest_record(_example("gaze_gaze_estimation", 1))
    assert record["question_id"] == "gaze_gaze_estimation_1"
    assert record["category"] == "gaze"
    assert record["correct_answer"] == "B"
    assert record["video_clip"][0]["video_id"] == "P00-synthetic"
