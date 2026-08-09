from src.dataset import parse_vqa_example
from src.experiment1.manifest import (
    available_experiment1_types,
    experiment1_manifest_record,
    select_experiment1_examples,
)


def _example(question_type: str, idx: int):
    return parse_vqa_example(
        f"{question_type}_{idx}",
        {
            "inputs": {"video 1": {"id": "P00-synthetic"}},
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
