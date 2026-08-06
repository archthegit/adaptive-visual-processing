from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from .dataset import VQAExample


CATEGORY_PREFIXES = (
    ("3d_perception_", "3d_perception"),
    ("fine_grained_", "fine_grained"),
    ("gaze_", "gaze"),
    ("ingredient_", "ingredient"),
    ("nutrition_", "nutrition"),
    ("object_motion_", "object_motion"),
    ("recipe_", "recipe"),
)

DEFAULT_CATEGORY_WEIGHTS = {
    "gaze": 10,
    "fine_grained": 12,
    "ingredient": 12,
    "object_motion": 10,
    "3d_perception": 6,
}

PREFERRED_TYPES = {
    "gaze": [
        "gaze_gaze_estimation",
        "gaze_interaction_anticipation",
    ],
    "fine_grained": [
        "fine_grained_action_localization",
        "fine_grained_action_recognition",
        "fine_grained_how_recognition",
        "fine_grained_why_recognition",
    ],
    "ingredient": [
        "ingredient_ingredient_adding_localization",
        "ingredient_exact_ingredient_recognition",
        "ingredient_ingredient_recognition",
        "ingredient_ingredient_retrieval",
        "ingredient_ingredients_order",
        "ingredient_ingredient_weight",
    ],
    "object_motion": [
        "object_motion_stationary_object_localization",
        "object_motion_object_movement_itinerary",
        "object_motion_object_movement_counting",
    ],
    "3d_perception": [
        "3d_perception_object_location",
        "3d_perception_fixture_location",
        "3d_perception_object_contents_retrieval",
        "3d_perception_fixture_interaction_counting",
    ],
}

SPATIAL_KEYWORDS = (
    "gaze",
    "location",
    "localization",
    "localisation",
    "object",
    "movement",
    "itinerary",
    "fixture",
    "contents",
    "retrieval",
    "ingredient",
    "exact",
    "action",
)


def broader_category(question_type: str) -> str:
    for prefix, category in CATEGORY_PREFIXES:
        if question_type.startswith(prefix):
            return category
    return question_type.split("_", 1)[0]


def count_by_type_and_category(examples: list[VQAExample]) -> dict[str, Any]:
    type_counts = Counter(example.question_type for example in examples)
    category_counts = Counter(broader_category(example.question_type) for example in examples)
    return {
        "by_question_type": dict(sorted(type_counts.items())),
        "by_category": dict(sorted(category_counts.items())),
    }


def spatial_priority(example: VQAExample) -> int:
    text = f"{example.question_type} {example.question}".lower()
    score = sum(1 for keyword in SPATIAL_KEYWORDS if keyword in text)
    if any(segment.start_seconds is not None for segment in example.inputs):
        score += 1
    if len(example.inputs) > 1:
        score += 1
    return score


def category_quotas(pilot_size: int, available_categories: set[str]) -> dict[str, int]:
    weights = {
        category: weight
        for category, weight in DEFAULT_CATEGORY_WEIGHTS.items()
        if category in available_categories
    }
    if not weights:
        return {}

    total_weight = sum(weights.values())
    quotas = {
        category: int(pilot_size * weight / total_weight)
        for category, weight in weights.items()
    }
    for category in weights:
        if quotas[category] == 0:
            quotas[category] = 1

    while sum(quotas.values()) < pilot_size:
        category = max(
            weights,
            key=lambda key: (
                pilot_size * weights[key] / total_weight - quotas[key],
                weights[key],
                key,
            ),
        )
        quotas[category] += 1
    while sum(quotas.values()) > pilot_size:
        category = max(quotas, key=lambda key: (quotas[key], -weights[key], key))
        quotas[category] -= 1
    return quotas


def _ordered_types(category: str, observed_types: set[str]) -> list[str]:
    preferred = [qtype for qtype in PREFERRED_TYPES.get(category, []) if qtype in observed_types]
    remaining = sorted(observed_types - set(preferred))
    return preferred + remaining


def select_pilot_examples(
    examples: list[VQAExample], pilot_size: int = 50, seed: int = 20260806
) -> list[VQAExample]:
    rng = random.Random(seed)
    by_category: dict[str, list[VQAExample]] = defaultdict(list)
    for example in examples:
        by_category[broader_category(example.question_type)].append(example)

    quotas = category_quotas(pilot_size, set(by_category))
    selected: list[VQAExample] = []
    selected_ids: set[str] = set()

    for category, quota in quotas.items():
        by_type: dict[str, list[VQAExample]] = defaultdict(list)
        for example in by_category[category]:
            by_type[example.question_type].append(example)

        for bucket in by_type.values():
            bucket.sort(
                key=lambda example: (
                    -spatial_priority(example),
                    rng.random(),
                    example.question_id,
                )
            )

        type_order = _ordered_types(category, set(by_type))
        while len([example for example in selected if broader_category(example.question_type) == category]) < quota:
            added = False
            for question_type in type_order:
                bucket = by_type[question_type]
                if not bucket:
                    continue
                example = bucket.pop(0)
                if example.question_id in selected_ids:
                    continue
                selected.append(example)
                selected_ids.add(example.question_id)
                added = True
                if len([item for item in selected if broader_category(item.question_type) == category]) >= quota:
                    break
            if not added:
                break

    if len(selected) < pilot_size:
        remaining = [example for example in examples if example.question_id not in selected_ids]
        remaining.sort(key=lambda example: (-spatial_priority(example), rng.random(), example.question_id))
        selected.extend(remaining[: pilot_size - len(selected)])

    selected.sort(key=lambda example: (broader_category(example.question_type), example.question_type, example.question_id))
    return selected[:pilot_size]


def manifest_record(example: VQAExample) -> dict[str, Any]:
    return {
        "question_id": example.question_id,
        "question_type": example.question_type,
        "category": broader_category(example.question_type),
        "question": example.question,
        "choices": list(example.choices),
        "correct_idx": example.correct_idx,
        "correct_answer": example.choices[example.correct_idx],
        "video_clip": [
            {
                "input_key": segment.input_key,
                "video_id": segment.video_id,
                "participant_id": segment.participant_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "image_time_seconds": segment.image_time_seconds,
                "raw": segment.raw,
            }
            for segment in example.inputs
        ],
        "raw_metadata": {
            key: value
            for key, value in example.raw.items()
            if key in {"others", "stat", "metadata"}
        },
    }


def pilot_summary(examples: list[VQAExample]) -> dict[str, Any]:
    counts = count_by_type_and_category(examples)
    return {
        "num_examples": len(examples),
        "by_category": counts["by_category"],
        "by_question_type": counts["by_question_type"],
    }
