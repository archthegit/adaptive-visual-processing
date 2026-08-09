from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any

from src.dataset import VQAExample
from src.pilot import broader_category, manifest_record, spatial_priority


EXPERIMENT1_CATEGORIES = ("gaze", "ingredient", "fine_grained", "object_motion")


def infer_experiment1_category(question_type: str) -> str | None:
    category = broader_category(question_type)
    return category if category in EXPERIMENT1_CATEGORIES else None


def available_experiment1_types(examples: list[VQAExample]) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        category = infer_experiment1_category(example.question_type)
        if category is not None:
            mapping[category].add(example.question_type)
    return {category: sorted(types) for category, types in sorted(mapping.items())}


def select_experiment1_examples(
    examples: list[VQAExample], examples_per_category: int = 12, seed: int = 20260808
) -> list[VQAExample]:
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[VQAExample]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        category = infer_experiment1_category(example.question_type)
        if category is not None:
            grouped[category][example.question_type].append(example)

    selected: list[VQAExample] = []
    seen: set[str] = set()
    for category in EXPERIMENT1_CATEGORIES:
        type_buckets = grouped.get(category, {})
        for bucket in type_buckets.values():
            bucket.sort(key=lambda example: (rng.random(), example.question_id))
        type_order = sorted(type_buckets)
        while len([example for example in selected if infer_experiment1_category(example.question_type) == category]) < examples_per_category:
            added = False
            for question_type in type_order:
                bucket = type_buckets[question_type]
                if not bucket:
                    continue
                example = bucket.pop(0)
                if example.question_id in seen:
                    continue
                selected.append(example)
                seen.add(example.question_id)
                added = True
                if len([item for item in selected if infer_experiment1_category(item.question_type) == category]) >= examples_per_category:
                    break
            if not added:
                break

    return sorted(selected, key=lambda example: (infer_experiment1_category(example.question_type) or "", example.question_type, example.question_id))


def _balanced_category_quotas(target_size: int, categories: list[str]) -> dict[str, int]:
    if target_size <= 0 or not categories:
        return {}
    base = target_size // len(categories)
    remainder = target_size % len(categories)
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }


def select_video_aware_experiment1_examples(
    examples: list[VQAExample],
    target_size: int = 12,
    max_new_videos: int = 8,
    seed: int = 20260808,
    preferred_video_ids: set[str] | None = None,
) -> list[VQAExample]:
    """Select a balanced Experiment 1 subset while limiting new MP4 downloads.

    ``preferred_video_ids`` are treated as already available. They do not count
    against ``max_new_videos`` but are still reported in the selected manifest.
    """
    preferred = set(preferred_video_ids or set())
    rng = random.Random(seed)
    eligible = [
        example
        for example in examples
        if infer_experiment1_category(example.question_type) in EXPERIMENT1_CATEGORIES
    ]
    categories = [category for category in EXPERIMENT1_CATEGORIES if any(infer_experiment1_category(ex.question_type) == category for ex in eligible)]
    quotas = _balanced_category_quotas(target_size, categories)
    selected: list[VQAExample] = []
    selected_ids: set[str] = set()
    selected_videos: set[str] = set()
    new_videos: set[str] = set()

    def can_add(example: VQAExample) -> bool:
        required = set(example.video_ids)
        return len(new_videos | (required - preferred)) <= max_new_videos

    def add_example(example: VQAExample) -> None:
        selected.append(example)
        selected_ids.add(example.question_id)
        selected_videos.update(example.video_ids)
        new_videos.update(set(example.video_ids) - preferred)

    def candidate_key(example: VQAExample) -> tuple[int, int, int, int, float, str]:
        required = set(example.video_ids)
        added_new = len((required - preferred) - new_videos)
        shared_available = len(required & (preferred | selected_videos))
        return (
            added_new,
            len(required),
            -shared_available,
            -spatial_priority(example),
            rng.random(),
            example.question_id,
        )

    by_category: dict[str, list[VQAExample]] = defaultdict(list)
    for example in eligible:
        category = infer_experiment1_category(example.question_type)
        if category is not None:
            by_category[category].append(example)
    for bucket in by_category.values():
        bucket.sort(key=candidate_key)

    made_progress = True
    while made_progress and len(selected) < target_size:
        made_progress = False
        for category in categories:
            if len([ex for ex in selected if infer_experiment1_category(ex.question_type) == category]) >= quotas.get(category, 0):
                continue
            for example in by_category[category]:
                if example.question_id in selected_ids or not can_add(example):
                    continue
                add_example(example)
                made_progress = True
                break

    if len(selected) < target_size:
        remaining = [example for example in eligible if example.question_id not in selected_ids]
        remaining.sort(key=candidate_key)
        for example in remaining:
            if len(selected) >= target_size:
                break
            if can_add(example):
                add_example(example)

    return sorted(
        selected,
        key=lambda example: (
            infer_experiment1_category(example.question_type) or "",
            example.question_type,
            example.question_id,
        ),
    )


def experiment1_manifest_record(example: VQAExample) -> dict[str, Any]:
    record = manifest_record(example)
    category = infer_experiment1_category(example.question_type)
    if category is None:
        raise ValueError(f"Example {example.question_id} is not in an Experiment 1 category.")
    record["category"] = category
    record["experiment"] = "experiment1_query_relevance_fusion"
    return record


def summarize_experiment1_manifest(examples: list[VQAExample]) -> dict[str, Any]:
    video_ids = sorted({video_id for example in examples for video_id in example.video_ids})
    return {
        "num_examples": len(examples),
        "by_category": dict(sorted(Counter(infer_experiment1_category(example.question_type) for example in examples).items())),
        "by_question_type": dict(sorted(Counter(example.question_type for example in examples).items())),
        "num_unique_videos": len(video_ids),
        "unique_videos": video_ids,
    }
