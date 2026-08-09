from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any

from src.dataset import VQAExample
from src.pilot import broader_category, manifest_record


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


def experiment1_manifest_record(example: VQAExample) -> dict[str, Any]:
    record = manifest_record(example)
    category = infer_experiment1_category(example.question_type)
    if category is None:
        raise ValueError(f"Example {example.question_id} is not in an Experiment 1 category.")
    record["category"] = category
    record["experiment"] = "experiment1_query_relevance_fusion"
    return record


def summarize_experiment1_manifest(examples: list[VQAExample]) -> dict[str, Any]:
    return {
        "num_examples": len(examples),
        "by_category": dict(sorted(Counter(infer_experiment1_category(example.question_type) for example in examples).items())),
        "by_question_type": dict(sorted(Counter(example.question_type for example in examples).items())),
    }
