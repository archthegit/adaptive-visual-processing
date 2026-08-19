from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Any

from src.dataset import VQAExample

from .manifest import EXPERIMENT1_CATEGORIES, infer_experiment1_category


@dataclass(frozen=True)
class TemporalSplitConfig:
    engineering_per_category: int = 2
    pilot_per_category: int = 12
    confirmatory_per_category: int = 60
    seed: int = 20260818
    max_per_source_video: int = 6


def single_video_temporal_examples(examples: list[VQAExample]) -> list[VQAExample]:
    selected = []
    for example in examples:
        category = infer_experiment1_category(example.question_type)
        if category not in EXPERIMENT1_CATEGORIES:
            continue
        if len(example.inputs) != 1:
            continue
        segment = example.inputs[0]
        if segment.is_image:
            continue
        selected.append(example)
    return selected


def bounded_duration_seconds(example: VQAExample) -> float | None:
    segment = example.inputs[0]
    if segment.start_seconds is None or segment.end_seconds is None:
        return None
    return max(0.0, float(segment.end_seconds - segment.start_seconds))


def duration_bucket(example: VQAExample) -> str:
    duration = bounded_duration_seconds(example)
    if duration is None:
        return "unbounded"
    if duration < 15:
        return "short"
    if duration < 120:
        return "medium"
    return "long"


def temporal_manifest_record(example: VQAExample) -> dict[str, Any]:
    segment = example.inputs[0]
    duration = bounded_duration_seconds(example)
    is_unbounded = segment.start_seconds is None or segment.end_seconds is None
    return {
        "question_id": example.question_id,
        "question_type": example.question_type,
        "category": infer_experiment1_category(example.question_type),
        "question": example.question,
        "choices": list(example.choices),
        "correct_idx": example.correct_idx,
        "correct_answer": example.choices[example.correct_idx],
        "source_video_id": segment.video_id,
        "participant_id": segment.participant_id,
        "input_key": segment.input_key,
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "bounded_duration_seconds": duration,
        "duration_bucket": duration_bucket(example),
        "is_unbounded_full_video": is_unbounded,
        "video_clip": [
            {
                "input_key": segment.input_key,
                "video_id": segment.video_id,
                "participant_id": segment.participant_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "image_time_seconds": segment.image_time_seconds,
                "is_unbounded_full_video": is_unbounded,
                "raw": segment.raw,
            }
        ],
        "raw_metadata": {
            key: value
            for key, value in example.raw.items()
            if key in {"others", "stat", "metadata"}
        },
        "experiment": "experiment1_temporal_query_relevance",
    }


def _category_counts(items: list[VQAExample]) -> dict[str, int]:
    return dict(sorted(Counter(infer_experiment1_category(item.question_type) for item in items).items()))


def _select_balanced(
    candidates: list[VQAExample],
    count: int,
    rng: random.Random,
    used_question_ids: set[str],
    max_per_source_video: int,
    source_counts: Counter[str],
    excluded_source_videos: set[str] | None = None,
) -> list[VQAExample]:
    excluded = set(excluded_source_videos or set())
    pool = [
        item
        for item in candidates
        if item.question_id not in used_question_ids and item.inputs[0].video_id not in excluded
    ]
    if len(pool) < count:
        raise ValueError(
            f"Could not select {count} examples while excluding source videos {sorted(excluded)}; "
            f"only {len(pool)} eligible candidates remain."
        )
    selected: list[VQAExample] = []
    subtype_counts: Counter[str] = Counter()
    participant_counts: Counter[str] = Counter()
    duration_counts: Counter[str] = Counter()
    remaining = list(pool)

    while len(selected) < count:
        allowed = [
            item
            for item in remaining
            if source_counts[item.inputs[0].video_id] < max_per_source_video
        ]
        if not allowed:
            raise ValueError(
                f"Could not select {count} examples with max_per_source_video={max_per_source_video}; "
                f"selected {len(selected)}."
            )

        def score(item: VQAExample) -> tuple[int, int, int, int, float, str]:
            segment = item.inputs[0]
            return (
                source_counts[segment.video_id] * 3,
                subtype_counts[item.question_type],
                participant_counts[segment.participant_id],
                duration_counts[duration_bucket(item)],
                rng.random(),
                item.question_id,
            )

        chosen = min(allowed, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
        segment = chosen.inputs[0]
        source_counts[segment.video_id] += 1
        subtype_counts[chosen.question_type] += 1
        participant_counts[segment.participant_id] += 1
        duration_counts[duration_bucket(chosen)] += 1

    used_question_ids.update(item.question_id for item in selected)
    return selected


def create_temporal_splits(
    examples: list[VQAExample],
    config: TemporalSplitConfig,
) -> tuple[dict[str, list[VQAExample]], dict[str, Any]]:
    if config.max_per_source_video <= 0:
        raise ValueError("max_per_source_video must be positive.")
    rng = random.Random(config.seed)
    eligible = single_video_temporal_examples(examples)
    by_category: dict[str, list[VQAExample]] = defaultdict(list)
    for example in eligible:
        category = infer_experiment1_category(example.question_type)
        if category is not None:
            by_category[category].append(example)

    requested_total_per_category = (
        config.engineering_per_category + config.pilot_per_category + config.confirmatory_per_category
    )
    missing = {
        category: len(by_category.get(category, []))
        for category in EXPERIMENT1_CATEGORIES
        if len(by_category.get(category, [])) < requested_total_per_category
    }
    if missing:
        raise ValueError(
            "Not enough eligible single-video temporal examples for requested split sizes. "
            f"Need {requested_total_per_category}/category, found {missing}."
        )

    for bucket in by_category.values():
        bucket.sort(key=lambda item: (item.question_type, item.inputs[0].participant_id, duration_bucket(item), rng.random(), item.question_id))

    splits = {"engineering": [], "pilot": [], "confirmatory": []}
    used_ids: set[str] = set()
    pilot_video_ids: set[str] = set()
    source_counts_by_split: dict[str, Counter[str]] = {
        "engineering": Counter(),
        "pilot": Counter(),
        "confirmatory": Counter(),
    }

    for category in EXPERIMENT1_CATEGORIES:
        bucket = by_category[category]
        splits["engineering"].extend(
            _select_balanced(
                bucket,
                config.engineering_per_category,
                rng,
                used_ids,
                config.max_per_source_video,
                source_counts_by_split["engineering"],
            )
        )
        pilot = _select_balanced(
            bucket,
            config.pilot_per_category,
            rng,
            used_ids,
            config.max_per_source_video,
            source_counts_by_split["pilot"],
        )
        splits["pilot"].extend(pilot)
        pilot_video_ids.update(item.inputs[0].video_id for item in pilot)

    for category in EXPERIMENT1_CATEGORIES:
        confirmatory = _select_balanced(
            by_category[category],
            config.confirmatory_per_category,
            rng,
            used_ids,
            config.max_per_source_video,
            source_counts_by_split["confirmatory"],
            excluded_source_videos=pilot_video_ids,
        )
        splits["confirmatory"].extend(confirmatory)

    for key in splits:
        splits[key].sort(key=lambda item: (infer_experiment1_category(item.question_type) or "", item.question_type, item.question_id))

    summary = build_temporal_split_summary(splits, eligible, config)
    return splits, summary


def _split_summary(items: list[VQAExample]) -> dict[str, Any]:
    source_ids = [item.inputs[0].video_id for item in items]
    participants = [item.inputs[0].participant_id for item in items]
    return {
        "num_examples": len(items),
        "by_category": _category_counts(items),
        "by_question_type": dict(sorted(Counter(item.question_type for item in items).items())),
        "by_participant": dict(sorted(Counter(participants).items())),
        "by_duration_bucket": dict(sorted(Counter(duration_bucket(item) for item in items).items())),
        "num_source_videos": len(set(source_ids)),
        "source_video_counts": dict(sorted(Counter(source_ids).items())),
        "unbounded_full_video_count": sum(1 for item in items if bounded_duration_seconds(item) is None),
    }


def build_temporal_split_summary(
    splits: dict[str, list[VQAExample]],
    eligible: list[VQAExample],
    config: TemporalSplitConfig,
) -> dict[str, Any]:
    pilot_videos = {item.inputs[0].video_id for item in splits["pilot"]}
    confirmatory_videos = {item.inputs[0].video_id for item in splits["confirmatory"]}
    ids_by_split = {name: {item.question_id for item in items} for name, items in splits.items()}
    return {
        "experiment": "experiment1_temporal_query_relevance",
        "seed": config.seed,
        "engineering_per_category": config.engineering_per_category,
        "pilot_per_category": config.pilot_per_category,
        "confirmatory_per_category": config.confirmatory_per_category,
        "max_per_source_video": config.max_per_source_video,
        "eligible_single_video_examples": len(eligible),
        "eligible_by_category": _category_counts(eligible),
        "splits": {name: _split_summary(items) for name, items in sorted(splits.items())},
        "question_id_overlap": {
            "engineering_pilot": sorted(ids_by_split["engineering"] & ids_by_split["pilot"]),
            "engineering_confirmatory": sorted(ids_by_split["engineering"] & ids_by_split["confirmatory"]),
            "pilot_confirmatory": sorted(ids_by_split["pilot"] & ids_by_split["confirmatory"]),
        },
        "pilot_confirmatory_source_video_overlap": sorted(pilot_videos & confirmatory_videos),
        "pilot_confirmatory_source_video_overlap_avoided": not (pilot_videos & confirmatory_videos),
    }
