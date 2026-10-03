from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.dataset import VQAExample

from .manifest import EXPERIMENT1_CATEGORIES, infer_experiment1_category
from .temporal_splits import bounded_duration_seconds, temporal_manifest_record
from .temporal_handoff import question_token_positions, temporal_regions_from_layout, visual_token_positions


ADAPTIVE_SPLITS = ("train", "development", "test")
FRAME_COUNTS = (8, 16, 32)
COMPACTION_LAYERS = (4, 8, 12, 16, 20)
RETENTION_FRACTIONS = (0.25, 0.50, 0.75)
MAX_CANDIDATE_ROUTES = 24
ADAPTIVE_SCHEMA_VERSION = "qwen_adaptive_temporal_compaction_v1"


@dataclass(frozen=True)
class AdaptiveSplitConfig:
    train_per_category: int = 50
    development_per_category: int = 20
    test_per_category: int = 50
    seed: int = 20260928
    max_questions_per_source_video: int = 4


@dataclass(frozen=True)
class CandidateRoute:
    action_id: str
    retained_cells: tuple[int, ...]
    route_family: str
    frame_count: int
    compaction_layer: int
    native_temporal_cell_count: int
    retention_fraction: float
    retention_count: int
    seed: int
    question_id: str

    def to_json(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "retained_cells": list(self.retained_cells),
            "route_family": self.route_family,
            "frame_count": self.frame_count,
            "compaction_layer": self.compaction_layer,
            "native_temporal_cell_count": self.native_temporal_cell_count,
            "retention_fraction": self.retention_fraction,
            "retention_count": self.retention_count,
            "seed": self.seed,
            "question_id": self.question_id,
        }


def eligible_adaptive_examples(examples: Sequence[VQAExample]) -> list[VQAExample]:
    output = []
    for example in examples:
        category = infer_experiment1_category(example.question_type)
        if category not in EXPERIMENT1_CATEGORIES:
            continue
        if len(example.inputs) != 1:
            continue
        if example.inputs[0].is_image:
            continue
        output.append(example)
    return output


def create_source_video_disjoint_splits(
    examples: Sequence[VQAExample],
    config: AdaptiveSplitConfig,
) -> tuple[dict[str, list[VQAExample]], dict[str, Any]]:
    if config.max_questions_per_source_video <= 0:
        raise ValueError("max_questions_per_source_video must be positive.")
    eligible = eligible_adaptive_examples(examples)
    targets = {
        "train": config.train_per_category,
        "development": config.development_per_category,
        "test": config.test_per_category,
    }
    by_source: dict[str, list[VQAExample]] = defaultdict(list)
    for example in eligible:
        by_source[example.inputs[0].video_id].append(example)
    diagnostics = adaptive_feasibility_diagnostics(eligible, config)
    shortages = {
        category: {
            "required": diagnostics["required_counts_per_category"][category],
            "capped_upper_bound": diagnostics["capped_per_category_upper_bounds"].get(category, 0),
        }
        for category in EXPERIMENT1_CATEGORIES
        if diagnostics["capped_per_category_upper_bounds"].get(category, 0)
        < diagnostics["required_counts_per_category"][category]
    }
    if shortages:
        raise ValueError(f"Adaptive split request is infeasible before allocation; shortages={shortages}; diagnostics={diagnostics}")

    source_records: list[dict[str, Any]] = []
    for source_video_id, source_examples in by_source.items():
        by_category: dict[str, list[VQAExample]] = {
            category: sorted(
                [item for item in source_examples if infer_experiment1_category(item.question_type) == category],
                key=lambda item: (item.question_type, item.question_id),
            )
            for category in EXPERIMENT1_CATEGORIES
        }
        category_counts = Counter(
            {
                category: len(items)
                for category, items in by_category.items()
                if items
            }
        )
        source_records.append(
            {
                "source_video_id": source_video_id,
                "examples_by_category": by_category,
                "category_counts": category_counts,
                "participant_id": source_examples[0].inputs[0].participant_id if source_examples else "",
                "duration": sum((bounded_duration_seconds(item) or 0.0) for item in source_examples) / max(1, len(source_examples)),
                "num_categories": sum(1 for items in by_category.values() if items),
                "total_examples": len(source_examples),
            }
        )
    allocation = _deficit_aware_allocate_sources(source_records, targets, config, diagnostics)
    if allocation is None:
        failure = _allocation_failure_diagnostics(source_records, targets, config, diagnostics)
        raise ValueError(f"Could not construct requested source-video-disjoint adaptive splits; diagnostics={failure}")
    selected = allocation
    for split in ADAPTIVE_SPLITS:
        selected[split].sort(key=lambda item: (infer_experiment1_category(item.question_type) or "", item.question_type, item.question_id))
    summary = adaptive_split_summary(selected, eligible, config)
    summary["feasibility_diagnostics"] = diagnostics
    overlaps = summary["source_video_overlap_counts"]
    if any(value != 0 for value in overlaps.values()):
        raise AssertionError(f"Source-video overlap detected: {overlaps}")
    for split, per_category in summary["splits"].items():
        expected = targets[split]
        if per_category["num_examples"] != expected * len(EXPERIMENT1_CATEGORIES):
            raise AssertionError(f"{split} has wrong total: {per_category['num_examples']}")
        for category in EXPERIMENT1_CATEGORIES:
            actual = int(per_category["by_category"].get(category, 0))
            if actual != expected:
                raise AssertionError(f"{split}/{category} has {actual}, expected {expected}")
    return selected, summary


def adaptive_feasibility_diagnostics(
    eligible: Sequence[VQAExample],
    config: AdaptiveSplitConfig,
) -> dict[str, Any]:
    by_category = Counter(infer_experiment1_category(item.question_type) for item in eligible)
    sources_by_category: dict[str, set[str]] = {category: set() for category in EXPERIMENT1_CATEGORIES}
    by_source: dict[str, list[VQAExample]] = defaultdict(list)
    for example in eligible:
        category = infer_experiment1_category(example.question_type)
        by_source[example.inputs[0].video_id].append(example)
        if category in sources_by_category:
            sources_by_category[category].add(example.inputs[0].video_id)
    capped_upper: Counter[str] = Counter()
    source_category_cardinality = Counter()
    total_capped_capacity = 0
    for source_examples in by_source.values():
        categories = {
            category: sum(1 for item in source_examples if infer_experiment1_category(item.question_type) == category)
            for category in EXPERIMENT1_CATEGORIES
        }
        source_category_cardinality[sum(1 for value in categories.values() if value)] += 1
        total_capped_capacity += min(config.max_questions_per_source_video, len(source_examples))
        for category, count in categories.items():
            capped_upper[category] += min(config.max_questions_per_source_video, count)
    required = config.train_per_category + config.development_per_category + config.test_per_category
    return {
        "eligible_questions": len(eligible),
        "raw_eligible_questions_by_category": {category: int(by_category.get(category, 0)) for category in EXPERIMENT1_CATEGORIES},
        "unique_source_videos_by_category": {category: len(sources_by_category[category]) for category in EXPERIMENT1_CATEGORIES},
        "capped_per_category_upper_bounds": {category: int(capped_upper.get(category, 0)) for category in EXPERIMENT1_CATEGORIES},
        "total_capped_capacity": int(total_capped_capacity),
        "required_counts_per_category": {category: required for category in EXPERIMENT1_CATEGORIES},
        "videos_by_number_of_represented_target_categories": dict(sorted(source_category_cardinality.items())),
        "num_source_videos": len(by_source),
    }


def _target_deficits(counts: dict[str, Counter[str]], targets: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        split: {
            category: max(0, targets[split] - int(counts[split][category]))
            for category in EXPERIMENT1_CATEGORIES
        }
        for split in ADAPTIVE_SPLITS
    }


def _select_source_examples_for_split(
    source: dict[str, Any],
    split_deficits: dict[str, int],
    *,
    max_questions_per_source_video: int,
    category_rarity_weights: dict[str, float],
    seed: int,
) -> list[VQAExample]:
    selected: list[VQAExample] = []
    used_by_category: Counter[str] = Counter()
    while len(selected) < max_questions_per_source_video:
        candidates = []
        for category in EXPERIMENT1_CATEGORIES:
            remaining_deficit = int(split_deficits.get(category, 0)) - used_by_category[category]
            available = len(source["examples_by_category"].get(category, ())) - used_by_category[category]
            if remaining_deficit <= 0 or available <= 0:
                continue
            tie = hashlib.sha1(f"{seed}:{source['source_video_id']}:{category}:{len(selected)}".encode("utf-8")).hexdigest()
            candidates.append(
                (
                    category_rarity_weights.get(category, 1.0) * remaining_deficit,
                    remaining_deficit,
                    available,
                    tie,
                    category,
                )
            )
        if not candidates:
            break
        category = max(candidates)[-1]
        example = source["examples_by_category"][category][used_by_category[category]]
        selected.append(example)
        used_by_category[category] += 1
    return selected


def _remaining_capacity_bounds(
    remaining_sources: Sequence[dict[str, Any]],
    *,
    max_questions_per_source_video: int,
) -> tuple[Counter[str], int]:
    per_category: Counter[str] = Counter()
    total = 0
    for source in remaining_sources:
        total += min(max_questions_per_source_video, int(source["total_examples"]))
        for category in EXPERIMENT1_CATEGORIES:
            per_category[category] += min(
                max_questions_per_source_video,
                len(source["examples_by_category"].get(category, ())),
            )
    return per_category, total


def _source_capacity_bounds(
    source: dict[str, Any],
    *,
    max_questions_per_source_video: int,
) -> tuple[Counter[str], int]:
    per_category: Counter[str] = Counter()
    total = min(max_questions_per_source_video, int(source["total_examples"]))
    for category in EXPERIMENT1_CATEGORIES:
        per_category[category] = min(
            max_questions_per_source_video,
            len(source["examples_by_category"].get(category, ())),
        )
    return per_category, total


def _passes_capacity_guard_with_bounds(
    counts: dict[str, Counter[str]],
    targets: dict[str, int],
    remaining_per_category: Counter[str],
    remaining_total: int,
) -> bool:
    deficits = _target_deficits(counts, targets)
    total_deficit = 0
    category_total_deficit: Counter[str] = Counter()
    for split in ADAPTIVE_SPLITS:
        split_total = sum(deficits[split].values())
        total_deficit += split_total
        if split_total > remaining_total:
            return False
        for category in EXPERIMENT1_CATEGORIES:
            category_total_deficit[category] += deficits[split][category]
            if deficits[split][category] > remaining_per_category[category]:
                return False
    if total_deficit > remaining_total:
        return False
    for category in EXPERIMENT1_CATEGORIES:
        if category_total_deficit[category] > remaining_per_category[category]:
            return False
    return True


def _score_assignment(
    selected: Sequence[VQAExample],
    split: str,
    counts: dict[str, Counter[str]],
    targets: dict[str, int],
    category_rarity_weights: dict[str, float],
    source: dict[str, Any],
    *,
    jitter: float,
) -> float:
    if not selected:
        return float("-inf")
    selection_counts = Counter(infer_experiment1_category(item.question_type) for item in selected)
    score = 0.0
    for category, amount in selection_counts.items():
        deficit = max(0, targets[split] - counts[split][category])
        normalized = deficit / max(1.0, float(targets[split]))
        score += amount * normalized * category_rarity_weights.get(category, 1.0)
        if amount > deficit:
            score -= 100.0 * (amount - deficit)
    unused_selected_capacity = max(0, len(selected) - sum(selection_counts.values()))
    score -= 0.01 * unused_selected_capacity
    score += 0.05 * int(source["num_categories"])
    score += jitter
    return score


def _allocation_complete(counts: dict[str, Counter[str]], targets: dict[str, int]) -> bool:
    return all(
        int(counts[split][category]) >= int(targets[split])
        for split in ADAPTIVE_SPLITS
        for category in EXPERIMENT1_CATEGORIES
    )


def _deficit_aware_allocate_sources(
    source_records: Sequence[dict[str, Any]],
    targets: dict[str, int],
    config: AdaptiveSplitConfig,
    diagnostics: dict[str, Any],
) -> dict[str, list[VQAExample]] | None:
    capped = diagnostics["capped_per_category_upper_bounds"]
    category_rarity_weights = {
        category: 1.0 + diagnostics["required_counts_per_category"][category] / max(1.0, float(capped.get(category, 0)))
        for category in EXPERIMENT1_CATEGORIES
    }
    attempts = 96
    best_selected: dict[str, list[VQAExample]] | None = None
    best_remaining = math.inf
    for attempt in range(attempts):
        rng = random.Random(f"{config.seed}:{attempt}")
        remaining = list(source_records)
        remaining.sort(
            key=lambda source: (
                -sum(category_rarity_weights.get(category, 1.0) * min(config.max_questions_per_source_video, count)
                     for category, count in source["category_counts"].items()),
                -int(source["num_categories"]),
                source["participant_id"],
                round(float(source["duration"]), 3),
                rng.random(),
                source["source_video_id"],
            )
        )
        selected: dict[str, list[VQAExample]] = {name: [] for name in ADAPTIVE_SPLITS}
        counts: dict[str, Counter[str]] = {name: Counter() for name in ADAPTIVE_SPLITS}
        while remaining and not _allocation_complete(counts, targets):
            deficits = _target_deficits(counts, targets)
            best = None
            all_remaining_per_category, all_remaining_total = _remaining_capacity_bounds(
                remaining,
                max_questions_per_source_video=config.max_questions_per_source_video,
            )
            for source_index, source in enumerate(remaining):
                source_per_category, source_total = _source_capacity_bounds(
                    source,
                    max_questions_per_source_video=config.max_questions_per_source_video,
                )
                rest_per_category = Counter(all_remaining_per_category)
                rest_per_category.subtract(source_per_category)
                rest_total = all_remaining_total - source_total
                for split in ADAPTIVE_SPLITS:
                    chosen = _select_source_examples_for_split(
                        source,
                        deficits[split],
                        max_questions_per_source_video=config.max_questions_per_source_video,
                        category_rarity_weights=category_rarity_weights,
                        seed=config.seed + attempt,
                    )
                    if not chosen:
                        continue
                    trial_counts = {name: Counter(counter) for name, counter in counts.items()}
                    for example in chosen:
                        category = infer_experiment1_category(example.question_type)
                        trial_counts[split][category] += 1
                    if any(trial_counts[split][category] > targets[split] for category in EXPERIMENT1_CATEGORIES):
                        continue
                    if not _passes_capacity_guard_with_bounds(trial_counts, targets, rest_per_category, rest_total):
                        continue
                    jitter_seed = f"{config.seed}:{attempt}:{source['source_video_id']}:{split}"
                    jitter = int(hashlib.sha1(jitter_seed.encode("utf-8")).hexdigest()[:8], 16) / 16**8 * 1e-6
                    score = _score_assignment(
                        chosen,
                        split,
                        counts,
                        targets,
                        category_rarity_weights,
                        source,
                        jitter=jitter,
                    )
                    candidate = (score, -source_index, split, source_index, chosen)
                    if best is None or candidate > best:
                        best = candidate
            if best is None:
                break
            _score, _neg_source_index, split, source_index, chosen = best
            source = remaining.pop(source_index)
            for example in chosen:
                category = infer_experiment1_category(example.question_type)
                selected[split].append(example)
                counts[split][category] += 1
            assert source["source_video_id"] not in {
                item.inputs[0].video_id
                for split_examples in selected.values()
                for item in split_examples
                if item not in chosen
            }
        remaining_deficit = sum(
            max(0, targets[split] - counts[split][category])
            for split in ADAPTIVE_SPLITS
            for category in EXPERIMENT1_CATEGORIES
        )
        if remaining_deficit < best_remaining:
            best_remaining = remaining_deficit
            best_selected = selected
        if remaining_deficit == 0:
            return selected
    return None


def _allocation_failure_diagnostics(
    source_records: Sequence[dict[str, Any]],
    targets: dict[str, int],
    config: AdaptiveSplitConfig,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    remaining_per_category, total_remaining = _remaining_capacity_bounds(
        source_records,
        max_questions_per_source_video=config.max_questions_per_source_video,
    )
    return {
        "feasibility_diagnostics": diagnostics,
        "required_targets": targets,
        "remaining_capacity_if_no_sources_used": {
            "per_category": dict(remaining_per_category),
            "total": total_remaining,
        },
        "note": "Independent per-category upper bounds are necessary diagnostics, not a joint-feasibility proof.",
    }


def adaptive_split_summary(
    splits: dict[str, list[VQAExample]],
    eligible: Sequence[VQAExample],
    config: AdaptiveSplitConfig,
) -> dict[str, Any]:
    source_sets = {split: {item.inputs[0].video_id for item in items} for split, items in splits.items()}
    question_sets = {split: {item.question_id for item in items} for split, items in splits.items()}
    pairwise_source_counts: dict[str, int] = {}
    pairwise_question_counts: dict[str, int] = {}
    for left, right in itertools.combinations(ADAPTIVE_SPLITS, 2):
        pairwise_source_counts[f"{left}_{right}"] = len(source_sets[left] & source_sets[right])
        pairwise_question_counts[f"{left}_{right}"] = len(question_sets[left] & question_sets[right])

    def split_summary(items: Sequence[VQAExample]) -> dict[str, Any]:
        sources = [item.inputs[0].video_id for item in items]
        return {
            "num_examples": len(items),
            "num_source_videos": len(set(sources)),
            "by_category": dict(sorted(Counter(infer_experiment1_category(item.question_type) for item in items).items())),
            "by_question_type": dict(sorted(Counter(item.question_type for item in items).items())),
            "by_participant": dict(sorted(Counter(item.inputs[0].participant_id for item in items).items())),
            "source_video_counts": dict(sorted(Counter(sources).items())),
        }

    return {
        "schema_version": "adaptive_compaction_splits_v1",
        "seed": config.seed,
        "max_questions_per_source_video": config.max_questions_per_source_video,
        "targets_per_category": {
            "train": config.train_per_category,
            "development": config.development_per_category,
            "test": config.test_per_category,
        },
        "eligible_single_video_examples": len(eligible),
        "splits": {split: split_summary(splits[split]) for split in ADAPTIVE_SPLITS},
        "source_video_overlap_counts": pairwise_source_counts,
        "question_id_overlap_counts": pairwise_question_counts,
        "source_video_disjoint_assertion": all(value == 0 for value in pairwise_source_counts.values()),
        "question_id_disjoint_assertion": all(value == 0 for value in pairwise_question_counts.values()),
    }


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def retention_count(native_cell_count: int, retention_fraction: float) -> int:
    if native_cell_count <= 0:
        raise ValueError("native_cell_count must be positive.")
    if not 0 < retention_fraction <= 1:
        raise ValueError("retention_fraction must be in (0, 1].")
    return max(1, min(native_cell_count, int(round(native_cell_count * retention_fraction))))


def stable_action_id(
    *,
    question_id: str,
    frame_count: int,
    compaction_layer: int,
    native_temporal_cell_count: int,
    retention_fraction: float,
    retained_cells: Sequence[int],
    route_family: str,
) -> str:
    payload = {
        "question_id": question_id,
        "frame_count": int(frame_count),
        "compaction_layer": int(compaction_layer),
        "native_temporal_cell_count": int(native_temporal_cell_count),
        "retention_fraction": float(retention_fraction),
        "retained_cells": [int(item) for item in retained_cells],
        "route_family": route_family,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"act_{digest}"


def candidate_routes(
    *,
    question_id: str,
    frame_count: int,
    compaction_layer: int,
    native_temporal_cell_count: int,
    retention_fraction: float,
    seed: int,
    max_candidates: int = MAX_CANDIDATE_ROUTES,
) -> list[CandidateRoute]:
    keep = retention_count(native_temporal_cell_count, retention_fraction)
    all_cells = tuple(range(native_temporal_cell_count))
    all_subsets = list(itertools.combinations(all_cells, keep))
    routes: dict[tuple[int, ...], str] = {}
    if len(all_subsets) <= max_candidates:
        for subset in all_subsets:
            routes[tuple(subset)] = "exhaustive"
    else:
        def add(cells: Sequence[int], family: str) -> None:
            if len(routes) >= max_candidates:
                return
            subset = tuple(sorted(set(int(cell) for cell in cells)))
            if len(subset) == keep and all(0 <= cell < native_temporal_cell_count for cell in subset):
                routes.setdefault(subset, family)

        if keep == 1:
            uniform = (native_temporal_cell_count // 2,)
        else:
            uniform = tuple(round(idx * (native_temporal_cell_count - 1) / (keep - 1)) for idx in range(keep))
        add(uniform, "uniform_temporal_coverage")
        add(range(keep), "prefix_retention")
        add(range(native_temporal_cell_count - keep, native_temporal_cell_count), "suffix_retention")
        window_starts = list(range(0, native_temporal_cell_count - keep + 1))
        remaining_slots = max(1, max_candidates - len(routes))
        if len(window_starts) > remaining_slots:
            if remaining_slots == 1:
                window_starts = [window_starts[len(window_starts) // 2]]
            else:
                window_starts = [
                    window_starts[round(idx * (len(window_starts) - 1) / (remaining_slots - 1))]
                    for idx in range(remaining_slots)
                ]
        for start in window_starts:
            add(range(start, start + keep), "contiguous_window")
        rng = random.Random(f"{seed}:{question_id}:{frame_count}:{compaction_layer}:{native_temporal_cell_count}:{retention_fraction}")
        attempts = 0
        while len(routes) < max_candidates and attempts < max_candidates * 20:
            attempts += 1
            add(rng.sample(list(all_cells), keep), "deterministic_random_subset")
    output = []
    for retained, family in sorted(routes.items(), key=lambda item: (item[1], item[0])):
        output.append(
            CandidateRoute(
                action_id=stable_action_id(
                    question_id=question_id,
                    frame_count=frame_count,
                    compaction_layer=compaction_layer,
                    native_temporal_cell_count=native_temporal_cell_count,
                    retention_fraction=retention_fraction,
                    retained_cells=retained,
                    route_family=family,
                ),
                retained_cells=retained,
                route_family=family,
                frame_count=frame_count,
                compaction_layer=compaction_layer,
                native_temporal_cell_count=native_temporal_cell_count,
                retention_fraction=retention_fraction,
                retention_count=keep,
                seed=seed,
                question_id=question_id,
            )
        )
    if len({route.retained_cells for route in output}) != len(output):
        raise AssertionError("Duplicate candidate routes were generated.")
    if len(output) > max_candidates:
        raise AssertionError("Candidate route cap was exceeded.")
    return output


def shard_records(records: Sequence[dict[str, Any]], *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards.")
    selected = []
    for record in sorted(records, key=lambda item: str(item["question_id"])):
        digest = int(hashlib.sha1(str(record["question_id"]).encode("utf-8")).hexdigest(), 16)
        if digest % num_shards == shard_index:
            selected.append(record)
    return selected


def extract_router_features(hidden_states: Any, layout: Any, *, layer: int, frame_count: int) -> dict[str, Any]:
    import torch

    regions = temporal_regions_from_layout(layout)
    question_positions = question_token_positions(layout)
    if not question_positions:
        raise ValueError("Question-token positions are required for router features.")
    question_idx = torch.tensor(question_positions, dtype=torch.long, device=hidden_states.device)
    q = hidden_states.index_select(1, question_idx)
    question_mean = q.mean(dim=1).detach().float().cpu()
    cell_means = []
    cell_norms = []
    cell_dispersion = []
    token_counts = []
    for cell_id in sorted(regions):
        positions = regions[cell_id]
        idx = torch.tensor(positions, dtype=torch.long, device=hidden_states.device)
        values = hidden_states.index_select(1, idx)
        mean = values.mean(dim=1)
        cell_means.append(mean.detach().float().cpu())
        centered = values - mean[:, None, :]
        cell_norms.append(float(torch.linalg.vector_norm(mean.float(), dim=-1).mean().item()))
        cell_dispersion.append(float(torch.sqrt(torch.mean(centered.float() * centered.float())).item()))
        token_counts.append(len(positions))
    cell_mean_tensor = torch.cat(cell_means, dim=0)
    if not torch.isfinite(cell_mean_tensor).all() or not torch.isfinite(question_mean).all():
        raise ValueError("Router features contain non-finite values.")
    return {
        "schema_version": "adaptive_compaction_router_features_v1",
        "layer": int(layer),
        "frame_count": int(frame_count),
        "native_temporal_cell_ids": list(sorted(regions)),
        "token_count_per_cell": token_counts,
        "cell_mean_residual": cell_mean_tensor,
        "question_mean_residual": question_mean.squeeze(0),
        "cell_residual_norm": cell_norms,
        "cell_residual_dispersion": cell_dispersion,
    }


def router_feature_metadata(feature_path: str | Path, features: dict[str, Any]) -> dict[str, Any]:
    tensor = features["cell_mean_residual"]
    question = features["question_mean_residual"]
    return {
        "feature_file": str(feature_path),
        "schema_version": features["schema_version"],
        "layer": features["layer"],
        "frame_count": features["frame_count"],
        "native_temporal_cell_count": len(features["native_temporal_cell_ids"]),
        "cell_mean_residual_shape": list(tensor.shape),
        "question_mean_residual_shape": list(question.shape),
        "dtype": str(tensor.dtype),
        "token_count_per_cell": list(features["token_count_per_cell"]),
    }


def validate_compact_action_artifact(artifact: dict[str, Any]) -> None:
    required = [
        "question_id",
        "source_video_id",
        "split",
        "frame_count",
        "compaction_layer",
        "native_temporal_cell_count",
        "retention_fraction",
        "retained_cell_ids",
        "route_family",
        "correct_choice_log_probability",
        "delta_correct_choice_log_probability_from_dense",
        "answer_margin",
        "delta_answer_margin_from_dense",
        "predicted_answer",
        "correct",
        "prediction_changed",
        "original_sequence_length",
        "compacted_sequence_length",
        "active_sequence_length_by_layer",
        "estimated_attention_flops",
        "paired_flop_reduction_from_dense",
        "memory_token_count",
        "status",
        "git_commit",
        "run_config",
    ]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"Compact action artifact is missing fields: {missing}")
    if artifact["status"] != "complete":
        raise ValueError("Only complete artifacts can pass schema validation.")
    if not 0 < float(artifact["retention_fraction"]) <= 1:
        raise ValueError("retention_fraction is out of range.")
    if len(set(int(item) for item in artifact["retained_cell_ids"])) != len(artifact["retained_cell_ids"]):
        raise ValueError("retained_cell_ids contain duplicates.")
    for field in ("correct_choice_log_probability", "answer_margin", "paired_flop_reduction_from_dense"):
        if not math.isfinite(float(artifact[field])):
            raise ValueError(f"{field} is non-finite.")


def manifest_records(examples: Sequence[VQAExample], split: str) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        record = temporal_manifest_record(example)
        record["split"] = split
        record["experiment"] = "adaptive_temporal_compaction"
        records.append(record)
    return records
