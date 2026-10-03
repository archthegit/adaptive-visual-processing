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
    by_source: dict[str, list[VQAExample]] = defaultdict(list)
    for example in eligible:
        by_source[example.inputs[0].video_id].append(example)
    rng = random.Random(config.seed)
    source_records: list[dict[str, Any]] = []
    for source_video_id, source_examples in by_source.items():
        capped = sorted(source_examples, key=lambda item: (item.question_type, item.question_id))
        rng.shuffle(capped)
        capped = sorted(capped[: config.max_questions_per_source_video], key=lambda item: item.question_id)
        category_counts = Counter(infer_experiment1_category(item.question_type) for item in capped)
        source_records.append(
            {
                "source_video_id": source_video_id,
                "examples": capped,
                "category_counts": category_counts,
                "participant_id": capped[0].inputs[0].participant_id if capped else "",
                "duration": sum((bounded_duration_seconds(item) or 0.0) for item in capped) / max(1, len(capped)),
            }
        )
    rng.shuffle(source_records)
    source_records.sort(
        key=lambda item: (
            -sum(item["category_counts"].values()),
            item["participant_id"],
            round(float(item["duration"]), 3),
            rng.random(),
            item["source_video_id"],
        )
    )
    targets = {
        "train": config.train_per_category,
        "development": config.development_per_category,
        "test": config.test_per_category,
    }
    selected: dict[str, list[VQAExample]] = {name: [] for name in ADAPTIVE_SPLITS}
    counts: dict[str, Counter[str]] = {name: Counter() for name in ADAPTIVE_SPLITS}
    used_questions: set[str] = set()
    used_sources: set[str] = set()

    def deficits(split: str) -> int:
        return sum(max(0, targets[split] - counts[split][category]) for category in EXPERIMENT1_CATEGORIES)

    for source in source_records:
        source_id = source["source_video_id"]
        if source_id in used_sources:
            continue
        best_split = None
        best_gain = 0
        for split in ADAPTIVE_SPLITS:
            gain = 0
            for example in source["examples"]:
                category = infer_experiment1_category(example.question_type)
                if counts[split][category] < targets[split]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_split = split
            elif gain == best_gain and gain > 0 and best_split is not None:
                if deficits(split) > deficits(best_split):
                    best_split = split
        if best_split is None or best_gain <= 0:
            continue
        for example in source["examples"]:
            category = infer_experiment1_category(example.question_type)
            if counts[best_split][category] >= targets[best_split]:
                continue
            if example.question_id in used_questions:
                continue
            selected[best_split].append(example)
            counts[best_split][category] += 1
            used_questions.add(example.question_id)
        used_sources.add(source_id)
        if all(counts[split][category] >= targets[split] for split in ADAPTIVE_SPLITS for category in EXPERIMENT1_CATEGORIES):
            break

    shortages = {
        split: {
            category: targets[split] - counts[split][category]
            for category in EXPERIMENT1_CATEGORIES
            if counts[split][category] < targets[split]
        }
        for split in ADAPTIVE_SPLITS
    }
    shortages = {split: value for split, value in shortages.items() if value}
    if shortages:
        raise ValueError(f"Could not construct requested source-video-disjoint adaptive splits; shortages={shortages}.")
    for split in ADAPTIVE_SPLITS:
        selected[split].sort(key=lambda item: (infer_experiment1_category(item.question_type) or "", item.question_type, item.question_id))
    summary = adaptive_split_summary(selected, eligible, config)
    overlaps = summary["source_video_overlap_counts"]
    if any(value != 0 for value in overlaps.values()):
        raise AssertionError(f"Source-video overlap detected: {overlaps}")
    return selected, summary


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
