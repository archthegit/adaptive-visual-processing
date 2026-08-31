from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from src.io import write_jsonl

from .v2_metrics import rank_order


def fusion_condition_layer(condition: str) -> int | None:
    prefix = "fusion_block_"
    marker = "_after_layer_"
    if not condition.startswith(prefix):
        return None
    if marker not in condition:
        raise ValueError(f"Fusion condition must include '{marker}': {condition}")
    suffix = condition.rsplit(marker, 1)[1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"Fusion condition has non-integer layer suffix: {condition}") from exc


def bin_budget(num_bins: int, fraction: float) -> int:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    return max(1, int(math.ceil(num_bins * fraction)))


def contiguous_high_attention_cluster(scores: Sequence[float], budget: int) -> list[int]:
    values = list(float(value) for value in scores)
    if budget <= 0 or budget > len(values):
        raise ValueError("budget must be in [1, num_bins].")
    best_start = 0
    best_score = None
    for start in range(0, len(values) - budget + 1):
        score = sum(values[start : start + budget])
        if best_score is None or score > best_score:
            best_score = score
            best_start = start
    return list(range(best_start, best_start + budget))


def select_temporal_bins(
    scores: Sequence[float],
    strategy: str,
    fraction: float = 0.2,
    seed: int = 20260830,
    question_id: str | None = None,
) -> list[int]:
    budget = bin_budget(len(scores), fraction)
    order = list(rank_order(scores))
    if strategy == "top":
        return sorted(order[:budget])
    if strategy == "bottom":
        return sorted(order[-budget:])
    if strategy == "random":
        rng = random.Random(f"{seed}:{question_id or ''}:{len(scores)}")
        return sorted(rng.sample(range(len(scores)), budget))
    if strategy == "contiguous_high_cluster":
        return contiguous_high_attention_cluster(scores, budget)
    raise ValueError(f"Unsupported intervention strategy: {strategy}")


def temporal_scores_from_artifact(path: str | Path, ranking_layer: int = -1) -> list[float]:
    data = json.loads(Path(path).read_text())
    temporal = data.get("temporal_relevance") or {}
    scores = temporal.get("normalized_temporal_bin_scores")
    if not scores:
        raise ValueError(f"Artifact {path} does not contain normalized_temporal_bin_scores.")
    layer_index = ranking_layer if ranking_layer >= 0 else len(scores) + ranking_layer
    if layer_index < 0 or layer_index >= len(scores):
        raise ValueError(f"ranking_layer {ranking_layer} is outside available layers in {path}.")
    return [float(value) for value in scores[layer_index]]


def artifact_for_question(output_dir: str | Path, question_id: str) -> Path:
    return Path(output_dir) / f"{question_id}.json"


def intervention_record(
    primary_record: dict[str, Any],
    condition: str,
    selected_bins: list[int],
    ranking_layer: int,
    removal_fraction: float,
    seed: int,
    selection_source: str,
) -> dict[str, Any]:
    record = dict(primary_record)
    record["condition"] = condition
    record["selected_temporal_bins"] = list(selected_bins)
    record["ranking_layer"] = ranking_layer
    record["removal_fraction"] = removal_fraction
    record["selection_seed"] = seed
    record["selection_source"] = selection_source
    if condition.startswith("mask_"):
        record["pre_encoder_mask_temporal_bins"] = list(selected_bins)
    if condition.startswith("fusion_block_"):
        record["decoder_direct_access_mask_temporal_bins"] = list(selected_bins)
        record["decoder_direct_access_through_layer"] = fusion_condition_layer(condition)
    if condition.startswith("keep_"):
        record["keep_temporal_bins"] = list(selected_bins)
    return record


def build_v2_intervention_records(
    primary_manifest: list[dict[str, Any]],
    baseline_output_dir: str | Path,
    condition: str,
    strategy: str,
    removal_fraction: float = 0.2,
    ranking_layer: int = -1,
    seed: int = 20260830,
    mismatched_output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    records = []
    for primary in primary_manifest:
        question_id = primary["question_id"]
        score_source = "baseline"
        score_path = artifact_for_question(baseline_output_dir, question_id)
        if strategy == "mismatched_top":
            if mismatched_output_dir is None:
                raise ValueError("mismatched_top strategy requires mismatched_output_dir.")
            score_path = artifact_for_question(mismatched_output_dir, question_id)
            strategy_for_selection = "top"
            score_source = "mismatched_query"
        else:
            strategy_for_selection = strategy
        scores = temporal_scores_from_artifact(score_path, ranking_layer=ranking_layer)
        selected = select_temporal_bins(
            scores,
            strategy_for_selection,
            fraction=removal_fraction,
            seed=seed,
            question_id=question_id,
        )
        records.append(
            intervention_record(
                primary,
                condition=condition,
                selected_bins=selected,
                ranking_layer=ranking_layer,
                removal_fraction=removal_fraction,
                seed=seed,
                selection_source=score_source,
            )
        )
    return records


def write_v2_intervention_manifest(
    primary_manifest_path: str | Path,
    baseline_output_dir: str | Path,
    output_jsonl: str | Path,
    condition: str,
    strategy: str,
    removal_fraction: float = 0.2,
    ranking_layer: int = -1,
    seed: int = 20260830,
    mismatched_output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    primary = []
    with Path(primary_manifest_path).open("r") as handle:
        for line in handle:
            if line.strip():
                primary.append(json.loads(line))
    records = build_v2_intervention_records(
        primary,
        baseline_output_dir=baseline_output_dir,
        condition=condition,
        strategy=strategy,
        removal_fraction=removal_fraction,
        ranking_layer=ranking_layer,
        seed=seed,
        mismatched_output_dir=mismatched_output_dir,
    )
    write_jsonl(output_jsonl, records)
    return records
