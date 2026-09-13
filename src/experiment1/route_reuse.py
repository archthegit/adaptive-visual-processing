from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ANCHOR_LAYERS = (8, 12, 16, 20, 24, 28)
ROUTE_REUSE_SEED = 20260913
NUM_TEMPORAL_BINS = 8
RETAINED_BINS = 4
RETENTION_RATIO = 0.5
UNIFORM_BINS = (0, 2, 5, 7)


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def topk_bins(distribution: Sequence[float], k: int = RETAINED_BINS) -> tuple[int, ...]:
    indexed = [(float(value), int(index)) for index, value in enumerate(distribution)]
    indexed.sort(key=lambda item: (-item[0], item[1]))
    return tuple(index for _value, index in indexed[:k])


def normalize_allowed_bins(values: Sequence[int]) -> tuple[int, ...]:
    bins = tuple(sorted({int(item) for item in values}))
    if any(item < 0 or item >= NUM_TEMPORAL_BINS for item in bins):
        raise ValueError(f"Route bins must be in [0,{NUM_TEMPORAL_BINS - 1}], got {bins}.")
    return bins


def routed_layers_for_anchor(anchor_layer: int, num_layers: int) -> tuple[int, ...]:
    if anchor_layer >= num_layers:
        return tuple()
    return tuple(layer for layer in range(anchor_layer + 1, min(anchor_layer + 4, num_layers)))


def route_anchor_layers(num_layers: int) -> tuple[int, ...]:
    return tuple(anchor for anchor in ANCHOR_LAYERS if anchor < num_layers and routed_layers_for_anchor(anchor, num_layers))


def _condition_anchor_bins(
    condition: str,
    scores: Sequence[Sequence[float]],
    anchor: int,
    *,
    question_id: str,
    model: str,
    seed: int,
) -> tuple[int, ...]:
    if condition == "route_reuse_gap4_top50":
        return topk_bins(scores[anchor], RETAINED_BINS)
    if condition == "uniform_reuse_gap4_top50":
        return UNIFORM_BINS
    if condition == "random_reuse_gap4_top50":
        rng = random.Random(f"{seed}:{model}:{question_id}:{anchor}")
        return tuple(sorted(rng.sample(range(NUM_TEMPORAL_BINS), RETAINED_BINS)))
    raise ValueError(f"Unsupported route-reuse condition: {condition}")


def route_layers_from_anchor_bins(
    anchor_bins: Mapping[int, Sequence[int]],
    num_layers: int,
) -> dict[int, dict[str, Any]]:
    layer_routes: dict[int, dict[str, Any]] = {}
    for anchor, bins in sorted((int(k), normalize_allowed_bins(v)) for k, v in anchor_bins.items()):
        if anchor < 0 or anchor >= num_layers:
            continue
        omitted = tuple(index for index in range(NUM_TEMPORAL_BINS) if index not in set(bins))
        for layer in routed_layers_for_anchor(anchor, num_layers):
            layer_routes[int(layer)] = {
                "source_anchor_layer": int(anchor),
                "selected_bins": list(bins),
                "omitted_bins": list(omitted),
            }
    return layer_routes


def route_spec_from_baseline_artifact(
    artifact: Mapping[str, Any],
    *,
    model: str,
    condition: str,
    baseline_artifact: str,
    seed: int = ROUTE_REUSE_SEED,
    git_commit: str | None = None,
) -> dict[str, Any]:
    scores = artifact["temporal_relevance"]["normalized_temporal_bin_scores"]
    if not scores or any(len(row) != NUM_TEMPORAL_BINS for row in scores):
        raise ValueError("Route reuse requires normalized temporal scores with exactly 8 temporal bins.")
    num_layers = len(scores)
    question_id = str(artifact["question_id"])
    anchors = route_anchor_layers(num_layers)
    anchor_bins = {
        anchor: _condition_anchor_bins(
            condition,
            scores,
            anchor,
            question_id=question_id,
            model=model,
            seed=seed,
        )
        for anchor in anchors
    }
    layer_routes = route_layers_from_anchor_bins(anchor_bins, num_layers)
    dense_layers = sorted(set(range(0, min(9, num_layers))) | set(anchors))
    return {
        "type": "causal_route_reuse",
        "model": model,
        "condition": condition,
        "num_temporal_bins": NUM_TEMPORAL_BINS,
        "retained_bins": RETAINED_BINS,
        "retention_ratio": RETENTION_RATIO,
        "dense_prefix_through_layer": 8,
        "anchor_layers": list(anchors),
        "dense_layers": dense_layers,
        "layer_routes": {str(layer): route for layer, route in sorted(layer_routes.items())},
        "baseline_artifact": baseline_artifact,
        "seed": int(seed),
        "git_commit": git_commit,
        "mask_rule": "Only text-token queries are blocked from omitted visual-bin columns at routed decoder layers; anchors and layers 0-8 remain dense.",
    }


def visual_tokens_by_temporal_bin_from_cells(cells: Sequence[Any]) -> dict[int, tuple[int, ...]]:
    by_bin: dict[int, list[int]] = {index: [] for index in range(NUM_TEMPORAL_BINS)}
    for cell in cells:
        if isinstance(cell, dict):
            temporal_index = int(cell.get("temporal_index", cell.get("analysis_bin", -1)))
            token_index = int(cell.get("token_index", -1))
            modality = cell.get("modality", "video")
        else:
            temporal_index = int(getattr(cell, "temporal_index", -1))
            token_index = int(getattr(cell, "token_index", -1))
            modality = getattr(cell, "modality", "video")
        if modality == "video" and temporal_index in by_bin:
            by_bin[temporal_index].append(token_index)
    return {key: tuple(sorted(values)) for key, values in by_bin.items()}


def route_mask_summary(route_spec: Mapping[str, Any], visual_tokens_by_bin: Mapping[int, Sequence[int]]) -> dict[str, Any]:
    layer_summaries = {}
    for raw_layer, route in sorted(route_spec.get("layer_routes", {}).items(), key=lambda item: int(item[0])):
        selected = normalize_allowed_bins(route["selected_bins"])
        omitted = normalize_allowed_bins(route["omitted_bins"])
        allowed_tokens = sorted(
            int(token)
            for bin_index in selected
            for token in visual_tokens_by_bin.get(int(bin_index), ())
        )
        blocked_tokens = sorted(
            int(token)
            for bin_index in omitted
            for token in visual_tokens_by_bin.get(int(bin_index), ())
        )
        layer_summaries[str(int(raw_layer))] = {
            "source_anchor_layer": int(route["source_anchor_layer"]),
            "selected_bins": list(selected),
            "omitted_bins": list(omitted),
            "num_allowed_visual_tokens": len(allowed_tokens),
            "num_blocked_visual_tokens": len(blocked_tokens),
        }
    return {
        "condition": route_spec.get("condition"),
        "retention_ratio": route_spec.get("retention_ratio"),
        "anchor_layers": list(route_spec.get("anchor_layers", [])),
        "layer_routes": layer_summaries,
        "baseline_artifact": route_spec.get("baseline_artifact"),
        "seed": route_spec.get("seed"),
        "git_commit": route_spec.get("git_commit"),
    }


@dataclass(frozen=True)
class LayerRouteMask:
    layer_to_blocked_visual_indices: dict[int, tuple[int, ...]]
    all_visual_token_indices: tuple[int, ...]

    @classmethod
    def from_route_spec(
        cls,
        route_spec: Mapping[str, Any] | None,
        visual_tokens_by_bin: Mapping[int, Sequence[int]],
        all_visual_token_indices: Sequence[int],
    ) -> "LayerRouteMask | None":
        if not route_spec:
            return None
        blocked_by_layer: dict[int, tuple[int, ...]] = {}
        for raw_layer, route in route_spec.get("layer_routes", {}).items():
            omitted = normalize_allowed_bins(route.get("omitted_bins", ()))
            blocked = sorted(
                int(token)
                for bin_index in omitted
                for token in visual_tokens_by_bin.get(int(bin_index), ())
            )
            if blocked:
                blocked_by_layer[int(raw_layer)] = tuple(blocked)
        if not blocked_by_layer:
            return None
        return cls(
            layer_to_blocked_visual_indices=blocked_by_layer,
            all_visual_token_indices=tuple(int(item) for item in all_visual_token_indices),
        )

    def blocked_indices_for_layer(self, layer: int, key_len: int) -> tuple[int, ...]:
        return tuple(index for index in self.layer_to_blocked_visual_indices.get(int(layer), ()) if index < key_len)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
