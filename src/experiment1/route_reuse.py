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
RETENTION_RATIO = 0.5
UNIFORM_BINS = (0, 2, 5, 7)
TOKEN_BUDGET_TOLERANCE = 0.05


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


def normalize_bins(values: Sequence[int]) -> tuple[int, ...]:
    bins = tuple(sorted({int(item) for item in values}))
    if any(item < 0 or item >= NUM_TEMPORAL_BINS for item in bins):
        raise ValueError(f"Analysis bins must be in [0,{NUM_TEMPORAL_BINS - 1}], got {bins}.")
    return bins


def routed_layers_for_anchor(anchor_layer: int, num_layers: int) -> tuple[int, ...]:
    if anchor_layer >= num_layers:
        return tuple()
    return tuple(layer for layer in range(anchor_layer + 1, min(anchor_layer + 4, num_layers)))


def route_anchor_layers(num_layers: int) -> tuple[int, ...]:
    return tuple(anchor for anchor in ANCHOR_LAYERS if anchor < num_layers and routed_layers_for_anchor(anchor, num_layers))


def _token_layout(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    layout = artifact.get("token_layout") or {}
    if not layout:
        raise ValueError(f"{artifact.get('question_id')}: route generation requires token_layout in baseline artifact.")
    return layout


def _all_video_visual_tokens(layout: Mapping[str, Any]) -> tuple[int, ...]:
    cells = layout.get("visual_token_cells") or []
    tokens = sorted({int(cell["token_index"]) for cell in cells if cell.get("modality", "video") == "video"})
    if not tokens:
        raise ValueError("Route generation requires video visual token cells.")
    return tuple(tokens)


def _frame_mapping_by_sample(artifact: Mapping[str, Any]) -> dict[int, int]:
    mappings = (artifact.get("frame_bin_mappings") or [[]])[0] or []
    return {int(item["sample_position"]): int(item["analysis_bin"]) for item in mappings}


def _qwen_cell_analysis_bins(cell: Mapping[str, Any], artifact: Mapping[str, Any]) -> tuple[int, ...]:
    direct = cell.get("analysis_bin")
    if direct is not None:
        return (int(direct),)
    qwen_temporal = int(cell.get("temporal_bin", cell.get("temporal_index", 0)))
    grid_t = int(cell.get("grid_t", NUM_TEMPORAL_BINS))
    sampled = (artifact.get("sampled_frame_indices") or [[]])[0] or []
    position_to_analysis = _frame_mapping_by_sample(artifact)
    if not sampled or not position_to_analysis:
        return (qwen_temporal,)
    start = int(round(qwen_temporal * len(sampled) / grid_t))
    end = int(round((qwen_temporal + 1) * len(sampled) / grid_t))
    if end <= start:
        end = min(len(sampled), start + 1)
    bins = sorted({position_to_analysis[pos] for pos in range(start, end) if pos in position_to_analysis})
    return tuple(bins or [qwen_temporal])


def native_routing_units_from_artifact(artifact: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    layout = _token_layout(artifact)
    cells = [cell for cell in layout.get("visual_token_cells", []) if cell.get("modality", "video") == "video"]
    grouped: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    if model == "vila":
        for cell in cells:
            grouped.setdefault((int(cell["analysis_bin"]),), []).append(cell)
    elif model == "qwen":
        by_native: dict[int, list[Mapping[str, Any]]] = {}
        for cell in cells:
            by_native.setdefault(int(cell.get("temporal_bin", cell.get("temporal_index", 0))), []).append(cell)
        for native_index, native_cells in by_native.items():
            bins = sorted({bin_id for cell in native_cells for bin_id in _qwen_cell_analysis_bins(cell, artifact)})
            grouped.setdefault(tuple(bins), []).extend(native_cells)
    else:
        raise ValueError(f"Unsupported route-reuse model: {model}")
    units = []
    for index, (analysis_bins, unit_cells) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        token_indices = sorted({int(cell["token_index"]) for cell in unit_cells})
        units.append(
            {
                "unit_id": index,
                "analysis_bins": list(analysis_bins),
                "visual_token_indices": token_indices,
                "num_visual_tokens": len(token_indices),
            }
        )
    validate_native_units(units, _all_video_visual_tokens(layout))
    return units


def validate_native_units(units: Sequence[Mapping[str, Any]], all_visual_tokens: Sequence[int]) -> None:
    seen: list[int] = []
    for unit in units:
        tokens = [int(token) for token in unit.get("visual_token_indices", ())]
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"Native routing unit has duplicate tokens: {unit}")
        seen.extend(tokens)
    if sorted(seen) != sorted(int(token) for token in all_visual_tokens):
        raise ValueError("Every video visual token must belong to exactly one native routing unit.")


def _unit_scores(distribution: Sequence[float], units: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    scores = {}
    for unit in units:
        bins = [int(item) for item in unit["analysis_bins"]]
        if not bins:
            raise ValueError(f"Native routing unit has no analysis bins: {unit}")
        scores[int(unit["unit_id"])] = float(sum(float(distribution[bin_id]) for bin_id in bins) / len(bins))
    return scores


def _retained_unit_count(units: Sequence[Mapping[str, Any]]) -> int:
    return max(1, int(round(len(units) * RETENTION_RATIO)))


def _select_top_units(distribution: Sequence[float], units: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    scores = _unit_scores(distribution, units)
    ordered = sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))
    return tuple(ordered[: _retained_unit_count(units)])


def _select_uniform_units(units: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    uniform_bins = set(UNIFORM_BINS)
    scored = []
    for unit in units:
        bins = set(int(item) for item in unit["analysis_bins"])
        overlap = len(bins & uniform_bins)
        center_distance = min(abs(bin_id - target) for bin_id in bins for target in uniform_bins)
        scored.append((-overlap, center_distance, int(unit["unit_id"])))
    scored.sort()
    return tuple(item[2] for item in scored[: _retained_unit_count(units)])


def _select_random_units(units: Sequence[Mapping[str, Any]], *, seed: int, model: str, question_id: str, anchor: int) -> tuple[int, ...]:
    rng = random.Random(f"{seed}:{model}:{question_id}:{anchor}")
    unit_ids = [int(unit["unit_id"]) for unit in units]
    return tuple(sorted(rng.sample(unit_ids, _retained_unit_count(units))))


def _condition_units(
    condition: str,
    distribution: Sequence[float],
    units: Sequence[Mapping[str, Any]],
    *,
    question_id: str,
    model: str,
    seed: int,
    anchor: int,
) -> tuple[int, ...]:
    if condition == "route_reuse_gap4_top50":
        return _select_top_units(distribution, units)
    if condition == "uniform_reuse_gap4_top50":
        return _select_uniform_units(units)
    if condition == "random_reuse_gap4_top50":
        return _select_random_units(units, seed=seed, model=model, question_id=question_id, anchor=anchor)
    raise ValueError(f"Unsupported route-reuse condition: {condition}")


def _unit_lookup(units: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(unit["unit_id"]): unit for unit in units}


def _route_for_units(
    selected_unit_ids: Sequence[int],
    units: Sequence[Mapping[str, Any]],
    source_anchor_layer: int,
    total_visual_tokens: int,
) -> dict[str, Any]:
    selected = tuple(sorted(int(item) for item in selected_unit_ids))
    lookup = _unit_lookup(units)
    omitted = tuple(unit_id for unit_id in sorted(lookup) if unit_id not in set(selected))
    allowed_tokens = sorted(
        int(token)
        for unit_id in selected
        for token in lookup[unit_id]["visual_token_indices"]
    )
    blocked_tokens = sorted(
        int(token)
        for unit_id in omitted
        for token in lookup[unit_id]["visual_token_indices"]
    )
    overlap = set(allowed_tokens) & set(blocked_tokens)
    if overlap:
        raise ValueError(f"Allowed and blocked route tokens overlap: {sorted(overlap)}")
    if len(allowed_tokens) + len(blocked_tokens) != total_visual_tokens:
        raise ValueError("Allowed plus blocked visual tokens must equal all video visual tokens.")
    selected_bins = sorted({bin_id for unit_id in selected for bin_id in lookup[unit_id]["analysis_bins"]})
    omitted_bins = sorted({bin_id for unit_id in omitted for bin_id in lookup[unit_id]["analysis_bins"]})
    return {
        "source_anchor_layer": int(source_anchor_layer),
        "selected_bins": selected_bins,
        "omitted_bins": omitted_bins,
        "selected_native_unit_ids": list(selected),
        "omitted_native_unit_ids": list(omitted),
        "allowed_visual_token_indices": allowed_tokens,
        "blocked_visual_token_indices": blocked_tokens,
        "num_allowed_visual_tokens": len(allowed_tokens),
        "num_blocked_visual_tokens": len(blocked_tokens),
        "actual_retained_visual_token_fraction": len(allowed_tokens) / float(total_visual_tokens),
    }


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
    units = native_routing_units_from_artifact(artifact, model)
    total_visual_tokens = sum(int(unit["num_visual_tokens"]) for unit in units)
    anchors = route_anchor_layers(num_layers)
    anchor_routes = {}
    layer_routes = {}
    for anchor in anchors:
        selected_units = _condition_units(
            condition,
            scores[anchor],
            units,
            question_id=question_id,
            model=model,
            seed=seed,
            anchor=anchor,
        )
        route = _route_for_units(selected_units, units, anchor, total_visual_tokens)
        anchor_routes[str(anchor)] = route
        for layer in routed_layers_for_anchor(anchor, num_layers):
            layer_routes[str(layer)] = dict(route)
    dense_layers = sorted(set(range(0, min(9, num_layers))) | set(anchors))
    retained_fractions = [route["actual_retained_visual_token_fraction"] for route in anchor_routes.values()]
    return {
        "type": "baseline_derived_causal_route_replay",
        "model": model,
        "condition": condition,
        "num_temporal_bins": NUM_TEMPORAL_BINS,
        "routing_unit_type": "qwen_native_temporal_cell" if model == "qwen" else "vila_frame_bin",
        "native_routing_units": units,
        "retention_ratio": RETENTION_RATIO,
        "retained_native_units": _retained_unit_count(units),
        "dense_prefix_through_layer": 8,
        "anchor_layers": list(anchors),
        "dense_layers": dense_layers,
        "anchor_routes": anchor_routes,
        "layer_routes": {str(layer): route for layer, route in sorted(layer_routes.items(), key=lambda item: int(item[0]))},
        "baseline_artifact": baseline_artifact,
        "seed": int(seed),
        "git_commit": git_commit,
        "actual_retained_visual_token_fraction_by_anchor": retained_fractions,
        "mean_actual_retained_visual_token_fraction": (
            sum(retained_fractions) / len(retained_fractions) if retained_fractions else None
        ),
        "mask_rule": (
            "Baseline-derived causal route replay: text-token queries are blocked from omitted native "
            "visual routing units at routed decoder layers. This is not online routing and not a measured speedup."
        ),
    }


def route_mask_summary(route_spec: Mapping[str, Any], _visual_tokens_by_bin: Mapping[int, Sequence[int]] | None = None) -> dict[str, Any]:
    layer_summaries = {}
    for raw_layer, route in sorted(route_spec.get("layer_routes", {}).items(), key=lambda item: int(item[0])):
        layer_summaries[str(int(raw_layer))] = {
            "source_anchor_layer": int(route["source_anchor_layer"]),
            "selected_bins": list(route["selected_bins"]),
            "omitted_bins": list(route["omitted_bins"]),
            "selected_native_unit_ids": list(route["selected_native_unit_ids"]),
            "omitted_native_unit_ids": list(route["omitted_native_unit_ids"]),
            "num_allowed_visual_tokens": int(route["num_allowed_visual_tokens"]),
            "num_blocked_visual_tokens": int(route["num_blocked_visual_tokens"]),
            "actual_retained_visual_token_fraction": float(route["actual_retained_visual_token_fraction"]),
        }
    return {
        "condition": route_spec.get("condition"),
        "type": route_spec.get("type"),
        "routing_unit_type": route_spec.get("routing_unit_type"),
        "retention_ratio": route_spec.get("retention_ratio"),
        "anchor_layers": list(route_spec.get("anchor_layers", [])),
        "native_routing_units": route_spec.get("native_routing_units", []),
        "layer_routes": layer_summaries,
        "baseline_artifact": route_spec.get("baseline_artifact"),
        "seed": route_spec.get("seed"),
        "git_commit": route_spec.get("git_commit"),
        "mean_actual_retained_visual_token_fraction": route_spec.get("mean_actual_retained_visual_token_fraction"),
    }


@dataclass(frozen=True)
class LayerRouteMask:
    layer_to_blocked_visual_indices: dict[int, tuple[int, ...]]
    all_visual_token_indices: tuple[int, ...]

    @classmethod
    def from_route_spec(
        cls,
        route_spec: Mapping[str, Any] | None,
        _visual_tokens_by_bin: Mapping[int, Sequence[int]] | None,
        all_visual_token_indices: Sequence[int],
    ) -> "LayerRouteMask | None":
        if not route_spec:
            return None
        blocked_by_layer: dict[int, tuple[int, ...]] = {}
        for raw_layer, route in route_spec.get("layer_routes", {}).items():
            blocked = tuple(sorted(int(token) for token in route.get("blocked_visual_token_indices", ())))
            if blocked:
                blocked_by_layer[int(raw_layer)] = blocked
        if not blocked_by_layer:
            return None
        return cls(
            layer_to_blocked_visual_indices=blocked_by_layer,
            all_visual_token_indices=tuple(int(item) for item in all_visual_token_indices),
        )

    def blocked_indices_for_layer(self, layer: int, key_len: int) -> tuple[int, ...]:
        return tuple(index for index in self.layer_to_blocked_visual_indices.get(int(layer), ()) if index < key_len)


def assert_matched_condition_budgets(route_specs: Sequence[Mapping[str, Any]], tolerance: float = TOKEN_BUDGET_TOLERANCE) -> None:
    fractions = [
        float(route["actual_retained_visual_token_fraction"])
        for spec in route_specs
        for route in spec.get("anchor_routes", {}).values()
    ]
    if not fractions:
        raise ValueError("No route budgets were available to compare.")
    if max(fractions) - min(fractions) > tolerance:
        raise ValueError(
            f"Route conditions have unmatched retained-token budgets: min={min(fractions):.4f}, max={max(fractions):.4f}."
        )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
