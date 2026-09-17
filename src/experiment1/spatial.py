from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import numpy as np

from .route_reuse import current_git_commit, routed_layers_for_anchor
from .temporal import represented_sampled_frames
from .token_layout import TokenLayout


SPATIAL_SCHEMA_VERSION = "spatial_relevance_v1"
SPATIAL_ROUTE_SCHEMA_VERSION = "spatial_route_v1"
SPATIAL_ROUTE_SEED = 20260917
SPATIAL_ROUTE_CONDITIONS = {
    "adaptive_spatial_top50": "spatial_route_gap4_top50",
    "random_spatial_top50": "spatial_random_gap4_top50",
    "uniform_spatial_top50": "spatial_uniform_gap4_top50",
}
SPATIAL_ROUTE_ANCHORS = (8, 12, 16, 20, 24)
SPATIAL_ROUTE_GAP = 4
SPATIAL_ROUTE_TARGETS = (9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sums = arr.sum(axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float64)
    np.divide(arr, sums, out=out, where=sums > 0.0)
    return out


def visual_token_mappings(layout: TokenLayout, input_index: int = 0) -> list[dict[str, Any]]:
    mappings = []
    for cell in layout.visual_cells:
        if cell.modality != "video" or cell.input_index != input_index:
            continue
        mappings.append(
            {
                "token_index": int(cell.token_index),
                "visual_index": int(cell.visual_index),
                "input_index": int(cell.input_index),
                "temporal_index": int(cell.temporal_index),
                "spatial_y": int(cell.spatial_y),
                "spatial_x": int(cell.spatial_x),
                "grid_h": int(cell.grid_h),
                "grid_w": int(cell.grid_w),
            }
        )
    mappings.sort(key=lambda item: item["visual_index"])
    return mappings


def native_temporal_cell_metadata(layout: TokenLayout, frame_batches: Sequence[Any] | None = None, input_index: int = 0) -> dict[str, Any]:
    mappings = visual_token_mappings(layout, input_index=input_index)
    native_cells = sorted({int(item["temporal_index"]) for item in mappings})
    batch = frame_batches[input_index] if frame_batches is not None and input_index < len(frame_batches) else None
    sampled_input_frames = [int(item) for item in getattr(batch, "frame_indices", ())] if batch is not None else []
    sampled_by_cell: dict[str, list[int]] = {}
    for temporal_index in native_cells:
        if batch is not None:
            represented = represented_sampled_frames(
                batch,
                temporal_index,
                max(native_cells) + 1 if native_cells else 0,
            )
            sampled_by_cell[str(temporal_index)] = [int(item) for item in represented.get("sampled_frame_indices", [])]
        else:
            sampled_by_cell[str(temporal_index)] = []
    return {
        "sampled_input_frames": sampled_input_frames,
        "native_temporal_cells": native_cells,
        "sampled_frames_by_native_temporal_cell": sampled_by_cell,
    }


def build_spatial_relevance_from_token_scores(
    token_scores: np.ndarray,
    layout: TokenLayout,
    extraction_method: str,
    frame_batches: Sequence[Any] | None = None,
    input_index: int = 0,
) -> dict[str, Any]:
    raw = np.asarray(token_scores, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"token_scores must have shape [layers, visual_tokens], got {raw.shape}.")
    mappings = visual_token_mappings(layout, input_index=input_index)
    if raw.shape[1] != len(mappings):
        raise ValueError(f"Every stored score must map to exactly one visual token: scores={raw.shape[1]}, mappings={len(mappings)}.")
    if not np.isfinite(raw).all():
        raise ValueError("Spatial token scores contain non-finite values.")

    normalized_global = normalize_rows(raw)
    native_metadata = native_temporal_cell_metadata(layout, frame_batches, input_index=input_index)
    temporal_indices = list(native_metadata["native_temporal_cells"])
    by_frame: list[list[list[float]]] = []
    for layer_idx in range(raw.shape[0]):
        layer_frames = []
        for temporal_index in temporal_indices:
            visual_indices = [int(item["visual_index"]) for item in mappings if int(item["temporal_index"]) == temporal_index]
            values = raw[layer_idx, visual_indices]
            total = float(values.sum())
            normalized = (values / total).tolist() if total > 0.0 else [0.0 for _ in visual_indices]
            if visual_indices and not np.isclose(sum(normalized), 1.0, atol=1e-8):
                raise ValueError(f"Per-frame spatial distribution does not sum to one for layer {layer_idx}, frame {temporal_index}.")
            layer_frames.append(normalized)
        by_frame.append(layer_frames)

    if normalized_global.shape[0] != raw.shape[0] or any(len(row) != raw.shape[1] for row in normalized_global):
        raise ValueError("Every layer must have the same visual-token count.")
    if not np.isfinite(normalized_global).all():
        raise ValueError("Normalized spatial token scores contain non-finite values.")

    return {
        "schema_version": SPATIAL_SCHEMA_VERSION,
        "extraction_method": extraction_method,
        "normalized_token_scores": normalized_global.tolist(),
        "absolute_spatial_visual_attention_mass": raw.sum(axis=1).tolist(),
        "visual_token_mappings": mappings,
        "sampled_input_frames": native_metadata["sampled_input_frames"],
        "native_temporal_cells": native_metadata["native_temporal_cells"],
        "sampled_frames_by_native_temporal_cell": native_metadata["sampled_frames_by_native_temporal_cell"],
        "normalized_spatial_distribution_by_temporal_frame": by_frame,
        "metadata": {
            "num_layers": int(raw.shape[0]),
            "num_visual_tokens": int(raw.shape[1]),
            "num_native_temporal_cells": int(len(temporal_indices)),
            "global_spatial_distribution_field": "normalized_token_scores",
            "input_index": int(input_index),
            "query_scope": layout.query_scope,
        },
    }


def assert_spatial_relevance_valid(spatial: Mapping[str, Any]) -> None:
    scores = np.asarray(spatial.get("normalized_token_scores"), dtype=np.float64)
    mappings = spatial.get("visual_token_mappings") or []
    if scores.ndim != 2:
        raise ValueError("normalized_token_scores must have shape [layers, visual_tokens].")
    if scores.shape[1] != len(mappings):
        raise ValueError("Every normalized token score must map to exactly one visual token.")
    if not np.isfinite(scores).all():
        raise ValueError("Spatial relevance contains non-finite token scores.")
    for layer_idx, frames in enumerate(spatial.get("normalized_spatial_distribution_by_temporal_frame") or []):
        for frame_idx, distribution in enumerate(frames):
            values = np.asarray(distribution, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Spatial per-frame distribution contains non-finite values.")
            if values.size and not np.isclose(values.sum(), 1.0, atol=1e-8):
                raise ValueError(f"Spatial per-frame distribution does not sum to one at layer {layer_idx}, frame {frame_idx}.")
    if scores.shape[0] != int((spatial.get("metadata") or {}).get("num_layers", scores.shape[0])):
        raise ValueError("Spatial relevance layer count metadata is inconsistent.")
    if spatial.get("normalized_global_spatial_distribution") is not None:
        raise ValueError("Do not serialize duplicate normalized_global_spatial_distribution; use normalized_token_scores.")
    native_cells = [int(item) for item in spatial.get("native_temporal_cells", ())]
    sampled_by_cell = spatial.get("sampled_frames_by_native_temporal_cell") or {}
    if set(str(item) for item in native_cells) != set(sampled_by_cell):
        raise ValueError("Spatial relevance native cell metadata is incomplete.")


def _frame_groups(mappings: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for mapping in mappings:
        groups.setdefault(int(mapping["temporal_index"]), []).append(dict(mapping))
    for values in groups.values():
        values.sort(key=lambda item: (int(item["spatial_y"]), int(item["spatial_x"]), int(item["visual_index"])))
    return dict(sorted(groups.items()))


def _retained_count(frame_tokens: Sequence[Mapping[str, Any]]) -> int:
    count = len(frame_tokens)
    if count <= 0 or count % 2 != 0:
        raise ValueError(f"Spatial top50 routing requires an even nonzero token count per frame, got {count}.")
    return count // 2


def _uniform_frame_selection(frame_tokens: Sequence[Mapping[str, Any]], count: int) -> list[int]:
    ordered = sorted(
        frame_tokens,
        key=lambda item: (
            (int(item["spatial_y"]) + int(item["spatial_x"])) % 2,
            -min(
                int(item["spatial_y"]),
                int(item["spatial_x"]),
                int(item["grid_h"]) - 1 - int(item["spatial_y"]),
                int(item["grid_w"]) - 1 - int(item["spatial_x"]),
            ),
            int(item["spatial_y"]),
            int(item["spatial_x"]),
            int(item["visual_index"]),
        ),
    )
    return sorted(int(item["token_index"]) for item in ordered[:count])


def _select_frame_tokens(
    *,
    condition: str,
    question_id: str,
    anchor_layer: int,
    frame_tokens: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    seed: int,
) -> list[int]:
    count = _retained_count(frame_tokens)
    if condition == "spatial_route_gap4_top50":
        ordered = sorted(
            frame_tokens,
            key=lambda item: (-float(scores[int(item["visual_index"])]), int(item["spatial_y"]), int(item["spatial_x"]), int(item["visual_index"])),
        )
        return sorted(int(item["token_index"]) for item in ordered[:count])
    if condition == "spatial_random_gap4_top50":
        rng = random.Random(f"{seed}:{question_id}:{anchor_layer}:{frame_tokens[0]['temporal_index']}")
        return sorted(int(item["token_index"]) for item in rng.sample(list(frame_tokens), count))
    if condition == "spatial_uniform_gap4_top50":
        return _uniform_frame_selection(frame_tokens, count)
    raise ValueError(f"Unsupported spatial route condition: {condition}")


def _route_for_allowed(
    allowed_tokens: Sequence[int],
    all_tokens: Sequence[int],
    source_anchor_layer: int,
) -> dict[str, Any]:
    allowed = sorted({int(item) for item in allowed_tokens})
    all_set = set(int(item) for item in all_tokens)
    blocked = sorted(all_set - set(allowed))
    if set(allowed) & set(blocked):
        raise ValueError("Spatial route allowed and blocked tokens overlap.")
    if set(allowed) | set(blocked) != all_set:
        raise ValueError("Spatial route allowed and blocked tokens do not cover all visual tokens.")
    return {
        "source_anchor_layer": int(source_anchor_layer),
        "allowed_visual_token_indices": allowed,
        "blocked_visual_token_indices": blocked,
        "num_allowed_visual_tokens": len(allowed),
        "num_blocked_visual_tokens": len(blocked),
        "actual_retained_visual_token_fraction": len(allowed) / float(len(all_set)),
    }


def spatial_route_spec_from_baseline_artifact(
    artifact: Mapping[str, Any],
    *,
    condition: str,
    baseline_artifact: str,
    seed: int = SPATIAL_ROUTE_SEED,
    git_commit: str | None = None,
) -> dict[str, Any]:
    if condition not in SPATIAL_ROUTE_CONDITIONS.values():
        raise ValueError(f"Unsupported spatial route condition: {condition}")
    spatial = artifact.get("spatial_relevance")
    if not isinstance(spatial, Mapping):
        raise ValueError(f"{artifact.get('question_id')}: dense artifact has no spatial_relevance object.")
    assert_spatial_relevance_valid(spatial)
    scores = np.asarray(spatial["normalized_token_scores"], dtype=np.float64)
    mappings = list(spatial["visual_token_mappings"])
    question_id = str(artifact["question_id"])
    all_tokens = [int(item["token_index"]) for item in mappings]
    frame_groups = _frame_groups(mappings)
    native_meta = {
        "sampled_input_frames": [int(item) for item in spatial.get("sampled_input_frames", [])],
        "native_temporal_cells": [int(item) for item in spatial.get("native_temporal_cells", sorted(frame_groups))],
        "sampled_frames_by_native_temporal_cell": {
            str(key): [int(item) for item in value]
            for key, value in (spatial.get("sampled_frames_by_native_temporal_cell") or {}).items()
        },
    }
    if set(native_meta["native_temporal_cells"]) != set(frame_groups):
        raise ValueError("Spatial relevance native temporal cells do not match visual-token mappings.")
    per_cell_budget = {str(frame): _retained_count(tokens) for frame, tokens in frame_groups.items()}
    anchors = tuple(anchor for anchor in SPATIAL_ROUTE_ANCHORS if anchor < scores.shape[0] and routed_layers_for_anchor(anchor, scores.shape[0], SPATIAL_ROUTE_GAP))
    anchor_routes: dict[str, Any] = {}
    layer_routes: dict[str, Any] = {}
    for anchor in anchors:
        allowed: list[int] = []
        selected_by_cell: dict[str, list[int]] = {}
        for frame, frame_tokens in frame_groups.items():
            selected = _select_frame_tokens(
                condition=condition,
                question_id=question_id,
                anchor_layer=anchor,
                frame_tokens=frame_tokens,
                scores=scores[anchor],
                seed=seed,
            )
            selected_by_cell[str(frame)] = selected
            allowed.extend(selected)
        route = _route_for_allowed(allowed, all_tokens, anchor)
        route["selected_visual_tokens_by_native_temporal_cell"] = selected_by_cell
        route["per_native_temporal_cell_retained_visual_tokens"] = dict(per_cell_budget)
        anchor_routes[str(anchor)] = route
        for layer in routed_layers_for_anchor(anchor, scores.shape[0], SPATIAL_ROUTE_GAP):
            layer_routes[str(layer)] = dict(route)
    retained_fractions = [float(route["actual_retained_visual_token_fraction"]) for route in anchor_routes.values()]
    return {
        "type": "baseline_derived_spatial_route_replay",
        "schema_version": SPATIAL_ROUTE_SCHEMA_VERSION,
        "condition": condition,
        "route_family": "spatial",
        "model": "qwen",
        "retention_ratio": 0.5,
        "refresh_gap": SPATIAL_ROUTE_GAP,
        "anchor_layers": list(anchors),
        "dense_prefix_through_layer": 8,
        "anchor_routes": anchor_routes,
        "layer_routes": {str(layer): route for layer, route in sorted(layer_routes.items(), key=lambda item: int(item[0]))},
        "sampled_input_frames": native_meta["sampled_input_frames"],
        "native_temporal_cells": native_meta["native_temporal_cells"],
        "sampled_frames_by_native_temporal_cell": native_meta["sampled_frames_by_native_temporal_cell"],
        "native_temporal_cells_preserved": sorted(int(frame) for frame in frame_groups),
        "per_native_temporal_cell_retained_visual_tokens": dict(per_cell_budget),
        "baseline_artifact": baseline_artifact,
        "seed": int(seed),
        "git_commit": git_commit,
        "actual_retained_visual_token_fraction_by_anchor": retained_fractions,
        "mean_actual_retained_visual_token_fraction": sum(retained_fractions) / len(retained_fractions) if retained_fractions else None,
        "mask_rule": (
            "Baseline-derived spatial route replay: text-token queries are blocked from omitted explicit visual-token "
            "indices at routed decoder layers. All sampled input frames remain present; every Qwen native temporal "
            "cell retains its own 50% spatial-token budget."
        ),
    }


def validate_spatial_route_spec(route: Mapping[str, Any], baseline_visual_token_indices: Sequence[int] | None = None) -> None:
    if route.get("type") != "baseline_derived_spatial_route_replay":
        raise ValueError("Spatial route type must be baseline_derived_spatial_route_replay.")
    if route.get("condition") not in SPATIAL_ROUTE_CONDITIONS.values():
        raise ValueError(f"Unsupported spatial route condition: {route.get('condition')}.")
    if route.get("route_family") != "spatial":
        raise ValueError("Spatial route must not overload temporal route semantics.")
    anchors = tuple(int(item) for item in route.get("anchor_layers", ()))
    if anchors != SPATIAL_ROUTE_ANCHORS:
        raise ValueError(f"Spatial route anchors must be exactly {SPATIAL_ROUTE_ANCHORS}, got {anchors}.")
    targets = tuple(sorted(int(layer) for layer in route.get("layer_routes", {})))
    if targets != SPATIAL_ROUTE_TARGETS:
        raise ValueError(f"Spatial routed targets must be exactly {SPATIAL_ROUTE_TARGETS}, got {targets}.")
    native_cells = set(int(item) for item in route.get("native_temporal_cells_preserved", ()))
    if not native_cells:
        raise ValueError("Spatial route must preserve at least one native temporal cell.")
    declared_native = set(int(item) for item in route.get("native_temporal_cells", ()))
    if declared_native != native_cells:
        raise ValueError("Spatial route native_temporal_cells and native_temporal_cells_preserved differ.")
    sampled_by_cell = route.get("sampled_frames_by_native_temporal_cell") or {}
    if set(str(item) for item in native_cells) != set(sampled_by_cell):
        raise ValueError("Spatial route must record sampled frames for every native temporal cell.")
    sampled_input_frames = set(int(item) for item in route.get("sampled_input_frames", ()))
    represented = {int(frame) for frames in sampled_by_cell.values() for frame in frames}
    if sampled_input_frames and represented != sampled_input_frames:
        raise ValueError("Spatial route sampled-frame mapping does not cover exactly the sampled input frames.")
    budgets = {int(k): int(v) for k, v in (route.get("per_native_temporal_cell_retained_visual_tokens") or {}).items()}
    if set(budgets) != native_cells:
        raise ValueError("Spatial route per-native-cell budgets must match preserved native temporal cells.")
    baseline_tokens = set(int(item) for item in (baseline_visual_token_indices or ()))
    for layer, layer_route in (route.get("layer_routes") or {}).items():
        layer_int = int(layer)
        source_anchor = int(layer_route.get("source_anchor_layer", -1))
        if layer_int <= source_anchor or layer_int - source_anchor not in {1, 2, 3}:
            raise ValueError(f"Spatial route layer {layer} is not causally after its source anchor.")
        by_cell = {int(k): [int(token) for token in v] for k, v in (layer_route.get("selected_visual_tokens_by_native_temporal_cell") or {}).items()}
        if set(by_cell) != native_cells:
            raise ValueError(f"Spatial route layer {layer} dropped a native temporal cell.")
        selected_union: list[int] = []
        for frame, tokens in by_cell.items():
            if len(tokens) != len(set(tokens)):
                raise ValueError(f"Spatial route layer {layer} has duplicate selected tokens in cell {frame}.")
            if len(tokens) != budgets[frame]:
                raise ValueError(f"Spatial route layer {layer} has wrong budget for native cell {frame}.")
            selected_union.extend(tokens)
        if len(selected_union) != len(set(selected_union)):
            raise ValueError(f"Spatial route layer {layer} has duplicate selected tokens across native cells.")
        allowed = set(int(item) for item in layer_route.get("allowed_visual_token_indices", ()))
        blocked = set(int(item) for item in layer_route.get("blocked_visual_token_indices", ()))
        if set(selected_union) != allowed:
            raise ValueError(f"Spatial route layer {layer} selected-token union does not equal allowed tokens.")
        if allowed & blocked:
            raise ValueError(f"Spatial route layer {layer} allowed/blocked tokens overlap.")
        if baseline_tokens and allowed | blocked != baseline_tokens:
            raise ValueError(f"Spatial route layer {layer} does not cover the dense visual-token set.")
        if baseline_tokens:
            for token in set(selected_union) | allowed | blocked:
                if token not in baseline_tokens:
                    raise ValueError(f"Spatial route layer {layer} references invalid dense visual token {token}.")
        if len(allowed) != int(layer_route.get("num_allowed_visual_tokens", -1)):
            raise ValueError(f"Spatial route layer {layer} allowed token count is inconsistent.")
        if len(blocked) != int(layer_route.get("num_blocked_visual_tokens", -1)):
            raise ValueError(f"Spatial route layer {layer} blocked token count is inconsistent.")
        if not np.isclose(float(layer_route.get("actual_retained_visual_token_fraction", -1.0)), 0.5, atol=1e-9):
            raise ValueError(f"Spatial route layer {layer} does not retain exactly 50% of visual tokens.")


def current_spatial_git_commit() -> str | None:
    return current_git_commit()
