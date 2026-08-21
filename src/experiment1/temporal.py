from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np

from src.frame_sampling import FrameBatch

from .relevance import normalize_distribution
from .token_layout import TokenLayout


@dataclass(frozen=True)
class TemporalLayerStats:
    layer: int
    normalized_temporal_entropy: float
    top1_temporal_bin_mass: float
    bins_to_80pct_mass: int
    fraction_bins_to_80pct_mass: float
    temporal_bin_rank_order: tuple[int, ...]
    spearman_with_final_layer_ordering: float
    topk_overlap_with_final_layer: int
    topk_overlap_fraction_with_final_layer: float


@dataclass(frozen=True)
class TemporalRelevance:
    raw_temporal_bin_scores: np.ndarray
    normalized_temporal_bin_scores: np.ndarray
    absolute_question_to_visual_attention_mass: np.ndarray
    temporal_bins: tuple[dict[str, Any], ...]
    layer_metrics: tuple[TemporalLayerStats, ...]
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "raw_temporal_bin_scores": self.raw_temporal_bin_scores.tolist(),
            "normalized_temporal_bin_scores": self.normalized_temporal_bin_scores.tolist(),
            "absolute_question_to_visual_attention_mass": self.absolute_question_to_visual_attention_mass.tolist(),
            "temporal_bins": list(self.temporal_bins),
            "layer_metrics": [asdict(item) for item in self.layer_metrics],
            "metadata": self.metadata,
        }


def temporal_bin_count(layout: TokenLayout, input_index: int = 0) -> int:
    cells = [cell for cell in layout.visual_cells if cell.modality == "video" and cell.input_index == input_index]
    if not cells:
        raise ValueError(f"No video visual tokens found for input_index={input_index}.")
    grid_counts = {cell.grid_t for cell in cells}
    if len(grid_counts) != 1:
        raise ValueError(f"Expected one temporal grid size for input_index={input_index}, got {sorted(grid_counts)}.")
    return int(next(iter(grid_counts)))


def pool_token_scores_to_temporal_bins(
    token_scores: np.ndarray,
    layout: TokenLayout,
    input_index: int = 0,
) -> np.ndarray:
    raw = np.asarray(token_scores, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"token_scores must have shape [layers, visual_tokens], got {raw.shape}.")
    num_bins = temporal_bin_count(layout, input_index=input_index)
    temporal = np.zeros((raw.shape[0], num_bins), dtype=np.float64)
    for cell in layout.visual_cells:
        if cell.modality != "video" or cell.input_index != input_index:
            continue
        temporal[:, cell.temporal_index] += raw[:, cell.visual_index]
    return temporal


def represented_sampled_frames(batch: FrameBatch, temporal_index: int, grid_t: int) -> dict[str, Any]:
    if grid_t <= 0:
        return {"sampled_frame_indices": [], "sampled_timestamps": []}
    count = len(batch.frame_indices)
    start = int(round(temporal_index * count / grid_t))
    end = int(round((temporal_index + 1) * count / grid_t))
    if end <= start:
        end = min(count, start + 1)
    return {
        "sampled_frame_indices": list(batch.frame_indices[start:end]),
        "sampled_timestamps": list(batch.timestamps[start:end]),
        "note": (
            "Qwen temporal bins can represent multiple sampled frames; "
            f"with {count} sampled frames and {grid_t} bins, this bin represents {end - start} sampled frame(s)."
        ),
    }


def temporal_bin_metadata(layout: TokenLayout, frame_batches: Sequence[FrameBatch], input_index: int = 0) -> tuple[dict[str, Any], ...]:
    num_bins = temporal_bin_count(layout, input_index=input_index)
    batch = frame_batches[input_index]
    metadata: list[dict[str, Any]] = []
    for temporal_index in range(num_bins):
        cells = [
            cell
            for cell in layout.visual_cells
            if cell.modality == "video" and cell.input_index == input_index and cell.temporal_index == temporal_index
        ]
        if not cells:
            raise ValueError(f"Temporal bin {temporal_index} has no visual tokens.")
        first = cells[0]
        metadata.append(
            {
                "input_index": input_index,
                "temporal_bin": temporal_index,
                "grid_t": first.grid_t,
                "grid_h": first.grid_h,
                "grid_w": first.grid_w,
                "num_visual_tokens": len(cells),
                "seconds_per_grid": first.seconds_per_grid,
                "qwen_timestamp": first.timestamp,
                **represented_sampled_frames(batch, temporal_index, first.grid_t),
            }
        )
    return tuple(metadata)


def normalized_temporal_entropy(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("normalized_temporal_entropy expects a 1D score vector.")
    if values.size <= 1:
        return 0.0
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    norm = values / total
    nonzero = norm[norm > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / math.log(values.size))


def top1_mass(scores: Sequence[float]) -> float:
    norm = normalize_distribution(np.asarray(scores, dtype=np.float64), axis=0)
    return float(norm.max()) if norm.size else 0.0


def bins_to_attention_mass(scores: Sequence[float], target_mass: float = 0.8) -> tuple[int, float]:
    if not 0.0 < target_mass <= 1.0:
        raise ValueError("target_mass must be in (0, 1].")
    values = np.asarray(scores, dtype=np.float64)
    total = float(values.sum())
    if values.size == 0 or total <= 0.0:
        return 0, 0.0
    norm = values / total
    cumulative = np.cumsum(np.sort(norm)[::-1])
    count = int(np.searchsorted(cumulative, target_mass, side="left") + 1)
    return count, float(count / norm.size)


def temporal_rank_order(scores: Sequence[float]) -> tuple[int, ...]:
    values = np.asarray(scores, dtype=np.float64)
    return tuple(int(index) for index in np.lexsort((np.arange(values.size), -values)))


def _rank_positions(order: Sequence[int]) -> np.ndarray:
    ranks = np.zeros(len(order), dtype=np.float64)
    for rank, index in enumerate(order):
        ranks[int(index)] = float(rank)
    return ranks


def spearman_rank_correlation(order: Sequence[int], final_order: Sequence[int]) -> float:
    if len(order) != len(final_order):
        raise ValueError("Rank orderings must have the same length.")
    if len(order) <= 1:
        return 1.0
    left = _rank_positions(order)
    right = _rank_positions(final_order)
    left -= left.mean()
    right -= right.mean()
    denom = float(np.sqrt((left * left).sum() * (right * right).sum()))
    return float((left * right).sum() / denom) if denom > 0 else 0.0


def topk_overlap(order: Sequence[int], final_order: Sequence[int], k: int) -> tuple[int, float]:
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, len(order), len(final_order))
    if k == 0:
        return 0, 0.0
    overlap = len(set(order[:k]) & set(final_order[:k]))
    return overlap, float(overlap / k)


def build_temporal_relevance_from_token_scores(
    token_scores: np.ndarray,
    layout: TokenLayout,
    frame_batches: Sequence[FrameBatch],
    extraction_method: str,
    input_index: int = 0,
    topk: int = 3,
) -> TemporalRelevance:
    raw_temporal = pool_token_scores_to_temporal_bins(token_scores, layout, input_index=input_index)
    normalized_temporal = normalize_distribution(raw_temporal, axis=1)
    absolute_mass = np.asarray(token_scores, dtype=np.float64).sum(axis=1)
    final_order = temporal_rank_order(normalized_temporal[-1])
    metrics: list[TemporalLayerStats] = []
    for layer_index, layer_scores in enumerate(normalized_temporal):
        order = temporal_rank_order(layer_scores)
        count80, fraction80 = bins_to_attention_mass(layer_scores, 0.8)
        overlap, overlap_fraction = topk_overlap(order, final_order, topk)
        metrics.append(
            TemporalLayerStats(
                layer=layer_index,
                normalized_temporal_entropy=normalized_temporal_entropy(layer_scores),
                top1_temporal_bin_mass=top1_mass(layer_scores),
                bins_to_80pct_mass=count80,
                fraction_bins_to_80pct_mass=fraction80,
                temporal_bin_rank_order=order,
                spearman_with_final_layer_ordering=spearman_rank_correlation(order, final_order),
                topk_overlap_with_final_layer=overlap,
                topk_overlap_fraction_with_final_layer=overlap_fraction,
            )
        )
    return TemporalRelevance(
        raw_temporal_bin_scores=raw_temporal,
        normalized_temporal_bin_scores=normalized_temporal,
        absolute_question_to_visual_attention_mass=absolute_mass,
        temporal_bins=temporal_bin_metadata(layout, frame_batches, input_index=input_index),
        layer_metrics=tuple(metrics),
        metadata={
            "num_layers": int(raw_temporal.shape[0]),
            "num_temporal_bins": int(raw_temporal.shape[1]),
            "num_visual_tokens": int(token_scores.shape[1]),
            "num_question_tokens": len(layout.question_token_indices),
            "query_scope": layout.query_scope,
            "extraction_method": extraction_method,
            "input_index": input_index,
            "topk": int(topk),
        },
    )
