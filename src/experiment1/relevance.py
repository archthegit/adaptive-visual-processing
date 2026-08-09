from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

from .token_layout import TokenLayout


@dataclass(frozen=True)
class ConcentrationStats:
    top1_frame_mass: float
    top3_frame_mass: float
    normalized_entropy: float


@dataclass(frozen=True)
class LayerwiseRelevance:
    raw_token_scores: np.ndarray
    normalized_token_scores: np.ndarray
    raw_frame_scores: np.ndarray
    normalized_frame_scores: np.ndarray
    raw_spatial_scores: np.ndarray
    normalized_spatial_scores: np.ndarray
    aggregate_frame_scores: np.ndarray
    cumulative_frame_curve: np.ndarray
    concentration_by_layer: tuple[ConcentrationStats, ...]
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "raw_token_scores": self.raw_token_scores.tolist(),
            "normalized_token_scores": self.normalized_token_scores.tolist(),
            "raw_frame_scores": self.raw_frame_scores.tolist(),
            "normalized_frame_scores": self.normalized_frame_scores.tolist(),
            "raw_spatial_scores": self.raw_spatial_scores.tolist(),
            "normalized_spatial_scores": self.normalized_spatial_scores.tolist(),
            "aggregate_frame_scores": self.aggregate_frame_scores.tolist(),
            "cumulative_frame_curve": self.cumulative_frame_curve.tolist(),
            "concentration_by_layer": [asdict(item) for item in self.concentration_by_layer],
            "metadata": self.metadata,
        }


def normalize_distribution(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    totals = values.sum(axis=axis, keepdims=True)
    return np.divide(values, totals + eps, out=np.zeros_like(values), where=totals > eps)


def _attention_array(attention: Any) -> np.ndarray:
    if hasattr(attention, "detach"):
        attention = attention.detach().cpu().numpy()
    arr = np.asarray(attention, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected attention shape [heads, query, key] or [batch, heads, query, key], got {arr.shape}.")
    return arr


def aggregate_question_to_visual_attention(
    attentions: Sequence[Any],
    question_token_indices: Sequence[int],
    visual_token_indices: Sequence[int],
) -> np.ndarray:
    if not question_token_indices:
        raise ValueError("question_token_indices is empty.")
    if not visual_token_indices:
        raise ValueError("visual_token_indices is empty.")
    per_layer = []
    for attention in attentions:
        arr = _attention_array(attention)
        qv = arr[:, list(question_token_indices), :][:, :, list(visual_token_indices)]
        per_layer.append(qv.mean(axis=(0, 1)))
    return np.stack(per_layer, axis=0)


def token_scores_to_frame_scores(token_scores: np.ndarray, layout: TokenLayout) -> np.ndarray:
    max_t = max(cell.temporal_index for cell in layout.visual_cells) + 1
    frame_scores = np.zeros((token_scores.shape[0], max_t), dtype=np.float64)
    for cell in layout.visual_cells:
        frame_scores[:, cell.temporal_index] += token_scores[:, cell.visual_index]
    return frame_scores


def token_scores_to_spatial_scores(token_scores: np.ndarray, layout: TokenLayout) -> np.ndarray:
    max_t = max(cell.temporal_index for cell in layout.visual_cells) + 1
    max_h = max(cell.spatial_y for cell in layout.visual_cells) + 1
    max_w = max(cell.spatial_x for cell in layout.visual_cells) + 1
    spatial = np.zeros((token_scores.shape[0], max_t, max_h, max_w), dtype=np.float64)
    for cell in layout.visual_cells:
        spatial[:, cell.temporal_index, cell.spatial_y, cell.spatial_x] += token_scores[:, cell.visual_index]
    return spatial


def cumulative_curve(frame_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(frame_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("cumulative_curve expects a 1D frame score array.")
    norm = normalize_distribution(scores, axis=0)
    ranked = np.sort(norm)[::-1]
    return np.cumsum(ranked)


def concentration_stats(frame_scores: np.ndarray) -> ConcentrationStats:
    norm = normalize_distribution(frame_scores, axis=0)
    ranked = np.sort(norm)[::-1]
    entropy = 0.0
    nonzero = norm[norm > 0]
    if len(nonzero) > 0 and len(norm) > 1:
        entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(len(norm)))
    return ConcentrationStats(
        top1_frame_mass=float(ranked[:1].sum()) if len(ranked) else 0.0,
        top3_frame_mass=float(ranked[:3].sum()) if len(ranked) else 0.0,
        normalized_entropy=entropy,
    )


def compute_layerwise_relevance(attentions: Sequence[Any], layout: TokenLayout) -> LayerwiseRelevance:
    raw_token = aggregate_question_to_visual_attention(
        attentions, layout.question_token_indices, layout.visual_token_indices
    )
    norm_token = normalize_distribution(raw_token, axis=1)
    raw_frame = token_scores_to_frame_scores(raw_token, layout)
    norm_frame = normalize_distribution(raw_frame, axis=1)
    raw_spatial = token_scores_to_spatial_scores(raw_token, layout)
    flat_spatial = raw_spatial.reshape(raw_spatial.shape[0], -1)
    norm_spatial = normalize_distribution(flat_spatial, axis=1).reshape(raw_spatial.shape)
    aggregate_frame = normalize_distribution(raw_frame.mean(axis=0), axis=0)
    return LayerwiseRelevance(
        raw_token_scores=raw_token,
        normalized_token_scores=norm_token,
        raw_frame_scores=raw_frame,
        normalized_frame_scores=norm_frame,
        raw_spatial_scores=raw_spatial,
        normalized_spatial_scores=norm_spatial,
        aggregate_frame_scores=aggregate_frame,
        cumulative_frame_curve=cumulative_curve(aggregate_frame),
        concentration_by_layer=tuple(concentration_stats(layer) for layer in raw_frame),
        metadata={
            "num_layers": int(raw_token.shape[0]),
            "num_visual_tokens": int(raw_token.shape[1]),
            "num_question_tokens": len(layout.question_token_indices),
            "query_scope": layout.query_scope,
        },
    )
