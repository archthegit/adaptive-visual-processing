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
    absolute_visual_mass_by_layer: np.ndarray
    raw_frame_scores: np.ndarray
    normalized_frame_scores: np.ndarray
    raw_frame_scores_by_input: np.ndarray
    normalized_frame_scores_by_input: np.ndarray
    raw_spatial_scores: np.ndarray
    normalized_spatial_scores: np.ndarray
    raw_spatial_scores_by_input: np.ndarray
    normalized_spatial_scores_by_input: np.ndarray
    aggregate_frame_scores: np.ndarray
    cumulative_frame_curve: np.ndarray
    concentration_by_layer: tuple[ConcentrationStats, ...]
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "raw_token_scores": self.raw_token_scores.tolist(),
            "normalized_token_scores": self.normalized_token_scores.tolist(),
            "absolute_visual_mass_by_layer": self.absolute_visual_mass_by_layer.tolist(),
            "raw_frame_scores": self.raw_frame_scores.tolist(),
            "normalized_frame_scores": self.normalized_frame_scores.tolist(),
            "raw_frame_scores_by_input": self.raw_frame_scores_by_input.tolist(),
            "normalized_frame_scores_by_input": self.normalized_frame_scores_by_input.tolist(),
            "raw_spatial_scores": self.raw_spatial_scores.tolist(),
            "normalized_spatial_scores": self.normalized_spatial_scores.tolist(),
            "raw_spatial_scores_by_input": self.raw_spatial_scores_by_input.tolist(),
            "normalized_spatial_scores_by_input": self.normalized_spatial_scores_by_input.tolist(),
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
        attention = attention.detach()
        if hasattr(attention, "float"):
            attention = attention.float()
        attention = attention.cpu().numpy()
    arr = np.asarray(attention, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected attention shape [heads, query, key] or [batch, heads, query, key], got {arr.shape}.")
    return arr


def _aggregate_single_attention(
    attention: Any,
    question_token_indices: Sequence[int],
    visual_token_indices: Sequence[int],
) -> np.ndarray:
    if hasattr(attention, "detach"):
        tensor = attention.detach()
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        if len(tensor.shape) != 3:
            raise ValueError(
                "Expected attention shape [heads, query, key] or "
                f"[batch, heads, query, key], got {tuple(tensor.shape)}."
            )
        qv = tensor[:, list(question_token_indices), :][:, :, list(visual_token_indices)]
        if hasattr(qv, "float"):
            qv = qv.float()
        return qv.mean(dim=(0, 1)).cpu().numpy().astype(np.float64)

    arr = _attention_array(attention)
    qv = arr[:, list(question_token_indices), :][:, :, list(visual_token_indices)]
    return np.asarray(qv.mean(axis=(0, 1)), dtype=np.float64)


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
        per_layer.append(_aggregate_single_attention(attention, question_token_indices, visual_token_indices))
    return np.stack(per_layer, axis=0)


def token_scores_to_frame_scores(token_scores: np.ndarray, layout: TokenLayout) -> np.ndarray:
    max_t = max(cell.temporal_index for cell in layout.visual_cells) + 1
    frame_scores = np.zeros((token_scores.shape[0], max_t), dtype=np.float64)
    for cell in layout.visual_cells:
        frame_scores[:, cell.temporal_index] += token_scores[:, cell.visual_index]
    return frame_scores


def token_scores_to_frame_scores_by_input(token_scores: np.ndarray, layout: TokenLayout) -> np.ndarray:
    max_input = max(cell.input_index for cell in layout.visual_cells) + 1
    max_t = max(cell.temporal_index for cell in layout.visual_cells) + 1
    frame_scores = np.zeros((token_scores.shape[0], max_input, max_t), dtype=np.float64)
    for cell in layout.visual_cells:
        frame_scores[:, cell.input_index, cell.temporal_index] += token_scores[:, cell.visual_index]
    return frame_scores


def token_scores_to_spatial_scores(token_scores: np.ndarray, layout: TokenLayout) -> np.ndarray:
    max_t = max(cell.temporal_index for cell in layout.visual_cells) + 1
    max_h = max(cell.spatial_y for cell in layout.visual_cells) + 1
    max_w = max(cell.spatial_x for cell in layout.visual_cells) + 1
    spatial = np.zeros((token_scores.shape[0], max_t, max_h, max_w), dtype=np.float64)
    for cell in layout.visual_cells:
        spatial[:, cell.temporal_index, cell.spatial_y, cell.spatial_x] += token_scores[:, cell.visual_index]
    return spatial


def token_scores_to_spatial_scores_by_input(token_scores: np.ndarray, layout: TokenLayout) -> np.ndarray:
    max_input = max(cell.input_index for cell in layout.visual_cells) + 1
    max_t = max(cell.temporal_index for cell in layout.visual_cells) + 1
    max_h = max(cell.spatial_y for cell in layout.visual_cells) + 1
    max_w = max(cell.spatial_x for cell in layout.visual_cells) + 1
    spatial = np.zeros((token_scores.shape[0], max_input, max_t, max_h, max_w), dtype=np.float64)
    for cell in layout.visual_cells:
        spatial[:, cell.input_index, cell.temporal_index, cell.spatial_y, cell.spatial_x] += token_scores[:, cell.visual_index]
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


def build_layerwise_relevance_from_token_scores(
    token_scores: np.ndarray,
    layout: TokenLayout,
    extraction_method: str,
) -> LayerwiseRelevance:
    raw_token = np.asarray(token_scores, dtype=np.float64)
    norm_token = normalize_distribution(raw_token, axis=1)
    absolute_visual_mass = raw_token.sum(axis=1)
    raw_frame = token_scores_to_frame_scores(raw_token, layout)
    norm_frame = normalize_distribution(raw_frame, axis=1)
    raw_frame_by_input = token_scores_to_frame_scores_by_input(raw_token, layout)
    flat_frame_by_input = raw_frame_by_input.reshape(raw_frame_by_input.shape[0], -1)
    norm_frame_by_input = normalize_distribution(flat_frame_by_input, axis=1).reshape(raw_frame_by_input.shape)
    raw_spatial = token_scores_to_spatial_scores(raw_token, layout)
    flat_spatial = raw_spatial.reshape(raw_spatial.shape[0], -1)
    norm_spatial = normalize_distribution(flat_spatial, axis=1).reshape(raw_spatial.shape)
    raw_spatial_by_input = token_scores_to_spatial_scores_by_input(raw_token, layout)
    flat_spatial_by_input = raw_spatial_by_input.reshape(raw_spatial_by_input.shape[0], -1)
    norm_spatial_by_input = normalize_distribution(flat_spatial_by_input, axis=1).reshape(raw_spatial_by_input.shape)
    aggregate_frame = normalize_distribution(raw_frame.mean(axis=0), axis=0)
    return LayerwiseRelevance(
        raw_token_scores=raw_token,
        normalized_token_scores=norm_token,
        absolute_visual_mass_by_layer=absolute_visual_mass,
        raw_frame_scores=raw_frame,
        normalized_frame_scores=norm_frame,
        raw_frame_scores_by_input=raw_frame_by_input,
        normalized_frame_scores_by_input=norm_frame_by_input,
        raw_spatial_scores=raw_spatial,
        normalized_spatial_scores=norm_spatial,
        raw_spatial_scores_by_input=raw_spatial_by_input,
        normalized_spatial_scores_by_input=norm_spatial_by_input,
        aggregate_frame_scores=aggregate_frame,
        cumulative_frame_curve=cumulative_curve(aggregate_frame),
        concentration_by_layer=tuple(concentration_stats(layer) for layer in raw_frame),
        metadata={
            "num_layers": int(raw_token.shape[0]),
            "num_visual_tokens": int(raw_token.shape[1]),
            "num_question_tokens": len(layout.question_token_indices),
            "query_scope": layout.query_scope,
            "extraction_method": extraction_method,
            "absolute_visual_mass_mean": float(absolute_visual_mass.mean()) if absolute_visual_mass.size else 0.0,
        },
    )


def compute_layerwise_relevance(attentions: Sequence[Any], layout: TokenLayout) -> LayerwiseRelevance:
    raw_token = aggregate_question_to_visual_attention(
        attentions, layout.question_token_indices, layout.visual_token_indices
    )
    return build_layerwise_relevance_from_token_scores(raw_token, layout, "returned_full_attention_reduced_after_forward")
