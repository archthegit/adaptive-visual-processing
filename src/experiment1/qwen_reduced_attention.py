from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .masking import cutoff_alias_to_layer
from .token_layout import TokenLayout


ATTENTION_IMPLEMENTATION = "qwen_relevance_reduced_sdpa"
MASKED_EAGER_IMPLEMENTATION = "qwen_relevance_masked_eager"
_ACTIVE_CAPTURE: "ReducedAttentionCapture | None" = None
_ACTIVE_VISUAL_ACCESS: "VisualAccessIntervention | None" = None
_ACTIVE_TEMPORAL_REMOVAL: "TemporalBinRemoval | None" = None


def _repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


@dataclass
class ReducedAttentionCapture:
    question_token_indices: tuple[int, ...]
    visual_token_indices: tuple[int, ...]
    reduced_by_layer: dict[int, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_layout(cls, layout: TokenLayout) -> "ReducedAttentionCapture":
        return cls(
            question_token_indices=tuple(layout.question_token_indices),
            visual_token_indices=tuple(layout.visual_token_indices),
        )

    def ordered_token_scores(self, expected_layers: int | None = None) -> np.ndarray:
        if not self.reduced_by_layer:
            raise RuntimeError("No reduced attention scores were captured.")
        layer_ids = sorted(self.reduced_by_layer)
        if expected_layers is not None and layer_ids != list(range(expected_layers)):
            raise RuntimeError(f"Captured layers {layer_ids}, expected {list(range(expected_layers))}.")
        return np.stack([self.reduced_by_layer[layer_idx] for layer_idx in layer_ids], axis=0)


@dataclass(frozen=True)
class VisualAccessIntervention:
    through_layer: int | None
    visual_token_indices: tuple[int, ...]
    prompt_seq_len: int

    @classmethod
    def from_layout(cls, layout: TokenLayout, cutoff: str | int | None, num_layers: int) -> "VisualAccessIntervention":
        if cutoff is None or cutoff == "none":
            through_layer = None
        elif isinstance(cutoff, int):
            through_layer = cutoff
        else:
            through_layer = cutoff_alias_to_layer(str(cutoff), num_layers)
        return cls(
            through_layer=through_layer,
            visual_token_indices=tuple(layout.visual_token_indices),
            prompt_seq_len=len(layout.prompt_token_indices),
        )

    @property
    def enabled(self) -> bool:
        return self.through_layer is not None


@dataclass(frozen=True)
class TemporalBinRemoval:
    visual_token_indices: tuple[int, ...]
    prompt_seq_len: int

    @classmethod
    def from_layout(cls, layout: TokenLayout, temporal_bins: tuple[int, ...] | None) -> "TemporalBinRemoval | None":
        if not temporal_bins:
            return None
        requested = set(int(item) for item in temporal_bins)
        visual_indices = tuple(
            cell.token_index
            for cell in layout.visual_cells
            if cell.modality == "video" and cell.temporal_index in requested
        )
        if not visual_indices:
            raise ValueError(f"Requested temporal bins {sorted(requested)} do not map to any video visual tokens.")
        return cls(visual_token_indices=visual_indices, prompt_seq_len=len(layout.prompt_token_indices))


def register_reduced_attention() -> None:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, qwen_relevance_reduced_sdpa_forward)
    ALL_ATTENTION_FUNCTIONS.register(MASKED_EAGER_IMPLEMENTATION, qwen_relevance_masked_eager_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, eager_mask)
    ALL_MASK_ATTENTION_FUNCTIONS.register(MASKED_EAGER_IMPLEMENTATION, eager_mask)


def _set_attention_implementation(model: Any, implementation: str) -> list[tuple[Any, str]]:
    changed: list[tuple[Any, str]] = []
    seen: set[int] = set()
    for module in model.modules():
        if not hasattr(module, "layer_idx") or module.layer_idx is None:
            continue
        config = getattr(module, "config", None)
        if config is None or not hasattr(config, "_attn_implementation") or id(config) in seen:
            continue
        seen.add(id(config))
        changed.append((config, config._attn_implementation))
        config._attn_implementation = implementation
    return changed


def _num_decoder_layers(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return int(text_config.num_hidden_layers)
    if config is not None and hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    layers = getattr(getattr(model, "model", model), "language_model", None)
    if layers is not None and hasattr(layers, "layers"):
        return len(layers.layers)
    raise ValueError("Could not infer Qwen decoder layer count.")


@contextmanager
def reduced_attention_context(
    model: Any,
    layout: TokenLayout,
    vision_access_through_layer: str | int | None = None,
    remove_temporal_bins: tuple[int, ...] | None = None,
) -> Iterator[ReducedAttentionCapture]:
    global _ACTIVE_CAPTURE, _ACTIVE_VISUAL_ACCESS, _ACTIVE_TEMPORAL_REMOVAL
    register_reduced_attention()
    capture = ReducedAttentionCapture.from_layout(layout)
    previous_capture = _ACTIVE_CAPTURE
    previous_intervention = _ACTIVE_VISUAL_ACCESS
    previous_removal = _ACTIVE_TEMPORAL_REMOVAL
    _ACTIVE_VISUAL_ACCESS = VisualAccessIntervention.from_layout(
        layout, vision_access_through_layer, _num_decoder_layers(model)
    )
    _ACTIVE_TEMPORAL_REMOVAL = TemporalBinRemoval.from_layout(layout, remove_temporal_bins)
    previous_configs = _set_attention_implementation(model, ATTENTION_IMPLEMENTATION)
    _ACTIVE_CAPTURE = capture
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE = previous_capture
        _ACTIVE_VISUAL_ACCESS = previous_intervention
        _ACTIVE_TEMPORAL_REMOVAL = previous_removal
        for config, previous_implementation in previous_configs:
            config._attn_implementation = previous_implementation


@contextmanager
def masked_eager_attention_context(
    model: Any,
    layout: TokenLayout,
    vision_access_through_layer: str | int | None,
    remove_temporal_bins: tuple[int, ...] | None = None,
) -> Iterator[None]:
    global _ACTIVE_VISUAL_ACCESS, _ACTIVE_TEMPORAL_REMOVAL
    register_reduced_attention()
    previous_intervention = _ACTIVE_VISUAL_ACCESS
    previous_removal = _ACTIVE_TEMPORAL_REMOVAL
    _ACTIVE_VISUAL_ACCESS = VisualAccessIntervention.from_layout(
        layout, vision_access_through_layer, _num_decoder_layers(model)
    )
    _ACTIVE_TEMPORAL_REMOVAL = TemporalBinRemoval.from_layout(layout, remove_temporal_bins)
    previous_configs = _set_attention_implementation(model, MASKED_EAGER_IMPLEMENTATION)
    try:
        yield
    finally:
        _ACTIVE_VISUAL_ACCESS = previous_intervention
        _ACTIVE_TEMPORAL_REMOVAL = previous_removal
        for config, previous_implementation in previous_configs:
            config._attn_implementation = previous_implementation


def _slice_attention_mask(attention_mask: Any, question_indices: Any) -> Any:
    if attention_mask is None:
        return None
    if len(attention_mask.shape) == 4:
        return attention_mask[:, :, question_indices, :]
    if len(attention_mask.shape) == 3:
        return attention_mask[:, question_indices, :]
    return attention_mask


def _query_absolute_positions(query: Any, key_states: Any, position_ids: Any | None) -> list[int]:
    q_len = int(query.shape[2])
    key_len = int(key_states.shape[2])
    if position_ids is not None:
        ids = position_ids
        if hasattr(ids, "detach"):
            ids = ids.detach()
        if len(ids.shape) == 3:
            ids = ids[0]
        if len(ids.shape) == 2:
            ids = ids[0]
        if len(ids.shape) == 1 and int(ids.shape[0]) == q_len:
            return [int(item) for item in ids.cpu().tolist()]
    return list(range(key_len - q_len, key_len))


def _visual_access_block_mask(module: Any, query: Any, key_states: Any, position_ids: Any | None) -> Any | None:
    intervention = _ACTIVE_VISUAL_ACCESS
    if intervention is None or not intervention.enabled:
        return None
    if int(module.layer_idx) <= int(intervention.through_layer):
        return None
    key_len = int(key_states.shape[2])
    visual_indices = [idx for idx in intervention.visual_token_indices if idx < key_len]
    if not visual_indices:
        return None
    query_positions = _query_absolute_positions(query, key_states, position_ids)
    blocked_rows = [
        row_idx
        for row_idx, absolute_pos in enumerate(query_positions)
        if absolute_pos not in intervention.visual_token_indices
    ]
    if not blocked_rows:
        return None
    import torch

    mask = torch.zeros((1, 1, int(query.shape[2]), key_len), dtype=query.dtype, device=query.device)
    blocked_value = torch.finfo(query.dtype).min
    for row_idx in blocked_rows:
        mask[:, :, row_idx, visual_indices] = blocked_value
    return mask


def _temporal_removal_block_mask(query: Any, key_states: Any, position_ids: Any | None) -> Any | None:
    removal = _ACTIVE_TEMPORAL_REMOVAL
    if removal is None:
        return None
    key_len = int(key_states.shape[2])
    removed_indices = [idx for idx in removal.visual_token_indices if idx < key_len]
    if not removed_indices:
        return None
    query_positions = _query_absolute_positions(query, key_states, position_ids)
    blocked_rows = [
        row_idx
        for row_idx, absolute_pos in enumerate(query_positions)
        if absolute_pos not in removal.visual_token_indices
    ]
    if not blocked_rows:
        return None
    import torch

    mask = torch.zeros((1, 1, int(query.shape[2]), key_len), dtype=query.dtype, device=query.device)
    blocked_value = torch.finfo(query.dtype).min
    for row_idx in blocked_rows:
        mask[:, :, row_idx, removed_indices] = blocked_value
    return mask


def _apply_experiment1_blocks(attention_mask: Any, module: Any, query: Any, key_states: Any, position_ids: Any | None) -> Any:
    block = _visual_access_block_mask(module, query, key_states, position_ids)
    temporal_block = _temporal_removal_block_mask(query, key_states, position_ids)
    for candidate in (block, temporal_block):
        if candidate is not None:
            attention_mask = candidate if attention_mask is None else attention_mask + candidate
    return attention_mask


def qwen_relevance_masked_eager_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, Any]:
    import torch
    import torch.nn as nn

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attention_mask = _apply_experiment1_blocks(attention_mask, module, query, key_states, kwargs.get("position_ids"))

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def qwen_relevance_reduced_sdpa_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, None]:
    import torch
    import torch.nn.functional as F

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attention_mask = _apply_experiment1_blocks(attention_mask, module, query, key_states, kwargs.get("position_ids"))

    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    capture = _ACTIVE_CAPTURE
    if capture is not None and not module.training:
        device = query.device
        question_indices = torch.as_tensor(capture.question_token_indices, device=device, dtype=torch.long)
        visual_indices = torch.as_tensor(capture.visual_token_indices, device=device, dtype=torch.long)
        query_rows = query.index_select(2, question_indices)
        logits = torch.matmul(query_rows, key_states.transpose(2, 3)) * scaling
        reduced_mask = _slice_attention_mask(attention_mask, question_indices)
        if reduced_mask is not None:
            logits = logits + reduced_mask
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        qv = probs.index_select(-1, visual_indices)
        capture.reduced_by_layer[int(module.layer_idx)] = qv.mean(dim=(0, 1, 2)).detach().cpu().numpy()

    return attn_output, None
