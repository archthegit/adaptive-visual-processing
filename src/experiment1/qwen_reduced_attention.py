from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .token_layout import TokenLayout


ATTENTION_IMPLEMENTATION = "qwen_relevance_reduced_sdpa"
_ACTIVE_CAPTURE: "ReducedAttentionCapture | None" = None


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


def register_reduced_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, qwen_relevance_reduced_sdpa_forward)


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


@contextmanager
def reduced_attention_context(model: Any, layout: TokenLayout) -> Iterator[ReducedAttentionCapture]:
    global _ACTIVE_CAPTURE
    register_reduced_attention()
    capture = ReducedAttentionCapture.from_layout(layout)
    previous_capture = _ACTIVE_CAPTURE
    previous_configs = _set_attention_implementation(model, ATTENTION_IMPLEMENTATION)
    _ACTIVE_CAPTURE = capture
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE = previous_capture
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
