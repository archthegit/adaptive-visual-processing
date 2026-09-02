from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .encoder_temporal import (
    aggregate_encoder_attention_to_temporal_bins,
    expanded_reverse_indices,
)


VISION_ATTENTION_IMPLEMENTATION = "qwen_experiment1_vision_attention_capture"
_ACTIVE_VISION_CAPTURE: "VisionAttentionCapture | None" = None


def _repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


@dataclass
class VisionAttentionCapture:
    module_to_layer: dict[int, int]
    grid_thw: list[int]
    spatial_merge_size: int
    reverse_indices: Any | None
    chunks_by_layer: dict[int, list[np.ndarray]] = field(default_factory=dict)
    reduced_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    tokens_seen_by_layer: dict[int, int] = field(default_factory=dict)
    tensor_shapes_by_layer: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def num_temporal_bins(self) -> int:
        return int(self.grid_thw[0])

    @property
    def num_tokens(self) -> int:
        t, h, w = [int(item) for item in self.grid_thw]
        return t * h * w

    def _window_to_temporal_bins_numpy(self) -> np.ndarray:
        t, h, w = [int(item) for item in self.grid_thw]
        canonical_temporal = np.repeat(np.arange(t, dtype=np.int64), h * w)
        if self.reverse_indices is None:
            return canonical_temporal
        expanded = expanded_reverse_indices(self.reverse_indices, self.spatial_merge_size * self.spatial_merge_size)
        if expanded.shape[0] != canonical_temporal.shape[0]:
            raise ValueError(
                f"Reverse indices length {expanded.shape[0]} does not match canonical visual token count {canonical_temporal.shape[0]}."
            )
        window_to_temporal = np.zeros_like(canonical_temporal)
        window_to_temporal[expanded] = canonical_temporal
        return window_to_temporal

    def record(self, module: Any, attention_weights: Any) -> None:
        layer = self.module_to_layer.get(id(module))
        if layer is None:
            return
        cursor = self.tokens_seen_by_layer.get(layer, 0)
        shape = tuple(int(item) for item in attention_weights.shape)
        self.tensor_shapes_by_layer.setdefault(layer, []).append(
            {"stage": "vision_encoder_attention_chunk", "shape": list(shape), "cursor": cursor}
        )
        if hasattr(attention_weights, "detach"):
            reduced = self._reduce_torch_chunk(attention_weights, cursor)
        else:
            reduced = self._reduce_numpy_chunk(attention_weights, cursor)
        current = self.reduced_by_layer.get(layer)
        self.reduced_by_layer[layer] = reduced if current is None else current + reduced
        self.tokens_seen_by_layer[layer] = cursor + int(shape[-1])

    def _reduce_numpy_chunk(self, attention_weights: Any, cursor: int) -> np.ndarray:
        arr = np.asarray(attention_weights, dtype=np.float64)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected vision attention chunk [heads, q, k], got {arr.shape}.")
        if arr.shape[1] != arr.shape[2]:
            raise ValueError("Vision attention chunks must be square.")
        key_mass = arr.sum(axis=1) / max(1, self.num_tokens)
        key_bins = self._window_to_temporal_bins_numpy()[cursor : cursor + arr.shape[2]]
        reduced = np.zeros((arr.shape[0], self.num_temporal_bins), dtype=np.float64)
        for local_key, temporal_bin in enumerate(key_bins):
            reduced[:, int(temporal_bin)] += key_mass[:, local_key]
        return reduced

    def _reduce_torch_chunk(self, attention_weights: Any, cursor: int) -> np.ndarray:
        import torch

        weights = attention_weights
        if weights.ndim == 4 and int(weights.shape[0]) == 1:
            weights = weights[0]
        if weights.ndim != 3:
            raise ValueError(f"Expected vision attention chunk [heads, q, k], got {tuple(weights.shape)}.")
        if int(weights.shape[1]) != int(weights.shape[2]):
            raise ValueError("Vision attention chunks must be square.")
        key_mass = weights.float().sum(dim=1) / max(1, self.num_tokens)
        key_bins_np = self._window_to_temporal_bins_numpy()[cursor : cursor + int(weights.shape[2])]
        if key_bins_np.shape[0] != int(weights.shape[2]):
            raise ValueError("Vision attention chunk cursor exceeded the visual token count.")
        key_bins = torch.as_tensor(key_bins_np, dtype=torch.long, device=weights.device)
        reduced = torch.zeros(
            (int(weights.shape[0]), self.num_temporal_bins),
            dtype=torch.float32,
            device=weights.device,
        )
        reduced.scatter_add_(1, key_bins.unsqueeze(0).expand(int(weights.shape[0]), -1), key_mass)
        return reduced.detach().cpu().numpy()

    def ordered_temporal_attention(self, expected_layers: int | None = None) -> np.ndarray:
        if not self.reduced_by_layer and not self.chunks_by_layer:
            raise RuntimeError("No Qwen vision attention chunks were captured.")
        layer_ids = sorted(self.reduced_by_layer or self.chunks_by_layer)
        if expected_layers is not None and layer_ids != list(range(expected_layers)):
            raise RuntimeError(f"Captured vision layers {layer_ids}, expected {list(range(expected_layers))}.")
        return np.stack([self.temporal_attention_for_layer(layer) for layer in layer_ids], axis=0)

    def temporal_attention_for_layer(self, layer: int) -> np.ndarray:
        reduced = self.reduced_by_layer.get(layer)
        if reduced is not None:
            totals = reduced.sum(axis=1, keepdims=True)
            return np.divide(reduced, totals, out=np.zeros_like(reduced), where=totals > 0)
        chunks = self.chunks_by_layer.get(layer)
        if not chunks:
            raise ValueError(f"No attention chunks captured for vision layer {layer}.")
        full_attention = block_diagonal_attention(chunks)
        return aggregate_encoder_attention_to_temporal_bins(
            full_attention,
            grid_thw=self.grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            reverse_indices=self.reverse_indices,
        )

    def to_json_dict(self, expected_layers: int | None = None) -> dict[str, Any]:
        temporal = self.ordered_temporal_attention(expected_layers=expected_layers)
        return {
            "available": True,
            "capture_method": VISION_ATTENTION_IMPLEMENTATION,
            "grid_thw": list(self.grid_thw),
            "spatial_merge_size": int(self.spatial_merge_size),
            "num_layers": int(temporal.shape[0]),
            "num_heads": int(temporal.shape[1]),
            "num_temporal_bins": int(temporal.shape[2]),
            "tensor_shapes_reduced": {
                str(layer): shapes for layer, shapes in sorted(self.tensor_shapes_by_layer.items())
            },
            "normalized_incoming_temporal_attention": temporal.tolist(),
        }


def block_diagonal_attention(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        raise ValueError("No attention chunks supplied.")
    num_heads = chunks[0].shape[0]
    total = sum(chunk.shape[1] for chunk in chunks)
    output = np.zeros((num_heads, total, total), dtype=np.float64)
    cursor = 0
    for chunk in chunks:
        if chunk.ndim != 3 or chunk.shape[0] != num_heads or chunk.shape[1] != chunk.shape[2]:
            raise ValueError("All attention chunks must have shape [heads, seq, seq] with matching heads.")
        size = chunk.shape[1]
        output[:, cursor : cursor + size, cursor : cursor + size] = chunk
        cursor += size
    return output


def register_vision_attention_capture() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(VISION_ATTENTION_IMPLEMENTATION, qwen_vision_attention_capture_forward)


def _find_visual(model: Any) -> Any | None:
    if hasattr(model, "named_modules"):
        for name, module in model.named_modules():
            if name == "visual" or name.endswith(".visual"):
                return module
    return getattr(model, "visual", None)


def _set_vision_attention_implementation(model: Any, implementation: str) -> list[tuple[Any, str]]:
    visual = _find_visual(model)
    if visual is None:
        raise ValueError("Could not find Qwen visual module for attention capture.")
    changed = []
    seen: set[int] = set()
    for block in getattr(visual, "blocks", []) or []:
        attn = getattr(block, "attn", None)
        config = getattr(attn, "config", None)
        if config is None or not hasattr(config, "_attn_implementation") or id(config) in seen:
            continue
        seen.add(id(config))
        changed.append((config, config._attn_implementation))
        config._attn_implementation = implementation
    return changed


def vision_module_to_layer(model: Any) -> dict[int, int]:
    visual = _find_visual(model)
    if visual is None:
        raise ValueError("Could not find Qwen visual module for attention capture.")
    mapping = {}
    for layer, block in enumerate(getattr(visual, "blocks", []) or []):
        attn = getattr(block, "attn", None)
        if attn is not None:
            mapping[id(attn)] = layer
    if not mapping:
        raise ValueError("No Qwen vision attention modules were found.")
    return mapping


@contextmanager
def qwen_vision_attention_capture_context(
    model: Any,
    grid_thw: list[int],
    spatial_merge_size: int,
    reverse_indices: Any | None,
) -> Iterator[VisionAttentionCapture]:
    global _ACTIVE_VISION_CAPTURE
    register_vision_attention_capture()
    previous_capture = _ACTIVE_VISION_CAPTURE
    capture = VisionAttentionCapture(
        module_to_layer=vision_module_to_layer(model),
        grid_thw=list(map(int, grid_thw)),
        spatial_merge_size=int(spatial_merge_size),
        reverse_indices=reverse_indices,
    )
    previous_configs = _set_vision_attention_implementation(model, VISION_ATTENTION_IMPLEMENTATION)
    _ACTIVE_VISION_CAPTURE = capture
    try:
        yield capture
    finally:
        _ACTIVE_VISION_CAPTURE = previous_capture
        for config, previous in previous_configs:
            config._attn_implementation = previous


def qwen_vision_attention_capture_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs: Any,
) -> tuple[Any, Any]:
    import torch
    import torch.nn as nn

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    capture = _ACTIVE_VISION_CAPTURE
    if capture is not None and not module.training:
        capture.record(module, attn_weights)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
