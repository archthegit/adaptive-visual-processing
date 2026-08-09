from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VisionAccessMaskSpec:
    through_layer: int | None
    num_layers: int
    question_token_indices: tuple[int, ...]
    text_token_indices: tuple[int, ...]
    visual_token_indices: tuple[int, ...]


def build_text_to_visual_block_mask(
    spec: VisionAccessMaskSpec,
    seq_len: int,
    blocked_value: float = -1e9,
) -> np.ndarray:
    masks = np.zeros((spec.num_layers, seq_len, seq_len), dtype=np.float32)
    if spec.through_layer is None:
        return masks
    for layer in range(spec.num_layers):
        if layer <= spec.through_layer:
            continue
        for query_idx in spec.text_token_indices:
            masks[layer, query_idx, list(spec.visual_token_indices)] = blocked_value
    return masks


def cutoff_alias_to_layer(alias: str, num_layers: int) -> int | None:
    if alias == "none":
        return None
    if alias == "early":
        return max(0, num_layers // 4 - 1)
    if alias == "middle":
        return max(0, num_layers // 2 - 1)
    if alias == "late":
        return max(0, (3 * num_layers) // 4 - 1)
    try:
        value = int(alias)
    except ValueError as exc:
        raise ValueError("vision access cutoff must be one of none/early/middle/late or an integer layer.") from exc
    if value < 0 or value >= num_layers:
        raise ValueError(f"Layer cutoff {value} is outside [0, {num_layers - 1}].")
    return value


def mask_blocks_text_to_visual(mask: np.ndarray, layer: int, text_indices: Sequence[int], visual_indices: Sequence[int]) -> bool:
    values = mask[layer, list(text_indices), :][:, list(visual_indices)]
    return bool(np.all(values < -1e8))
