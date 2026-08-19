from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        value = value.cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected token representations with shape [tokens, dim], got {arr.shape}.")
    return arr


def pool_temporal_representations(
    token_representations: Any,
    grid_thw: list[int] | tuple[int, int, int],
    spatial_merge_size: int = 1,
) -> np.ndarray:
    reps = _as_numpy(token_representations)
    t, h, w = [int(item) for item in grid_thw]
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive.")
    if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
        raise ValueError(f"Grid {(t, h, w)} is not divisible by spatial_merge_size={spatial_merge_size}.")
    pooled_h = h // spatial_merge_size
    pooled_w = w // spatial_merge_size
    expected = t * pooled_h * pooled_w
    if reps.shape[0] != expected:
        raise ValueError(f"Expected {expected} visual tokens from grid {(t, h, w)}, got {reps.shape[0]}.")
    return reps.reshape(t, pooled_h * pooled_w, reps.shape[-1]).mean(axis=1)


def adjacent_cosine_similarity(temporal_representations: Any) -> list[float]:
    reps = np.asarray(temporal_representations, dtype=np.float64)
    if reps.ndim != 2:
        raise ValueError("adjacent_cosine_similarity expects [temporal_bins, dim].")
    if reps.shape[0] <= 1:
        return []
    sims = []
    for idx in range(reps.shape[0] - 1):
        left = reps[idx]
        right = reps[idx + 1]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        sims.append(float(np.dot(left, right) / denom) if denom > 0 else 0.0)
    return sims


def temporal_representation_summary(name: str, temporal_representations: np.ndarray) -> dict[str, Any]:
    similarities = adjacent_cosine_similarity(temporal_representations)
    return {
        "name": name,
        "temporal_representations": temporal_representations.tolist(),
        "adjacent_bin_cosine_similarity": similarities,
        "mean_adjacent_bin_cosine_similarity": float(np.mean(similarities)) if similarities else None,
        "min_adjacent_bin_cosine_similarity": float(np.min(similarities)) if similarities else None,
        "max_adjacent_bin_cosine_similarity": float(np.max(similarities)) if similarities else None,
        "num_temporal_bins": int(temporal_representations.shape[0]),
        "embedding_dim": int(temporal_representations.shape[1]) if temporal_representations.ndim == 2 else 0,
    }


def _find_named_module(model: Any, suffixes: tuple[str, ...]) -> tuple[str, Any] | None:
    for name, module in model.named_modules():
        if any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes):
            return name, module
    return None


@dataclass
class VisionTemporalCapture:
    video_grid_thw: list[list[int]]
    spatial_merge_size: int
    raw_outputs: dict[str, np.ndarray] = field(default_factory=dict)
    hook_handles: list[Any] = field(default_factory=list)

    def _hook(self, label: str):
        def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            try:
                self.raw_outputs[label] = _as_numpy(output)
            except ValueError:
                return

        return capture

    def register(self, model: Any) -> None:
        targets = {
            "vision_patch_embed": ("visual.patch_embed", "patch_embed"),
            "vision_merger": ("visual.merger", "merger"),
        }
        for label, suffixes in targets.items():
            found = _find_named_module(model, suffixes)
            if found is None:
                continue
            _name, module = found
            self.hook_handles.append(module.register_forward_hook(self._hook(label)))

    def close(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def to_json_dict(self) -> dict[str, Any]:
        if not self.video_grid_thw:
            return {"available": False, "reason": "no video_grid_thw"}
        grid = self.video_grid_thw[0]
        stages: dict[str, Any] = {}
        for label, raw in self.raw_outputs.items():
            merge = self.spatial_merge_size if label == "vision_merger" else 1
            try:
                temporal = pool_temporal_representations(raw, grid, merge)
            except ValueError as exc:
                stages[label] = {"available": False, "error": str(exc)}
                continue
            stages[label] = temporal_representation_summary(label, temporal)
        return {
            "available": bool(stages),
            "video_grid_thw": [list(map(int, item)) for item in self.video_grid_thw],
            "spatial_merge_size": int(self.spatial_merge_size),
            "stages": stages,
        }


@contextmanager
def vision_temporal_capture_context(
    model: Any,
    video_grid_thw: list[list[int]],
    spatial_merge_size: int,
) -> Iterator[VisionTemporalCapture]:
    capture = VisionTemporalCapture(video_grid_thw=video_grid_thw, spatial_merge_size=spatial_merge_size)
    capture.register(model)
    try:
        yield capture
    finally:
        capture.close()
