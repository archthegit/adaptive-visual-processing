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


def canonicalize_window_ordered_groups(token_representations: Any, reverse_indices: Any | None) -> np.ndarray:
    reps = _as_numpy(token_representations)
    if reverse_indices is None:
        return reps
    indices = np.asarray(reverse_indices.detach().cpu().numpy() if hasattr(reverse_indices, "detach") else reverse_indices, dtype=np.int64)
    if reps.shape[0] == indices.shape[0]:
        return reps[indices]
    if reps.shape[0] % indices.shape[0] != 0:
        raise ValueError(
            f"Cannot canonicalize {reps.shape[0]} tokens with {indices.shape[0]} reverse indices."
        )
    group_size = reps.shape[0] // indices.shape[0]
    grouped = reps.reshape(indices.shape[0], group_size, reps.shape[-1])
    return grouped[indices].reshape(reps.shape[0], reps.shape[-1])


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


def vision_stage_indices(num_blocks: int) -> dict[str, int]:
    if num_blocks <= 0:
        return {}
    return {
        "early": 0,
        "middle": num_blocks // 2,
        "late": num_blocks - 1,
    }


def qwen_vision_window_indices(model: Any, video_grid_thw: Any) -> tuple[Any | None, Any | None]:
    try:
        from transformers.vision_utils import get_vision_window_index
    except Exception:
        return None, None
    visual = getattr(model, "visual", None)
    if visual is None:
        return None, None
    try:
        window_index, _cu_window = get_vision_window_index(
            video_grid_thw,
            spatial_merge_size=visual.spatial_merge_size,
            window_size=visual.window_size,
            patch_size=visual.patch_size,
            kwargs={},
        )
    except Exception:
        return None, None
    try:
        import torch

        return window_index, torch.argsort(window_index)
    except Exception:
        return window_index, np.argsort(np.asarray(window_index))


@dataclass
class VisionTemporalCapture:
    video_grid_thw: list[list[int]]
    spatial_merge_size: int
    reverse_indices: Any | None = None
    raw_outputs: dict[str, np.ndarray] = field(default_factory=dict)
    stage_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    hook_handles: list[Any] = field(default_factory=list)

    def _hook(self, label: str):
        def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            try:
                self.raw_outputs[label] = _as_numpy(output)
            except ValueError:
                return

        return capture

    def _visual_hook(self, label: str):
        def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            value = getattr(output, "pooler_output", None)
            if value is None and isinstance(output, dict):
                value = output.get("pooler_output")
            if value is None:
                value = output
            try:
                self.raw_outputs[label] = _as_numpy(value)
            except ValueError:
                return

        return capture

    def register(self, model: Any) -> None:
        visual_found = _find_named_module(model, ("visual",))
        visual = visual_found[1] if visual_found is not None else getattr(model, "visual", None)
        if visual is None:
            return
        blocks = list(getattr(visual, "blocks", []) or [])
        for stage, block_index in vision_stage_indices(len(blocks)).items():
            label = f"vision_block_{stage}"
            self.stage_specs[label] = {"order": "window", "spatial_merge_size": 1}
            self.hook_handles.append(blocks[block_index].register_forward_hook(self._hook(label)))

        merger = getattr(visual, "merger", None)
        if merger is not None:
            self.stage_specs["vision_merger_pre_reverse"] = {
                "order": "window",
                "spatial_merge_size": self.spatial_merge_size,
            }
            self.hook_handles.append(merger.register_forward_hook(self._hook("vision_merger_pre_reverse")))

        self.stage_specs["vision_final"] = {"order": "canonical", "spatial_merge_size": self.spatial_merge_size}
        self.hook_handles.append(visual.register_forward_hook(self._visual_hook("vision_final")))

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
            spec = self.stage_specs.get(label, {"order": "canonical", "spatial_merge_size": self.spatial_merge_size})
            ordered = raw
            if spec["order"] == "window":
                try:
                    ordered = canonicalize_window_ordered_groups(raw, self.reverse_indices)
                except ValueError as exc:
                    stages[label] = {"available": False, "error": str(exc)}
                    continue
            try:
                temporal = pool_temporal_representations(ordered, grid, spec["spatial_merge_size"])
            except ValueError as exc:
                stages[label] = {"available": False, "error": str(exc)}
                continue
            stage_summary = temporal_representation_summary(label, temporal)
            stage_summary["canonical_order_recovered"] = spec["order"] == "window" and self.reverse_indices is not None
            stage_summary["source_order"] = spec["order"]
            stages[label] = stage_summary
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
    video_grid_tensor: Any | None = None,
) -> Iterator[VisionTemporalCapture]:
    _window_index, reverse_indices = qwen_vision_window_indices(
        model,
        video_grid_tensor if video_grid_tensor is not None else video_grid_thw,
    )
    capture = VisionTemporalCapture(
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        reverse_indices=reverse_indices,
    )
    capture.register(model)
    try:
        yield capture
    finally:
        capture.close()
