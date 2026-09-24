from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "qwen_temporal_handoff_prefill_v1"
SUPPORTED_CONDITIONS = {"dense_custom", "handoff_mean", "hard_evict", "random_handoff"}


@dataclass(frozen=True)
class TemporalHandoffConfig:
    condition: str
    handoff_layer: int = 8
    retained_temporal_regions: tuple[int, ...] = ()
    memory_tokens_per_region: int = 2
    random_seed: int = 20260830
    num_hidden_layers: int = 28

    def __post_init__(self) -> None:
        if self.condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported handoff condition {self.condition!r}.")
        if self.memory_tokens_per_region not in {0, 1, 2, 4}:
            raise ValueError("memory_tokens_per_region must be one of 0, 1, 2 or 4.")
        if self.condition == "hard_evict" and self.memory_tokens_per_region != 0:
            raise ValueError("hard_evict must use memory_tokens_per_region=0.")
        if self.condition in {"handoff_mean", "random_handoff"} and self.memory_tokens_per_region == 0:
            raise ValueError(f"{self.condition} requires memory tokens.")
        if self.handoff_layer < 0:
            raise ValueError("handoff_layer must be non-negative.")


@dataclass(frozen=True)
class HandoffOutputEntry:
    kind: str
    old_position: int | None
    memory_region: int | None = None
    memory_index: int | None = None
    source_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class TemporalRegionPlan:
    temporal_region: int
    action: str
    original_token_positions: tuple[int, ...]
    retained_token_positions: tuple[int, ...] = ()
    memory_token_count: int = 0
    memory_source_position_groups: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class NativeTemporalAggregation:
    analysis_bin_count: int
    native_temporal_cell_count: int
    analysis_bin_to_native_cell: dict[int, int]
    native_cell_to_analysis_bins: dict[int, tuple[int, ...]]
    aggregation_method: str
    native_temporal_scores: tuple[tuple[float, ...], ...]
    expected_native_temporal_cell_count: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "analysis_bin_count": self.analysis_bin_count,
            "native_temporal_cell_count": self.native_temporal_cell_count,
            "expected_native_temporal_cell_count": self.expected_native_temporal_cell_count,
            "analysis_bin_to_native_cell": {
                str(key): int(value) for key, value in sorted(self.analysis_bin_to_native_cell.items())
            },
            "native_cell_to_analysis_bins": {
                str(key): list(value) for key, value in sorted(self.native_cell_to_analysis_bins.items())
            },
            "aggregation_method": self.aggregation_method,
            "native_temporal_scores": [list(layer) for layer in self.native_temporal_scores],
        }


@dataclass(frozen=True)
class CompactionPlan:
    condition: str
    canonical_order: str
    original_sequence_length: int
    compacted_sequence_length: int
    retained_temporal_regions: tuple[int, ...]
    handed_off_temporal_regions: tuple[int, ...]
    memory_tokens_per_region: int
    regions: tuple[TemporalRegionPlan, ...]
    output_entries: tuple[HandoffOutputEntry, ...]
    old_to_new: dict[int, int]
    new_to_old: dict[int, int | None]
    retained_visual_token_positions: tuple[int, ...]
    removed_visual_token_positions: tuple[int, ...]
    memory_token_positions: tuple[int, ...]
    text_old_positions: tuple[int, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "schema_version": SCHEMA_VERSION,
            "canonical_order": self.canonical_order,
            "original_sequence_length": self.original_sequence_length,
            "compacted_sequence_length": self.compacted_sequence_length,
            "retained_temporal_regions": list(self.retained_temporal_regions),
            "handed_off_temporal_regions": list(self.handed_off_temporal_regions),
            "memory_tokens_per_region": self.memory_tokens_per_region,
            "retained_visual_token_positions": list(self.retained_visual_token_positions),
            "removed_visual_token_positions": list(self.removed_visual_token_positions),
            "memory_token_positions": list(self.memory_token_positions),
            "text_old_positions": list(self.text_old_positions),
            "regions": [
                {
                    "temporal_region": region.temporal_region,
                    "action": region.action,
                    "original_token_positions": list(region.original_token_positions),
                    "retained_token_positions": list(region.retained_token_positions),
                    "memory_token_count": region.memory_token_count,
                    "memory_source_position_groups": [
                        list(group) for group in region.memory_source_position_groups
                    ],
                }
                for region in self.regions
            ],
            "output_entries": [
                {
                    "new_position": idx,
                    "kind": entry.kind,
                    "old_position": entry.old_position,
                    "memory_region": entry.memory_region,
                    "memory_index": entry.memory_index,
                    "source_positions": list(entry.source_positions),
                }
                for idx, entry in enumerate(self.output_entries)
            ],
        }


@dataclass
class LayerInstrumentation:
    layer: int
    sequence_length_in: int
    sequence_length_out: int
    visual_token_count_in: int
    visual_token_count_out: int
    memory_token_count_in: int
    memory_token_count_out: int
    text_token_count_in: int
    text_token_count_out: int
    attention_q_len: int
    attention_k_len: int
    estimated_qk_flops: int
    estimated_av_flops: int
    compaction_applied_after_layer: bool = False


@dataclass
class DecoderPrefillResult:
    logits: Any
    final_hidden_states: Any
    final_token_hidden_state: Any
    instrumentation: list[LayerInstrumentation]
    compaction_plan: CompactionPlan | None
    final_question_token_indices: tuple[int, ...]
    final_visual_token_indices: tuple[int, ...]
    final_memory_token_indices: tuple[int, ...]
    total_estimated_attention_flops: int
    attention_mask_mode: str
    layer_types: tuple[str, ...]
    rotary_embedding_computations: int

    def instrumentation_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "layers": [vars(item) for item in self.instrumentation],
            "total_estimated_attention_flops": self.total_estimated_attention_flops,
            "final_question_token_indices": list(self.final_question_token_indices),
            "final_visual_token_indices": list(self.final_visual_token_indices),
            "final_memory_token_indices": list(self.final_memory_token_indices),
            "attention_mask_mode": self.attention_mask_mode,
            "layer_types": list(self.layer_types),
            "rotary_embedding_computations": self.rotary_embedding_computations,
        }


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _visual_cells(layout: Any) -> tuple[Any, ...]:
    cells = _get_field(layout, "visual_cells")
    if cells is None:
        cells = _get_field(layout, "visual_token_cells")
    if cells is None and isinstance(layout, dict):
        cells = layout.get("token_layout", {}).get("visual_token_cells")
    if cells is None:
        raise ValueError("Layout does not contain visual token cells.")
    return tuple(cells)


def visual_token_positions(layout: Any) -> tuple[int, ...]:
    return tuple(int(_get_field(cell, "token_index")) for cell in _visual_cells(layout))


def question_token_positions(layout: Any) -> tuple[int, ...]:
    positions = _get_field(layout, "question_token_indices", ())
    return tuple(int(item) for item in positions)


def temporal_regions_from_layout(layout: Any) -> dict[int, tuple[int, ...]]:
    regions: dict[int, list[int]] = {}
    for cell in _visual_cells(layout):
        temporal_index = int(_get_field(cell, "temporal_index"))
        token_index = int(_get_field(cell, "token_index"))
        regions.setdefault(temporal_index, []).append(token_index)
    if not regions:
        raise ValueError("No visual temporal regions found.")
    return {key: tuple(sorted(value)) for key, value in sorted(regions.items())}


def _split_evenly(items: Sequence[int], parts: int) -> tuple[tuple[int, ...], ...]:
    if parts <= 0:
        return ()
    if not items:
        raise ValueError("Cannot create memory tokens for an empty temporal region.")
    groups: list[tuple[int, ...]] = []
    n = len(items)
    for idx in range(parts):
        start = math.floor(idx * n / parts)
        end = math.floor((idx + 1) * n / parts)
        if end <= start:
            end = min(start + 1, n)
        groups.append(tuple(int(item) for item in items[start:end]))
    return tuple(groups)


def select_top_temporal_regions(
    normalized_temporal_scores: Sequence[Sequence[float]],
    handoff_layer: int,
    retain_count: int,
) -> tuple[int, ...]:
    layer_scores = list(float(item) for item in normalized_temporal_scores[handoff_layer])
    if not 0 < retain_count <= len(layer_scores):
        raise ValueError("retain_count must be between one and the number of temporal regions.")
    ranked = sorted(range(len(layer_scores)), key=lambda idx: (-layer_scores[idx], idx))
    return tuple(sorted(ranked[:retain_count]))


def select_random_temporal_regions(num_regions: int, retain_count: int, seed: int, question_id: str = "") -> tuple[int, ...]:
    if not 0 < retain_count <= num_regions:
        raise ValueError("retain_count must be between one and num_regions.")
    rng = random.Random(f"{seed}:{question_id}:{num_regions}:{retain_count}")
    regions = list(range(num_regions))
    rng.shuffle(regions)
    return tuple(sorted(regions[:retain_count]))


def aggregate_analysis_scores_to_native_cells(
    analysis_scores: Sequence[Sequence[float]],
    artifact: dict[str, Any],
    *,
    tolerance: float = 1e-6,
) -> NativeTemporalAggregation:
    """Sum decoder analysis-bin distributions into Qwen-native temporal-cell distributions.

    The mapping is derived only from artifact metadata:
    frame_bin_mappings[*].{analysis_bin, source_frame_index} and
    token_layout.visual_token_cells[*].{temporal_bin, sampled_frame_indices}.
    """
    scores = tuple(tuple(float(value) for value in layer) for layer in analysis_scores)
    if not scores:
        raise ValueError("No analysis-bin temporal scores were provided.")
    analysis_bin_count = len(scores[0])
    if analysis_bin_count <= 0:
        raise ValueError("Analysis temporal score layers are empty.")
    for layer_idx, layer in enumerate(scores):
        if len(layer) != analysis_bin_count:
            raise ValueError(
                f"Analysis temporal score layer {layer_idx} has length {len(layer)}, expected {analysis_bin_count}."
            )
        if not all(math.isfinite(value) for value in layer):
            raise ValueError(f"Analysis temporal score layer {layer_idx} contains non-finite values.")
        total = sum(layer)
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Analysis temporal score layer {layer_idx} sums to {total}, expected 1.")

    frame_to_analysis = _frame_to_analysis_bin_mapping(artifact)
    frame_to_native = _frame_to_native_cell_mapping(artifact)
    missing_analysis_frames = sorted(set(frame_to_native).difference(frame_to_analysis))
    if missing_analysis_frames:
        raise ValueError(
            "Native temporal-cell metadata references sampled source frames missing from frame_bin_mappings: "
            f"{missing_analysis_frames[:8]}"
        )
    analysis_to_native_sets: dict[int, set[int]] = {idx: set() for idx in range(analysis_bin_count)}
    for frame_index, analysis_bin in frame_to_analysis.items():
        if analysis_bin < 0 or analysis_bin >= analysis_bin_count:
            raise ValueError(
                f"Frame {frame_index} maps to analysis bin {analysis_bin}, outside score range 0..{analysis_bin_count - 1}."
            )
        if frame_index not in frame_to_native:
            raise ValueError(f"Sampled source frame {frame_index} has no native temporal-cell owner.")
        analysis_to_native_sets[analysis_bin].add(frame_to_native[frame_index])

    analysis_bin_to_native: dict[int, int] = {}
    for analysis_bin in range(analysis_bin_count):
        owners = analysis_to_native_sets.get(analysis_bin, set())
        if not owners:
            raise ValueError(f"Analysis bin {analysis_bin} is not covered by any native temporal cell.")
        if len(owners) != 1:
            raise ValueError(f"Analysis bin {analysis_bin} maps to multiple native temporal cells: {sorted(owners)}.")
        analysis_bin_to_native[analysis_bin] = next(iter(owners))

    native_ids = sorted(set(analysis_bin_to_native.values()))
    if native_ids != list(range(len(native_ids))):
        raise ValueError(f"Native temporal-bin IDs are not contiguous from zero: {native_ids}.")
    expected_native_count = _expected_native_temporal_cell_count(artifact)
    if expected_native_count is not None and expected_native_count != len(native_ids):
        raise ValueError(
            "Derived native temporal-cell count does not match video_grid_thw: "
            f"{len(native_ids)} != {expected_native_count}."
        )
    native_cell_to_analysis: dict[int, tuple[int, ...]] = {
        native: tuple(idx for idx, owner in sorted(analysis_bin_to_native.items()) if owner == native)
        for native in native_ids
    }
    native_scores: list[tuple[float, ...]] = []
    for layer_idx, layer in enumerate(scores):
        aggregated = [0.0 for _ in native_ids]
        for analysis_bin, native in analysis_bin_to_native.items():
            aggregated[native] += layer[analysis_bin]
        if not all(math.isfinite(value) for value in aggregated):
            raise ValueError(f"Native temporal score layer {layer_idx} contains non-finite values.")
        total = sum(aggregated)
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Native temporal score layer {layer_idx} sums to {total}, expected 1.")
        native_scores.append(tuple(float(value) for value in aggregated))
    return NativeTemporalAggregation(
        analysis_bin_count=analysis_bin_count,
        native_temporal_cell_count=len(native_ids),
        analysis_bin_to_native_cell=analysis_bin_to_native,
        native_cell_to_analysis_bins=native_cell_to_analysis,
        aggregation_method="sum",
        native_temporal_scores=tuple(native_scores),
        expected_native_temporal_cell_count=expected_native_count,
    )


def _frame_to_analysis_bin_mapping(artifact: dict[str, Any]) -> dict[int, int]:
    batches = artifact.get("frame_bin_mappings")
    if not batches or not isinstance(batches, list):
        raise ValueError("Artifact is missing frame_bin_mappings.")
    mapping: dict[int, int] = {}
    for batch in batches:
        if not isinstance(batch, list):
            continue
        for item in batch:
            if "source_frame_index" not in item or "analysis_bin" not in item:
                raise ValueError(f"Malformed frame_bin_mappings entry: {item}")
            frame = int(item["source_frame_index"])
            analysis_bin = int(item["analysis_bin"])
            previous = mapping.get(frame)
            if previous is not None and previous != analysis_bin:
                raise ValueError(f"Source frame {frame} maps to multiple analysis bins: {previous}, {analysis_bin}.")
            mapping[frame] = analysis_bin
    if not mapping:
        raise ValueError("No source-frame to analysis-bin mappings found.")
    return mapping


def _frame_to_native_cell_mapping(artifact: dict[str, Any]) -> dict[int, int]:
    cells = (artifact.get("token_layout") or {}).get("visual_token_cells")
    if not cells:
        raise ValueError("Artifact is missing token_layout.visual_token_cells.")
    mapping: dict[int, int] = {}
    for cell in cells:
        if "temporal_bin" not in cell:
            raise ValueError(f"Visual token cell is missing temporal_bin: {cell}")
        sampled = cell.get("sampled_frame_indices")
        if sampled is None:
            raise ValueError(f"Visual token cell is missing sampled_frame_indices: {cell}")
        native = int(cell["temporal_bin"])
        for frame_value in sampled:
            frame = int(frame_value)
            previous = mapping.get(frame)
            if previous is not None and previous != native:
                raise ValueError(f"Source frame {frame} belongs to multiple native temporal cells: {previous}, {native}.")
            mapping[frame] = native
    if not mapping:
        raise ValueError("No sampled-frame to native temporal-cell mappings found.")
    return mapping


def _expected_native_temporal_cell_count(artifact: dict[str, Any]) -> int | None:
    token_layout = artifact.get("token_layout") or {}
    metadata = token_layout.get("visual_grid_metadata") or {}
    candidates = [
        token_layout.get("video_grid_thw"),
        metadata.get("video_grid_thw") if isinstance(metadata, dict) else None,
        (artifact.get("metadata") or {}).get("video_grid_thw"),
    ]
    for value in candidates:
        count = _first_grid_t(value)
        if count is not None:
            return count
    return None


def _first_grid_t(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("video_grid_thw", "grid", "thw"):
            count = _first_grid_t(value.get(key))
            if count is not None:
                return count
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        first = value[0]
        if isinstance(first, (list, tuple)):
            return _first_grid_t(first)
        return int(first)
    return None


def build_compaction_plan(
    layout: Any,
    sequence_length: int,
    retained_temporal_regions: Sequence[int],
    memory_tokens_per_region: int,
    condition: str,
) -> CompactionPlan:
    if condition == "dense_custom":
        raise ValueError("dense_custom does not require a compaction plan.")
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported handoff condition {condition!r}.")
    regions = temporal_regions_from_layout(layout)
    retained = tuple(sorted(int(item) for item in retained_temporal_regions))
    unknown = set(retained).difference(regions)
    if unknown:
        raise ValueError(f"Retained temporal regions not present in layout: {sorted(unknown)}")
    if not retained:
        raise ValueError("At least one temporal region must be retained.")
    memory_count = 0 if condition == "hard_evict" else int(memory_tokens_per_region)
    if condition != "hard_evict" and memory_count not in {1, 2, 4}:
        raise ValueError("Handoff conditions require 1, 2 or 4 memory tokens per region.")

    retained_set = set(retained)
    removed_positions: set[int] = set()
    region_plans: list[TemporalRegionPlan] = []
    first_removed_position_to_region: dict[int, int] = {}
    for temporal_region, positions in regions.items():
        if temporal_region in retained_set:
            region_plans.append(
                TemporalRegionPlan(
                    temporal_region=temporal_region,
                    action="retained",
                    original_token_positions=positions,
                    retained_token_positions=positions,
                )
            )
            continue
        removed_positions.update(positions)
        groups = _split_evenly(positions, memory_count) if memory_count else ()
        if positions:
            first_removed_position_to_region[min(positions)] = temporal_region
        region_plans.append(
            TemporalRegionPlan(
                temporal_region=temporal_region,
                action="compressed" if memory_count else "evicted",
                original_token_positions=positions,
                memory_token_count=memory_count,
                memory_source_position_groups=groups,
            )
        )

    output_entries: list[HandoffOutputEntry] = []
    skipped_regions: set[int] = set()
    region_by_id = {region.temporal_region: region for region in region_plans}
    for old_position in range(sequence_length):
        if old_position not in removed_positions:
            output_entries.append(HandoffOutputEntry(kind="original", old_position=old_position))
            continue
        temporal_region = first_removed_position_to_region.get(old_position)
        if temporal_region is None or temporal_region in skipped_regions:
            continue
        skipped_regions.add(temporal_region)
        for memory_index, source_group in enumerate(region_by_id[temporal_region].memory_source_position_groups):
            output_entries.append(
                HandoffOutputEntry(
                    kind="memory",
                    old_position=None,
                    memory_region=temporal_region,
                    memory_index=memory_index,
                    source_positions=source_group,
                )
            )

    old_to_new = {
        int(entry.old_position): idx
        for idx, entry in enumerate(output_entries)
        if entry.old_position is not None
    }
    new_to_old = {idx: entry.old_position for idx, entry in enumerate(output_entries)}
    memory_positions = tuple(idx for idx, entry in enumerate(output_entries) if entry.kind == "memory")
    retained_visual = tuple(
        old_to_new[position]
        for region in region_plans
        if region.action == "retained"
        for position in region.retained_token_positions
    )
    removed_visual = tuple(sorted(removed_positions))
    original_visual = set(visual_token_positions(layout))
    text_positions = tuple(position for position in range(sequence_length) if position not in original_visual)
    plan = CompactionPlan(
        condition=condition,
        canonical_order="original_sequence_order_with_memory_tokens_inserted_at_each_handed_off_region_start",
        original_sequence_length=sequence_length,
        compacted_sequence_length=len(output_entries),
        retained_temporal_regions=retained,
        handed_off_temporal_regions=tuple(sorted(set(regions).difference(retained_set))),
        memory_tokens_per_region=memory_count,
        regions=tuple(sorted(region_plans, key=lambda item: item.temporal_region)),
        output_entries=tuple(output_entries),
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        retained_visual_token_positions=tuple(sorted(retained_visual)),
        removed_visual_token_positions=removed_visual,
        memory_token_positions=memory_positions,
        text_old_positions=text_positions,
    )
    _validate_compaction_plan(plan, layout)
    return plan


def _validate_compaction_plan(plan: CompactionPlan, layout: Any) -> None:
    if plan.compacted_sequence_length >= plan.original_sequence_length and plan.condition != "handoff_mean":
        if plan.condition == "hard_evict":
            raise ValueError("hard_evict compaction did not reduce sequence length.")
    visual_positions = set(visual_token_positions(layout))
    retained_old = {
        position
        for region in plan.regions
        if region.action == "retained"
        for position in region.retained_token_positions
    }
    removed_old = set(plan.removed_visual_token_positions)
    if retained_old.intersection(removed_old):
        raise ValueError("A visual token cannot be both retained and removed.")
    if retained_old.union(removed_old) != visual_positions:
        raise ValueError("Retained plus removed visual tokens must equal the original visual-token set.")
    if len(plan.output_entries) != plan.compacted_sequence_length:
        raise ValueError("Output entry count does not match compacted sequence length.")
    memory_expected = sum(region.memory_token_count for region in plan.regions)
    if len(plan.memory_token_positions) != memory_expected:
        raise ValueError("Memory token count does not match region plan.")


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in dependency-free environments.
        raise RuntimeError("Temporal handoff execution requires torch.") from exc
    return torch


def build_additive_causal_mask(seq_len: int, dtype: Any, device: Any) -> Any:
    torch = _torch_module()
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    blocked = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    mask[:, :, blocked] = torch.finfo(dtype).min
    return mask


def build_additive_sliding_causal_mask(seq_len: int, sliding_window: int, dtype: Any, device: Any) -> Any:
    torch = _torch_module()
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive.")
    mask = build_additive_causal_mask(seq_len, dtype, device)
    row = torch.arange(seq_len, device=device).view(seq_len, 1)
    col = torch.arange(seq_len, device=device).view(1, seq_len)
    too_old = col < (row - sliding_window + 1)
    mask[:, :, too_old] = torch.finfo(dtype).min
    return mask


def compact_hidden_states_and_positions(
    hidden_states: Any,
    position_ids: Any | None,
    plan: CompactionPlan,
) -> tuple[Any, Any | None]:
    torch = _torch_module()
    hidden_pieces: list[Any] = []
    position_pieces: list[Any] = []
    for entry in plan.output_entries:
        if entry.kind == "original":
            assert entry.old_position is not None
            hidden_pieces.append(hidden_states[:, entry.old_position : entry.old_position + 1, :])
            if position_ids is not None:
                position_pieces.append(_position_slice(position_ids, entry.old_position))
        elif entry.kind == "memory":
            if not entry.source_positions:
                raise ValueError("Memory entry has no source positions.")
            source = torch.tensor(entry.source_positions, dtype=torch.long, device=hidden_states.device)
            pooled = hidden_states.index_select(1, source).mean(dim=1, keepdim=True)
            hidden_pieces.append(pooled)
            if position_ids is not None:
                medoid = entry.source_positions[len(entry.source_positions) // 2]
                position_pieces.append(_position_slice(position_ids, medoid))
        else:
            raise ValueError(f"Unknown output entry kind {entry.kind!r}.")
    compacted_hidden = torch.cat(hidden_pieces, dim=1)
    compacted_positions = torch.cat(position_pieces, dim=-1) if position_ids is not None else None
    if compacted_hidden.shape[1] != plan.compacted_sequence_length:
        raise AssertionError("Compacted hidden-state length does not match plan.")
    return compacted_hidden, compacted_positions


def _position_slice(position_ids: Any, position: int) -> Any:
    if position_ids.dim() == 3:
        return position_ids[:, :, position : position + 1]
    if position_ids.dim() == 2:
        return position_ids[:, position : position + 1]
    raise ValueError(f"Unsupported position_ids shape {tuple(position_ids.shape)}.")


def remap_indices(indices: Iterable[int], old_to_new: dict[int, int]) -> tuple[int, ...]:
    missing = [int(index) for index in indices if int(index) not in old_to_new]
    if missing:
        raise ValueError(f"Cannot remap removed token indices: {missing[:8]}")
    return tuple(old_to_new[int(index)] for index in indices)


def _call_decoder_layer(
    layer: Any,
    hidden_states: Any,
    attention_mask: Any,
    position_embeddings: Any,
    text_position_ids: Any | None,
) -> Any:
    import inspect

    signature = inspect.signature(layer.forward if hasattr(layer, "forward") else layer)
    parameters = signature.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    required = {"attention_mask", "position_embeddings"}
    missing = [name for name in required if name not in parameters and not accepts_kwargs]
    if missing:
        raise RuntimeError(
            "Installed Qwen decoder layer does not expose the expected Transformers 5.14.1 "
            f"arguments: missing {missing}."
        )
    kwargs = {
        "attention_mask": attention_mask,
        "position_embeddings": position_embeddings,
        "position_ids": text_position_ids,
        "past_key_values": None,
        "use_cache": False,
    }
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None and (key in parameters or accepts_kwargs)
    }
    output = layer(hidden_states, **kwargs)
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _estimate_attention_flops(seq_len: int, num_heads: int, head_dim: int, batch_size: int = 1) -> tuple[int, int]:
    # One multiply-add pair is counted as two FLOPs. QK and AV have the same leading cost.
    qk = int(2 * batch_size * num_heads * seq_len * seq_len * head_dim)
    av = int(2 * batch_size * num_heads * seq_len * seq_len * head_dim)
    return qk, av


def _counts(seq_len: int, visual_indices: Sequence[int], memory_indices: Sequence[int]) -> tuple[int, int, int]:
    visual = len(tuple(visual_indices))
    memory = len(tuple(memory_indices))
    text = seq_len - visual - memory
    if text < 0:
        raise ValueError("Visual plus memory token counts exceed sequence length.")
    return visual, memory, text


def run_custom_decoder_prefill(
    *,
    layers: Sequence[Any],
    hidden_states: Any,
    position_ids: Any | None,
    layout: Any,
    config: TemporalHandoffConfig,
    lm_head: Any | None,
    norm: Any | None,
    num_attention_heads: int,
    head_dim: int,
    rotary_emb: Any | None = None,
    layer_types: Sequence[str] | None = None,
    sliding_window: int | None = None,
) -> DecoderPrefillResult:
    torch = _torch_module()
    if config.condition != "dense_custom" and not config.retained_temporal_regions:
        raise ValueError("Compaction conditions require retained_temporal_regions.")
    if config.handoff_layer >= len(layers):
        raise ValueError("handoff_layer is outside decoder layer range.")
    current_visual_indices = visual_token_positions(layout)
    current_question_indices = question_token_positions(layout)
    current_memory_indices: tuple[int, ...] = ()
    compaction_plan: CompactionPlan | None = None
    instrumentation: list[LayerInstrumentation] = []
    total_flops = 0
    if rotary_emb is None and layers:
        # Tests may use tiny fake layers. Real Qwen callers must pass rotary_emb.
        if position_ids is not None:
            raise RuntimeError("Real Qwen execution requires language_model.rotary_emb.")
    active_layer_types = tuple(layer_types or ("full_attention",) * len(layers))
    if len(active_layer_types) != len(layers):
        raise ValueError("layer_types length must match decoder layer count.")
    supported_layer_types = {"full_attention", "sliding_attention"}
    unsupported = sorted(set(active_layer_types).difference(supported_layer_types))
    if unsupported:
        raise RuntimeError(f"Unsupported Qwen attention layer types: {unsupported}")
    if "sliding_attention" in active_layer_types and not sliding_window:
        raise RuntimeError("Qwen sliding_attention layers require a sliding_window value.")
    text_position_ids = _official_text_position_ids(position_ids)
    attention_mask_mode = "full_attention_only" if set(active_layer_types) == {"full_attention"} else "mixed_full_and_sliding_attention"
    position_embeddings = None
    rotary_embedding_computations = 0

    with torch.inference_mode():
        position_embeddings = _position_embeddings(rotary_emb, hidden_states, position_ids)
        rotary_embedding_computations = 1 if position_embeddings is not None else 0
        for layer_idx, layer in enumerate(layers):
            seq_in = int(hidden_states.shape[1])
            visual_in, memory_in, text_in = _counts(seq_in, current_visual_indices, current_memory_indices)
            qk, av = _estimate_attention_flops(
                seq_in,
                num_attention_heads=num_attention_heads,
                head_dim=head_dim,
                batch_size=int(hidden_states.shape[0]),
            )
            total_flops += qk + av
            full_attention_mask = build_additive_causal_mask(seq_in, hidden_states.dtype, hidden_states.device)
            if active_layer_types[layer_idx] == "full_attention":
                attention_mask = full_attention_mask
            else:
                attention_mask = build_additive_sliding_causal_mask(
                    seq_in,
                    int(sliding_window),
                    hidden_states.dtype,
                    hidden_states.device,
                )
            hidden_states = _call_decoder_layer(
                layer,
                hidden_states,
                attention_mask,
                position_embeddings,
                text_position_ids,
            )
            seq_out = int(hidden_states.shape[1])
            compaction_applied = False
            if config.condition != "dense_custom" and layer_idx == config.handoff_layer:
                compaction_plan = build_compaction_plan(
                    layout=layout,
                    sequence_length=seq_out,
                    retained_temporal_regions=config.retained_temporal_regions,
                    memory_tokens_per_region=config.memory_tokens_per_region,
                    condition=config.condition,
                )
                hidden_states, position_ids = compact_hidden_states_and_positions(
                    hidden_states,
                    position_ids,
                    compaction_plan,
                )
                current_visual_indices = compaction_plan.retained_visual_token_positions
                current_memory_indices = compaction_plan.memory_token_positions
                current_question_indices = remap_indices(current_question_indices, compaction_plan.old_to_new)
                text_position_ids = _official_text_position_ids(position_ids)
                position_embeddings = _position_embeddings(rotary_emb, hidden_states, position_ids)
                rotary_embedding_computations += 1 if position_embeddings is not None else 0
                seq_out = int(hidden_states.shape[1])
                compaction_applied = True
            visual_out, memory_out, text_out = _counts(seq_out, current_visual_indices, current_memory_indices)
            instrumentation.append(
                LayerInstrumentation(
                    layer=layer_idx,
                    sequence_length_in=seq_in,
                    sequence_length_out=seq_out,
                    visual_token_count_in=visual_in,
                    visual_token_count_out=visual_out,
                    memory_token_count_in=memory_in,
                    memory_token_count_out=memory_out,
                    text_token_count_in=text_in,
                    text_token_count_out=text_out,
                    attention_q_len=seq_in,
                    attention_k_len=seq_in,
                    estimated_qk_flops=qk,
                    estimated_av_flops=av,
                    compaction_applied_after_layer=compaction_applied,
                )
            )
        if norm is not None:
            hidden_states = norm(hidden_states)
        final_token_hidden_state = hidden_states[:, -1:, :]
        logits = lm_head(final_token_hidden_state) if lm_head is not None else final_token_hidden_state
    return DecoderPrefillResult(
        logits=logits,
        final_hidden_states=hidden_states,
        final_token_hidden_state=final_token_hidden_state,
        instrumentation=instrumentation,
        compaction_plan=compaction_plan,
        final_question_token_indices=current_question_indices,
        final_visual_token_indices=tuple(current_visual_indices),
        final_memory_token_indices=tuple(current_memory_indices),
        total_estimated_attention_flops=total_flops,
        attention_mask_mode=attention_mask_mode,
        layer_types=active_layer_types,
        rotary_embedding_computations=rotary_embedding_computations,
    )


def _official_text_position_ids(position_ids: Any | None) -> Any | None:
    if position_ids is None:
        return None
    if position_ids.dim() == 3 and position_ids.shape[0] == 4:
        return position_ids[0]
    return None


def _position_embeddings(rotary_emb: Any | None, hidden_states: Any, position_ids: Any | None) -> Any | None:
    if rotary_emb is None:
        return None
    if position_ids is None:
        raise RuntimeError("Qwen rotary_emb requires multimodal position_ids.")
    if position_ids.dim() != 3:
        raise RuntimeError(f"Expected Qwen multimodal position_ids rank 3, got shape {tuple(position_ids.shape)}.")
    rotary_position_ids = position_ids[1:] if position_ids.shape[0] == 4 else position_ids
    if rotary_position_ids.shape[0] != 3:
        raise RuntimeError(
            "Expected Qwen rotary position_ids with three multimodal axes "
            f"after optional text row removal, got shape {tuple(rotary_position_ids.shape)}."
        )
    if rotary_position_ids.shape[-1] != hidden_states.shape[1]:
        raise RuntimeError(
            "Qwen position_ids sequence length does not match hidden_states: "
            f"{rotary_position_ids.shape[-1]} != {hidden_states.shape[1]}"
        )
    return rotary_emb(hidden_states, rotary_position_ids)


def cuda_profile_prefill(callable_obj: Any, *, warmup: int = 3, repeats: int = 10) -> tuple[Any, dict[str, Any]]:
    torch = _torch_module()
    repeats = max(1, repeats)
    warmup = max(0, warmup)
    if not torch.cuda.is_available():
        for _ in range(warmup):
            callable_obj()
        timings: list[float] = []
        result = None
        for _ in range(repeats):
            started = time.perf_counter()
            result = callable_obj()
            timings.append(time.perf_counter() - started)
        return result, {
            "cuda_available": False,
            "warmup": warmup,
            "repeats": repeats,
            "prefill_latency_seconds_median": statistics.median(timings),
            "prefill_latency_seconds_mean": statistics.fmean(timings),
            "prefill_latency_seconds_stddev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "incremental_peak_allocated_bytes": None,
            "incremental_peak_reserved_bytes": None,
            "absolute_peak_allocated_bytes": None,
            "absolute_peak_reserved_bytes": None,
        }
    for _ in range(max(0, warmup)):
        callable_obj()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    result = None
    timings_ms: list[float] = []
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    peak_allocated = baseline_allocated
    peak_reserved = baseline_reserved
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = callable_obj()
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(float(start.elapsed_time(end)))
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
    return result, {
        "cuda_available": True,
        "warmup": warmup,
        "repeats": repeats,
        "prefill_latency_seconds_median": statistics.median(timings_ms) / 1000.0,
        "prefill_latency_seconds_mean": statistics.fmean(timings_ms) / 1000.0,
        "prefill_latency_seconds_stddev": (statistics.stdev(timings_ms) / 1000.0) if len(timings_ms) > 1 else 0.0,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
        "incremental_peak_reserved_bytes": max(0, peak_reserved - baseline_reserved),
        "absolute_peak_allocated_bytes": peak_allocated,
        "absolute_peak_reserved_bytes": peak_reserved,
    }


def cuda_profile_stages(
    stage_callables: dict[str, Any],
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    for name, callable_obj in stage_callables.items():
        result, profile = cuda_profile_prefill(callable_obj, warmup=warmup, repeats=repeats)
        results[name] = result
        profiles[name] = profile
    return results, profiles


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def condition_from_baseline_scores(
    *,
    condition: str,
    baseline_temporal_scores: Sequence[Sequence[float]],
    handoff_layer: int,
    retain_count: int,
    num_regions: int,
    memory_tokens_per_region: int,
    seed: int,
    question_id: str,
) -> TemporalHandoffConfig:
    if condition in {"dense_custom"}:
        retained: tuple[int, ...] = ()
        memory = 0
    elif condition in {"handoff_mean", "hard_evict"}:
        retained = select_top_temporal_regions(baseline_temporal_scores, handoff_layer, retain_count)
        memory = 0 if condition == "hard_evict" else memory_tokens_per_region
    elif condition == "random_handoff":
        retained = select_random_temporal_regions(num_regions, retain_count, seed, question_id)
        memory = memory_tokens_per_region
    else:
        raise ValueError(f"Unsupported condition {condition!r}.")
    return TemporalHandoffConfig(
        condition=condition,
        handoff_layer=handoff_layer,
        retained_temporal_regions=retained,
        memory_tokens_per_region=memory,
        random_seed=seed,
    )


def qwen_decoder_stack(model: Any) -> dict[str, Any]:
    """Return the text decoder components used by Qwen2.5-VL HF checkpoints.

    This intentionally fails loudly instead of silently using a wrong module. The
    handoff prototype must run the same decoder layers as the dense model.
    """
    core = getattr(model, "model", None)
    if core is None:
        raise RuntimeError("Could not locate Qwen core model at model.model.")
    language_model = getattr(core, "language_model", None)
    if language_model is None:
        raise RuntimeError("Could not locate Qwen language_model.")
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate Qwen decoder layers.")
    norm = getattr(language_model, "norm", None)
    rotary_emb = getattr(language_model, "rotary_emb", None)
    if rotary_emb is None:
        raise RuntimeError("Could not locate Qwen language_model.rotary_emb.")
    lm_config = getattr(language_model, "config", None)
    if lm_config is None:
        lm_config = getattr(getattr(model, "config", None), "text_config", None)
    if lm_config is None:
        raise RuntimeError("Could not locate Qwen text config.")
    layer_types = tuple(getattr(lm_config, "layer_types", ("full_attention",) * len(tuple(layers))))
    if len(layer_types) != len(tuple(layers)):
        raise RuntimeError("Qwen text config layer_types length does not match decoder layer count.")
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Could not locate Qwen lm_head.")
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    num_heads = int(getattr(text_config, "num_attention_heads"))
    hidden_size = int(getattr(text_config, "hidden_size"))
    head_dim = int(getattr(text_config, "head_dim", hidden_size // num_heads))
    return {
        "layers": tuple(layers),
        "norm": norm,
        "lm_head": lm_head,
        "language_model": language_model,
        "rotary_emb": rotary_emb,
        "layer_types": layer_types,
        "sliding_window": getattr(lm_config, "sliding_window", None),
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
    }


def qwen_multimodal_decoder_inputs(model: Any, inputs: Any) -> tuple[Any, Any | None]:
    """Construct decoder input embeddings and multimodal RoPE positions.

    The implementation mirrors the current Qwen2.5-VL HF forward contract:
    embed text input IDs, replace image/video placeholder embeddings with
    visual encoder features, then obtain mRoPE position IDs from get_rope_index.
    """
    torch = _torch_module()
    core = getattr(model, "model", None)
    if core is None:
        raise RuntimeError("Could not locate Qwen core model at model.model.")
    with torch.inference_mode():
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise RuntimeError("Phase-1 temporal handoff prototype supports batch size one only.")
        embed_tokens = getattr(core, "embed_tokens", None)
        if embed_tokens is None and hasattr(model, "get_input_embeddings"):
            embed_tokens = model.get_input_embeddings()
        if embed_tokens is None:
            raise RuntimeError("Could not locate Qwen text embedding module.")
        inputs_embeds = embed_tokens(input_ids)
        if inputs.get("pixel_values") is not None:
            if not hasattr(core, "get_image_features"):
                raise RuntimeError("Qwen core model lacks get_image_features.")
            image_outputs = core.get_image_features(inputs["pixel_values"], inputs.get("image_grid_thw"))
            image_pooler = getattr(image_outputs, "pooler_output", None)
            if image_pooler is None:
                raise RuntimeError("Qwen get_image_features did not return pooler_output.")
            image_embeds = _cat_qwen_pooler_output(image_pooler).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = _placeholder_masks(core, input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if inputs.get("pixel_values_videos") is not None:
            if not hasattr(core, "get_video_features"):
                raise RuntimeError("Qwen core model lacks get_video_features.")
            video_outputs = core.get_video_features(inputs["pixel_values_videos"], inputs.get("video_grid_thw"))
            video_pooler = getattr(video_outputs, "pooler_output", None)
            if video_pooler is None:
                raise RuntimeError("Qwen get_video_features did not return pooler_output.")
            video_embeds = _cat_qwen_pooler_output(video_pooler).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = _placeholder_masks(core, input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        position_ids = None
        if hasattr(core, "compute_3d_position_ids"):
            attention_mask = inputs.get("attention_mask")
            second_per_grid_ts = inputs.get("second_per_grid_ts")
            position_ids = core.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
        elif hasattr(core, "get_rope_index"):
            rope_result = core.get_rope_index(
                input_ids,
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=inputs.get("second_per_grid_ts"),
                attention_mask=inputs.get("attention_mask"),
            )
            position_ids = rope_result[0] if isinstance(rope_result, tuple) else rope_result
        if position_ids is None and (inputs.get("pixel_values") is not None or inputs.get("pixel_values_videos") is not None):
            raise RuntimeError("Qwen multimodal input requires official 3D position IDs; scalar arange fallback is forbidden.")
        if position_ids is None:
            seq_len = int(input_ids.shape[1])
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        _assert_qwen_position_ids(position_ids, inputs_embeds.shape[1])
    return inputs_embeds, position_ids


def _cat_qwen_pooler_output(pooler_output: Any) -> Any:
    torch = _torch_module()
    if isinstance(pooler_output, torch.Tensor):
        raise RuntimeError(
            "Qwen get_*_features pooler_output was a tensor, but Transformers 5.14.1 "
            "should return per-input feature chunks. Refusing to guess feature boundaries."
        )
    chunks = tuple(pooler_output)
    if not chunks:
        raise RuntimeError("Qwen get_*_features returned no visual feature chunks.")
    return torch.cat(chunks, dim=0)


def _placeholder_masks(
    core: Any,
    input_ids: Any,
    inputs_embeds: Any,
    *,
    image_features: Any | None = None,
    video_features: Any | None = None,
) -> tuple[Any, Any]:
    if hasattr(core, "get_placeholder_mask"):
        image_mask, video_mask = core.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            video_features=video_features,
        )
    else:
        config = getattr(core, "config", None)
        if config is None:
            raise RuntimeError("Cannot build placeholder masks without core.config.")
        image_mask = (input_ids == int(config.image_token_id)).unsqueeze(-1).to(inputs_embeds.device)
        video_mask = (input_ids == int(config.video_token_id)).unsqueeze(-1).to(inputs_embeds.device)
    hidden = int(inputs_embeds.shape[-1])
    if image_features is not None:
        expected = int(image_mask.sum().item())
        actual = int(image_features.numel() // hidden)
        if expected != actual:
            raise RuntimeError(f"Image features and placeholders do not match: {actual} != {expected}.")
    if video_features is not None:
        expected = int(video_mask.sum().item())
        actual = int(video_features.numel() // hidden)
        if expected != actual:
            raise RuntimeError(f"Video features and placeholders do not match: {actual} != {expected}.")
    return image_mask, video_mask


def _assert_qwen_position_ids(position_ids: Any, seq_len: int) -> None:
    if position_ids.dim() != 3:
        raise RuntimeError(f"Expected Qwen position_ids rank 3, got shape {tuple(position_ids.shape)}.")
    if position_ids.shape[0] not in {3, 4}:
        raise RuntimeError(f"Expected Qwen position_ids first dimension 3 or 4, got {position_ids.shape[0]}.")
    if int(position_ids.shape[-1]) != int(seq_len):
        raise RuntimeError(f"Qwen position_ids length {position_ids.shape[-1]} does not match sequence length {seq_len}.")
