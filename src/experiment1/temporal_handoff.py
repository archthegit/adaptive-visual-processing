from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
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
    instrumentation: list[LayerInstrumentation]
    compaction_plan: CompactionPlan | None
    final_question_token_indices: tuple[int, ...]
    final_visual_token_indices: tuple[int, ...]
    final_memory_token_indices: tuple[int, ...]
    total_estimated_attention_flops: int

    def instrumentation_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "layers": [vars(item) for item in self.instrumentation],
            "total_estimated_attention_flops": self.total_estimated_attention_flops,
            "final_question_token_indices": list(self.final_question_token_indices),
            "final_visual_token_indices": list(self.final_visual_token_indices),
            "final_memory_token_indices": list(self.final_memory_token_indices),
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


def _call_decoder_layer(layer: Any, hidden_states: Any, attention_mask: Any, position_ids: Any | None) -> Any:
    kwargs = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": False,
        "output_attentions": False,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        output = layer(hidden_states, **kwargs)
    except TypeError:
        kwargs.pop("output_attentions", None)
        try:
            output = layer(hidden_states, **kwargs)
        except TypeError:
            kwargs.pop("use_cache", None)
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

    with torch.inference_mode():
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
            attention_mask = build_additive_causal_mask(seq_in, hidden_states.dtype, hidden_states.device)
            hidden_states = _call_decoder_layer(layer, hidden_states, attention_mask, position_ids)
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
        logits = lm_head(hidden_states) if lm_head is not None else hidden_states
    return DecoderPrefillResult(
        logits=logits,
        final_hidden_states=hidden_states,
        instrumentation=instrumentation,
        compaction_plan=compaction_plan,
        final_question_token_indices=current_question_indices,
        final_visual_token_indices=tuple(current_visual_indices),
        final_memory_token_indices=tuple(current_memory_indices),
        total_estimated_attention_flops=total_flops,
    )


def cuda_profile_prefill(callable_obj: Any, *, warmup: int = 1, repeats: int = 1) -> tuple[Any, dict[str, Any]]:
    torch = _torch_module()
    if not torch.cuda.is_available():
        started = time.perf_counter()
        result = callable_obj()
        elapsed = time.perf_counter() - started
        return result, {
            "cuda_available": False,
            "warmup": warmup,
            "repeats": repeats,
            "prefill_latency_seconds": elapsed,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    for _ in range(max(0, warmup)):
        callable_obj()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    result = None
    start.record()
    for _ in range(max(1, repeats)):
        result = callable_obj()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / max(1, repeats)
    return result, {
        "cuda_available": True,
        "warmup": warmup,
        "repeats": repeats,
        "prefill_latency_seconds": elapsed_ms / 1000.0,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


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
    layers = getattr(core, "layers", None)
    if layers is None:
        layers = getattr(getattr(core, "language_model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate Qwen decoder layers.")
    norm = getattr(core, "norm", None)
    if norm is None and getattr(core, "language_model", None) is not None:
        norm = getattr(core.language_model, "norm", None)
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
        visual = getattr(model, "visual", None)
        if visual is None:
            visual = getattr(core, "visual", None)
        if visual is None:
            raise RuntimeError("Could not locate Qwen visual encoder.")

        if inputs.get("pixel_values") is not None:
            image_embeds = visual(inputs["pixel_values"], grid_thw=inputs.get("image_grid_thw"))
            image_token_id = int(getattr(getattr(model, "config", None), "image_token_id"))
            image_mask = input_ids == image_token_id
            inputs_embeds = _scatter_visual_embeds(inputs_embeds, image_mask, image_embeds)
        if inputs.get("pixel_values_videos") is not None:
            video_embeds = visual(inputs["pixel_values_videos"], grid_thw=inputs.get("video_grid_thw"))
            video_token_id = int(getattr(getattr(model, "config", None), "video_token_id"))
            video_mask = input_ids == video_token_id
            inputs_embeds = _scatter_visual_embeds(inputs_embeds, video_mask, video_embeds)

        position_ids = None
        if hasattr(core, "get_rope_index"):
            attention_mask = inputs.get("attention_mask")
            second_per_grid_ts = inputs.get("second_per_grid_ts")
            candidates = [
                {
                    "input_ids": input_ids,
                    "image_grid_thw": inputs.get("image_grid_thw"),
                    "video_grid_thw": inputs.get("video_grid_thw"),
                    "second_per_grid_ts": second_per_grid_ts,
                    "attention_mask": attention_mask,
                },
                {
                    "input_ids": input_ids,
                    "image_grid_thw": inputs.get("image_grid_thw"),
                    "video_grid_thw": inputs.get("video_grid_thw"),
                    "attention_mask": attention_mask,
                },
            ]
            last_error: Exception | None = None
            for kwargs in candidates:
                try:
                    rope_result = core.get_rope_index(**kwargs)
                    position_ids = rope_result[0] if isinstance(rope_result, tuple) else rope_result
                    break
                except TypeError as exc:
                    last_error = exc
            if position_ids is None and last_error is not None:
                raise RuntimeError("Could not call Qwen get_rope_index with known signatures.") from last_error
        if position_ids is None:
            seq_len = int(input_ids.shape[1])
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    return inputs_embeds, position_ids


def _scatter_visual_embeds(inputs_embeds: Any, placeholder_mask: Any, visual_embeds: Any) -> Any:
    if visual_embeds.dim() == 3:
        visual_embeds = visual_embeds.reshape(-1, visual_embeds.shape[-1])
    expected = int(placeholder_mask.sum().item())
    actual = int(visual_embeds.shape[0])
    if expected != actual:
        raise RuntimeError(f"Visual feature count mismatch: placeholders={expected}, visual_embeds={actual}.")
    mask = placeholder_mask.unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(mask, visual_embeds.to(inputs_embeds.device, inputs_embeds.dtype))
