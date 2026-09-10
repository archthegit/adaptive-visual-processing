from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.frame_sampling import FrameBatch
from src.models.base import format_multiple_choice_prompt, parse_choice_response

from .answer_scoring import score_answer_choices_from_outputs
from .resolution import ResolutionConfig
from .temporal import (
    TemporalLayerStats,
    TemporalRelevance,
    bins_to_attention_mass,
    gini_coefficient,
    normalized_temporal_entropy,
    spearman_rank_correlation,
    temporal_rank_order,
    top1_mass,
    top_fraction_mass,
    topk_overlap,
)


VILA_DEFAULT_CHECKPOINT = "Efficient-Large-Model/Llama-3-VILA1.5-8B"


@dataclass(frozen=True)
class PreparedVILAInputs:
    model_inputs: dict[str, Any]
    rendered_prompt: str
    input_ids: Sequence[int]
    question_token_indices: tuple[int, ...]
    visual_token_indices: tuple[int, ...]
    visual_token_frame_indices: tuple[int, ...]
    prepared_frame_indices: tuple[int, ...]
    truncation_occurred: bool = False


class VILALlama3Wrapper:
    """Thin adapter for the VILA checkpoint.

    The public VILA preprocessing APIs are not stable enough to infer temporal
    token mappings in this repository. A real integration must expose
    ``prepare_experiment1_inputs`` returning the exact fields in
    ``PreparedVILAInputs``. This wrapper fails loudly when that mapping is not
    available rather than assuming equal tokens per frame.
    """

    def __init__(self, checkpoint: str = VILA_DEFAULT_CHECKPOINT, max_new_tokens: int = 16):
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "VILA inference requires an isolated environment with torch, transformers, and VILA-compatible "
                "remote-code dependencies. See requirements-vila.txt and scripts/setup_vila_environment.sh."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; refusing to run VILA inference.")
        self._processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self._model.eval()

    @property
    def tokenizer(self) -> Any:
        self._load()
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self._processor
        return tokenizer

    def prepare_inputs(self, example: Any, prompt: str, frame_batches: list[FrameBatch]) -> PreparedVILAInputs:
        self._load()
        if hasattr(self._processor, "prepare_experiment1_inputs"):
            payload = self._processor.prepare_experiment1_inputs(prompt=prompt, frame_batches=frame_batches)
        elif hasattr(self._model, "prepare_experiment1_inputs"):
            payload = self._model.prepare_experiment1_inputs(prompt=prompt, frame_batches=frame_batches)
        else:
            raise RuntimeError(
                "The loaded VILA processor/model does not expose prepare_experiment1_inputs. "
                "This repository requires an exact frame->visual-token mapping to prevent silent VILA "
                "resampling, truncation, duplication, or reordering."
            )
        return coerce_prepared_vila_inputs(payload)

    def forward(self, prepared: PreparedVILAInputs, output_attentions: bool = True) -> Any:
        self._load()
        return self._model(**prepared.model_inputs, output_attentions=output_attentions, use_cache=False)

    def generate(self, prepared: PreparedVILAInputs) -> Any:
        self._load()
        return self._model.generate(**prepared.model_inputs, max_new_tokens=self.max_new_tokens)

    def decode_new_tokens(self, output_ids: Any, input_length: int) -> str:
        tokenizer = self.tokenizer
        if hasattr(output_ids, "detach"):
            generated = output_ids[:, input_length:]
        else:
            generated = output_ids
        if hasattr(tokenizer, "batch_decode"):
            return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return ""


def coerce_prepared_vila_inputs(payload: Any) -> PreparedVILAInputs:
    if isinstance(payload, PreparedVILAInputs):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("VILA preparer must return PreparedVILAInputs or a mapping with equivalent fields.")
    required = {
        "model_inputs",
        "rendered_prompt",
        "input_ids",
        "question_token_indices",
        "visual_token_indices",
        "visual_token_frame_indices",
        "prepared_frame_indices",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"VILA prepared input payload is missing required fields: {missing}")
    return PreparedVILAInputs(
        model_inputs=dict(payload["model_inputs"]),
        rendered_prompt=str(payload["rendered_prompt"]),
        input_ids=tuple(int(item) for item in _as_sequence(payload["input_ids"])),
        question_token_indices=tuple(int(item) for item in payload["question_token_indices"]),
        visual_token_indices=tuple(int(item) for item in payload["visual_token_indices"]),
        visual_token_frame_indices=tuple(int(item) for item in payload["visual_token_frame_indices"]),
        prepared_frame_indices=tuple(int(item) for item in payload["prepared_frame_indices"]),
        truncation_occurred=bool(payload.get("truncation_occurred", False)),
    )


def _as_sequence(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _all_frame_indices(frame_batches: Sequence[FrameBatch]) -> tuple[int, ...]:
    indices: list[int] = []
    for batch in frame_batches:
        presented = batch.metadata.get("presented_source_frame_indices")
        if presented is not None:
            indices.extend(int(index) for index in presented)
        else:
            indices.extend(int(index) for index in batch.frame_indices)
    return tuple(indices)


def sample_position_to_analysis_bin(batch: FrameBatch) -> dict[int, int]:
    return {
        int(item["sample_position"]): int(item["analysis_bin"])
        for item in (batch.metadata.get("frame_bin_mapping", []) or [])
    }


def validate_prepared_vila_mapping(prepared: PreparedVILAInputs, frame_batches: Sequence[FrameBatch]) -> None:
    expected_frames = _all_frame_indices(frame_batches)
    if prepared.prepared_frame_indices != expected_frames:
        raise ValueError(
            "VILA frame order mismatch: prepared frames do not match decoded Experiment 1 frames. "
            f"expected={expected_frames}, prepared={prepared.prepared_frame_indices}"
        )
    if prepared.truncation_occurred:
        raise ValueError("VILA reported truncation; refusing to run Experiment 1 temporal replication.")
    if len(prepared.visual_token_indices) != len(prepared.visual_token_frame_indices):
        raise ValueError("VILA visual token/frame mapping length mismatch.")
    allowed_positions = set(range(len(expected_frames)))
    bad = [idx for idx in prepared.visual_token_frame_indices if idx not in allowed_positions]
    if bad:
        raise ValueError(f"VILA visual tokens reference nonexistent frame positions: {bad[:5]}")
    if not prepared.question_token_indices:
        raise ValueError("VILA prepared inputs did not identify question-token rows.")
    if not prepared.visual_token_indices:
        raise ValueError("VILA prepared inputs did not identify visual-token columns.")


def visual_token_analysis_bins(prepared: PreparedVILAInputs, batch: FrameBatch) -> tuple[int, ...]:
    mapping = sample_position_to_analysis_bin(batch)
    if not mapping:
        raise ValueError("FrameBatch is missing frame_bin_mapping required for VILA temporal aggregation.")
    bins: list[int] = []
    for frame_position in prepared.visual_token_frame_indices:
        if int(frame_position) not in mapping:
            raise ValueError(f"Visual token references sample position {frame_position}, absent from frame_bin_mapping.")
        bins.append(mapping[int(frame_position)])
    return tuple(bins)


def extract_temporal_scores_from_vila_attentions(
    attentions: Sequence[Any],
    question_token_indices: Sequence[int],
    visual_token_indices: Sequence[int],
    visual_token_bins: Sequence[int],
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(visual_token_indices) != len(visual_token_bins):
        raise ValueError("visual_token_indices and visual_token_bins must have the same length.")
    raw = np.zeros((len(attentions), num_bins), dtype=np.float64)
    absolute = np.zeros((len(attentions),), dtype=np.float64)
    q_idx = np.asarray(tuple(int(item) for item in question_token_indices), dtype=np.int64)
    v_idx = np.asarray(tuple(int(item) for item in visual_token_indices), dtype=np.int64)
    bin_idx = np.asarray(tuple(int(item) for item in visual_token_bins), dtype=np.int64)
    for layer_index, attention in enumerate(attentions):
        array = _to_numpy(attention)
        if array.ndim != 4:
            raise ValueError(f"Expected VILA attention shape [batch, heads, q, k], got {array.shape}.")
        if array.shape[0] != 1:
            raise ValueError("Experiment 1 VILA extraction expects batch size 1.")
        selected = array[0][:, q_idx, :][:, :, v_idx]
        token_mass = selected.mean(axis=(0, 1))
        absolute[layer_index] = float(token_mass.sum())
        for visual_position, analysis_bin in enumerate(bin_idx):
            raw[layer_index, int(analysis_bin)] += float(token_mass[visual_position])
    return raw, absolute


def temporal_relevance_from_raw_scores(
    raw_temporal: np.ndarray,
    absolute_mass: np.ndarray,
    temporal_bins: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    topk: int = 3,
) -> TemporalRelevance:
    raw = np.asarray(raw_temporal, dtype=np.float64)
    normalized = np.zeros_like(raw)
    for layer_index, scores in enumerate(raw):
        total = float(scores.sum())
        normalized[layer_index] = scores / total if total > 0.0 else scores
    final_order = temporal_rank_order(normalized[-1]) if len(normalized) else tuple()
    metrics: list[TemporalLayerStats] = []
    for layer_index, layer_scores in enumerate(normalized):
        order = temporal_rank_order(layer_scores)
        count80, fraction80 = bins_to_attention_mass(layer_scores, 0.8)
        overlap, overlap_fraction = topk_overlap(order, final_order, topk)
        metrics.append(
            TemporalLayerStats(
                layer=layer_index,
                normalized_temporal_entropy=normalized_temporal_entropy(layer_scores),
                top1_temporal_bin_mass=top1_mass(layer_scores),
                top20_temporal_bin_mass=top_fraction_mass(layer_scores, 0.2),
                temporal_gini=gini_coefficient(layer_scores),
                first_bin_mass=float(layer_scores[0]) if len(layer_scores) else 0.0,
                last_bin_mass=float(layer_scores[-1]) if len(layer_scores) else 0.0,
                bins_to_80pct_mass=count80,
                fraction_bins_to_80pct_mass=fraction80,
                temporal_bin_rank_order=order,
                spearman_with_final_layer_ordering=spearman_rank_correlation(order, final_order)
                if final_order
                else 0.0,
                topk_overlap_with_final_layer=overlap,
                topk_overlap_fraction_with_final_layer=overlap_fraction,
            )
        )
    return TemporalRelevance(
        raw_temporal_bin_scores=raw,
        normalized_temporal_bin_scores=normalized,
        absolute_question_to_visual_attention_mass=np.asarray(absolute_mass, dtype=np.float64),
        temporal_bins=tuple(dict(item) for item in temporal_bins),
        layer_metrics=tuple(metrics),
        metadata=dict(metadata),
    )


def vila_temporal_bin_metadata(batch: FrameBatch, visual_token_bins: Sequence[int]) -> tuple[dict[str, Any], ...]:
    mapping = batch.metadata.get("frame_bin_mapping", []) or []
    num_bins = max(int(item["analysis_bin"]) for item in mapping) + 1
    token_counts = {index: 0 for index in range(num_bins)}
    for analysis_bin in visual_token_bins:
        token_counts[int(analysis_bin)] += 1
    metadata: list[dict[str, Any]] = []
    for bin_index in range(num_bins):
        frames = [item for item in mapping if int(item["analysis_bin"]) == bin_index]
        metadata.append(
            {
                "input_index": 0,
                "temporal_bin": bin_index,
                "analysis_bin": bin_index,
                "num_visual_tokens": int(token_counts[bin_index]),
                "sampled_frame_indices": [int(item["source_frame_index"]) for item in frames],
                "sampled_timestamps": [float(item["timestamp_seconds"]) for item in frames],
                "note": "VILA visual tokens are aggregated by verified frame-position-to-analysis-bin mapping.",
            }
        )
    return tuple(metadata)


def vila_visual_token_cell_metadata(prepared: PreparedVILAInputs, visual_token_bins: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "token_index": int(token_index),
            "visual_index": visual_index,
            "sample_position": int(frame_position),
            "analysis_bin": int(analysis_bin),
            "modality": "video",
        }
        for visual_index, (token_index, frame_position, analysis_bin) in enumerate(
            zip(prepared.visual_token_indices, prepared.visual_token_frame_indices, visual_token_bins)
        )
    ]


def _profile_memory() -> dict[str, Any]:
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = None
    cuda: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            cuda = {
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    except Exception:
        pass
    return {"cpu_rss_bytes": rss, **cuda}


def run_vila_relevance_example(
    model: Any,
    example: Any,
    frame_batches: list[FrameBatch],
    resolution: ResolutionConfig,
    query_scope: str = "question",
    attention_extraction: str = "reduced_sdpa",
    vision_access_through_layer: str | None = None,
    decoder_direct_access_mask_temporal_bins: tuple[int, ...] = (),
    decoder_direct_access_through_layer: int | None = None,
    pre_encoder_remove_temporal_bins: tuple[int, ...] = (),
    pre_encoder_keep_temporal_bins: tuple[int, ...] = (),
    condition: str | None = None,
    profiler: Any | None = None,
) -> dict[str, Any]:
    if query_scope != "question":
        raise ValueError("VILA cross-model replication currently supports query_scope='question' only.")
    if decoder_direct_access_mask_temporal_bins or pre_encoder_remove_temporal_bins or pre_encoder_keep_temporal_bins:
        raise ValueError("VILA replication currently supports descriptive controls, not causal interventions.")
    if attention_extraction not in {"reduced_sdpa", "full"}:
        raise ValueError("attention_extraction must be 'full' or 'reduced_sdpa'.")
    if vision_access_through_layer not in {None, "none"}:
        raise ValueError("VILA replication does not implement Qwen decoder-access cutoff conditions.")

    from .qwen_execution import apply_frame_control

    stage = profiler.stage if profiler is not None else None
    null_stage = _NullStage()
    stage_fn = (lambda name: stage(name)) if stage is not None else (lambda name: null_stage)

    started = time.time()
    controlled_batches = apply_frame_control(frame_batches, condition)
    prompt = format_multiple_choice_prompt(example)
    with stage_fn("vila_prepare_inputs"):
        prepared = model.prepare_inputs(example, prompt, controlled_batches)
        prepared = coerce_prepared_vila_inputs(prepared)
        validate_prepared_vila_mapping(prepared, controlled_batches)
    visual_bins = visual_token_analysis_bins(prepared, controlled_batches[0])
    num_bins = max(visual_bins) + 1

    with stage_fn("vila_decoder_prefill_attention_extraction"):
        outputs = model.forward(prepared, output_attentions=True)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            raise ValueError("VILA forward did not return decoder attentions.")
        raw_temporal, absolute_mass = extract_temporal_scores_from_vila_attentions(
            attentions,
            prepared.question_token_indices,
            prepared.visual_token_indices,
            visual_bins,
            num_bins,
        )
    temporal_relevance = temporal_relevance_from_raw_scores(
        raw_temporal,
        absolute_mass,
        vila_temporal_bin_metadata(controlled_batches[0], visual_bins),
        {
            "num_layers": int(raw_temporal.shape[0]),
            "num_temporal_bins": int(raw_temporal.shape[1]),
            "num_visual_tokens": len(prepared.visual_token_indices),
            "num_question_tokens": len(prepared.question_token_indices),
            "query_scope": query_scope,
            "extraction_method": "vila_llama3_question_visual_rows",
            "input_index": 0,
            "topk": 3,
        },
    )
    with stage_fn("vila_answer_scoring"):
        scoring_outputs = model.forward(prepared, output_attentions=False)
        answer_choice_scores = score_answer_choices_from_outputs(
            scoring_outputs,
            model.tokenizer,
            example.correct_idx,
            len(example.choices),
        )
    with stage_fn("vila_generation"):
        output_ids = model.generate(prepared)
        raw_response = model.decode_new_tokens(output_ids, len(prepared.input_ids))
    predicted_idx = parse_choice_response(raw_response, len(example.choices))

    return {
        "question_id": example.question_id,
        "question_type": example.question_type,
        "question": example.question,
        "choices": list(example.choices),
        "correct_idx": example.correct_idx,
        "correct_answer": example.choices[example.correct_idx],
        "video_clip": [
            {
                "input_key": segment.input_key,
                "video_id": segment.video_id,
                "participant_id": segment.participant_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "image_time_seconds": segment.image_time_seconds,
            }
            for segment in example.inputs
        ],
        "raw_response": raw_response,
        "predicted_idx": predicted_idx,
        "correct": predicted_idx == example.correct_idx,
        "answer_choice_scores": answer_choice_scores,
        "intervention_answer_choice_scores": {},
        "sampled_frame_indices": [batch.frame_indices for batch in controlled_batches],
        "sampled_timestamps": [batch.timestamps for batch in controlled_batches],
        "frame_bin_mappings": [batch.metadata.get("frame_bin_mapping", []) for batch in controlled_batches],
        "presented_to_original_frame_bin_mappings": [
            batch.metadata.get("presented_to_original_frame_bin_mapping", []) for batch in controlled_batches
        ],
        "sampling_metadata": [batch.metadata.get("sampling", {}) for batch in controlled_batches],
        "token_layout": {
            "question_token_indices": list(prepared.question_token_indices),
            "visual_token_indices": list(prepared.visual_token_indices),
            "visual_grid_metadata": {
                "backend": "vila_llama3",
                "visual_token_frame_indices": list(prepared.visual_token_frame_indices),
            },
            "num_visual_tokens": len(prepared.visual_token_indices),
            "query_scope": query_scope,
            "visual_token_cells": vila_visual_token_cell_metadata(prepared, visual_bins),
        },
        "temporal_relevance": temporal_relevance.to_json_dict(),
        "encoder_temporal": {"available": False, "reason": "VILA replication measures decoder prefill only."},
        "encoder_attention_temporal": {"available": False, "reason": "VILA replication measures decoder prefill only."},
        "metadata": {
            "model_backend": "vila_llama3",
            "model_id": getattr(model, "checkpoint", VILA_DEFAULT_CHECKPOINT),
            "checkpoint": getattr(model, "checkpoint", VILA_DEFAULT_CHECKPOINT),
            "actual_num_frames": sum(len(batch.frame_indices) for batch in controlled_batches),
            "actual_num_visual_tokens": len(prepared.visual_token_indices),
            "num_decoder_layers": int(raw_temporal.shape[0]),
            "question_token_indices": list(prepared.question_token_indices),
            "visual_token_indices": list(prepared.visual_token_indices),
            "truncation_occurred": bool(prepared.truncation_occurred),
            "condition": condition or "baseline",
            "resolution": resolution.to_metadata(),
            "prefill_runtime_seconds": time.time() - started,
            "source_video_paths": [str(batch.video_path) if batch.video_path else None for batch in controlled_batches],
            "attention_extraction": attention_extraction,
            "answer_choice_score_source": "separate_unmodified_vila_prefill_forward",
            **_profile_memory(),
        },
    }


class _NullStage:
    def __enter__(self) -> "_NullStage":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False
