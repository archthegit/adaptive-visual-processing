from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.dataset import VQAExample
from src.frame_sampling import FrameBatch
from src.models.base import format_multiple_choice_prompt, parse_choice_response, parse_question_tags
from src.models.qwen import Qwen25VLWrapper
from src.models.qwen_temporal_rope import temporal_position_interval

from .answer_scoring import score_answer_choices_from_outputs
from .encoder_temporal import vision_temporal_capture_context
from .qwen_vision_attention import qwen_vision_attention_capture_context
from .qwen_reduced_attention import masked_eager_attention_context, reduced_attention_context
from .relevance import aggregate_question_to_visual_attention
from .resolution import ResolutionConfig
from .temporal import build_temporal_relevance_from_token_scores, represented_sampled_frames
from .token_layout import build_token_layout


def _pil_frames(batch: FrameBatch) -> list[Image.Image]:
    return [Image.fromarray(frame) for frame in batch.frames]


def _pil_image(batch: FrameBatch) -> Image.Image:
    return Image.fromarray(batch.frames[0])


def frame_indices_for_temporal_bins(num_frames: int, num_temporal_bins: int, temporal_bins: tuple[int, ...]) -> list[int]:
    selected: set[int] = set()
    for temporal_bin in temporal_bins:
        if temporal_bin < 0 or temporal_bin >= num_temporal_bins:
            raise ValueError(f"Temporal bin {temporal_bin} is outside available bins 0..{num_temporal_bins - 1}.")
        start = int(round(temporal_bin * num_frames / num_temporal_bins))
        end = int(round((temporal_bin + 1) * num_frames / num_temporal_bins))
        if end <= start:
            end = min(num_frames, start + 1)
        selected.update(range(start, end))
    return sorted(selected)


def mask_frame_batch_temporal_bins(
    batch: FrameBatch,
    num_temporal_bins: int,
    temporal_bins: tuple[int, ...],
) -> FrameBatch:
    if not temporal_bins:
        return batch
    frames = batch.frames.copy()
    indices = frame_indices_for_temporal_bins(len(frames), num_temporal_bins, temporal_bins)
    for index in indices:
        frames[index] = 0
    metadata = dict(batch.metadata)
    metadata["pre_encoder_removed_temporal_bins"] = list(temporal_bins)
    metadata["pre_encoder_masked_sample_positions"] = indices
    return replace(batch, frames=frames, metadata=metadata)


def keep_frame_batch_temporal_bins(
    batch: FrameBatch,
    num_temporal_bins: int,
    temporal_bins: tuple[int, ...],
) -> FrameBatch:
    if not temporal_bins:
        raise ValueError("At least one temporal bin must be selected for keep/pruning.")
    positions = frame_indices_for_temporal_bins(len(batch.frames), num_temporal_bins, temporal_bins)
    if not positions:
        raise ValueError("Selected temporal bins did not map to any sampled frame positions.")
    metadata = dict(batch.metadata)
    metadata["pre_encoder_kept_temporal_bins"] = list(temporal_bins)
    metadata["pre_encoder_kept_sample_positions"] = positions
    return replace(
        batch,
        frames=batch.frames[positions],
        frame_indices=tuple(batch.frame_indices[position] for position in positions),
        timestamps=tuple(batch.timestamps[position] for position in positions),
        metadata=metadata,
    )


def repeat_frame_batch(batch: FrameBatch, sample_position: int = 0) -> FrameBatch:
    if len(batch.frames) == 0:
        raise ValueError("Cannot repeat a frame from an empty FrameBatch.")
    if sample_position < 0 or sample_position >= len(batch.frames):
        raise ValueError(f"sample_position {sample_position} is outside sampled frame range.")
    frames = batch.frames.copy()
    frames[:] = frames[sample_position]
    metadata = dict(batch.metadata)
    metadata["control"] = "repeated_frame"
    metadata["repeated_sample_position"] = int(sample_position)
    metadata["repeated_source_frame_index"] = int(batch.frame_indices[sample_position])
    return replace(batch, frames=frames, metadata=metadata)


def reverse_frame_batch(batch: FrameBatch) -> FrameBatch:
    metadata = dict(batch.metadata)
    metadata["control"] = "reversed_video"
    metadata["temporal_positions_preserved"] = True
    metadata["presented_source_frame_indices"] = list(reversed(batch.frame_indices))
    metadata["presented_source_timestamps"] = list(reversed(batch.timestamps))
    mapping = list(batch.metadata.get("frame_bin_mapping", []) or [])
    if mapping:
        by_position = {int(item["sample_position"]): item for item in mapping}
        count = len(batch.frame_indices)
        metadata["presented_to_original_frame_bin_mapping"] = [
            {
                "presented_sample_position": presented,
                "original_sample_position": count - 1 - presented,
                "original_source_frame_index": int(by_position[count - 1 - presented]["source_frame_index"]),
                "original_analysis_bin": int(by_position[count - 1 - presented]["analysis_bin"]),
                "presented_analysis_bin": int(by_position[presented]["analysis_bin"]),
            }
            for presented in range(count)
            if presented in by_position and count - 1 - presented in by_position
        ]
    return replace(
        batch,
        frames=batch.frames[::-1].copy(),
        metadata=metadata,
    )


def apply_frame_control(frame_batches: list[FrameBatch], condition: str | None) -> list[FrameBatch]:
    if condition == "repeated_frame":
        return [
            repeat_frame_batch(batch) if batch.metadata.get("input_modality") != "image" else batch
            for batch in frame_batches
        ]
    if condition == "reversed_video":
        return [
            reverse_frame_batch(batch) if batch.metadata.get("input_modality") != "image" else batch
            for batch in frame_batches
        ]
    return frame_batches


def normalize_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep qwen-vl-utils kwargs compatible with strict processor validators."""
    normalized = dict(video_kwargs)
    fps = normalized.get("fps")
    if isinstance(fps, list) and len(fps) >= 1 and all(item == fps[0] for item in fps):
        normalized["fps"] = fps[0]
    elif isinstance(fps, list):
        raise ValueError(
            "Qwen processor returned per-video FPS values, but this installed processor validates `fps` as a scalar. "
            f"Refusing to collapse different FPS values silently: {fps}"
        )
    return normalized


def effective_sample_fps(batch: FrameBatch) -> float:
    if len(batch.timestamps) <= 1:
        return 1.0
    duration = float(batch.timestamps[-1] - batch.timestamps[0])
    if duration <= 0:
        return 1.0
    return float((len(batch.timestamps) - 1) / duration)


def validate_scalar_video_fps_compatibility(frame_batches: list[FrameBatch], tolerance: float = 1e-6) -> None:
    video_fps = [
        effective_sample_fps(batch)
        for batch in frame_batches
        if batch.metadata.get("input_modality") != "image"
    ]
    if len(video_fps) <= 1:
        return
    first = video_fps[0]
    if any(abs(fps - first) > tolerance for fps in video_fps[1:]):
        raise ValueError(
            "Multiple video inputs have different effective FPS values, but the pinned Qwen processor validates "
            f"`fps` as a scalar. Use a single-video debug manifest or a processor version with per-video FPS support. "
            f"effective_fps={video_fps}"
        )


def corrected_second_per_grid_ts(frame_batches: list[FrameBatch], temporal_patch_size: int) -> list[float]:
    return [
        float(temporal_patch_size) / effective_sample_fps(batch)
        for batch in frame_batches
        if batch.metadata.get("input_modality") != "image"
    ]


def apply_corrected_second_per_grid_ts(inputs: Any, frame_batches: list[FrameBatch], model: Any, torch_module: Any) -> list[float]:
    temporal_patch_size = int(model.config.vision_config.temporal_patch_size)
    seconds = corrected_second_per_grid_ts(frame_batches, temporal_patch_size)
    if not seconds:
        return []
    inputs["second_per_grid_ts"] = torch_module.as_tensor(seconds, dtype=torch_module.float32, device=model.device)
    return seconds


def temporal_position_intervals(model: Any, second_per_grid_ts: list[float]) -> list[int]:
    tokens_per_second = model.config.vision_config.tokens_per_second
    return [temporal_position_interval(tokens_per_second, seconds) for seconds in second_per_grid_ts]


def cuda_memory_metadata(torch_module: Any) -> dict[str, int]:
    if not torch_module.cuda.is_available():
        return {}
    return {
        "cuda_memory_allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "cuda_memory_reserved_bytes": int(torch_module.cuda.memory_reserved()),
        "cuda_max_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "cuda_max_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    }


def visual_token_cell_metadata(layout: Any, frame_batches: list[FrameBatch]) -> list[dict[str, Any]]:
    records = []
    for cell in layout.visual_cells:
        batch = frame_batches[cell.input_index]
        represented = represented_sampled_frames(batch, cell.temporal_index, cell.grid_t)
        records.append(
            {
                "token_index": cell.token_index,
                "visual_index": cell.visual_index,
                "video_input_index": cell.input_index,
                "temporal_bin": cell.temporal_index,
                "spatial_row": cell.spatial_y,
                "spatial_col": cell.spatial_x,
                "grid_t": cell.grid_t,
                "grid_h": cell.grid_h,
                "grid_w": cell.grid_w,
                "seconds_per_grid": cell.seconds_per_grid,
                "qwen_timestamp": cell.timestamp,
                **represented,
            }
        )
    return records


def next_token_topk_from_outputs(outputs: Any, k: int = 10) -> list[dict[str, float | int]]:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        return []
    next_logits = logits[0, -1].detach().float().cpu()
    values, indices = next_logits.topk(min(k, next_logits.shape[0]))
    return [
        {"token_id": int(token_id), "logit": float(logit)}
        for token_id, logit in zip(indices.tolist(), values.tolist())
    ]


def remap_encoder_attention_to_analysis_bins(encoder_json: dict[str, Any], batch: FrameBatch) -> dict[str, Any]:
    mapping_items = batch.metadata.get("frame_bin_mapping", []) or []
    if not mapping_items or not encoder_json.get("available"):
        return encoder_json
    qwen_layers = encoder_json.get("normalized_incoming_temporal_attention") or []
    if not qwen_layers:
        return encoder_json
    num_qwen_bins = int(encoder_json.get("num_temporal_bins") or len(qwen_layers[0][0]))
    num_samples = len(batch.frame_indices)
    position_to_analysis = {int(item["sample_position"]): int(item["analysis_bin"]) for item in mapping_items}
    num_analysis_bins = max(position_to_analysis.values()) + 1
    qwen_to_analysis: list[dict[int, float]] = []
    for temporal_index in range(num_qwen_bins):
        start = int(round(temporal_index * num_samples / num_qwen_bins))
        end = int(round((temporal_index + 1) * num_samples / num_qwen_bins))
        if end <= start:
            end = min(num_samples, start + 1)
        bins = [position_to_analysis[pos] for pos in range(start, end) if pos in position_to_analysis]
        weights: dict[int, float] = {}
        for analysis_bin in bins:
            weights[analysis_bin] = weights.get(analysis_bin, 0.0) + 1.0 / max(1, len(bins))
        qwen_to_analysis.append(weights or {temporal_index: 1.0})

    remapped_layers = []
    for layer in qwen_layers:
        remapped_heads = []
        for head in layer:
            values = np.asarray(head, dtype=np.float64)
            remapped = np.zeros(num_analysis_bins, dtype=np.float64)
            for qwen_bin, weights in enumerate(qwen_to_analysis):
                for analysis_bin, weight in weights.items():
                    if analysis_bin < num_analysis_bins and qwen_bin < values.size:
                        remapped[analysis_bin] += values[qwen_bin] * weight
            total = float(remapped.sum())
            if total > 0:
                remapped /= total
            remapped_heads.append(remapped.tolist())
        remapped_layers.append(remapped_heads)
    updated = dict(encoder_json)
    updated["qwen_num_temporal_bins"] = encoder_json.get("num_temporal_bins")
    updated["num_temporal_bins"] = num_analysis_bins
    updated["analysis_bin_remap_active"] = True
    updated["normalized_incoming_temporal_attention"] = remapped_layers
    return updated


def run_qwen_relevance_example(
    model: Qwen25VLWrapper,
    example: VQAExample,
    frame_batches: list[FrameBatch],
    resolution: ResolutionConfig,
    query_scope: str = "question",
    attention_extraction: str = "full",
    vision_access_through_layer: str | int | None = None,
    decoder_direct_access_mask_temporal_bins: tuple[int, ...] | None = None,
    decoder_direct_access_through_layer: int | None = None,
    pre_encoder_remove_temporal_bins: tuple[int, ...] | None = None,
    pre_encoder_keep_temporal_bins: tuple[int, ...] | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("Real Qwen execution requires torch and qwen-vl-utils.") from exc

    model._load()
    assert model._model is not None
    assert model._processor is not None
    frame_batches = apply_frame_control(frame_batches, condition)
    validate_scalar_video_fps_compatibility(frame_batches)

    prompt = format_multiple_choice_prompt(example)
    question_text = parse_question_tags(example.question, example)

    def build_messages(batches: list[FrameBatch]) -> tuple[list[dict[str, Any]], str, Any, Any, dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for batch in batches:
            if batch.metadata.get("input_modality") == "image":
                content.append(
                    {
                        "type": "image",
                        "image": _pil_image(batch),
                        **resolution.to_processor_kwargs(),
                    }
                )
            else:
                content.append(
                    {
                        "type": "video",
                        "video": _pil_frames(batch),
                        "fps": effective_sample_fps(batch),
                        **resolution.to_processor_kwargs(),
                    }
                )
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered_prompt = model._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        video_batches = [batch for batch in batches if batch.metadata.get("input_modality") != "image"]
        if video_batches:
            video_kwargs["fps"] = effective_sample_fps(video_batches[0])
        return messages, rendered_prompt, image_inputs, video_inputs, normalize_video_kwargs(video_kwargs)

    _messages, rendered, image_inputs, video_inputs, video_kwargs = build_messages(frame_batches)

    inputs = model._processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model._model.device)
    corrected_seconds = apply_corrected_second_per_grid_ts(inputs, frame_batches, model._model, torch)

    video_grid_tensor = inputs.get("video_grid_thw")
    video_grid_thw = video_grid_tensor.detach().cpu().tolist() if video_grid_tensor is not None else []
    if pre_encoder_remove_temporal_bins or pre_encoder_keep_temporal_bins:
        if not video_grid_thw:
            raise ValueError("Pre-encoder temporal intervention requires video_grid_thw.")
        masked_batches = list(frame_batches)
        if pre_encoder_remove_temporal_bins:
            masked_batches[0] = mask_frame_batch_temporal_bins(
                masked_batches[0],
                int(video_grid_thw[0][0]),
                tuple(pre_encoder_remove_temporal_bins),
            )
        if pre_encoder_keep_temporal_bins:
            masked_batches[0] = keep_frame_batch_temporal_bins(
                masked_batches[0],
                int(video_grid_thw[0][0]),
                tuple(pre_encoder_keep_temporal_bins),
            )
        frame_batches = masked_batches
        _messages, rendered, image_inputs, video_inputs, video_kwargs = build_messages(frame_batches)
        inputs = model._processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model._model.device)
        corrected_seconds = apply_corrected_second_per_grid_ts(inputs, frame_batches, model._model, torch)

    temporal_patch_size = float(model._processor.video_processor.temporal_patch_size)
    actual_seconds = [temporal_patch_size / effective_sample_fps(batch) for batch in frame_batches if batch.metadata.get("input_modality") != "image"]
    if actual_seconds and inputs.get("second_per_grid_ts") is not None:
        current_seconds = inputs["second_per_grid_ts"]
        inputs["second_per_grid_ts"] = torch.tensor(actual_seconds, dtype=current_seconds.dtype, device=current_seconds.device)
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    mm_ids = mm_token_type_ids[0].detach().cpu().tolist() if mm_token_type_ids is not None else None
    video_grid_tensor = inputs.get("video_grid_thw")
    image_grid_tensor = inputs.get("image_grid_thw")
    video_grid_thw = video_grid_tensor.detach().cpu().tolist() if video_grid_tensor is not None else []
    image_grid_thw = image_grid_tensor.detach().cpu().tolist() if image_grid_tensor is not None else []
    second_per_grid_ts = inputs.get("second_per_grid_ts")
    seconds = second_per_grid_ts.detach().cpu().tolist() if second_per_grid_ts is not None else []
    temporal_intervals = temporal_position_intervals(model._model, [float(item) for item in seconds])
    spatial_merge_size = int(model._model.config.vision_config.spatial_merge_size)
    visual_input_modalities = [
        "image" if batch.metadata.get("input_modality") == "image" else "video"
        for batch in frame_batches
    ]
    layout = build_token_layout(
        input_ids=input_ids,
        tokenizer=model._processor.tokenizer,
        rendered_prompt=rendered,
        question_text=question_text,
        video_grid_thw=video_grid_thw,
        image_grid_thw=image_grid_thw,
        visual_input_modalities=visual_input_modalities,
        spatial_merge_size=spatial_merge_size,
        video_token_id=getattr(model._processor, "video_token_id", None),
        image_token_id=getattr(model._processor, "image_token_id", None),
        mm_token_type_ids=mm_ids,
        second_per_grid_ts=seconds,
        query_scope=query_scope,
        user_prompt_text=prompt,
    )

    def vision_attention_context(encoder_capture: Any):
        if not video_grid_thw:
            return nullcontext(None)
        return qwen_vision_attention_capture_context(
            model._model,
            video_grid_thw[0],
            spatial_merge_size,
            encoder_capture.reverse_indices,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    prefill_next_token_topk: list[dict[str, float | int]] = []
    intervention_answer_choice_scores: dict[str, Any] = {}
    decoder_mask_bins = tuple(decoder_direct_access_mask_temporal_bins or ())
    pre_encoder_bins = tuple(pre_encoder_remove_temporal_bins or ())
    keep_bins = tuple(pre_encoder_keep_temporal_bins or ())
    decoder_intervention_active = bool(decoder_mask_bins) or vision_access_through_layer not in {None, "none"}
    if attention_extraction == "full":
        context = (
            masked_eager_attention_context(
                model._model,
                layout,
                vision_access_through_layer,
                decoder_direct_access_mask_temporal_bins=decoder_mask_bins,
                decoder_direct_access_through_layer=decoder_direct_access_through_layer,
            )
            if vision_access_through_layer not in {None, "none"} or decoder_mask_bins
            else None
        )
        if context is None:
            with vision_temporal_capture_context(
                model._model,
                video_grid_thw,
                spatial_merge_size,
                video_grid_tensor=video_grid_tensor,
            ) as encoder_capture:
                with vision_attention_context(encoder_capture) as vision_attention_capture:
                    with torch.inference_mode():
                        outputs = model._model(**inputs, output_attentions=True, use_cache=False)
        else:
            with context:
                with vision_temporal_capture_context(
                    model._model,
                    video_grid_thw,
                    spatial_merge_size,
                    video_grid_tensor=video_grid_tensor,
                ) as encoder_capture:
                    with vision_attention_context(encoder_capture) as vision_attention_capture:
                        with torch.inference_mode():
                            outputs = model._model(**inputs, output_attentions=True, use_cache=False)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            raise RuntimeError(
                "Qwen did not return decoder attentions. Ensure attn_implementation='eager' and output_attentions=True."
            )
        prefill_next_token_topk = next_token_topk_from_outputs(outputs)
        token_scores = aggregate_question_to_visual_attention(
            attentions, layout.question_token_indices, layout.visual_token_indices
        )
        temporal_relevance = build_temporal_relevance_from_token_scores(
            token_scores,
            layout,
            frame_batches,
            "returned_full_attention_temporally_reduced_after_forward",
        )
        del outputs, attentions
    elif attention_extraction == "reduced_sdpa":
        expected_layers = int(model._model.config.text_config.num_hidden_layers)
        with reduced_attention_context(
            model._model,
            layout,
            vision_access_through_layer,
            decoder_direct_access_mask_temporal_bins=decoder_mask_bins,
            decoder_direct_access_through_layer=decoder_direct_access_through_layer,
        ) as capture:
            with vision_temporal_capture_context(
                model._model,
                video_grid_thw,
                spatial_merge_size,
                video_grid_tensor=video_grid_tensor,
            ) as encoder_capture:
                with vision_attention_context(encoder_capture) as vision_attention_capture:
                    with torch.inference_mode():
                        outputs = model._model(**inputs, output_attentions=False, use_cache=False)
        prefill_next_token_topk = next_token_topk_from_outputs(outputs)
        del outputs
        token_scores = capture.ordered_token_scores(expected_layers=expected_layers)
        temporal_relevance = build_temporal_relevance_from_token_scores(
            token_scores,
            layout,
            frame_batches,
            "qwen_reduced_sdpa_temporal_question_visual_rows",
        )
    else:
        raise ValueError("attention_extraction must be 'full' or 'reduced_sdpa'.")

    prefill_runtime = time.time() - started

    scoring_started = time.time()
    with torch.inference_mode():
        scoring_outputs = model._model(**inputs, output_attentions=False, use_cache=False)
    answer_choice_scores = score_answer_choices_from_outputs(
        scoring_outputs, model._processor.tokenizer, example.correct_idx, len(example.choices)
    )
    unmodified_prefill_next_token_topk = next_token_topk_from_outputs(scoring_outputs)
    del scoring_outputs
    if decoder_intervention_active:
        with masked_eager_attention_context(
            model._model,
            layout,
            vision_access_through_layer,
            decoder_direct_access_mask_temporal_bins=decoder_mask_bins,
            decoder_direct_access_through_layer=decoder_direct_access_through_layer,
        ):
            with torch.inference_mode():
                intervention_scoring_outputs = model._model(**inputs, output_attentions=False, use_cache=False)
        intervention_answer_choice_scores = score_answer_choices_from_outputs(
            intervention_scoring_outputs,
            model._processor.tokenizer,
            example.correct_idx,
            len(example.choices),
        )
        del intervention_scoring_outputs
    answer_scoring_runtime = time.time() - scoring_started

    memory_after_prefill = cuda_memory_metadata(torch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen_started = time.time()
    if vision_access_through_layer in {None, "none"}:
        if decoder_mask_bins:
            with masked_eager_attention_context(
                model._model,
                layout,
                vision_access_through_layer,
                decoder_direct_access_mask_temporal_bins=decoder_mask_bins,
                decoder_direct_access_through_layer=decoder_direct_access_through_layer,
            ):
                with torch.inference_mode():
                    output_ids = model._model.generate(**inputs, max_new_tokens=model.config.max_new_tokens)
        else:
            with torch.inference_mode():
                output_ids = model._model.generate(**inputs, max_new_tokens=model.config.max_new_tokens)
    else:
        with masked_eager_attention_context(
            model._model,
            layout,
            vision_access_through_layer,
            decoder_direct_access_mask_temporal_bins=decoder_mask_bins,
            decoder_direct_access_through_layer=decoder_direct_access_through_layer,
        ):
            with torch.inference_mode():
                output_ids = model._model.generate(**inputs, max_new_tokens=model.config.max_new_tokens)
    raw_response = model._processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    predicted_idx = parse_choice_response(raw_response, len(example.choices))
    temporal_rope_metadata = (
        model._temporal_rope_patch_info.to_metadata()
        if getattr(model, "_temporal_rope_patch_info", None) is not None
        else {"transformers_version": "unknown", "temporal_rope_patch_active": False}
    )
    if vision_attention_capture is None:
        encoder_attention_temporal = {"available": False, "reason": "no video_grid_thw"}
    else:
        try:
            expected_vision_layers = int(model._model.config.vision_config.depth)
        except Exception:
            expected_vision_layers = None
        try:
            encoder_attention_temporal = vision_attention_capture.to_json_dict(expected_layers=expected_vision_layers)
            encoder_attention_temporal = remap_encoder_attention_to_analysis_bins(
                encoder_attention_temporal,
                frame_batches[0],
            )
        except Exception as exc:
            encoder_attention_temporal = {"available": False, "error": str(exc)}

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
        "intervention_answer_choice_scores": intervention_answer_choice_scores,
        "sampled_frame_indices": [batch.frame_indices for batch in frame_batches],
        "sampled_timestamps": [batch.timestamps for batch in frame_batches],
        "frame_bin_mappings": [batch.metadata.get("frame_bin_mapping", []) for batch in frame_batches],
        "presented_to_original_frame_bin_mappings": [
            batch.metadata.get("presented_to_original_frame_bin_mapping", []) for batch in frame_batches
        ],
        "sampling_metadata": [batch.metadata.get("sampling", {}) for batch in frame_batches],
        "token_layout": {
            "question_token_indices": layout.question_token_indices,
            "visual_token_indices": layout.visual_token_indices,
            "visual_grid_metadata": layout.visual_grid_metadata,
            "num_visual_tokens": layout.num_visual_tokens,
            "query_scope": layout.query_scope,
            "visual_token_cells": visual_token_cell_metadata(layout, frame_batches),
        },
        "temporal_relevance": temporal_relevance.to_json_dict(),
        "encoder_temporal": encoder_capture.to_json_dict(),
        "encoder_attention_temporal": encoder_attention_temporal,
        "metadata": {
            "model_id": model.config.model_id,
            **temporal_rope_metadata,
            "attn_implementation": model.config.attn_implementation,
            "attention_extraction": attention_extraction,
            "vision_access_through_layer": vision_access_through_layer or "none",
            "decoder_direct_access_mask_temporal_bins": list(decoder_mask_bins),
            "decoder_direct_access_through_layer": decoder_direct_access_through_layer,
            "pre_encoder_removed_temporal_bins": list(pre_encoder_bins),
            "pre_encoder_kept_temporal_bins": list(keep_bins),
            "pre_encoder_masked_sample_positions": [
                batch.metadata.get("pre_encoder_masked_sample_positions", []) for batch in frame_batches
            ],
            "pre_encoder_kept_sample_positions": [
                batch.metadata.get("pre_encoder_kept_sample_positions", []) for batch in frame_batches
            ],
            "condition": condition or "baseline",
            "resolution": resolution.to_metadata(),
            "prefill_runtime_seconds": prefill_runtime,
            "answer_scoring_runtime_seconds": answer_scoring_runtime,
            "generation_runtime_seconds": time.time() - gen_started,
            "input_token_count": len(input_ids),
            "video_grid_thw": video_grid_thw,
            "image_grid_thw": image_grid_thw,
            "visual_input_modalities": visual_input_modalities,
            "second_per_grid_ts": seconds,
            "corrected_second_per_grid_ts": corrected_seconds,
            "temporal_position_interval": temporal_intervals,
            "source_video_paths": [str(batch.video_path) if batch.video_path else None for batch in frame_batches],
            "effective_sample_fps": [effective_sample_fps(batch) for batch in frame_batches],
            "prefill_next_token_topk": prefill_next_token_topk,
            "unmodified_prefill_next_token_topk": unmodified_prefill_next_token_topk,
            "answer_choice_score_source": (
                "separate_unmodified_prefill_forward"
                if not pre_encoder_bins and not keep_bins
                else "separate_prefill_forward_on_pre_encoder_modified_inputs"
            ),
            "intervention_answer_choice_score_source": (
                "separate_masked_eager_prefill_forward"
                if decoder_intervention_active
                else None
            ),
            "answer_choice_comparison_scope": (
                "compare_answer_choice_scores_with_matching_unmasked_baseline_artifact"
                if pre_encoder_bins or keep_bins
                else "same_artifact_intervention_answer_choice_scores"
                if decoder_intervention_active
                else "baseline_only"
            ),
            **memory_after_prefill,
        },
    }
