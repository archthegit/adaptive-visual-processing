from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.dataset import VQAExample
from src.frame_sampling import FrameBatch
from src.models.base import format_multiple_choice_prompt, parse_choice_response, parse_question_tags
from src.models.qwen import Qwen25VLWrapper

from .qwen_reduced_attention import reduced_attention_context
from .relevance import build_layerwise_relevance_from_token_scores, compute_layerwise_relevance
from .resolution import ResolutionConfig
from .token_layout import build_token_layout


def _pil_frames(batch: FrameBatch) -> list[Image.Image]:
    return [Image.fromarray(frame) for frame in batch.frames]


def normalize_video_kwargs(video_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep qwen-vl-utils kwargs compatible with strict processor validators."""
    normalized = dict(video_kwargs)
    fps = normalized.get("fps")
    if isinstance(fps, list) and len(fps) >= 1 and all(item == fps[0] for item in fps):
        normalized["fps"] = fps[0]
    return normalized


def cuda_memory_metadata(torch_module: Any) -> dict[str, int]:
    if not torch_module.cuda.is_available():
        return {}
    return {
        "cuda_memory_allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "cuda_memory_reserved_bytes": int(torch_module.cuda.memory_reserved()),
        "cuda_max_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "cuda_max_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    }


def run_qwen_relevance_example(
    model: Qwen25VLWrapper,
    example: VQAExample,
    frame_batches: list[FrameBatch],
    resolution: ResolutionConfig,
    query_scope: str = "question",
    attention_extraction: str = "full",
) -> dict[str, Any]:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("Real Qwen execution requires torch and qwen-vl-utils.") from exc

    model._load()
    assert model._model is not None
    assert model._processor is not None

    prompt = format_multiple_choice_prompt(example)
    question_text = parse_question_tags(example.question, example)
    content: list[dict[str, Any]] = []
    for batch in frame_batches:
        content.append(
            {
                "type": "video",
                "video": _pil_frames(batch),
                "sample_fps": 1.0,
                **resolution.to_processor_kwargs(),
            }
        )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = model._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    video_kwargs = normalize_video_kwargs(video_kwargs)

    inputs = model._processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model._model.device)

    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    mm_ids = mm_token_type_ids[0].detach().cpu().tolist() if mm_token_type_ids is not None else None
    video_grid_thw = inputs["video_grid_thw"].detach().cpu().tolist()
    second_per_grid_ts = inputs.get("second_per_grid_ts")
    seconds = second_per_grid_ts.detach().cpu().tolist() if second_per_grid_ts is not None else []
    spatial_merge_size = int(model._model.config.vision_config.spatial_merge_size)
    layout = build_token_layout(
        input_ids=input_ids,
        tokenizer=model._processor.tokenizer,
        rendered_prompt=rendered,
        question_text=question_text,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        video_token_id=getattr(model._processor, "video_token_id", None),
        image_token_id=getattr(model._processor, "image_token_id", None),
        mm_token_type_ids=mm_ids,
        second_per_grid_ts=seconds,
        query_scope=query_scope,
        user_prompt_text=prompt,
    )

    started = time.time()
    if attention_extraction == "full":
        with torch.inference_mode():
            outputs = model._model(**inputs, output_attentions=True, use_cache=False)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            raise RuntimeError(
                "Qwen did not return decoder attentions. Ensure attn_implementation='eager' and output_attentions=True."
            )
        relevance = compute_layerwise_relevance(attentions, layout)
        del outputs, attentions
    elif attention_extraction == "reduced_sdpa":
        expected_layers = int(model._model.config.text_config.num_hidden_layers)
        with reduced_attention_context(model._model, layout) as capture:
            with torch.inference_mode():
                model._model(**inputs, output_attentions=False, use_cache=False)
        token_scores = capture.ordered_token_scores(expected_layers=expected_layers)
        relevance = build_layerwise_relevance_from_token_scores(
            token_scores, layout, "qwen_reduced_sdpa_question_visual_rows"
        )
    else:
        raise ValueError("attention_extraction must be 'full' or 'reduced_sdpa'.")

    prefill_runtime = time.time() - started
    memory_after_prefill = cuda_memory_metadata(torch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen_started = time.time()
    with torch.inference_mode():
        output_ids = model._model.generate(**inputs, max_new_tokens=model.config.max_new_tokens)
    raw_response = model._processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    predicted_idx = parse_choice_response(raw_response, len(example.choices))

    return {
        "question_id": example.question_id,
        "question_type": example.question_type,
        "question": example.question,
        "choices": list(example.choices),
        "correct_idx": example.correct_idx,
        "correct_answer": example.choices[example.correct_idx],
        "raw_response": raw_response,
        "predicted_idx": predicted_idx,
        "correct": predicted_idx == example.correct_idx,
        "sampled_frame_indices": [batch.frame_indices for batch in frame_batches],
        "sampled_timestamps": [batch.timestamps for batch in frame_batches],
        "token_layout": {
            "question_token_indices": layout.question_token_indices,
            "visual_token_indices": layout.visual_token_indices,
            "visual_grid_metadata": layout.visual_grid_metadata,
            "num_visual_tokens": layout.num_visual_tokens,
            "query_scope": layout.query_scope,
        },
        "relevance": relevance.to_json_dict(),
        "metadata": {
            "model_id": model.config.model_id,
            "attn_implementation": model.config.attn_implementation,
            "attention_extraction": attention_extraction,
            "resolution": resolution.to_metadata(),
            "prefill_runtime_seconds": prefill_runtime,
            "generation_runtime_seconds": time.time() - gen_started,
            "input_token_count": len(input_ids),
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": seconds,
            "source_video_paths": [str(batch.video_path) if batch.video_path else None for batch in frame_batches],
            **memory_after_prefill,
        },
    }
