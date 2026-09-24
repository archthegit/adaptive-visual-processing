#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment1 import (
    current_git_commit,
    frame_batches_for_example,
    load_examples_by_id,
    load_manifest,
)
from src.experiment1.answer_scoring import score_answer_choice_logits
from src.experiment1.qwen_execution import (
    apply_corrected_second_per_grid_ts,
    effective_sample_fps,
    normalize_video_kwargs,
    parse_question_tags,
)
from src.experiment1.resolution import get_resolution_config
from src.experiment1.temporal_handoff import (
    TemporalHandoffConfig,
    append_jsonl,
    condition_from_baseline_scores,
    cuda_profile_prefill,
    qwen_decoder_stack,
    qwen_multimodal_decoder_inputs,
    run_custom_decoder_prefill,
    write_json,
)
from src.experiment1.token_layout import build_token_layout
from src.models.base import format_multiple_choice_prompt
from src.models.qwen import Qwen25VLWrapper, QwenConfig


DEFAULT_QUESTION_ID = "fine_grained_action_localization_3168"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the one-example Qwen temporal-handoff prefill smoke test.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--baseline-dir", default="outputs/experiment1_v3_cross_model/runs/qwen/baseline")
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID)
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_temporal_handoff/smoke_qwen")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resolution-config", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--handoff-layer", type=int, default=8)
    parser.add_argument("--retain-regions", type=int, default=2)
    parser.add_argument("--memory-tokens-per-region", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dense-equivalence-atol", type=float, default=1e-4)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _latest_complete_record(records_path: Path, question_id: str) -> dict[str, Any]:
    if not records_path.exists():
        raise FileNotFoundError(f"Baseline records file not found: {records_path}")
    matches = [
        record
        for record in _read_jsonl(records_path)
        if record.get("question_id") == question_id and record.get("status") == "complete"
    ]
    if not matches:
        raise RuntimeError(f"No complete baseline record found for {question_id} in {records_path}.")
    return matches[-1]


def _load_artifact_from_record(record: dict[str, Any]) -> dict[str, Any]:
    artifact_path = record.get("artifact")
    if not artifact_path:
        raise RuntimeError("Complete baseline record does not contain an artifact path.")
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline artifact not found: {path}")
    return json.loads(path.read_text())


def _temporal_scores_from_artifact(artifact: dict[str, Any]) -> list[list[float]]:
    temporal = artifact.get("temporal_relevance", {})
    scores = temporal.get("normalized_temporal_bin_scores")
    if scores is None:
        scores = temporal.get("normalized_scores")
    if scores is None:
        raise RuntimeError("Baseline artifact lacks normalized temporal scores.")
    return [[float(value) for value in layer] for layer in scores]


def _prediction_from_scores(scores: dict[str, Any]) -> int | None:
    logits = scores.get("choice_logits")
    if not logits:
        return None
    return int(max(range(len(logits)), key=lambda idx: float(logits[idx])))


def _max_abs_diff(a: Any, b: Any) -> float:
    import torch

    return float(torch.max(torch.abs(a.detach().float() - b.detach().float())).item())


def _build_messages(frame_batches: list[Any], resolution: Any, prompt: str, processor: Any) -> tuple[str, Any, Any, dict[str, Any]]:
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    def pil_frames(batch: Any) -> list[Any]:
        return [Image.fromarray(frame) for frame in batch.frames]

    content = []
    for batch in frame_batches:
        content.append(
            {
                "type": "video",
                "video": pil_frames(batch),
                "fps": effective_sample_fps(batch),
                **resolution.to_processor_kwargs(),
            }
        )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    video_kwargs["fps"] = effective_sample_fps(frame_batches[0])
    return rendered, image_inputs, video_inputs, normalize_video_kwargs(video_kwargs)


def _prepare_inputs(model: Qwen25VLWrapper, example: Any, frame_batches: list[Any], resolution: Any) -> tuple[Any, Any, str, Any]:
    import torch

    assert model._model is not None
    assert model._processor is not None
    prompt = format_multiple_choice_prompt(example)
    rendered, image_inputs, video_inputs, video_kwargs = _build_messages(frame_batches, resolution, prompt, model._processor)
    inputs = model._processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model._model.device)
    apply_corrected_second_per_grid_ts(inputs, frame_batches, model._model, torch)
    return inputs, rendered, prompt, video_kwargs


def _build_layout(model: Qwen25VLWrapper, example: Any, inputs: Any, rendered: str, frame_batches: list[Any]) -> Any:
    assert model._model is not None
    assert model._processor is not None
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    mm_ids = inputs.get("mm_token_type_ids")
    mm_token_type_ids = mm_ids[0].detach().cpu().tolist() if mm_ids is not None else None
    video_grid_tensor = inputs.get("video_grid_thw")
    image_grid_tensor = inputs.get("image_grid_thw")
    second_per_grid_ts = inputs.get("second_per_grid_ts")
    video_grid_thw = video_grid_tensor.detach().cpu().tolist() if video_grid_tensor is not None else []
    image_grid_thw = image_grid_tensor.detach().cpu().tolist() if image_grid_tensor is not None else []
    seconds = second_per_grid_ts.detach().cpu().tolist() if second_per_grid_ts is not None else []
    prompt = format_multiple_choice_prompt(example)
    question_text = parse_question_tags(example.question, example)
    spatial_merge_size = int(model._model.config.vision_config.spatial_merge_size)
    return build_token_layout(
        input_ids=input_ids,
        tokenizer=model._processor.tokenizer,
        rendered_prompt=rendered,
        question_text=question_text,
        video_grid_thw=video_grid_thw,
        image_grid_thw=image_grid_thw,
        visual_input_modalities=["video" for _ in frame_batches],
        spatial_merge_size=spatial_merge_size,
        video_token_id=getattr(model._processor, "video_token_id", None),
        image_token_id=getattr(model._processor, "image_token_id", None),
        mm_token_type_ids=mm_token_type_ids,
        second_per_grid_ts=seconds,
        query_scope="question",
        user_prompt_text=prompt,
    )


def _stock_logits(model: Qwen25VLWrapper, inputs: Any) -> Any:
    import torch

    assert model._model is not None
    with torch.inference_mode():
        return model._model(**inputs, output_attentions=False, use_cache=False, logits_to_keep=1).logits


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()

    manifest_records = load_manifest(args.manifest)
    selected = [record for record in manifest_records if record["question_id"] == args.question_id]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one manifest record for {args.question_id}, found {len(selected)}.")
    manifest_record = selected[0]
    examples = load_examples_by_id(args.questions_dir, selected)
    example = examples[args.question_id]
    resolution = get_resolution_config(args.resolution_config)
    frame_batches = frame_batches_for_example(
        example,
        args.mp4_dir,
        num_frames=8,
        sampling_mode="cross_model_8",
        manifest_record=manifest_record,
        frames_per_bin_override=1,
    )
    if len(frame_batches[0].frame_indices) != 8:
        raise RuntimeError("Smoke test requires exactly eight sampled frames.")

    baseline_record = _latest_complete_record(Path(args.baseline_dir) / "records.jsonl", args.question_id)
    baseline_artifact = _load_artifact_from_record(baseline_record)
    baseline_scores = _temporal_scores_from_artifact(baseline_artifact)
    if len(baseline_scores[args.handoff_layer]) != 4:
        raise RuntimeError("Expected four native temporal cells in the cross-model Qwen baseline.")

    model = Qwen25VLWrapper(QwenConfig(model_id=args.model_id, max_new_tokens=1, attn_implementation="sdpa"))
    model._load()
    assert model._model is not None
    assert model._processor is not None
    inputs, rendered, prompt, _video_kwargs = _prepare_inputs(model, example, frame_batches, resolution)
    layout = _build_layout(model, example, inputs, rendered, frame_batches)
    decoder_inputs, position_ids = qwen_multimodal_decoder_inputs(model._model, inputs)
    stack = qwen_decoder_stack(model._model)
    stock_logits = _stock_logits(model, inputs)

    conditions = ("dense_custom", "handoff_mean", "hard_evict", "random_handoff")
    dense_custom_logits = None
    artifacts: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    git_commit = current_git_commit()
    for condition in conditions:
        config = condition_from_baseline_scores(
            condition=condition,
            baseline_temporal_scores=baseline_scores,
            handoff_layer=args.handoff_layer,
            retain_count=args.retain_regions,
            num_regions=4,
            memory_tokens_per_region=args.memory_tokens_per_region,
            seed=args.seed,
            question_id=args.question_id,
        )

        def execute_condition() -> Any:
            return run_custom_decoder_prefill(
                layers=stack["layers"],
                hidden_states=decoder_inputs.clone(),
                position_ids=position_ids.clone() if position_ids is not None else None,
                layout=layout,
                config=config,
                lm_head=stack["lm_head"],
                norm=stack["norm"],
                num_attention_heads=stack["num_attention_heads"],
                head_dim=stack["head_dim"],
                rotary_emb=stack["rotary_emb"],
                layer_types=stack["layer_types"],
                sliding_window=stack["sliding_window"],
            )

        def execute_stack_only() -> Any:
            return run_custom_decoder_prefill(
                layers=stack["layers"],
                hidden_states=decoder_inputs.clone(),
                position_ids=position_ids.clone() if position_ids is not None else None,
                layout=layout,
                config=config,
                lm_head=None,
                norm=stack["norm"],
                num_attention_heads=stack["num_attention_heads"],
                head_dim=stack["head_dim"],
                rotary_emb=stack["rotary_emb"],
                layer_types=stack["layer_types"],
                sliding_window=stack["sliding_window"],
            )

        result, profile = cuda_profile_prefill(execute_condition, warmup=args.warmup, repeats=args.repeats)
        stack_result, stack_profile = cuda_profile_prefill(execute_stack_only, warmup=args.warmup, repeats=args.repeats)
        lm_logits, lm_profile = cuda_profile_prefill(
            lambda: stack["lm_head"](stack_result.final_token_hidden_state),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        logits = result.logits
        next_logits = logits[0, -1]
        scores = score_answer_choice_logits(
            next_logits,
            model._processor.tokenizer,
            example.correct_idx,
            len(example.choices),
        ).to_json_dict()
        predicted_idx = _prediction_from_scores(scores)
        artifact = {
            "question_id": example.question_id,
            "condition": condition,
            "status": "complete",
            "model_backend": "qwen",
            "model_checkpoint": args.model_id,
            "generation_supported": False,
            "generation_unsupported_reason": "Phase-1 handoff prototype is prefill-only; layer-specific decoding cache is not implemented.",
            "sampled_frame_indices": [list(frame_batches[0].frame_indices)],
            "sampled_timestamps": [list(frame_batches[0].timestamps)],
            "frame_bin_mappings": [frame_batches[0].metadata.get("frame_bin_mapping", [])],
            "choices": list(example.choices),
            "correct_idx": example.correct_idx,
            "predicted_idx": predicted_idx,
            "correct": predicted_idx == example.correct_idx,
            "answer_choice_scores": scores,
            "temporal_handoff": {
                "schema_version": "qwen_temporal_handoff_prefill_v1",
                "handoff_layer": args.handoff_layer,
                "retained_temporal_regions": list(config.retained_temporal_regions),
                "memory_tokens_per_region": config.memory_tokens_per_region,
                "condition": condition,
                "baseline_artifact": baseline_record.get("artifact"),
                "baseline_selection_layer": args.handoff_layer,
                "compaction_plan": result.compaction_plan.to_metadata() if result.compaction_plan else None,
                "instrumentation": result.instrumentation_metadata(),
            },
            "metadata": {
                "git_commit": git_commit,
                "resolution": resolution.to_metadata(),
                "sampling_mode": "cross_model_8",
                "query_scope": "question",
                "input_token_count": int(inputs["input_ids"].shape[1]),
                "original_sequence_length": int(decoder_inputs.shape[1]),
                "final_sequence_length": int(result.final_hidden_states.shape[1]),
                "cuda_profile": {
                    "combined_prefill": profile,
                    "decoder_stack_excluding_lm_head": stack_profile,
                    "final_token_lm_head": lm_profile,
                },
                "final_token_lm_head_shape": list(lm_logits.shape),
            },
        }
        artifact_path = output_dir / f"{condition}.json"
        write_json(artifact_path, artifact)
        append_jsonl(
            records_path,
            {
                "question_id": example.question_id,
                "condition": condition,
                "status": "complete",
                "artifact": str(artifact_path),
            },
        )
        artifacts.append(artifact)
        profiles[condition] = profile
        if condition == "dense_custom":
            dense_custom_logits = logits

    if dense_custom_logits is None:
        raise RuntimeError("dense_custom condition did not run.")
    dense_max_logit_diff = _max_abs_diff(stock_logits, dense_custom_logits)
    dense_scores = artifacts[0]["answer_choice_scores"]
    stock_scores = score_answer_choice_logits(
        stock_logits[0, -1],
        model._processor.tokenizer,
        example.correct_idx,
        len(example.choices),
    ).to_json_dict()
    equivalence = {
        "question_id": example.question_id,
        "stock_vs_dense_custom_max_logit_difference": dense_max_logit_diff,
        "tolerance": args.dense_equivalence_atol,
        "passed": bool(dense_max_logit_diff <= args.dense_equivalence_atol),
        "stock_answer_choice_scores": stock_scores,
        "dense_custom_answer_choice_scores": dense_scores,
        "stock_predicted_idx": _prediction_from_scores(stock_scores),
        "dense_custom_predicted_idx": artifacts[0]["predicted_idx"],
    }
    write_json(output_dir / "dense_equivalence_report.json", equivalence)
    write_json(
        output_dir / "profile.json",
        {
            "question_id": example.question_id,
            "git_commit": git_commit,
            "profiles": profiles,
            "dense_equivalence": equivalence,
        },
    )
    if not equivalence["passed"]:
        raise RuntimeError(
            "dense_custom did not match stock dense forward within tolerance: "
            f"max_diff={dense_max_logit_diff}, tolerance={args.dense_equivalence_atol}"
        )


if __name__ == "__main__":
    main()
