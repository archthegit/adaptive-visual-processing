#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.resolution import get_resolution_config
from src.io import append_jsonl, write_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 query relevance + vision/query fusion.")
    parser.add_argument("--questions-dir", default=None)
    parser.add_argument("--mp4-dir", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--sampling-mode",
        choices=["legacy", "realtime", "fixed_budget"],
        default="legacy",
        help=(
            "legacy uses the historical uniform --num-frames sampler; realtime uses the Experiment 1 adaptive "
            "full-coverage policy; fixed_budget uses 128 frames, 16 bins, 8 frames per bin."
        ),
    )
    parser.add_argument(
        "--sampling-policy-json",
        default=None,
        help="Path to split_summary.json or a policy JSON containing realtime_sampling_policy for --sampling-mode realtime.",
    )
    parser.add_argument(
        "--frames-per-bin",
        type=int,
        default=None,
        help=(
            "Override the frozen realtime policy frames_per_bin. This is intended for matched cross-model "
            "replication conditions, e.g. one deterministic center frame per temporal bin."
        ),
    )
    parser.add_argument("--model-backend", choices=["qwen", "vila_llama3"], default="qwen")
    parser.add_argument(
        "--model-checkpoint",
        default=None,
        help="Model checkpoint identifier. Defaults to the backend's canonical Experiment 1 checkpoint.",
    )
    parser.add_argument(
        "--frame-budget-mode",
        default="total",
        choices=["total", "per-input"],
        help=(
            "Interpret --num-frames as a total budget split across visual inputs, "
            "or as the legacy per-input frame count."
        ),
    )
    parser.add_argument("--resolution-config", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--vision-access-through-layer", default="none")
    parser.add_argument(
        "--decoder-mask-temporal-bin",
        action="append",
        type=int,
        default=None,
        help="Block direct decoder attention from text/question rows to this Qwen temporal bin. Can be repeated.",
    )
    parser.add_argument(
        "--decoder-direct-access-through-layer",
        type=int,
        default=None,
        help=(
            "For decoder direct-access masking, allow selected visual bins through this decoder layer "
            "and block direct question access only in later layers."
        ),
    )
    parser.add_argument(
        "--pre-encoder-mask-temporal-bin",
        action="append",
        type=int,
        default=None,
        help="Mask sampled frames represented by this Qwen temporal bin before the vision encoder. Can be repeated.",
    )
    parser.add_argument(
        "--pre-encoder-keep-temporal-bin",
        action="append",
        type=int,
        default=None,
        help="Prune input to keep only sampled frames represented by this Qwen temporal bin. Can be repeated.",
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="Optional condition label/control, e.g. baseline, repeated_frame, reversed_video, mismatched_query.",
    )
    parser.add_argument("--query-scope", default="question", choices=["question", "full_user_prompt"])
    parser.add_argument("--attention-extraction", default="full", choices=["full", "reduced_sdpa"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=None, help="Run only this question id. Can be repeated.")
    parser.add_argument("--output-dir", default="outputs/experiment1_debug")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-7b-inference", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip examples whose complete artifact already exists.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--profile-one-example",
        action="store_true",
        help="Run only one selected example and write detailed stage-level memory/timing profiling.",
    )
    parser.add_argument(
        "--profile-output-json",
        default=None,
        help="Optional path for the one-example profiling JSON. Defaults to <output-dir>/profile.json.",
    )
    parser.add_argument(
        "--no-progress-log",
        action="store_true",
        help="Disable stage-level stderr progress logs during real inference.",
    )
    return parser.parse_args()


def load_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def filter_records(records: list[dict[str, Any]], question_ids: list[str] | None) -> list[dict[str, Any]]:
    if not question_ids:
        return records
    allowed = set(question_ids)
    filtered = [record for record in records if record["question_id"] in allowed]
    missing = allowed - {record["question_id"] for record in filtered}
    if missing:
        raise ValueError(f"Requested question IDs not found in manifest: {sorted(missing)}")
    return filtered


def shard_records(records: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    return [record for index, record in enumerate(records) if index % num_shards == shard_index]


def completed_question_ids(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    completed: set[str] = set()
    with records_path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            artifact = record.get("artifact")
            if record.get("status") == "complete" and artifact and Path(artifact).exists():
                completed.add(str(record["question_id"]))
    return completed


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def intervention_bins(record: dict[str, Any], args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    decoder_bins = args.decoder_mask_temporal_bin
    pre_encoder_bins = args.pre_encoder_mask_temporal_bin
    keep_bins = args.pre_encoder_keep_temporal_bin
    if decoder_bins is None:
        decoder_bins = record.get("decoder_direct_access_mask_temporal_bins")
    if pre_encoder_bins is None:
        pre_encoder_bins = record.get("pre_encoder_mask_temporal_bins")
    if keep_bins is None:
        keep_bins = record.get("keep_temporal_bins")
    decoder_tuple = tuple(int(item) for item in (decoder_bins or ()))
    pre_encoder_tuple = tuple(int(item) for item in (pre_encoder_bins or ()))
    keep_tuple = tuple(int(item) for item in (keep_bins or ()))
    active = sum(bool(item) for item in (decoder_tuple, pre_encoder_tuple, keep_tuple))
    if active > 1:
        raise ValueError(
            "Decoder direct-access masking, pre-encoder temporal masking, and pre-encoder keep/pruning are separate interventions; "
            "run them in separate output directories."
        )
    return decoder_tuple, pre_encoder_tuple, keep_tuple


def decoder_direct_access_through_layer(record: dict[str, Any], args: argparse.Namespace) -> int | None:
    value = args.decoder_direct_access_through_layer
    if value is None:
        value = record.get("decoder_direct_access_through_layer")
    return None if value is None else int(value)


def records_filename(shard_index: int, num_shards: int) -> str:
    return "records.jsonl" if num_shards == 1 else f"records_shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"


def summary_filename(shard_index: int, num_shards: int) -> str:
    return "summary.json" if num_shards == 1 else f"summary_shard-{shard_index:05d}-of-{num_shards:05d}.json"


def load_examples_by_id(questions_dir: str | None, records: list[dict[str, Any]]):
    if questions_dir is None:
        raise ValueError("--questions-dir is required for non-dry-run Experiment 1 execution.")
    from src.dataset import HDEpicVQADataset

    question_types = sorted({record["question_type"] for record in records})
    dataset = HDEpicVQADataset(questions_dir, question_types)
    examples = {example.question_id: example for example in dataset.examples}
    missing = [record["question_id"] for record in records if record["question_id"] not in examples]
    if missing:
        raise ValueError(f"Manifest examples were not found under --questions-dir: {missing[:5]}")
    return examples


def frames_per_video_input(num_frames: int, num_video_inputs: int, mode: str = "total") -> list[int]:
    if num_frames <= 0:
        raise ValueError("--num-frames must be positive.")
    if num_video_inputs <= 0:
        return []
    if mode == "per-input":
        return [num_frames for _ in range(num_video_inputs)]
    if mode != "total":
        raise ValueError("frame budget mode must be 'total' or 'per-input'.")
    if num_frames < num_video_inputs:
        raise ValueError(
            f"--num-frames={num_frames} is smaller than the {num_video_inputs} video inputs. "
            "Increase --num-frames or use --frame-budget-mode per-input."
        )
    base = num_frames // num_video_inputs
    remainder = num_frames % num_video_inputs
    return [base + (1 if input_idx < remainder else 0) for input_idx in range(num_video_inputs)]


def _probe_video_for_sampling(path: str | Path) -> dict[str, Any]:
    from src.frame_sampling import UniformFrameSampler

    return UniformFrameSampler(num_frames=1)._probe_video(Path(path))


def _sample_video_indices(path: str | Path, indices: list[int]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install numpy to sample exact Experiment 1 v2 frames.") from exc

    video_path = Path(path)
    try:
        import decord
    except ImportError:
        info = _probe_video_for_sampling(video_path)
        frames = []
        for index in indices:
            timestamp = index / float(info["fps"])
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
            frame = np.frombuffer(result.stdout, dtype=np.uint8)
            expected = int(info["height"]) * int(info["width"]) * 3
            if frame.size != expected:
                raise RuntimeError(f"ffmpeg returned {frame.size} bytes, expected {expected}.")
            frames.append(frame.reshape((int(info["height"]), int(info["width"]), 3)))
        return np.stack(frames, axis=0)

    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    return reader.get_batch(indices).asnumpy()


def _decord_video_length(path: str | Path) -> int | None:
    try:
        import decord
    except ImportError:
        return None
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    return int(len(reader))


def _load_realtime_sampling_policy(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("--sampling-mode realtime requires --sampling-policy-json.")
    payload = json.loads(Path(path).read_text())
    policy = payload.get("realtime_sampling_policy", payload)
    required = {"delta_t_seconds", "frames_per_bin", "max_bins", "max_frames"}
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Sampling policy JSON is missing required fields: {missing}")
    return policy


def _sampling_plan_for_record(
    record: dict[str, Any],
    video_path: str | Path,
    sampling_mode: str,
    frames_per_bin_override: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.experiment1.v2_sampling import (
        TemporalSamplingPolicy,
        fixed_budget_bin_plan,
        real_time_bin_plan,
        robustness_policy,
    )

    info = _probe_video_for_sampling(video_path)
    start = float(record.get("analyzed_start_seconds", 0.0) or 0.0)
    end = float(record.get("analyzed_end_seconds") or info["num_frames"] / float(info["fps"]))
    decord_length = _decord_video_length(video_path)
    if sampling_mode == "realtime":
        frozen = record.get("_realtime_sampling_policy")
        if frozen is None:
            raise ValueError("Realtime sampling requires a frozen policy from --sampling-policy-json.")
        frames_per_bin = int(frames_per_bin_override or frozen["frames_per_bin"])
        if frames_per_bin <= 0:
            raise ValueError("--frames-per-bin must be positive when provided.")
        policy = TemporalSamplingPolicy(
            name="adaptive_full_coverage_median_development",
            delta_t_seconds=float(frozen["delta_t_seconds"]),
            frames_per_bin=frames_per_bin,
            max_bins=int(frozen["max_bins"]),
            fixed_num_frames=None,
            fixed_num_bins=None,
            min_bins=int(frozen.get("min_bins", 8)),
        )
        plan = real_time_bin_plan(
            start,
            end,
            float(info["fps"]),
            policy,
            decord_length=decord_length,
            ffprobe_frame_count=int(info["num_frames"]),
        )
    elif sampling_mode == "fixed_budget":
        policy = robustness_policy()
        plan = fixed_budget_bin_plan(
            start,
            end,
            float(info["fps"]),
            policy,
            decord_length=decord_length,
            ffprobe_frame_count=int(info["num_frames"]),
        )
    else:
        raise ValueError(f"Unsupported v2 sampling mode: {sampling_mode}")
    return plan, {
        "mode": sampling_mode,
        "policy": policy.to_json(),
        "source_fps": float(info["fps"]),
        "source_num_frames": int(info["num_frames"]),
        "ffprobe_frame_count": int(info["num_frames"]),
        "decord_frame_count": plan[0].get("decord_frame_count") if plan else None,
        "start_seconds": start,
        "end_seconds": end,
        "effective_analyzed_start_seconds": plan[0].get("effective_analyzed_start_seconds") if plan else start,
        "effective_analyzed_end_seconds": plan[-1].get("effective_analyzed_end_seconds") if plan else end,
        "analyzed_end_adjustment": plan[0].get("analyzed_end_adjustment") if plan else None,
        "target_delta_t_seconds": plan[0].get("target_delta_t_seconds") if plan else None,
        "effective_seconds_per_bin": plan[0].get("effective_seconds_per_bin") if plan else None,
        "desired_bins": plan[0].get("desired_bins") if plan else None,
        "num_bins": len(plan),
        "min_bin_clipped": plan[0].get("min_bin_clipped") if plan else None,
        "max_bin_clipped": plan[0].get("max_bin_clipped") if plan else None,
        "first_sampled_timestamp_seconds": plan[0]["source_timestamps"][0] if plan else None,
        "last_sampled_timestamp_seconds": plan[-1]["source_timestamps"][-1] if plan else None,
        "frozen_policy_source": record.get("_sampling_policy_json"),
        "frames_per_bin_override": frames_per_bin_override,
    }


def _frame_batch_from_sampling_plan(
    example,
    record: dict[str, Any],
    mp4_dir: str | None,
    sampling_mode: str,
    frames_per_bin_override: int | None = None,
):
    if mp4_dir is None:
        raise ValueError("--mp4-dir is required for non-dry-run Experiment 1 execution.")
    from src.frame_sampling import FrameBatch

    segment = example.inputs[0]
    if segment.is_image:
        raise ValueError(f"{sampling_mode} sampling requires a single video input, got image input.")
    video_path = segment.path_under(mp4_dir)
    plan, sampling_metadata = _sampling_plan_for_record(
        record,
        video_path,
        sampling_mode,
        frames_per_bin_override=frames_per_bin_override,
    )
    indices = [int(index) for item in plan for index in item["source_frame_indices"]]
    frames = _sample_video_indices(video_path, indices)
    timestamps = tuple(float(timestamp) for item in plan for timestamp in item["source_timestamps"])
    frame_bin_mapping = []
    position = 0
    for item in plan:
        for source_frame_index, timestamp in zip(item["source_frame_indices"], item["source_timestamps"]):
            frame_bin_mapping.append(
                {
                    "sample_position": position,
                    "analysis_bin": int(item["analysis_bin"]),
                    "source_frame_index": int(source_frame_index),
                    "timestamp_seconds": float(timestamp),
                    "bin_start_seconds": float(item["bin_start_seconds"]),
                    "bin_end_seconds": float(item["bin_end_seconds"]),
                    "original_temporal_position": int(item["original_temporal_position"]),
                    "presented_temporal_position": int(item["presented_temporal_position"]),
                }
            )
            position += 1
    return FrameBatch(
        frames=frames,
        frame_indices=tuple(indices),
        timestamps=timestamps,
        video_path=Path(video_path),
        metadata={
            "backend": f"experiment1_v2_{sampling_mode}",
            "fps": sampling_metadata["source_fps"],
            "source_num_frames": sampling_metadata["source_num_frames"],
            "input_modality": "video",
            "sampling": sampling_metadata,
            "frame_bin_mapping": frame_bin_mapping,
            "temporal_bin_plan": plan,
        },
    )


def frame_batches_for_example(
    example,
    mp4_dir: str | None,
    num_frames: int,
    frame_budget_mode: str = "total",
    sampling_mode: str = "legacy",
    manifest_record: dict[str, Any] | None = None,
    frames_per_bin_override: int | None = None,
):
    if sampling_mode != "legacy":
        if manifest_record is None:
            raise ValueError("Experiment 1 v2 sampling requires the manifest record.")
        if len(example.inputs) != 1 or example.inputs[0].is_image:
            raise ValueError(f"{sampling_mode} sampling requires exactly one video input.")
        return [
            _frame_batch_from_sampling_plan(
                example,
                manifest_record,
                mp4_dir,
                sampling_mode,
                frames_per_bin_override=frames_per_bin_override,
            )
        ]

    if mp4_dir is None:
        raise ValueError("--mp4-dir is required for non-dry-run Experiment 1 execution.")
    from src.frame_sampling import FrameBatch, UniformFrameSampler

    def with_modality(batch: FrameBatch, modality: str) -> FrameBatch:
        metadata = dict(batch.metadata)
        metadata["input_modality"] = modality
        return replace(batch, metadata=metadata)

    video_allocations = iter(
        frames_per_video_input(
            num_frames,
            sum(1 for segment in example.inputs if not segment.is_image),
            frame_budget_mode,
        )
    )
    batches = []
    for segment in example.inputs:
        if segment.is_image:
            batch = UniformFrameSampler(num_frames=1).sample_video(segment.path_under(mp4_dir), segment)
            batches.append(with_modality(batch, "image"))
        else:
            batch = UniformFrameSampler(num_frames=next(video_allocations)).sample_video(
                segment.path_under(mp4_dir), segment
            )
            batches.append(with_modality(batch, "video"))
    return batches


def condition_for_record(record: dict[str, Any], cli_condition: str | None) -> str:
    return str(cli_condition or record.get("condition") or "baseline")


def example_for_record(example: Any, record: dict[str, Any]) -> Any:
    if "override_question" not in record:
        return example
    choices = tuple(str(choice) for choice in record.get("override_choices", example.choices))
    correct_idx = int(record.get("override_correct_idx", example.correct_idx))
    if not 0 <= correct_idx < len(choices):
        raise ValueError(f"override_correct_idx={correct_idx} is outside override choices.")
    raw = dict(getattr(example, "raw", {}) or {})
    raw["experiment1_v2_original_question_id"] = example.question_id
    raw["experiment1_v2_override_question_id"] = record.get("override_question_id")
    return replace(
        example,
        question=str(record["override_question"]),
        choices=choices,
        correct_idx=correct_idx,
        raw=raw,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = get_resolution_config(args.resolution_config)
    records = filter_records(load_manifest(args.manifest, args.limit), args.question_id)
    if args.profile_one_example:
        records = records[:1]
    if args.sampling_mode == "realtime":
        realtime_policy = _load_realtime_sampling_policy(args.sampling_policy_json)
        for record in records:
            record["_realtime_sampling_policy"] = realtime_policy
            record["_sampling_policy_json"] = args.sampling_policy_json
    records = shard_records(records, args.shard_index, args.num_shards)
    started = time.time()
    jsonl_path = output_dir / records_filename(args.shard_index, args.num_shards)
    if jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()
    complete_on_resume = completed_question_ids(jsonl_path) if args.resume else set()
    git_commit = current_git_commit()

    examples_by_id = None
    backend_runner = None
    if args.allow_7b_inference and not args.dry_run:
        from src.experiment1.model_backends import create_model_backend

        examples_by_id = load_examples_by_id(args.questions_dir, records)
        backend_runner = create_model_backend(
            args.model_backend,
            checkpoint=args.model_checkpoint,
            max_new_tokens=args.max_new_tokens,
        )

    for record in records:
        decoder_mask_bins, pre_encoder_mask_bins, keep_bins = intervention_bins(record, args)
        decoder_through_layer = decoder_direct_access_through_layer(record, args)
        condition = condition_for_record(record, args.condition)
        if record["question_id"] in complete_on_resume:
            append_jsonl(
                jsonl_path,
                {
                    "question_id": record["question_id"],
                    "category": record["category"],
                    "question_type": record["question_type"],
                    "status": "skipped_complete",
                    "resume": True,
                },
            )
            continue
        if args.dry_run or not args.allow_7b_inference:
            status = "dry_run" if args.dry_run else "blocked_requires_allow_7b_inference"
            append_jsonl(
                jsonl_path,
                {
                    "question_id": record["question_id"],
                    "category": record["category"],
                    "question_type": record["question_type"],
                    "status": status,
                    "num_frames": args.num_frames,
                    "sampling_mode": args.sampling_mode,
                    "sampling_policy_json": args.sampling_policy_json,
                    "frames_per_bin": args.frames_per_bin,
                    "model_backend": args.model_backend,
                    "model_checkpoint": args.model_checkpoint,
                    "frame_budget_mode": args.frame_budget_mode,
                    "resolution": resolution.to_metadata(),
                    "vision_access_through_layer": args.vision_access_through_layer,
                    "decoder_direct_access_mask_temporal_bins": list(decoder_mask_bins),
                    "decoder_direct_access_through_layer": decoder_through_layer,
                    "pre_encoder_mask_temporal_bins": list(pre_encoder_mask_bins),
                    "pre_encoder_keep_temporal_bins": list(keep_bins),
                    "condition": condition,
                    "query_scope": args.query_scope,
                    "attention_extraction": args.attention_extraction,
                    "seed": args.seed,
                    "git_commit": git_commit,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                },
            )
            continue

        assert examples_by_id is not None
        assert backend_runner is not None
        example = example_for_record(examples_by_id[record["question_id"]], record)
        from src.experiment1.profiling import StageProfiler

        profiler = StageProfiler(enabled=True, log_progress=not args.no_progress_log)
        try:
            with profiler.stage("video_decoding_sampling"):
                frame_batches = frame_batches_for_example(
                    example,
                    args.mp4_dir,
                    args.num_frames,
                    args.frame_budget_mode,
                    sampling_mode=args.sampling_mode,
                    manifest_record=record,
                    frames_per_bin_override=args.frames_per_bin,
                )
            artifact = backend_runner.run_example(
                example,
                frame_batches,
                resolution,
                query_scope=args.query_scope,
                attention_extraction=args.attention_extraction,
                vision_access_through_layer=args.vision_access_through_layer,
                decoder_direct_access_mask_temporal_bins=decoder_mask_bins,
                decoder_direct_access_through_layer=decoder_through_layer,
                pre_encoder_remove_temporal_bins=pre_encoder_mask_bins,
                pre_encoder_keep_temporal_bins=keep_bins,
                condition=condition,
                profiler=profiler,
            )
            artifact["category"] = record["category"]
            artifact["model_backend"] = args.model_backend
            artifact["model_checkpoint"] = backend_runner.checkpoint
            artifact["vision_access_through_layer"] = args.vision_access_through_layer
            artifact["condition"] = condition
            artifact["decoder_direct_access_through_layer"] = decoder_through_layer
            if record.get("intervention"):
                artifact["intervention"] = record["intervention"]
            artifact["run_config"] = {
                "manifest": args.manifest,
                "num_frames": args.num_frames,
                "sampling_mode": args.sampling_mode,
                "sampling_policy_json": args.sampling_policy_json,
                "frames_per_bin": args.frames_per_bin,
                "model_backend": args.model_backend,
                "model_checkpoint": backend_runner.checkpoint,
                "frame_budget_mode": args.frame_budget_mode,
                "resolution_config": args.resolution_config,
                "vision_access_through_layer": args.vision_access_through_layer,
                "decoder_direct_access_mask_temporal_bins": list(decoder_mask_bins),
                "decoder_direct_access_through_layer": decoder_through_layer,
                "pre_encoder_mask_temporal_bins": list(pre_encoder_mask_bins),
                "pre_encoder_keep_temporal_bins": list(keep_bins),
                "intervention": record.get("intervention"),
                "condition": condition,
                "query_scope": args.query_scope,
                "attention_extraction": args.attention_extraction,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "git_commit": git_commit,
            }
            artifact_path = output_dir / f"{record['question_id']}.json"
            artifact["metadata"]["profiling"] = profiler.to_json_dict()
            with profiler.stage("artifact_serialization"):
                write_json_atomic(artifact_path, artifact)
            profile_path = Path(args.profile_output_json) if args.profile_output_json else output_dir / "profile.json"
            if args.profile_one_example:
                profiler.write_json(profile_path)
            append_jsonl(
                jsonl_path,
                {
                    "question_id": record["question_id"],
                    "category": record["category"],
                    "question_type": record["question_type"],
                    "status": "complete",
                    "artifact": str(artifact_path),
                    "correct": artifact["correct"],
                    "predicted_idx": artifact["predicted_idx"],
                    "num_visual_tokens": artifact["token_layout"]["num_visual_tokens"],
                    "num_temporal_bins": artifact["temporal_relevance"]["metadata"]["num_temporal_bins"],
                    "peak_cuda_memory_bytes": artifact["metadata"].get("cuda_max_memory_allocated_bytes"),
                },
            )
        except Exception as exc:
            append_jsonl(
                jsonl_path,
                {
                    "question_id": record["question_id"],
                    "category": record["category"],
                    "question_type": record["question_type"],
                    "status": "failed",
                    "error": str(exc),
                    "retryable": True,
                },
            )

    summary = {
        "num_records": len(records),
        "output_jsonl": str(jsonl_path),
        "runtime_seconds": time.time() - started,
        "config": {
            "manifest": args.manifest,
            "num_frames": args.num_frames,
            "sampling_mode": args.sampling_mode,
            "sampling_policy_json": args.sampling_policy_json,
            "frames_per_bin": args.frames_per_bin,
            "model_backend": args.model_backend,
            "model_checkpoint": args.model_checkpoint,
            "frame_budget_mode": args.frame_budget_mode,
            "resolution": resolution.to_metadata(),
            "vision_access_through_layer": args.vision_access_through_layer,
            "decoder_direct_access_mask_temporal_bin_cli": list(args.decoder_mask_temporal_bin or ()),
            "decoder_direct_access_through_layer_cli": args.decoder_direct_access_through_layer,
            "pre_encoder_mask_temporal_bin_cli": list(args.pre_encoder_mask_temporal_bin or ()),
            "pre_encoder_keep_temporal_bin_cli": list(args.pre_encoder_keep_temporal_bin or ()),
            "condition": args.condition,
            "query_scope": args.query_scope,
            "attention_extraction": args.attention_extraction,
            "max_new_tokens": args.max_new_tokens,
            "dry_run": args.dry_run,
            "allow_7b_inference": args.allow_7b_inference,
            "resume": args.resume,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "seed": args.seed,
            "git_commit": git_commit,
        },
    }
    write_json(output_dir / summary_filename(args.shard_index, args.num_shards), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
