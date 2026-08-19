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


def frame_batches_for_example(example, mp4_dir: str | None, num_frames: int, frame_budget_mode: str = "total"):
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = get_resolution_config(args.resolution_config)
    records = filter_records(load_manifest(args.manifest, args.limit), args.question_id)
    records = shard_records(records, args.shard_index, args.num_shards)
    started = time.time()
    jsonl_path = output_dir / "records.jsonl"
    if jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()
    complete_on_resume = completed_question_ids(jsonl_path) if args.resume else set()
    git_commit = current_git_commit()

    examples_by_id = None
    qwen_model = None
    if args.allow_7b_inference and not args.dry_run:
        from src.experiment1.qwen_execution import run_qwen_relevance_example
        from src.models.qwen import Qwen25VLWrapper, QwenConfig

        examples_by_id = load_examples_by_id(args.questions_dir, records)
        qwen_model = Qwen25VLWrapper(QwenConfig(max_new_tokens=args.max_new_tokens))

    for record in records:
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
                    "frame_budget_mode": args.frame_budget_mode,
                    "resolution": resolution.to_metadata(),
                    "vision_access_through_layer": args.vision_access_through_layer,
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
        assert qwen_model is not None
        example = examples_by_id[record["question_id"]]
        try:
            frame_batches = frame_batches_for_example(example, args.mp4_dir, args.num_frames, args.frame_budget_mode)
            artifact = run_qwen_relevance_example(
                qwen_model,
                example,
                frame_batches,
                resolution,
                query_scope=args.query_scope,
                attention_extraction=args.attention_extraction,
                vision_access_through_layer=args.vision_access_through_layer,
            )
            artifact["category"] = record["category"]
            artifact["vision_access_through_layer"] = args.vision_access_through_layer
            artifact["run_config"] = {
                "manifest": args.manifest,
                "num_frames": args.num_frames,
                "frame_budget_mode": args.frame_budget_mode,
                "resolution_config": args.resolution_config,
                "vision_access_through_layer": args.vision_access_through_layer,
                "query_scope": args.query_scope,
                "attention_extraction": args.attention_extraction,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "git_commit": git_commit,
            }
            artifact_path = output_dir / f"{record['question_id']}.json"
            write_json_atomic(artifact_path, artifact)
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
            "frame_budget_mode": args.frame_budget_mode,
            "resolution": resolution.to_metadata(),
            "vision_access_through_layer": args.vision_access_through_layer,
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
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
