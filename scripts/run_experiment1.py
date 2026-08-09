#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.resolution import get_resolution_config
from src.io import append_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 query relevance + vision/query fusion.")
    parser.add_argument("--questions-dir", default=None)
    parser.add_argument("--mp4-dir", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resolution-config", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--vision-access-through-layer", default="none")
    parser.add_argument("--query-scope", default="question", choices=["question", "full_user_prompt"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=None, help="Run only this question id. Can be repeated.")
    parser.add_argument("--output-dir", default="outputs/experiment1_debug")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-7b-inference", action="store_true")
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


def frame_batches_for_example(example, mp4_dir: str | None, num_frames: int):
    if mp4_dir is None:
        raise ValueError("--mp4-dir is required for non-dry-run Experiment 1 execution.")
    from src.frame_sampling import UniformFrameSampler

    sampler = UniformFrameSampler(num_frames=num_frames)
    return [sampler.sample_video(segment.path_under(mp4_dir), segment) for segment in example.inputs]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = get_resolution_config(args.resolution_config)
    records = filter_records(load_manifest(args.manifest, args.limit), args.question_id)
    started = time.time()
    jsonl_path = output_dir / "records.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    examples_by_id = None
    qwen_model = None
    if args.allow_7b_inference and not args.dry_run:
        from src.experiment1.qwen_execution import run_qwen_relevance_example
        from src.models.qwen import Qwen25VLWrapper, QwenConfig

        examples_by_id = load_examples_by_id(args.questions_dir, records)
        qwen_model = Qwen25VLWrapper(QwenConfig(max_new_tokens=args.max_new_tokens))

    for record in records:
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
                    "resolution": resolution.to_metadata(),
                    "vision_access_through_layer": args.vision_access_through_layer,
                    "query_scope": args.query_scope,
                },
            )
            continue

        assert examples_by_id is not None
        assert qwen_model is not None
        example = examples_by_id[record["question_id"]]
        try:
            frame_batches = frame_batches_for_example(example, args.mp4_dir, args.num_frames)
            artifact = run_qwen_relevance_example(
                qwen_model,
                example,
                frame_batches,
                resolution,
                query_scope=args.query_scope,
            )
            artifact["category"] = record["category"]
            artifact["vision_access_through_layer"] = args.vision_access_through_layer
            artifact_path = output_dir / f"{record['question_id']}.json"
            write_json(artifact_path, artifact)
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
                },
            )

    summary = {
        "num_records": len(records),
        "output_jsonl": str(jsonl_path),
        "runtime_seconds": time.time() - started,
        "config": {
            "manifest": args.manifest,
            "num_frames": args.num_frames,
            "resolution": resolution.to_metadata(),
            "vision_access_through_layer": args.vision_access_through_layer,
            "query_scope": args.query_scope,
            "max_new_tokens": args.max_new_tokens,
            "dry_run": args.dry_run,
            "allow_7b_inference": args.allow_7b_inference,
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
