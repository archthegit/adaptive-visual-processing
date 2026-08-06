#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HDEpicVQADataset
from src.frame_sampling import FrameBatch, UniformFrameSampler
from src.io import append_jsonl, write_json
from src.models.base import ModelPrediction, format_multiple_choice_prompt
from src.models.qwen import Qwen25VLWrapper, QwenConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small HD-EPIC VQA baseline.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", default=None)
    parser.add_argument("--question-files", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/qwen_smoke")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse examples and write pending records without loading videos or Qwen.",
    )
    parser.add_argument(
        "--allow-7b-inference",
        action="store_true",
        help="Actually load and run Qwen2.5-VL-7B. Requires local checkpoint access and CUDA.",
    )
    return parser.parse_args()


def pending_prediction(example: Any, reason: str) -> ModelPrediction:
    return ModelPrediction(
        raw_response="",
        predicted_idx=-2,
        correct_idx=example.correct_idx,
        correct=False,
        frame_indices=tuple(),
        timestamps=tuple(),
        metadata={"status": "blocked", "reason": reason},
    )


def record_for_example(example: Any, prediction: ModelPrediction) -> dict[str, Any]:
    return {
        "question_id": example.question_id,
        "question_type": example.question_type,
        "video_ids": example.video_ids,
        "question": example.question,
        "choices": example.choices,
        "correct_idx": example.correct_idx,
        "raw_response": prediction.raw_response,
        "predicted_idx": prediction.predicted_idx,
        "correct": prediction.correct,
        "frame_indices": prediction.frame_indices,
        "timestamps": prediction.timestamps,
        "inputs": [segment.raw | {"input_key": segment.input_key} for segment in example.inputs],
        "metadata": prediction.metadata,
    }


def load_frame_batches(
    example: Any, mp4_dir: str | None, sampler: UniformFrameSampler
) -> list[FrameBatch]:
    if mp4_dir is None:
        raise RuntimeError("--mp4-dir is required for non-dry-run inference.")
    batches = []
    for segment in example.inputs:
        batches.append(sampler.sample_video(segment.path_under(mp4_dir), segment))
    return batches


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    jsonl_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if jsonl_path.exists():
        jsonl_path.unlink()

    dataset = HDEpicVQADataset(args.questions_dir, args.question_files)
    sampler = UniformFrameSampler(num_frames=args.num_frames, resize=args.resize)
    model = None
    if args.allow_7b_inference and not args.dry_run:
        model = Qwen25VLWrapper(QwenConfig(model_id=args.model_id))

    started = time.time()
    records = []
    for example in dataset.iter_limit(args.limit):
        try:
            if args.dry_run:
                prediction = pending_prediction(example, "dry_run")
            elif model is None:
                prediction = pending_prediction(
                    example, "Pass --allow-7b-inference to run Qwen2.5-VL-7B."
                )
            else:
                frame_batches = load_frame_batches(example, args.mp4_dir, sampler)
                prediction = model.predict(example, frame_batches)
        except Exception as exc:
            prediction = pending_prediction(example, str(exc))

        record = record_for_example(example, prediction)
        record["prompt"] = format_multiple_choice_prompt(example)
        append_jsonl(jsonl_path, record)
        records.append(record)

    attempted = [record for record in records if record["predicted_idx"] >= -1]
    correct = sum(1 for record in attempted if record["correct"])
    summary = {
        "num_examples": len(records),
        "num_attempted": len(attempted),
        "accuracy": correct / len(attempted) if attempted else None,
        "output_jsonl": str(jsonl_path),
        "runtime_seconds": time.time() - started,
        "config": {
            "num_frames": args.num_frames,
            "limit": args.limit,
            "resize": args.resize,
            "model_id": args.model_id,
            "dry_run": args.dry_run,
            "allow_7b_inference": args.allow_7b_inference,
            "python": platform.python_version(),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
