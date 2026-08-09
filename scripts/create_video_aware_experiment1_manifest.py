#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HDEpicVQADataset
from src.experiment1.manifest import (
    available_experiment1_types,
    experiment1_manifest_record,
    select_video_aware_experiment1_examples,
    summarize_experiment1_manifest,
)
from src.io import append_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smaller Experiment 1 manifest that limits unique MP4 downloads."
    )
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--target-size", type=int, default=12)
    parser.add_argument("--max-new-videos", type=int, default=8)
    parser.add_argument(
        "--max-video-inputs",
        type=int,
        default=1,
        help="Exclude examples with more than this many real video inputs. Image-time inputs do not count.",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--mp4-dir", default=None, help="Prefer videos already present under this directory.")
    parser.add_argument("--preferred-video-id", action="append", default=None)
    parser.add_argument("--output-jsonl", default="outputs/experiment1_video_aware_manifest.jsonl")
    parser.add_argument("--summary-json", default="outputs/experiment1_video_aware_summary.json")
    return parser.parse_args()


def existing_video_ids(mp4_dir: str | Path | None) -> set[str]:
    if mp4_dir is None:
        return set()
    root = Path(mp4_dir)
    if not root.exists():
        return set()
    return {path.stem for path in root.glob("*/*.mp4")}


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for record in records:
        append_jsonl(path, record)


def main() -> None:
    args = parse_args()
    preferred = existing_video_ids(args.mp4_dir) | set(args.preferred_video_id or [])
    dataset = HDEpicVQADataset(args.questions_dir)
    selected = select_video_aware_experiment1_examples(
        dataset.examples,
        target_size=args.target_size,
        max_new_videos=args.max_new_videos,
        seed=args.seed,
        preferred_video_ids=preferred,
        max_video_inputs=args.max_video_inputs,
    )
    records = [experiment1_manifest_record(example) for example in selected]
    output_path = Path(args.output_jsonl)
    write_manifest(output_path, records)

    summary = {
        "target_size": args.target_size,
        "actual_size": len(selected),
        "max_new_videos": args.max_new_videos,
        "max_video_inputs": args.max_video_inputs,
        "seed": args.seed,
        "preferred_video_count": len(preferred),
        "preferred_videos_used": sorted(
            preferred & {video_id for example in selected for video_id in example.video_ids}
        ),
        "available_question_types": available_experiment1_types(dataset.examples),
        "pilot": summarize_experiment1_manifest(selected),
        "representative_examples": records[:5],
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
