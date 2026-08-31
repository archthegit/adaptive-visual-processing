#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.v2_controls import write_v2_control_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Experiment 1 v2 control manifests.")
    parser.add_argument("--primary-manifest", default="outputs/experiment1_v2/primary_manifest.jsonl")
    parser.add_argument("--mismatched-queries", default="outputs/experiment1_v2/mismatched_queries.json")
    parser.add_argument("--additional-questions", default="outputs/experiment1_v2/additional_questions.jsonl")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--control",
        required=True,
        choices=["repeated_frame", "reversed_video", "mismatched_query", "same_video_different_query"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = write_v2_control_manifest(
        primary_manifest_path=args.primary_manifest,
        mismatched_queries_path=args.mismatched_queries,
        output_jsonl=args.output_jsonl,
        control=args.control,
        additional_questions_path=args.additional_questions,
    )
    print(json.dumps({"output_jsonl": args.output_jsonl, "num_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
