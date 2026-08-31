#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.v2_manifest import Experiment1V2Config, build_experiment1_v2_manifests
from src.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-video-level Experiment 1 v2 temporal manifests.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiment1_v2")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--min-test-per-category", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Experiment1V2Config(
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        min_test_per_category=args.min_test_per_category,
    )
    outputs = build_experiment1_v2_manifests(args.questions_dir, args.mp4_dir, config)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "duration_inventory.jsonl", outputs["duration_inventory"])
    write_jsonl(output_dir / "primary_manifest.jsonl", outputs["primary_manifest"])
    write_jsonl(output_dir / "additional_questions.jsonl", outputs["additional_questions"])
    write_json(output_dir / "mismatched_queries.json", outputs["mismatched_queries"])
    write_json(output_dir / "split_summary.json", outputs["split_summary"])
    write_jsonl(output_dir / "exclusions.jsonl", outputs["exclusions"])
    print(json.dumps(outputs["split_summary"], indent=2))


if __name__ == "__main__":
    main()
