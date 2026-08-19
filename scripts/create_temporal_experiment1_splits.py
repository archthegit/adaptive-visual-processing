#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HDEpicVQADataset
from src.experiment1.temporal_splits import (
    TemporalSplitConfig,
    create_temporal_splits,
    temporal_manifest_record,
)
from src.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic temporal-only Experiment 1 manifests.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiment1_temporal_splits")
    parser.add_argument("--engineering-per-category", type=int, default=2)
    parser.add_argument("--pilot-per-category", type=int, default=12)
    parser.add_argument("--confirmatory-per-category", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-per-source-video", type=int, default=6)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    config = TemporalSplitConfig(
        engineering_per_category=args.engineering_per_category,
        pilot_per_category=args.pilot_per_category,
        confirmatory_per_category=args.confirmatory_per_category,
        seed=args.seed,
        max_per_source_video=args.max_per_source_video,
    )
    dataset = HDEpicVQADataset(args.questions_dir)
    splits, summary = create_temporal_splits(dataset.examples, config)
    output_dir = Path(args.output_dir)
    for split_name, examples in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", [temporal_manifest_record(example) for example in examples])
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
