#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HDEpicVQADataset
from src.experiment1.adaptive_compaction import (
    AdaptiveSplitConfig,
    create_source_video_disjoint_splits,
    manifest_records,
    write_jsonl,
)
from src.experiment1.temporal_handoff import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create source-video-disjoint adaptive temporal compaction manifests.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiment1_v3_adaptive_compaction/manifests")
    parser.add_argument("--train-per-category", type=int, default=50)
    parser.add_argument("--development-per-category", type=int, default=20)
    parser.add_argument("--test-per-category", type=int, default=50)
    parser.add_argument("--max-questions-per-source-video", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AdaptiveSplitConfig(
        train_per_category=args.train_per_category,
        development_per_category=args.development_per_category,
        test_per_category=args.test_per_category,
        max_questions_per_source_video=args.max_questions_per_source_video,
        seed=args.seed,
    )
    dataset = HDEpicVQADataset(args.questions_dir)
    splits, summary = create_source_video_disjoint_splits(dataset.examples, config)
    output = Path(args.output_dir)
    for split, examples in splits.items():
        write_jsonl(output / f"{split}.jsonl", manifest_records(examples, split))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
