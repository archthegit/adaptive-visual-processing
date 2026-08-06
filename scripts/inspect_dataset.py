#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HDEpicVQADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HD-EPIC VQA annotation files.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--question-files", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = HDEpicVQADataset(args.questions_dir, args.question_files)
    type_counts = Counter(example.question_type for example in dataset.examples)
    input_counts = Counter(len(example.inputs) for example in dataset.examples)

    print(f"Loaded examples: {len(dataset)}")
    print(f"Question files: {len(dataset.question_files)}")
    print(f"Question types: {len(type_counts)}")
    print(f"Inputs per question: {dict(sorted(input_counts.items()))}")
    print()

    for example in dataset.iter_limit(args.limit):
        print(f"{example.question_id} [{example.question_type}]")
        print(f"  video_ids: {', '.join(example.video_ids)}")
        for segment in example.inputs:
            print(
                "  input:"
                f" {segment.input_key} id={segment.video_id}"
                f" start={segment.start_seconds} end={segment.end_seconds}"
            )
        print(f"  question: {example.question}")
        print(f"  correct: {example.correct_idx} ({chr(ord('A') + example.correct_idx)})")
        print()


if __name__ == "__main__":
    main()
