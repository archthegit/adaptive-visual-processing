#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HDEpicVQADataset
from src.experiment1.manifest import (
    available_experiment1_types,
    experiment1_manifest_record,
    select_experiment1_examples,
    summarize_experiment1_manifest,
)
from src.io import append_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Experiment 1 HD-EPIC VQA manifest.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--examples-per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-jsonl", default="outputs/experiment1_manifest.jsonl")
    parser.add_argument("--summary-json", default="outputs/experiment1_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = HDEpicVQADataset(args.questions_dir)
    selected = select_experiment1_examples(dataset.examples, args.examples_per_category, args.seed)
    output_path = Path(args.output_jsonl)
    if output_path.exists():
        output_path.unlink()
    for example in selected:
        append_jsonl(output_path, experiment1_manifest_record(example))
    summary = {
        "seed": args.seed,
        "examples_per_category": args.examples_per_category,
        "source_questions_dir": str(Path(args.questions_dir)),
        "available_question_types": available_experiment1_types(dataset.examples),
        "pilot": summarize_experiment1_manifest(selected),
        "output_jsonl": str(output_path),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
