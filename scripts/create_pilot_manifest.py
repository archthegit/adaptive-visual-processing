#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HDEpicVQADataset
from src.io import append_jsonl, write_json
from src.pilot import count_by_type_and_category, manifest_record, pilot_summary, select_pilot_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic HD-EPIC VQA pilot manifest.")
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--output-jsonl", default="outputs/pilot_manifest.jsonl")
    parser.add_argument("--summary-json", default="outputs/pilot_summary.json")
    parser.add_argument("--pilot-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = HDEpicVQADataset(args.questions_dir)
    all_counts = count_by_type_and_category(dataset.examples)
    pilot = select_pilot_examples(dataset.examples, pilot_size=args.pilot_size, seed=args.seed)

    output_path = Path(args.output_jsonl)
    if output_path.exists():
        output_path.unlink()
    for example in pilot:
        append_jsonl(output_path, manifest_record(example))

    summary = {
        "seed": args.seed,
        "pilot_size": args.pilot_size,
        "source_questions_dir": str(Path(args.questions_dir)),
        "total_dataset": {
            "num_examples": len(dataset),
            **all_counts,
        },
        "pilot": pilot_summary(pilot),
        "output_jsonl": str(output_path),
    }
    write_json(args.summary_json, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
