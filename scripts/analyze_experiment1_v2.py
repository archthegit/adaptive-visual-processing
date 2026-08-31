#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.v2_analysis import write_v2_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Experiment 1 v2 completeness and aggregate completed outputs.")
    parser.add_argument("--primary-manifest", default="outputs/experiment1_v2/primary_manifest.jsonl")
    parser.add_argument("--additional-questions", default="outputs/experiment1_v2/additional_questions.jsonl")
    parser.add_argument("--output-root", default="outputs/experiment1_v2/runs")
    parser.add_argument("--final-dir", default="outputs/experiment1_v2/final")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--no-fusion-depth", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_v2_analysis_outputs(
        primary_manifest_path=args.primary_manifest,
        output_root=args.output_root,
        final_dir=args.final_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        include_fusion_depth=not args.no_fusion_depth,
        additional_questions_path=args.additional_questions,
        seed=args.seed,
    )
    print(json.dumps(outputs["completeness"], indent=2))


if __name__ == "__main__":
    main()
