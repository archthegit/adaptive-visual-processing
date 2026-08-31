#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.v2_interventions import write_v2_intervention_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Experiment 1 v2 intervention manifests from baseline artifacts.")
    parser.add_argument("--primary-manifest", default="outputs/experiment1_v2/primary_manifest.jsonl")
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["top", "bottom", "random", "mismatched_top", "contiguous_high_cluster"],
    )
    parser.add_argument("--removal-fraction", type=float, default=0.2)
    parser.add_argument("--ranking-layer", type=int, default=None)
    parser.add_argument(
        "--frozen-reference-layer-json",
        default="outputs/experiment1_v2/frozen_reference_layer.json",
        help="Required frozen reference-layer file produced from development examples.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--mismatched-output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = write_v2_intervention_manifest(
        primary_manifest_path=args.primary_manifest,
        baseline_output_dir=args.baseline_output_dir,
        output_jsonl=args.output_jsonl,
        condition=args.condition,
        strategy=args.strategy,
        removal_fraction=args.removal_fraction,
        ranking_layer=args.ranking_layer,
        seed=args.seed,
        mismatched_output_dir=args.mismatched_output_dir,
        frozen_reference_layer_path=args.frozen_reference_layer_json,
    )
    print(json.dumps({"output_jsonl": args.output_jsonl, "num_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
