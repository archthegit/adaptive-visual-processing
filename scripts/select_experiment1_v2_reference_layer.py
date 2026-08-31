#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.v2_reference_layer import write_frozen_reference_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the Experiment 1 v2 decoder reference layer from development artifacts.")
    parser.add_argument("--primary-manifest", default="outputs/experiment1_v2/primary_manifest.jsonl")
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--mismatched-output-dir", default=None)
    parser.add_argument("--output-json", default="outputs/experiment1_v2/frozen_reference_layer.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_frozen_reference_layer(
        primary_manifest_path=args.primary_manifest,
        baseline_output_dir=args.baseline_output_dir,
        mismatched_output_dir=args.mismatched_output_dir,
        output_json=args.output_json,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
