#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.adaptive_router import evaluate_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen adaptive temporal compaction router.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--unsafe-probability-threshold", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_router(
        args.dataset_dir,
        args.model_dir,
        args.output_dir,
        split=args.split,
        allow_test=args.allow_test,
        unsafe_probability_threshold=args.unsafe_probability_threshold,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
