#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment1 import current_git_commit
from src.experiment1.adaptive_router import calibrate_router_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and freeze the adaptive temporal compaction router policy on development data.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unsafe-rate-bound", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = calibrate_router_policy(
        args.dataset_dir,
        args.model_dir,
        args.output_dir,
        unsafe_rate_bound=args.unsafe_rate_bound,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        git_commit=current_git_commit(),
    )
    print(json.dumps(policy, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
