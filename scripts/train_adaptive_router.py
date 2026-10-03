#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment1 import current_git_commit
from src.experiment1.adaptive_router import train_router_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive temporal compaction router models.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_router_models(
        args.dataset_dir,
        args.output_dir,
        seed=args.seed,
        git_commit=current_git_commit(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
