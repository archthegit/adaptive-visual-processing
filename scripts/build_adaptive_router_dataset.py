#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment1 import current_git_commit
from src.experiment1.adaptive_router import build_router_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adaptive temporal compaction router dataset.")
    parser.add_argument("--label-dir", default=None, help="Backward-compatible shared label directory for train/development only.")
    parser.add_argument("--train-label-dir", default=None)
    parser.add_argument("--development-label-dir", default=None)
    parser.add_argument("--test-label-dir", default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--allow-test", action="store_true", help="Include test rows for final frozen evaluation dataset construction.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_router_dataset(
        label_dir=args.label_dir,
        train_label_dir=args.train_label_dir,
        development_label_dir=args.development_label_dir,
        test_label_dir=args.test_label_dir,
        train_manifest=args.train_manifest,
        development_manifest=args.development_manifest,
        test_manifest=args.test_manifest,
        allow_test=args.allow_test,
        output_dir=args.output_dir,
        git_commit=current_git_commit(),
    )
    print(json.dumps({"num_rows": len(dataset.rows), "num_features": len(dataset.feature_names)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
