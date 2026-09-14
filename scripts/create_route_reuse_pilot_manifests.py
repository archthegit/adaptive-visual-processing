#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.route_reuse import (
    ROUTE_REUSE_SEED,
    assert_matched_condition_budgets,
    current_git_commit,
    read_jsonl,
    route_spec_from_baseline_artifact,
)
from src.io import write_json_atomic


CONDITIONS = (
    "route_reuse_gap4_top50",
    "random_reuse_gap4_top50",
    "uniform_reuse_gap4_top50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create development-only route-reuse causal pilot manifests.")
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--vila-dir", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=ROUTE_REUSE_SEED)
    parser.add_argument("--expected-dev-examples", type=int, default=15)
    return parser.parse_args()


def _artifact_path(raw: str, output_dir: Path) -> Path:
    path = Path(raw)
    candidates = (path, output_dir / path.name, output_dir.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve artifact {raw!r} from {output_dir}")


def load_latest_complete_artifacts(output_dir: str | Path) -> dict[str, tuple[dict[str, Any], Path]]:
    output_path = Path(output_dir)
    records_path = output_path / "records.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not records_path.is_file():
        raise FileNotFoundError(f"Baseline directory has no records.jsonl: {records_path}")
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "complete" and record.get("artifact"):
            latest[str(record["question_id"])] = record
    artifacts: dict[str, tuple[dict[str, Any], Path]] = {}
    for question_id, record in sorted(latest.items()):
        path = _artifact_path(record["artifact"], output_path)
        artifacts[question_id] = (json.loads(path.read_text()), path)
    return artifacts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def manifest_rows_for_model(
    *,
    model: str,
    baseline_dir: str | Path,
    dev_records: list[dict[str, Any]],
    condition: str,
    seed: int,
    git_commit: str | None,
) -> list[dict[str, Any]]:
    artifacts = load_latest_complete_artifacts(baseline_dir)
    dev_ids = {str(record["question_id"]) for record in dev_records}
    missing = sorted(dev_ids - set(artifacts))
    if missing:
        raise ValueError(f"{model} baseline is missing development artifacts: {missing}")
    rows = []
    for record in dev_records:
        question_id = str(record["question_id"])
        artifact, artifact_path = artifacts[question_id]
        row = dict(record)
        row["condition"] = condition
        row["model_backend"] = "qwen" if model == "qwen" else "vila_llama3"
        row["route_reuse"] = route_spec_from_baseline_artifact(
            artifact,
            model=model,
            condition=condition,
            baseline_artifact=str(artifact_path),
            seed=seed,
            git_commit=git_commit,
        )
        row["intervention"] = {
            "type": "causal_route_reuse",
            "condition": condition,
            "model": model,
            "baseline_artifact": str(artifact_path),
            "seed": seed,
            "git_commit": git_commit,
        }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_records = read_jsonl(args.dev_manifest)
    if len(dev_records) != args.expected_dev_examples:
        raise ValueError(
            f"Expected {args.expected_dev_examples} development records, found {len(dev_records)} in {args.dev_manifest}."
        )
    git_commit = current_git_commit()
    summary: dict[str, Any] = {
        "dev_manifest": args.dev_manifest,
        "seed": args.seed,
        "git_commit": git_commit,
        "expected_dev_examples": args.expected_dev_examples,
        "outputs": [],
    }
    for model, baseline_dir in (("qwen", args.qwen_dir), ("vila", args.vila_dir)):
        rows_by_condition = {}
        for condition in CONDITIONS:
            rows_by_condition[condition] = manifest_rows_for_model(
                model=model,
                baseline_dir=baseline_dir,
                dev_records=dev_records,
                condition=condition,
                seed=args.seed,
                git_commit=git_commit,
            )
        for question_id in [str(record["question_id"]) for record in dev_records]:
            assert_matched_condition_budgets(
                [
                    next(row for row in rows_by_condition[condition] if str(row["question_id"]) == question_id)[
                        "route_reuse"
                    ]
                    for condition in CONDITIONS
                ]
            )
        for condition, rows in rows_by_condition.items():
            path = output_dir / model / f"{condition}.jsonl"
            write_jsonl(path, rows)
            summary["outputs"].append(
                {
                    "model": model,
                    "condition": condition,
                    "path": str(path),
                    "num_records": len(rows),
                    "question_ids": [row["question_id"] for row in rows],
                }
            )
    write_json_atomic(output_dir / "route_reuse_pilot_manifest_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
