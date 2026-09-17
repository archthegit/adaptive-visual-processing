#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.spatial import (  # noqa: E402
    SPATIAL_ROUTE_CONDITIONS,
    SPATIAL_ROUTE_SEED,
    current_spatial_git_commit,
    spatial_route_spec_from_baseline_artifact,
    validate_spatial_route_spec,
)
from src.io import write_json_atomic  # noqa: E402


DEFAULT_CONDITIONS = (
    "spatial_route_gap4_top50",
    "spatial_random_gap4_top50",
    "spatial_uniform_gap4_top50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Qwen spatial-route development pilot manifests.")
    parser.add_argument("--qwen-dir", required=True, help="Dense Qwen baseline directory containing spatial_relevance artifacts.")
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SPATIAL_ROUTE_SEED)
    parser.add_argument("--expected-dev-examples", type=int, default=15)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        choices=tuple(SPATIAL_ROUTE_CONDITIONS.values()),
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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
    if not records_path.is_file():
        raise FileNotFoundError(f"Baseline directory has no records.jsonl: {records_path}")
    latest: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "complete" and record.get("artifact"):
            latest[str(record["question_id"])] = record
    return {
        question_id: (json.loads(_artifact_path(record["artifact"], output_path).read_text()), _artifact_path(record["artifact"], output_path))
        for question_id, record in sorted(latest.items())
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _budget_signature(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "frames": route["temporal_frames_preserved"],
        "per_frame": route["per_frame_retained_visual_tokens"],
        "retained_fraction": route["mean_actual_retained_visual_token_fraction"],
        "routed_layers": sorted(int(layer) for layer in route["layer_routes"]),
    }


def manifest_rows(
    *,
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
        raise ValueError(f"Qwen dense baseline is missing development artifacts: {missing}")
    rows = []
    for record in dev_records:
        question_id = str(record["question_id"])
        artifact, artifact_path = artifacts[question_id]
        route = spatial_route_spec_from_baseline_artifact(
            artifact,
            condition=condition,
            baseline_artifact=str(artifact_path),
            seed=seed,
            git_commit=git_commit,
        )
        validate_spatial_route_spec(route)
        row = dict(record)
        row["condition"] = condition
        row["model_backend"] = "qwen"
        row["spatial_route"] = route
        row["intervention"] = {
            "type": "baseline_derived_spatial_route_replay",
            "condition": condition,
            "model": "qwen",
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
        raise ValueError(f"Expected {args.expected_dev_examples} development records, found {len(dev_records)}.")
    git_commit = current_spatial_git_commit()
    rows_by_condition = {
        condition: manifest_rows(
            baseline_dir=args.qwen_dir,
            dev_records=dev_records,
            condition=condition,
            seed=args.seed,
            git_commit=git_commit,
        )
        for condition in args.conditions
    }
    for question_id in [str(row["question_id"]) for row in dev_records]:
        signatures = {
            condition: _budget_signature(next(row["spatial_route"] for row in rows if str(row["question_id"]) == question_id))
            for condition, rows in rows_by_condition.items()
        }
        if len({json.dumps(sig, sort_keys=True) for sig in signatures.values()}) != 1:
            raise ValueError(f"Spatial route budgets differ across conditions for {question_id}: {signatures}")
    summary = {
        "dev_manifest": args.dev_manifest,
        "qwen_dir": args.qwen_dir,
        "seed": args.seed,
        "git_commit": git_commit,
        "expected_dev_examples": args.expected_dev_examples,
        "outputs": [],
    }
    for condition, rows in rows_by_condition.items():
        path = output_dir / "qwen" / f"{condition}.jsonl"
        write_jsonl(path, rows)
        summary["outputs"].append(
            {
                "condition": condition,
                "path": str(path),
                "num_records": len(rows),
                "question_ids": [row["question_id"] for row in rows],
            }
        )
    write_json_atomic(output_dir / "spatial_route_pilot_manifest_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
