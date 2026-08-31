from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import json

from src.io import write_json, write_jsonl

from .v2_metrics import bootstrap_ci_clustered_by_video


BASELINE_CONDITION = "baseline"
CONTROL_CONDITIONS = (
    "repeated_frame",
    "reversed_video",
    "mismatched_query",
)
NECESSITY_CONDITIONS = (
    "mask_top20",
    "mask_bottom20",
    "mask_random20",
    "mask_mismatched_top20",
    "mask_contiguous_high_cluster",
)
SUFFICIENCY_CONDITIONS = (
    "keep_top20",
    "keep_uniform20",
    "keep_random20",
    "keep_mismatched_top20",
)
FUSION_DEPTH_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)


def expected_conditions(include_fusion_depth: bool = True) -> list[str]:
    conditions = [BASELINE_CONDITION]
    conditions.extend(CONTROL_CONDITIONS)
    conditions.extend(NECESSITY_CONDITIONS)
    conditions.extend(SUFFICIENCY_CONDITIONS)
    if include_fusion_depth:
        for layer in FUSION_DEPTH_LAYERS:
            conditions.append(f"fusion_block_top20_after_layer_{layer}")
            conditions.append(f"fusion_block_random20_after_layer_{layer}")
    return conditions


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open("r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def expected_run_matrix(primary_manifest: Iterable[dict[str, Any]], include_fusion_depth: bool = True) -> list[dict[str, Any]]:
    matrix = []
    for record in primary_manifest:
        for condition in expected_conditions(include_fusion_depth=include_fusion_depth):
            matrix.append(
                {
                    "split": record["split"],
                    "condition": condition,
                    "question_id": record["question_id"],
                    "source_video_id": record["source_video_id"],
                    "participant_id": record["participant_id"],
                    "category": record["category"],
                    "duration_group": record["duration_group"],
                }
            )
    return matrix


def artifact_path(output_root: str | Path, condition: str, question_id: str) -> Path:
    return Path(output_root) / condition / f"{question_id}.json"


def validate_completeness(
    primary_manifest: Iterable[dict[str, Any]],
    output_root: str | Path,
    include_fusion_depth: bool = True,
) -> dict[str, Any]:
    matrix = expected_run_matrix(primary_manifest, include_fusion_depth=include_fusion_depth)
    missing = []
    complete = []
    failed = []
    for expected in matrix:
        path = artifact_path(output_root, expected["condition"], expected["question_id"])
        if not path.exists():
            missing.append(expected)
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            failed.append(dict(expected, artifact=str(path), reason=f"invalid_json: {exc}"))
            continue
        if data.get("status") == "failed":
            failed.append(dict(expected, artifact=str(path), reason=data.get("error", "failed")))
        else:
            complete.append(dict(expected, artifact=str(path)))
    return {
        "expected": len(matrix),
        "complete": len(complete),
        "missing": len(missing),
        "failed": len(failed),
        "complete_by_condition": dict(sorted(Counter(item["condition"] for item in complete).items())),
        "missing_by_condition": dict(sorted(Counter(item["condition"] for item in missing).items())),
        "failed_by_condition": dict(sorted(Counter(item["condition"] for item in failed).items())),
        "missing_records": missing,
        "failed_records": failed,
    }


def flatten_completed_artifacts(output_root: str | Path, conditions: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for condition in conditions:
        condition_dir = Path(output_root) / condition
        if not condition_dir.exists():
            continue
        for path in sorted(condition_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            metadata = data.get("metadata", {})
            temporal = data.get("temporal_relevance", {})
            answer_scores = data.get("answer_choice_scores", {})
            row = {
                "condition": condition,
                "question_id": data.get("question_id"),
                "source_video_id": (data.get("video_clip") or [{}])[0].get("video_id"),
                "participant_id": (data.get("video_clip") or [{}])[0].get("participant_id"),
                "category": data.get("category") or data.get("question_type", "").split("_", 1)[0],
                "correct": bool(data.get("correct", False)),
                "correct_choice_log_probability": answer_scores.get("correct_choice_log_probability"),
                "correct_vs_best_incorrect_margin": answer_scores.get("correct_vs_best_incorrect_margin"),
                "visual_token_count": (data.get("token_layout") or {}).get("num_visual_tokens"),
                "latency_seconds": metadata.get("generation_runtime_seconds"),
            }
            layer_metrics = temporal.get("layer_metrics") or []
            if layer_metrics:
                final_layer = layer_metrics[-1]
                row["final_layer_entropy"] = final_layer.get("normalized_temporal_entropy")
                row["final_layer_top1_mass"] = final_layer.get("top1_temporal_bin_mass")
                row["final_layer_bins_to_80pct_mass"] = final_layer.get("bins_to_80pct_mass")
            rows.append(row)
    return rows


def summarize_completed_rows(rows: list[dict[str, Any]], bootstrap_replicates: int = 10000, seed: int = 20260830) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    summary: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        accuracy_values = [
            dict(row, accuracy=float(row["correct"]))
            for row in condition_rows
            if row.get("source_video_id") and row.get("participant_id")
        ]
        condition_summary = {
            "num_records": len(condition_rows),
            "num_source_videos": len({row.get("source_video_id") for row in condition_rows}),
            "accuracy": sum(1 for row in condition_rows if row.get("correct")) / len(condition_rows) if condition_rows else 0.0,
            "by_category": dict(sorted(Counter(row.get("category") for row in condition_rows).items())),
        }
        if accuracy_values:
            condition_summary["accuracy_bootstrap_ci"] = bootstrap_ci_clustered_by_video(
                accuracy_values,
                "accuracy",
                replicates=bootstrap_replicates,
                seed=seed,
            )
        summary[condition] = condition_summary
    return summary


def write_v2_analysis_outputs(
    primary_manifest_path: str | Path,
    output_root: str | Path,
    final_dir: str | Path,
    bootstrap_replicates: int = 10000,
    include_fusion_depth: bool = True,
    seed: int = 20260830,
) -> dict[str, Any]:
    primary = load_jsonl(primary_manifest_path)
    final = Path(final_dir)
    (final / "figures").mkdir(parents=True, exist_ok=True)
    (final / "tables").mkdir(parents=True, exist_ok=True)
    matrix = expected_run_matrix(primary, include_fusion_depth=include_fusion_depth)
    completeness = validate_completeness(primary, output_root, include_fusion_depth=include_fusion_depth)
    rows = flatten_completed_artifacts(output_root, expected_conditions(include_fusion_depth=include_fusion_depth))
    stats = {
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "condition_summaries": summarize_completed_rows(rows, bootstrap_replicates=bootstrap_replicates, seed=seed)
        if rows
        else {},
    }
    write_jsonl(final / "tables" / "expected_run_matrix.jsonl", matrix)
    write_jsonl(final / "tables" / "completed_rows.jsonl", rows)
    write_json(final / "completeness_report.json", completeness)
    write_json(final / "statistical_results.json", stats)
    report = {
        "primary_manifest": str(primary_manifest_path),
        "output_root": str(output_root),
        "final_dir": str(final),
        "complete": completeness["complete"],
        "expected": completeness["expected"],
        "missing": completeness["missing"],
        "failed": completeness["failed"],
        "interpretation_note": (
            "High temporal attention is descriptive unless separated from positional controls, "
            "mismatched-query controls, and causal interventions."
        ),
    }
    write_json(final / "experiment_report.json", report)
    (final / "experiment_report.md").write_text(
        "\n".join(
            [
                "# Experiment 1 v2 Report",
                "",
                f"Expected runs: {completeness['expected']}",
                f"Complete runs: {completeness['complete']}",
                f"Missing runs: {completeness['missing']}",
                f"Failed runs: {completeness['failed']}",
                "",
                "This report distinguishes temporal non-uniformity, positional bias, video saliency, "
                "query-conditioned relevance, causal importance, and pruning efficiency. Do not claim "
                "that high attention alone proves importance.",
                "",
            ]
        )
    )
    return {"completeness": completeness, "statistical_results": stats, "report": report}
