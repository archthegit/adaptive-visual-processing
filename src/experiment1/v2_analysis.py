from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np

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


def condition_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    table = []
    for condition, items in sorted(by_condition.items()):
        log_probs = [float(item["correct_choice_log_probability"]) for item in items if item.get("correct_choice_log_probability") is not None]
        margins = [float(item["correct_vs_best_incorrect_margin"]) for item in items if item.get("correct_vs_best_incorrect_margin") is not None]
        latencies = [float(item["latency_seconds"]) for item in items if item.get("latency_seconds") is not None]
        tokens = [float(item["visual_token_count"]) for item in items if item.get("visual_token_count") is not None]
        table.append(
            {
                "condition": condition,
                "num_records": len(items),
                "accuracy": sum(1 for item in items if item.get("correct")) / len(items) if items else 0.0,
                "mean_correct_choice_log_probability": float(np.mean(log_probs)) if log_probs else None,
                "mean_correct_vs_best_incorrect_margin": float(np.mean(margins)) if margins else None,
                "mean_latency_seconds": float(np.mean(latencies)) if latencies else None,
                "mean_visual_token_count": float(np.mean(tokens)) if tokens else None,
            }
        )
    return table


def write_markdown_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No completed rows available.\n")
        return
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(column) is None else str(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _resample_distribution(values: list[float], bins: int = 32) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(bins, dtype=np.float64)
    if arr.size == bins:
        return arr
    source_x = np.linspace(0.0, 1.0, arr.size)
    target_x = np.linspace(0.0, 1.0, bins)
    return np.interp(target_x, source_x, arr)


def average_decoder_heatmap(output_root: str | Path, condition: str = BASELINE_CONDITION, bins: int = 32) -> np.ndarray | None:
    arrays = []
    for path in sorted((Path(output_root) / condition).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        layers = (data.get("temporal_relevance") or {}).get("normalized_temporal_bin_scores") or []
        if not layers:
            continue
        arrays.append(np.stack([_resample_distribution(layer, bins=bins) for layer in layers], axis=0))
    if not arrays:
        return None
    min_layers = min(array.shape[0] for array in arrays)
    return np.mean([array[:min_layers] for array in arrays], axis=0)


def average_encoder_heatmap(output_root: str | Path, condition: str = BASELINE_CONDITION, bins: int = 32) -> np.ndarray | None:
    arrays = []
    for path in sorted((Path(output_root) / condition).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        encoder = data.get("encoder_attention_temporal") or {}
        layers = encoder.get("normalized_incoming_temporal_attention") or []
        if not layers:
            continue
        per_layer = []
        for layer in layers:
            head_mean = np.asarray(layer, dtype=np.float64).mean(axis=0)
            per_layer.append(_resample_distribution(head_mean.tolist(), bins=bins))
        arrays.append(np.stack(per_layer, axis=0))
    if not arrays:
        return None
    min_layers = min(array.shape[0] for array in arrays)
    return np.mean([array[:min_layers] for array in arrays], axis=0)


def write_heatmap(path: str | Path, matrix: np.ndarray, title: str) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Normalized temporal position")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_condition_barplot(path: str | Path, table: list[dict[str, Any]], metric: str, title: str) -> bool:
    plt = _pyplot()
    if plt is None or not table:
        return False
    labels = [row["condition"] for row in table]
    values = [row.get(metric) for row in table]
    if any(value is None for value in values):
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4))
    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_paper_artifacts(output_root: str | Path, final_dir: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = Path(final_dir)
    figures = final / "figures"
    tables = final / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    table = condition_table(rows)
    write_jsonl(tables / "condition_summary.jsonl", table)
    write_markdown_table(tables / "condition_summary.md", table)
    generated: dict[str, Any] = {"tables": ["condition_summary.jsonl", "condition_summary.md"], "figures": {}, "matplotlib_available": _pyplot() is not None}
    decoder = average_decoder_heatmap(output_root)
    if decoder is not None:
        generated["figures"]["decoder_layer_temporal_heatmap"] = {
            "path": "decoder_layer_temporal_heatmap.png",
            "generated": write_heatmap(figures / "decoder_layer_temporal_heatmap.png", decoder, "Decoder temporal attention"),
        }
    encoder = average_encoder_heatmap(output_root)
    if encoder is not None:
        generated["figures"]["encoder_layer_temporal_heatmap"] = {
            "path": "encoder_layer_temporal_heatmap.png",
            "generated": write_heatmap(figures / "encoder_layer_temporal_heatmap.png", encoder, "Encoder temporal attention"),
        }
    generated["figures"]["condition_accuracy"] = {
        "path": "condition_accuracy.png",
        "generated": write_condition_barplot(figures / "condition_accuracy.png", table, "accuracy", "Accuracy by condition"),
    }
    write_json(final / "paper_artifacts_manifest.json", generated)
    return generated


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
    paper_artifacts = write_paper_artifacts(output_root, final, rows)
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
        "paper_artifacts": paper_artifacts,
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
