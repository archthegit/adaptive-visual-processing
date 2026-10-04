#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.adaptive_compaction import candidate_routes


def read_jsonl_snapshot(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in read_jsonl_snapshot(path)}


def categorize_failure(error: str) -> str:
    lowered = error.lower()
    if any(token in lowered for token in ("corrupt", "unreadable", "decode", "decord", "video")):
        return "corrupted_or_unreadable_video"
    if "distinct" in lowered or "insufficient" in lowered or "duplicate frame" in lowered:
        return "insufficient_distinct_frames"
    return "incomplete_or_missing_action_artifacts"


def artifact_path(record: dict[str, Any]) -> Path | None:
    value = record.get("artifact") or record.get("artifact_path")
    return Path(value) if value else None


def read_artifact(record: dict[str, Any]) -> dict[str, Any] | None:
    path = artifact_path(record)
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def deduplicate_complete_records(records: Sequence[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    dedup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "complete":
            continue
        try:
            key = (str(record["question_id"]), int(record["frame_count"]), str(record["action_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        dedup[key] = record
    return dedup


def expected_action_ids(dense: dict[str, Any]) -> set[str]:
    run_config = dense.get("run_config", {})
    layers = [int(item) for item in run_config.get("compaction_layers", ())]
    fractions = [float(item) for item in run_config.get("retention_fractions", ())]
    seed = int(run_config.get("seed", dense.get("seed", 20260928)))
    if not layers or not fractions:
        raise ValueError(f"Dense artifact for {dense.get('question_id')} lacks compaction layer/fraction config.")
    output = set()
    for layer in layers:
        for fraction in fractions:
            for route in candidate_routes(
                question_id=str(dense["question_id"]),
                frame_count=int(dense["frame_count"]),
                compaction_layer=layer,
                native_temporal_cell_count=int(dense["native_temporal_cell_count"]),
                retention_fraction=fraction,
                seed=seed,
            ):
                output.add(route.action_id)
    return output


def route_is_safe(action: dict[str, Any], quality_floor: float) -> bool:
    return (
        float(action.get("delta_correct_choice_log_probability_from_dense", -math.inf)) >= quality_floor
        and not bool(action.get("prediction_changed", False))
    )


def source_cluster_bootstrap(rows: Sequence[dict[str, Any]], value_key: str, *, samples: int, seed: int) -> dict[str, Any]:
    clean = [row for row in rows if row.get(value_key) is not None and math.isfinite(float(row[value_key]))]
    if not clean:
        return {"mean": None, "median": None, "ci95": [None, None], "n": 0, "n_source_videos": 0}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_source[str(row["source_video_id"])].append(row)
    sources = sorted(by_source)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled_sources = rng.choice(sources, size=len(sources), replace=True)
        values = [float(row[value_key]) for source in sampled_sources for row in by_source[source]]
        estimates.append(float(np.mean(values)))
    values = np.asarray([float(row[value_key]) for row in clean], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n": len(clean),
        "n_source_videos": len(sources),
    }


def uniform_distance(retained: Sequence[int], native_cell_count: int) -> float:
    retained = sorted(int(item) for item in retained)
    if not retained:
        return float("inf")
    denom = max(1, native_cell_count - 1)
    actual = np.asarray([item / denom for item in retained], dtype=np.float64)
    ideal = np.linspace(0.0, 1.0, num=len(retained))
    return float(np.mean((actual - ideal) ** 2))


def choose_fixed_route(actions: Sequence[dict[str, Any]], kind: str, seed: int) -> dict[str, Any] | None:
    if not actions:
        return None
    if kind == "uniform":
        return min(actions, key=lambda row: (uniform_distance(row["retained_cell_ids"], int(row["native_temporal_cell_count"])), str(row["action_id"])))
    if kind == "prefix":
        return min(actions, key=lambda row: (max(int(item) for item in row["retained_cell_ids"]), str(row["action_id"])))
    if kind == "suffix":
        return max(actions, key=lambda row: (min(int(item) for item in row["retained_cell_ids"]), str(row["action_id"])))
    if kind == "seeded_random":
        template = actions[0]
        import hashlib

        def key(row: dict[str, Any]) -> str:
            payload = json.dumps(
                [
                    seed,
                    row["question_id"],
                    int(row["frame_count"]),
                    int(row["compaction_layer"]),
                    float(row["retention_fraction"]),
                    sorted(int(item) for item in row["retained_cell_ids"]),
                    row["action_id"],
                ],
                sort_keys=True,
            )
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()

        _ = template
        return min(actions, key=key)
    raise ValueError(f"Unknown fixed route kind: {kind}")


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    bootstrap_samples: int,
    seed: int,
    include_category_all: bool = False,
) -> list[dict[str, Any]]:
    expanded = list(rows)
    if include_category_all:
        expanded.extend(dict(row, category="all") for row in rows)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    output = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        logp = source_cluster_bootstrap(items, "delta_logp", samples=bootstrap_samples, seed=seed)
        flop = source_cluster_bootstrap(items, "flop_reduction", samples=bootstrap_samples, seed=seed + 1)
        out = {name: value for name, value in zip(group_keys, key)}
        out.update(
            {
                "n_questions": len({item["question_id"] for item in items}),
                "n_source_videos": len({item["source_video_id"] for item in items}),
                "n_routes": len(items),
                "mean_delta_logp": logp["mean"],
                "median_delta_logp": logp["median"],
                "delta_logp_ci95_low": logp["ci95"][0],
                "delta_logp_ci95_high": logp["ci95"][1],
                "prediction_flip_rate": float(np.mean([bool(item["prediction_changed"]) for item in items])),
                "safe_route_fraction": float(np.mean([bool(item["safe"]) for item in items])),
                "mean_flop_reduction": flop["mean"],
                "flop_reduction_ci95_low": flop["ci95"][0],
                "flop_reduction_ci95_high": flop["ci95"][1],
            }
        )
        output.append(out)
    return output


def analyze(records: str | Path, manifest: str | Path, output_dir: str | Path, *, bootstrap_samples: int, seed: int, quality_floor: float) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_qid = load_manifest(manifest)
    raw_records = read_jsonl_snapshot(records)
    failures: list[dict[str, Any]] = []
    for record in raw_records:
        if record.get("status") == "failed":
            failures.append(
                {
                    "question_id": record.get("question_id"),
                    "frame_count": record.get("frame_count"),
                    "reason": categorize_failure(str(record.get("error", ""))),
                    "error": record.get("error"),
                }
            )
    dedup = deduplicate_complete_records(raw_records)
    artifacts: dict[tuple[str, int, str], dict[str, Any]] = {}
    bad_complete_records: list[dict[str, Any]] = []
    for key, record in dedup.items():
        artifact = read_artifact(record)
        if artifact is None:
            bad_complete_records.append({"question_id": key[0], "frame_count": key[1], "action_id": key[2], "reason": "incomplete_or_missing_action_artifacts"})
            continue
        if artifact.get("split") == "test":
            raise RuntimeError(f"Refusing to analyze test artifact: {record.get('artifact')}")
        artifacts[key] = artifact

    dense_keys = [key for key, artifact in artifacts.items() if key[2] == "dense_custom" or artifact.get("condition") == "dense_custom"]
    valid_groups: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for qid, frame_count, _action in sorted(dense_keys):
        dense = artifacts[(qid, frame_count, _action)]
        manifest_row = manifest_by_qid.get(qid, {})
        source = dense.get("source_video_id") or manifest_row.get("source_video_id")
        category = dense.get("category") or manifest_row.get("category")
        try:
            expected = expected_action_ids(dense)
        except Exception as exc:  # noqa: BLE001 - reported as incomplete group diagnostic.
            exclusions.append({"question_id": qid, "frame_count": frame_count, "source_video_id": source, "category": category, "reason": "incomplete_or_missing_action_artifacts", "detail": str(exc)})
            continue
        present = {action_id for (aqid, aframe, action_id), artifact in artifacts.items() if aqid == qid and aframe == frame_count and artifact.get("condition") != "dense_custom"}
        missing = sorted(expected - present)
        if missing:
            exclusions.append(
                {
                    "question_id": qid,
                    "frame_count": frame_count,
                    "source_video_id": source,
                    "category": category,
                    "reason": "incomplete_or_missing_action_artifacts",
                    "missing_action_count": len(missing),
                }
            )
            continue
        group_actions = []
        for action_id in sorted(expected):
            action = artifacts[(qid, frame_count, action_id)]
            row = {
                "question_id": qid,
                "source_video_id": source,
                "category": category,
                "frame_count": frame_count,
                "compaction_layer": int(action["compaction_layer"]),
                "retention_fraction": float(action["retention_fraction"]),
                "route_family": action.get("route_family"),
                "retained_cell_ids": list(action.get("retained_cell_ids", [])),
                "action_id": action_id,
                "delta_logp": float(action["delta_correct_choice_log_probability_from_dense"]),
                "delta_margin": float(action["delta_answer_margin_from_dense"]),
                "prediction_changed": bool(action.get("prediction_changed")),
                "safe": route_is_safe(action, quality_floor),
                "flop_reduction": float(action["paired_flop_reduction_from_dense"]),
                "dense_prediction": dense.get("predicted_answer", dense.get("predicted_idx")),
                "compacted_prediction": action.get("predicted_answer", action.get("predicted_idx")),
                "dense_correct": bool(dense.get("correct")),
                "native_temporal_cell_count": int(action["native_temporal_cell_count"]),
            }
            group_actions.append(row)
            route_rows.append(row)
        valid_groups.append(
            {
                "question_id": qid,
                "source_video_id": source,
                "category": category,
                "frame_count": frame_count,
                "dense_correct": bool(dense.get("correct")),
                "actions": group_actions,
            }
        )
    exclusions.extend(bad_complete_records)
    exclusions.extend(failures)

    coverage_rows = coverage_table(valid_groups, exclusions, route_rows)
    layer_rows = aggregate_rows(
        route_rows,
        group_keys=("category", "frame_count", "compaction_layer", "retention_fraction"),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        include_category_all=True,
    )
    fixed_rows = fixed_route_table(valid_groups, bootstrap_samples=bootstrap_samples, seed=seed)
    oracle_rows, question_summary = oracle_tables(valid_groups, bootstrap_samples=bootstrap_samples, seed=seed)
    failure_rows = failure_cases(route_rows)

    write_csv(output_dir / "coverage.csv", coverage_rows)
    write_csv(output_dir / "layer_retention.csv", layer_rows)
    write_csv(output_dir / "fixed_routes.csv", fixed_rows)
    write_csv(output_dir / "oracle.csv", oracle_rows)
    write_csv(output_dir / "failure_cases.csv", failure_rows)
    write_csv(output_dir / "question_summary.csv", question_summary)
    summary = {
        "records": str(records),
        "manifest": str(manifest),
        "quality_floor": quality_floor,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "num_questions": len({group["question_id"] for group in valid_groups}),
        "num_question_frame_groups": len(valid_groups),
        "num_routes": len(route_rows),
        "num_source_videos": len({group["source_video_id"] for group in valid_groups}),
        "num_excluded_groups": len(exclusions),
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
        "statistical_unit": "source_video_id",
    }
    write_json(output_dir / "summary.json", summary)
    make_figures(output_dir, route_rows, fixed_rows, oracle_rows)
    return summary


def coverage_table(valid_groups: Sequence[dict[str, Any]], exclusions: Sequence[dict[str, Any]], route_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    grouped_routes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in route_rows:
        grouped_routes[(str(row["category"]), int(row["frame_count"]))].append(row)
    exclusion_counts = Counter((row.get("category"), row.get("frame_count"), row["reason"]) for row in exclusions)
    keys = set(grouped_routes)
    keys.update((category, frame) for category, frame, _reason in exclusion_counts if category is not None and frame is not None)
    for category, frame_count in sorted(keys, key=lambda item: (str(item[0]), int(item[1]))):
        items = grouped_routes.get((category, frame_count), [])
        for reason in sorted({reason for c, f, reason in exclusion_counts if c == category and f == frame_count} or {""}):
            reason_count = exclusion_counts.get((category, frame_count, reason), 0)
            rows.append(
                {
                    "category": category,
                    "frame_count": frame_count,
                    "num_questions": len({item["question_id"] for item in items}),
                    "num_question_frame_groups": len({(item["question_id"], item["frame_count"]) for item in items}),
                    "num_source_videos": len({item["source_video_id"] for item in items}),
                    "num_routes": len(items),
                    "num_safe_routes": sum(1 for item in items if item["safe"]),
                    "safe_route_fraction": float(np.mean([item["safe"] for item in items])) if items else 0.0,
                    "num_excluded_groups": reason_count,
                    "exclusion_reason": reason,
                }
            )
    return rows


def fixed_route_table(valid_groups: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    chosen_rows = []
    for group in valid_groups:
        by_setting: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
        for action in group["actions"]:
            by_setting[(action["compaction_layer"], action["retention_fraction"])].append(action)
        for (layer, fraction), actions in by_setting.items():
            for baseline in ("uniform", "prefix", "suffix", "seeded_random"):
                chosen = choose_fixed_route(actions, baseline, seed)
                if chosen is not None:
                    chosen_rows.append(dict(chosen, baseline=baseline))
    rows = aggregate_rows(
        chosen_rows,
        group_keys=("baseline", "category", "frame_count", "compaction_layer", "retention_fraction"),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        include_category_all=True,
    )
    for row in rows:
        row["training_descriptive_not_heldout"] = True
    return rows


def oracle_tables(valid_groups: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    question_summary = []
    for group in valid_groups:
        safe_actions = [action for action in group["actions"] if action["safe"]]
        if safe_actions:
            chosen = max(safe_actions, key=lambda row: (row["flop_reduction"], row["delta_logp"], str(row["action_id"])))
            fallback = False
        else:
            first = group["actions"][0]
            chosen = dict(first, action_id="dense_fallback", compaction_layer=None, retention_fraction=None, retained_cell_ids=[], route_family="dense_fallback", delta_logp=0.0, delta_margin=0.0, prediction_changed=False, safe=True, flop_reduction=0.0)
            fallback = True
        selected.append(dict(chosen, oracle_name="global_safe_oracle_upper_bound", dense_fallback=fallback))
        question_summary.append(
            {
                "question_id": group["question_id"],
                "source_video_id": group["source_video_id"],
                "category": group["category"],
                "frame_count": group["frame_count"],
                "num_safe_routes": len(safe_actions),
                "safe_route_fraction": len(safe_actions) / max(1, len(group["actions"])),
                "best_safe_flop_reduction": max((row["flop_reduction"] for row in safe_actions), default=0.0),
                "oracle_selected_action": chosen["action_id"],
                "dense_fallback_required": fallback,
                "dense_correctness": group["dense_correct"],
            }
        )
    aggregate = aggregate_rows(
        selected,
        group_keys=("oracle_name", "category"),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        include_category_all=True,
    )
    for row in aggregate:
        subset = [item for item in selected if item["category"] == row["category"] or row["category"] == "all"]
        row["feasible_compaction_fraction"] = float(np.mean([not item["dense_fallback"] for item in subset])) if subset else 0.0
        row["dense_fallback_fraction"] = float(np.mean([item["dense_fallback"] for item in subset])) if subset else 0.0
        row["selected_layer_distribution"] = json.dumps(dict(Counter(str(item["compaction_layer"]) for item in subset)), sort_keys=True)
        row["selected_retention_distribution"] = json.dumps(dict(Counter(str(item["retention_fraction"]) for item in subset)), sort_keys=True)
    return aggregate, question_summary


def failure_cases(route_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    flips = [row for row in route_rows if row["prediction_changed"]]
    worst = sorted(route_rows, key=lambda row: row["delta_logp"])[: min(50, len(route_rows))]
    seen = set()
    output = []
    for row in worst + flips:
        key = (row["question_id"], row["frame_count"], row["action_id"])
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "question_id": row["question_id"],
                "source_video_id": row["source_video_id"],
                "category": row["category"],
                "frame_count": row["frame_count"],
                "layer": row["compaction_layer"],
                "retention": row["retention_fraction"],
                "route_family": row["route_family"],
                "retained_cells": json.dumps(row["retained_cell_ids"]),
                "delta_logp": row["delta_logp"],
                "margin_delta": row["delta_margin"],
                "flop_reduction": row["flop_reduction"],
                "dense_prediction": row["dense_prediction"],
                "compacted_prediction": row["compacted_prediction"],
            }
        )
    return output


def make_figures(output_dir: Path, route_rows: Sequence[dict[str, Any]], fixed_rows: Sequence[dict[str, Any]], oracle_rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / f"{name}.png", dpi=180)
        plt.savefig(figures / f"{name}.pdf")
        plt.close()

    if route_rows:
        plt.figure(figsize=(6, 4))
        plt.scatter([row["flop_reduction"] for row in route_rows], [row["delta_logp"] for row in route_rows], s=8, alpha=0.35)
        plt.axhline(-0.10, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Attention-FLOP reduction")
        plt.ylabel("Delta correct-choice log probability")
        plt.title("Training split exploratory analysis: quality versus FLOP reduction")
        save("quality_vs_flop_pareto")

        for metric, name, title in (
            ("safe", "safe_route_fraction_heatmap", "safe-route fraction"),
            ("delta_logp", "mean_delta_logp_heatmap", "mean delta logp"),
        ):
            frames = sorted({int(row["frame_count"]) for row in route_rows})
            fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 4), squeeze=False)
            for ax, frame in zip(axes[0], frames):
                subset = [row for row in route_rows if int(row["frame_count"]) == frame]
                layers = sorted({int(row["compaction_layer"]) for row in subset})
                fracs = sorted({float(row["retention_fraction"]) for row in subset})
                matrix = np.zeros((len(layers), len(fracs)))
                for i, layer in enumerate(layers):
                    for j, frac in enumerate(fracs):
                        vals = [float(row[metric]) for row in subset if int(row["compaction_layer"]) == layer and float(row["retention_fraction"]) == frac]
                        matrix[i, j] = float(np.mean(vals)) if vals else np.nan
                im = ax.imshow(matrix, aspect="auto")
                ax.set_xticks(range(len(fracs)), [str(item) for item in fracs])
                ax.set_yticks(range(len(layers)), [str(item) for item in layers])
                ax.set_xlabel("Retention fraction")
                ax.set_ylabel("Compaction layer")
                ax.set_title(f"{title}, {frame} frames")
                fig.colorbar(im, ax=ax)
            fig.suptitle(f"Training split exploratory analysis: {title}")
            save(name)

    if fixed_rows and oracle_rows:
        plt.figure(figsize=(7, 4))
        labels = [row.get("baseline", row.get("oracle_name", "oracle")) for row in fixed_rows + oracle_rows]
        vals = [float(row.get("mean_flop_reduction", 0.0) or 0.0) for row in fixed_rows + oracle_rows]
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=90)
        plt.ylabel("Mean attention-FLOP reduction")
        plt.title("Training split exploratory analysis: oracle versus deterministic fixed routes")
        save("oracle_vs_fixed_route_flop_reduction")

    if oracle_rows:
        layer_counts = Counter()
        for row in oracle_rows:
            try:
                dist = json.loads(row.get("selected_layer_distribution", "{}"))
                layer_counts.update(dist)
            except json.JSONDecodeError:
                pass
        if layer_counts:
            plt.figure(figsize=(6, 4))
            keys = sorted(layer_counts, key=str)
            plt.bar(keys, [layer_counts[key] for key in keys])
            plt.xlabel("Selected layer")
            plt.ylabel("Count")
            plt.title("Training split exploratory analysis: oracle selected-layer distribution by category")
            save("oracle_selected_layer_distribution_by_category")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed adaptive-compaction action-label records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--quality-floor", type=float, default=-0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        args.records,
        args.manifest,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        quality_floor=args.quality_floor,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
