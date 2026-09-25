#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_qwen_temporal_handoff_dev import bootstrap_ci, write_json
from scripts.run_qwen_temporal_handoff_pair_sweep import (
    PAIR_SWEEP_CONDITIONS,
    RETAINED_PAIRS,
    latest_artifacts,
    manifest_by_id,
    pair_key,
    pair_rows_from_artifacts,
)


THRESHOLDS = (0.01, 0.05, 0.10, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen temporal-handoff retained-pair oracle results.")
    parser.add_argument("--input-dir", default="outputs/experiment1_v3_temporal_handoff/dev_qwen_pair_sweep")
    parser.add_argument("--manifest", default="outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def _condition_rows(rows: Sequence[dict[str, Any]], qid: str, condition: str) -> list[dict[str, Any]]:
    subset = [row for row in rows if row["question_id"] == qid and row["condition"] == condition]
    if len(subset) != len(RETAINED_PAIRS):
        raise RuntimeError(f"{qid}/{condition}: expected {len(RETAINED_PAIRS)} retained-pair rows, found {len(subset)}.")
    return sorted(subset, key=lambda row: row["retained_pair"])


def _attention_selected(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(subset, key=lambda row: (float(row["retained_attention_mass"]), str(row["retained_pair"])))


def _rank_desc(values: Sequence[float], selected: float) -> int:
    return 1 + sum(1 for value in values if float(value) > float(selected))


def oracle_per_example(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    qids = sorted({row["question_id"] for row in rows})
    output: list[dict[str, Any]] = []
    for qid in qids:
        for condition in PAIR_SWEEP_CONDITIONS:
            subset = _condition_rows(rows, qid, condition)
            dense_logp = float(subset[0]["dense_correct_choice_log_probability"])
            dense_margin = float(subset[0]["dense_answer_margin"])
            best = max(subset, key=lambda row: (float(row["correct_choice_log_probability"]), row["retained_pair"]))
            worst = min(subset, key=lambda row: (float(row["correct_choice_log_probability"]), row["retained_pair"]))
            selected = _attention_selected(subset)
            selected_logp = float(selected["correct_choice_log_probability"])
            deltas = [float(row["correct_choice_log_probability"]) - dense_logp for row in subset]
            row = {
                "question_id": qid,
                "condition": condition,
                "dense_correct_choice_log_probability": dense_logp,
                "dense_answer_margin": dense_margin,
                "best_pair": best["retained_pair"],
                "best_pair_logp": float(best["correct_choice_log_probability"]),
                "best_pair_margin": float(best["answer_margin"]),
                "best_pair_delta_logp_vs_dense": float(best["correct_choice_log_probability"]) - dense_logp,
                "best_pair_delta_margin_vs_dense": float(best["answer_margin"]) - dense_margin,
                "worst_pair": worst["retained_pair"],
                "worst_pair_logp": float(worst["correct_choice_log_probability"]),
                "worst_pair_margin": float(worst["answer_margin"]),
                "worst_pair_delta_logp_vs_dense": float(worst["correct_choice_log_probability"]) - dense_logp,
                "worst_pair_delta_margin_vs_dense": float(worst["answer_margin"]) - dense_margin,
                "attention_selected_pair": selected["retained_pair"],
                "attention_selected_logp": selected_logp,
                "attention_selected_margin": float(selected["answer_margin"]),
                "attention_selected_regret_logp": float(best["correct_choice_log_probability"]) - selected_logp,
                "attention_selected_rank_logp": _rank_desc(
                    [float(item["correct_choice_log_probability"]) for item in subset],
                    selected_logp,
                ),
                "any_pair_preserves_or_improves_dense_logp": any(delta >= 0 for delta in deltas),
            }
            for threshold in THRESHOLDS:
                count = sum(1 for delta in deltas if delta >= -threshold)
                suffix = str(threshold).replace(".", "_")
                row[f"num_pairs_within_{suffix}_logp_of_dense"] = count
                row[f"fraction_pairs_within_{suffix}_logp_of_dense"] = count / float(len(subset))
            output.append(row)
    return output


def fixed_pair_summary(rows: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in PAIR_SWEEP_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        mean_by_pair = {}
        for pair in (pair_key(pair) for pair in RETAINED_PAIRS):
            subset = [row for row in condition_rows if row["retained_pair"] == pair]
            if not subset:
                raise RuntimeError(f"{condition}/{pair}: no rows.")
            deltas = [float(row["delta_logp_vs_dense"]) for row in subset]
            margins = [float(row["delta_margin_vs_dense"]) for row in subset]
            accuracies = [bool(row["correct"]) for row in subset]
            catastrophes = [float(row["delta_logp_vs_dense"]) < -0.5 for row in subset]
            mean_by_pair[pair] = float(np.mean(deltas))
            logp_ci = bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed + len(output) * 13)
            margin_ci = bootstrap_ci(margins, samples=bootstrap_samples, seed=seed + len(output) * 13 + 1)
            output.append(
                {
                    "condition": condition,
                    "retained_pair": pair,
                    "mean_delta_logp_vs_dense": float(np.mean(deltas)),
                    "median_delta_logp_vs_dense": float(np.median(deltas)),
                    "delta_logp_ci95_low": logp_ci["ci95"][0],
                    "delta_logp_ci95_high": logp_ci["ci95"][1],
                    "mean_delta_answer_margin_vs_dense": float(np.mean(margins)),
                    "median_delta_answer_margin_vs_dense": float(np.median(margins)),
                    "delta_margin_ci95_low": margin_ci["ci95"][0],
                    "delta_margin_ci95_high": margin_ci["ci95"][1],
                    "accuracy": float(np.mean(accuracies)),
                    "catastrophic_degradation_rate_below_minus_0_5": float(np.mean(catastrophes)),
                }
            )
        ranks = {pair: rank for rank, pair in enumerate(sorted(mean_by_pair, key=lambda item: mean_by_pair[item], reverse=True), start=1)}
        for row in output:
            if row["condition"] == condition:
                row["rank_across_fixed_pairs"] = ranks[row["retained_pair"]]
    return output


def _ci_from_example_rows(rows: Sequence[dict[str, Any]], field: str, samples: int, seed: int) -> dict[str, Any]:
    return bootstrap_ci([float(row[field]) for row in rows], samples=samples, seed=seed)


def _fraction_ci(rows: Sequence[dict[str, Any]], field: str, samples: int, seed: int) -> dict[str, Any]:
    return bootstrap_ci([1.0 if row[field] else 0.0 for row in rows], samples=samples, seed=seed)


def best_pair_frequencies(per_example: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    subset = [row for row in per_example if row["condition"] == condition]
    counts = Counter(row["best_pair"] for row in subset)
    cell_counts = Counter()
    for pair, count in counts.items():
        for cell in pair.split("_"):
            cell_counts[cell] += count
    return {
        "pair_counts": dict(sorted(counts.items())),
        "pair_fractions": {pair: count / float(len(subset)) for pair, count in sorted(counts.items())},
        "cell_inclusion_counts": dict(sorted(cell_counts.items())),
        "cell_inclusion_fractions": {cell: count / float(len(subset)) for cell, count in sorted(cell_counts.items())},
        "interpretation": "Concentration is descriptive on the 15-example development set; no inferential position-universality claim is made.",
    }


def summarize(per_example: Sequence[dict[str, Any]], fixed_rows: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"conditions": {}, "fixed_pair_summary": fixed_rows}
    for cidx, condition in enumerate(PAIR_SWEEP_CONDITIONS):
        subset = [row for row in per_example if row["condition"] == condition]
        no_pair_within_010 = [
            row["num_pairs_within_0_1_logp_of_dense"] == 0
            for row in subset
        ]
        summary["conditions"][condition] = {
            "num_examples": len(subset),
            "best_pair_delta_logp_vs_dense": _ci_from_example_rows(subset, "best_pair_delta_logp_vs_dense", bootstrap_samples, seed + cidx * 100),
            "worst_pair_delta_logp_vs_dense": _ci_from_example_rows(subset, "worst_pair_delta_logp_vs_dense", bootstrap_samples, seed + cidx * 100 + 1),
            "attention_selected_regret_logp": _ci_from_example_rows(subset, "attention_selected_regret_logp", bootstrap_samples, seed + cidx * 100 + 2),
            "fraction_any_pair_preserves_or_improves_dense_logp": _fraction_ci(subset, "any_pair_preserves_or_improves_dense_logp", bootstrap_samples, seed + cidx * 100 + 3),
            "fraction_no_pair_within_0_10_logp_of_dense": bootstrap_ci([1.0 if item else 0.0 for item in no_pair_within_010], samples=bootstrap_samples, seed=seed + cidx * 100 + 4),
            "best_pair_frequency": best_pair_frequencies(per_example, condition),
        }
    handoff_best = summary["conditions"]["handoff_mean"]["best_pair_delta_logp_vs_dense"]["mean"]
    hard_best = summary["conditions"]["hard_evict"]["best_pair_delta_logp_vs_dense"]["mean"]
    fixed_safe = {
        row["retained_pair"]: row
        for row in fixed_rows
        if row["condition"] == "handoff_mean"
        and row["catastrophic_degradation_rate_below_minus_0_5"] == 0.0
        and row["mean_delta_logp_vs_dense"] >= -0.1
    }
    attention_regret = float(summary["conditions"]["handoff_mean"]["attention_selected_regret_logp"]["mean"])
    summary["answers"] = {
        "is_there_oracle_headroom_for_better_selector": attention_regret > 0.0,
        "is_there_globally_safe_fixed_temporal_pair": bool(fixed_safe),
        "globally_safe_fixed_pairs_by_development_criterion": sorted(fixed_safe),
        "does_compact_memory_improve_oracle_ceiling_over_hard_deletion": handoff_best > hard_best,
        "recommendation": recommendation(summary),
        "scope": "Development-only oracle analysis. This is not held-out evidence and does not test new inference.",
    }
    return summary


def recommendation(summary: dict[str, Any]) -> str:
    handoff = summary["conditions"]["handoff_mean"]
    regret = float(handoff["attention_selected_regret_logp"]["mean"])
    oracle = float(handoff["best_pair_delta_logp_vs_dense"]["mean"])
    no_close = float(handoff["fraction_no_pair_within_0_10_logp_of_dense"]["mean"])
    memory_beats = float(handoff["best_pair_delta_logp_vs_dense"]["mean"]) > float(summary["conditions"]["hard_evict"]["best_pair_delta_logp_vs_dense"]["mean"])
    if oracle >= -0.1 and regret > 0.05:
        return "develop a better selector"
    if no_close > 0.5:
        return "reduce the compaction ratio or improve the memory representation before further selector work"
    if memory_beats:
        return "improve the memory representation and selector jointly"
    return "terminate this direction unless a less aggressive compression setting creates oracle headroom"


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plt():
    import matplotlib.pyplot as plt

    return plt


def write_plots(output_dir: Path, per_example: Sequence[dict[str, Any]], fixed_rows: Sequence[dict[str, Any]]) -> None:
    plt = _plt()
    output_dir.mkdir(parents=True, exist_ok=True)

    handoff = [row for row in per_example if row["condition"] == "handoff_mean"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([row["question_id"] for row in handoff], [row["best_pair_delta_logp_vs_dense"] for row in handoff], label="best")
    ax.bar([row["question_id"] for row in handoff], [row["attention_selected_logp"] - row["dense_correct_choice_log_probability"] for row in handoff], alpha=0.6, label="attention-selected")
    ax.set_ylabel("Delta logp vs dense")
    ax.set_xticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_pair_delta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    hfixed = [row for row in fixed_rows if row["condition"] == "handoff_mean"]
    ax.bar([row["retained_pair"] for row in hfixed], [row["mean_delta_logp_vs_dense"] for row in hfixed])
    ax.set_ylabel("Mean delta logp vs dense")
    ax.set_xlabel("Fixed retained pair")
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_pair_quality.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = Counter(row["best_pair"] for row in handoff)
    ax.bar(list(counts), [counts[key] for key in counts])
    ax.set_ylabel("Best-pair count")
    ax.set_xlabel("Retained pair")
    fig.tight_layout()
    fig.savefig(output_dir / "best_pair_frequency.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    lines = [
        "# Qwen Temporal-Handoff Retained-Pair Oracle Analysis",
        "",
        "This CPU-only analysis uses the 15 development examples as independent units. Pair rows are not treated as independent samples.",
        "",
        "## Direct Answers",
        "",
        f"- Oracle headroom for a better temporal selector: `{answers['is_there_oracle_headroom_for_better_selector']}`.",
        f"- Globally safe fixed temporal pair: `{answers['is_there_globally_safe_fixed_temporal_pair']}`.",
        f"- Compact memory improves oracle ceiling over hard deletion: `{answers['does_compact_memory_improve_oracle_ceiling_over_hard_deletion']}`.",
        f"- Recommendation: **{answers['recommendation']}**.",
        "",
        "This is development-only evidence and does not claim held-out performance or end-to-end generation speedup.",
    ]
    path.write_text("\n".join(lines) + "\n")


def analyze(input_dir: str | Path, manifest: str | Path, *, bootstrap_samples: int, seed: int, write_outputs: bool = True) -> dict[str, Any]:
    output = Path(input_dir)
    if not output.exists():
        raise FileNotFoundError(f"Pair-sweep output directory not found: {output}")
    manifest_records = manifest_by_id(manifest)
    rows = pair_rows_from_artifacts(latest_artifacts(output), manifest_records)
    per_example = oracle_per_example(rows)
    fixed_rows = fixed_pair_summary(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    summary = summarize(per_example, fixed_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    if write_outputs:
        write_csv(output / "oracle_per_example.csv", per_example)
        write_csv(output / "fixed_pair_summary.csv", fixed_rows)
        write_json(output / "oracle_summary.json", summary)
        write_plots(output, per_example, fixed_rows)
        write_report(output / "oracle_report.md", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(args.input_dir, args.manifest, bootstrap_samples=args.bootstrap_samples, seed=args.seed, write_outputs=True)
    print(json.dumps({"input_dir": args.input_dir, "answers": summary["answers"]}, indent=2))


if __name__ == "__main__":
    main()
