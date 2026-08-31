#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.v2_controls import write_v2_control_manifest
from src.experiment1.v2_manifest import Experiment1V2Config, build_experiment1_v2_manifests
from src.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct Experiment 1 v2 manifests and print ordered GPU commands without running inference."
    )
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--output-root", default="outputs/experiment1_v2")
    parser.add_argument("--run-root", default="outputs/experiment1_v2/runs")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def qwen_run_command(
    manifest: str,
    output_dir: str,
    questions_dir: str,
    mp4_dir: str,
    condition: str,
    sampling_mode: str = "realtime",
    num_frames: int = 128,
    max_new_tokens: int = 16,
    sampling_policy_json: str | None = None,
) -> str:
    parts = [
            "python scripts/run_experiment1.py",
            f"--questions-dir {questions_dir}",
            f"--mp4-dir {mp4_dir}",
            f"--manifest {manifest}",
            f"--num-frames {num_frames}",
            f"--sampling-mode {sampling_mode}",
    ]
    if sampling_policy_json is not None:
        parts.append(f"--sampling-policy-json {sampling_policy_json}")
    parts.extend(
        [
            "--resolution-config low",
            "--vision-access-through-layer none",
            "--query-scope question",
            "--attention-extraction reduced_sdpa",
            f"--max-new-tokens {max_new_tokens}",
            f"--condition {condition}",
            f"--output-dir {output_dir}",
            "--resume",
            "--allow-7b-inference",
        ]
    )
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = build_experiment1_v2_manifests(
        args.questions_dir,
        args.mp4_dir,
        Experiment1V2Config(seed=args.seed, dev_fraction=args.dev_fraction),
    )
    write_jsonl(output_root / "duration_inventory.jsonl", manifests["duration_inventory"])
    write_jsonl(output_root / "primary_manifest.jsonl", manifests["primary_manifest"])
    write_jsonl(output_root / "additional_questions.jsonl", manifests["additional_questions"])
    write_json(output_root / "mismatched_queries.json", manifests["mismatched_queries"])
    write_json(output_root / "split_summary.json", manifests["split_summary"])
    write_jsonl(output_root / "exclusions.jsonl", manifests["exclusions"])

    controls_dir = output_root / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    for control in ("repeated_frame", "reversed_video", "mismatched_query", "same_video_different_query"):
        try:
            write_v2_control_manifest(
                output_root / "primary_manifest.jsonl",
                output_root / "mismatched_queries.json",
                controls_dir / f"{control}.jsonl",
                control,
                additional_questions_path=output_root / "additional_questions.jsonl",
            )
        except ValueError as exc:
            write_json(controls_dir / f"{control}_skipped.json", {"control": control, "reason": str(exc)})

    commands: list[str] = []
    primary = str(output_root / "primary_manifest.jsonl")
    sampling_policy = str(output_root / "split_summary.json")
    run_root = Path(args.run_root)
    commands.append(
        qwen_run_command(
            primary,
            str(run_root / "baseline"),
            args.questions_dir,
            args.mp4_dir,
            "baseline",
            max_new_tokens=args.max_new_tokens,
            sampling_policy_json=sampling_policy,
        )
    )
    commands.append(
        qwen_run_command(
            primary,
            str(run_root / "baseline_fixed_budget"),
            args.questions_dir,
            args.mp4_dir,
            "baseline_fixed_budget",
            sampling_mode="fixed_budget",
            max_new_tokens=args.max_new_tokens,
        )
    )
    for control in ("repeated_frame", "reversed_video", "mismatched_query", "same_video_different_query"):
        manifest_path = controls_dir / f"{control}.jsonl"
        if manifest_path.exists():
            commands.append(
                qwen_run_command(
                    str(manifest_path),
                    str(run_root / control),
                    args.questions_dir,
                    args.mp4_dir,
                    control,
                    max_new_tokens=args.max_new_tokens,
                    sampling_policy_json=sampling_policy,
                )
            )

    commands.append(
        "python scripts/select_experiment1_v2_reference_layer.py "
        f"--primary-manifest {primary} "
        f"--baseline-output-dir {run_root / 'baseline'} "
        f"--mismatched-output-dir {run_root / 'mismatched_query'} "
        f"--output-json {output_root / 'frozen_reference_layer.json'}"
    )

    interventions_dir = output_root / "interventions"
    intervention_specs = [
        ("mask_top20", "top", "realtime"),
        ("mask_bottom20", "bottom", "realtime"),
        ("mask_random20", "random", "realtime"),
        ("mask_mismatched_top20", "mismatched_top", "realtime"),
        ("mask_contiguous_high_cluster", "contiguous_high_cluster", "realtime"),
        ("keep_top20", "top", "realtime"),
        ("keep_uniform20", "uniform", "realtime"),
        ("keep_random20", "random", "realtime"),
        ("keep_mismatched_top20", "mismatched_top", "realtime"),
        ("mask_top20_fixed_budget", "top", "fixed_budget"),
        ("mask_random20_fixed_budget", "random", "fixed_budget"),
    ]
    for condition, strategy, sampling_mode in intervention_specs:
        baseline_for_condition = run_root / ("baseline_fixed_budget" if sampling_mode == "fixed_budget" else "baseline")
        create = (
            "python scripts/create_experiment1_v2_intervention_manifest.py "
            f"--primary-manifest {primary} "
            f"--baseline-output-dir {baseline_for_condition} "
            f"--output-jsonl {interventions_dir / (condition + '.jsonl')} "
            f"--condition {condition} "
            f"--strategy {strategy} "
            "--removal-fraction 0.2 "
            f"--frozen-reference-layer-json {output_root / 'frozen_reference_layer.json'}"
        )
        if strategy == "mismatched_top":
            create += f" --mismatched-output-dir {run_root / 'mismatched_query'}"
        commands.append(create)
        commands.append(
            qwen_run_command(
                str(interventions_dir / f"{condition}.jsonl"),
                str(run_root / condition),
                args.questions_dir,
                args.mp4_dir,
                condition,
                sampling_mode=sampling_mode,
                max_new_tokens=args.max_new_tokens,
                sampling_policy_json=sampling_policy if sampling_mode == "realtime" else None,
            )
        )
    for layer in (0, 4, 8, 12, 16, 20, 24, 27):
        for suffix, strategy in (("top20", "top"), ("random20", "random")):
            condition = f"fusion_block_{suffix}_after_layer_{layer}"
            commands.append(
                "python scripts/create_experiment1_v2_intervention_manifest.py "
                f"--primary-manifest {primary} "
                f"--baseline-output-dir {run_root / 'baseline'} "
                f"--output-jsonl {interventions_dir / (condition + '.jsonl')} "
                f"--condition {condition} "
                f"--strategy {strategy} "
                "--removal-fraction 0.2 "
                f"--frozen-reference-layer-json {output_root / 'frozen_reference_layer.json'}"
            )
            commands.append(
                qwen_run_command(
                    str(interventions_dir / f"{condition}.jsonl"),
                    str(run_root / condition),
                    args.questions_dir,
                    args.mp4_dir,
                    condition,
                    max_new_tokens=args.max_new_tokens,
                    sampling_policy_json=sampling_policy,
                )
            )

    commands.append(
        "python scripts/analyze_experiment1_v2.py "
        f"--primary-manifest {primary} "
        f"--output-root {run_root} "
        f"--final-dir {output_root / 'final'} "
        f"--bootstrap-replicates {args.bootstrap_replicates}"
    )
    write_json(output_root / "ordered_gpu_commands.json", {"commands": commands})
    print(json.dumps({"output_root": str(output_root), "commands": commands}, indent=2))


if __name__ == "__main__":
    main()
