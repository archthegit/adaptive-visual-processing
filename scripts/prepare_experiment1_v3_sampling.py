#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment1.v2_manifest import (  # noqa: E402
    Experiment1V2Config,
    assign_duration_group,
    build_experiment1_v2_manifests,
    duration_tertiles,
    inventory_local_mp4s,
)
from src.experiment1.v2_sampling import (  # noqa: E402
    TemporalSamplingPolicy,
    primary_policy_from_development_durations,
    real_time_bin_plan,
    validate_temporal_plan,
)
from src.io import write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Experiment 1 v3 adaptive full-coverage temporal sampling artifacts "
            "and run a CPU-only sampling audit."
        )
    )
    parser.add_argument("--questions-dir", required=True)
    parser.add_argument("--mp4-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiment1_v3")
    parser.add_argument("--frozen-primary-manifest", default="outputs/experiment1_v2/primary_manifest.jsonl")
    parser.add_argument("--frozen-additional-questions", default="outputs/experiment1_v2/additional_questions.jsonl")
    parser.add_argument("--frozen-mismatched-queries", default="outputs/experiment1_v2/mismatched_queries.json")
    parser.add_argument("--expected-size", type=int, default=77)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--allow-rebuild-frozen-cohort", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def video_path_for_record(record: dict[str, Any], mp4_dir: str | Path) -> Path:
    participant = str(record.get("participant_id") or str(record["source_video_id"]).split("-")[0])
    return Path(mp4_dir) / participant / f"{record['source_video_id']}.mp4"


def decord_length(path: Path) -> int:
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("Install decord to run the authoritative v3 sampling audit.") from exc
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    return int(len(reader))


def load_or_build_frozen_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    frozen_path = Path(args.frozen_primary_manifest)
    if frozen_path.exists():
        primary = read_jsonl(frozen_path)
        if args.expected_size > 0 and len(primary) != args.expected_size:
            raise ValueError(
                f"Frozen primary manifest has {len(primary)} examples, expected {args.expected_size}. "
                "Refusing to silently change the Experiment 1 cohort."
            )
        additional_path = Path(args.frozen_additional_questions)
        mismatch_path = Path(args.frozen_mismatched_queries)
        additional = read_jsonl(additional_path) if additional_path.exists() else []
        mismatches = read_json(mismatch_path) if mismatch_path.exists() else {"seed": args.seed, "mismatches": {}}
        return primary, additional, mismatches, []

    if not args.allow_rebuild_frozen_cohort:
        raise FileNotFoundError(
            f"Frozen primary manifest not found: {frozen_path}. "
            "Pass the existing v2 manifest or use --allow-rebuild-frozen-cohort only for non-final local smoke tests."
        )
    outputs = build_experiment1_v2_manifests(
        args.questions_dir,
        args.mp4_dir,
        Experiment1V2Config(seed=args.seed, dev_fraction=args.dev_fraction),
    )
    return (
        list(outputs["primary_manifest"]),
        list(outputs["additional_questions"]),
        dict(outputs["mismatched_queries"]),
        list(outputs["exclusions"]),
    )


def rewrite_records_for_v3(records: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    rewritten = []
    for record in records:
        duration = float(record["analyzed_duration_seconds"])
        updated = dict(record)
        updated["duration_group"] = assign_duration_group(duration, thresholds)
        updated["duration_group_basis"] = "analyzed_duration_seconds"
        updated["experiment"] = "experiment1_v3_adaptive_temporal_attention"
        updated["invalid_prior_pilot_note"] = (
            "outputs/experiment1_v2/runs/baseline used the invalid 17-second policy and is non-final."
        )
        rewritten.append(updated)
    return rewritten


def question_length(record: dict[str, Any]) -> int:
    return len(str(record.get("question", "")).split())


def recompute_v3_split(
    records: list[dict[str, Any]],
    seed: int = 20260830,
    dev_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1.")
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[(str(record["category"]), str(record["duration_group"]))].append(record)

    rewritten: list[dict[str, Any]] = []
    stratum_summary: dict[str, dict[str, int | bool]] = {}
    for (category, duration_group), items in sorted(by_stratum.items()):
        ordered = sorted(
            items,
            key=lambda item: (
                str(item.get("participant_id", "")),
                str(item["source_video_id"]),
                str(item["question_id"]),
                rng.random(),
            ),
        )
        if len(ordered) == 1:
            dev_count = 0
        else:
            dev_count = max(1, int(round(len(ordered) * dev_fraction)))
            dev_count = min(dev_count, len(ordered) - 1)
        for index, record in enumerate(ordered):
            updated = dict(record)
            updated["split"] = "dev" if index < dev_count else "test"
            rewritten.append(updated)
        key = f"{category}:{duration_group}"
        stratum_summary[key] = {
            "total": len(ordered),
            "dev": dev_count,
            "test": len(ordered) - dev_count,
            "dev_test_coverage_feasible": len(ordered) >= 2,
            "has_dev_and_test_when_feasible": (dev_count > 0 and len(ordered) - dev_count > 0) if len(ordered) >= 2 else True,
        }
    return (
        sorted(rewritten, key=lambda item: (item["split"], item["category"], item["duration_group"], item["question_id"])),
        stratum_summary,
    )


def recompute_v3_mismatched_queries(records: list[dict[str, Any]], seed: int = 20260830) -> dict[str, Any]:
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[(str(record["category"]), str(record["duration_group"]))].append(record)

    mismatches: dict[str, dict[str, Any]] = {}
    stratum_failures = []
    for key, items in sorted(by_stratum.items()):
        if len(items) < 2:
            stratum_failures.append({"stratum": f"{key[0]}:{key[1]}", "count": len(items)})
            continue
        ordered = sorted(
            items,
            key=lambda item: (question_length(item), str(item["source_video_id"]), str(item["question_id"])),
        )
        donor_by_question: dict[str, dict[str, Any]] = {}
        if len(ordered) % 2 == 0:
            pair_limit = len(ordered)
        else:
            pair_limit = len(ordered) - 3
        for index in range(0, pair_limit, 2):
            left = ordered[index]
            right = ordered[index + 1]
            donor_by_question[str(left["question_id"])] = right
            donor_by_question[str(right["question_id"])] = left
        if len(ordered) % 2 == 1:
            tail = ordered[-3:]
            donor_by_question[str(tail[0]["question_id"])] = tail[1]
            donor_by_question[str(tail[1]["question_id"])] = tail[2]
            donor_by_question[str(tail[2]["question_id"])] = tail[0]
        if set(donor_by_question) != {str(item["question_id"]) for item in ordered}:
            raise ValueError(f"Could not construct complete v3 mismatch derangement for stratum {key}.")
        for item in ordered:
            donor = donor_by_question[str(item["question_id"])]
            if donor["source_video_id"] == item["source_video_id"]:
                raise ValueError(f"No different-video v3 mismatch derangement exists for stratum {key}.")
            mismatches[str(item["question_id"])] = {
                "mismatched_question_id": donor["question_id"],
                "mismatched_source_video_id": donor["source_video_id"],
                "category": item["category"],
                "duration_group": item["duration_group"],
                "question": donor["question"],
                "choices": list(donor["choices"]),
                "correct_idx": int(donor["correct_idx"]),
                "token_length_difference": abs(question_length(donor) - question_length(item)),
                "derangement_source": "adjacent_question_length_derangement_with_odd_tail_cycle",
            }
    if stratum_failures:
        raise ValueError(f"Cannot derange v3 mismatched queries for singleton strata: {stratum_failures}")
    return {
        "seed": seed,
        "source": "recomputed_from_v3_primary_manifest_after_analyzed_duration_grouping",
        "mismatches": mismatches,
    }


def assert_v3_protocol_invariants(
    frozen_records: list[dict[str, Any]],
    v3_records: list[dict[str, Any]],
    mismatches: dict[str, Any],
    split_strata: dict[str, dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> None:
    frozen_question_ids = {str(record["question_id"]) for record in frozen_records}
    v3_question_ids = {str(record["question_id"]) for record in v3_records}
    if frozen_question_ids != v3_question_ids:
        raise ValueError("v3 primary manifest does not preserve the exact frozen question IDs.")
    frozen_video_ids = {str(record["source_video_id"]) for record in frozen_records}
    v3_video_ids = {str(record["source_video_id"]) for record in v3_records}
    if frozen_video_ids != v3_video_ids:
        raise ValueError("v3 primary manifest does not preserve the exact frozen source-video IDs.")
    split_by_video: dict[str, str] = {}
    for record in v3_records:
        previous = split_by_video.setdefault(str(record["source_video_id"]), str(record["split"]))
        if previous != record["split"]:
            raise ValueError(f"Source video crosses dev/test split: {record['source_video_id']}")
    bad_strata = [
        key
        for key, item in split_strata.items()
        if item["dev_test_coverage_feasible"] and not item["has_dev_and_test_when_feasible"]
    ]
    if bad_strata:
        raise ValueError(f"Feasible v3 strata lack both dev and test coverage: {bad_strata}")
    mismatch_map = mismatches.get("mismatches", {})
    by_question = {str(record["question_id"]): record for record in v3_records}
    if set(mismatch_map) != set(by_question):
        raise ValueError("v3 mismatched_queries keys do not exactly match primary_manifest question IDs.")
    for question_id, mismatch in mismatch_map.items():
        record = by_question[question_id]
        if mismatch["category"] != record["category"]:
            raise ValueError(f"Mismatch category changed for {question_id}.")
        if mismatch["duration_group"] != record["duration_group"]:
            raise ValueError(f"Mismatch duration group changed for {question_id}.")
        if mismatch["mismatched_source_video_id"] == record["source_video_id"]:
            raise ValueError(f"Mismatch uses same source video for {question_id}.")
    if summary is not None:
        expected = summarize_records(v3_records)
        if summary.get("primary_summary") != expected:
            raise ValueError("split_summary primary_summary does not exactly match primary_manifest.")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_examples": len(records),
        "by_category": dict(sorted(Counter(record["category"] for record in records).items())),
        "by_duration_group": dict(sorted(Counter(record["duration_group"] for record in records).items())),
        "by_category_and_duration": {
            f"{category}:{group}": count
            for (category, group), count in sorted(
                Counter((record["category"], record["duration_group"]) for record in records).items()
            )
        },
        "by_split": dict(sorted(Counter(record.get("split", "unknown") for record in records).items())),
    }


def audit_sampling(
    records: list[dict[str, Any]],
    inventory_by_video: dict[str, Any],
    mp4_dir: str | Path,
    policy: TemporalSamplingPolicy,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for record in records:
        path = video_path_for_record(record, mp4_dir)
        if not path.exists():
            failures.append({"question_id": record["question_id"], "reason": f"missing_mp4: {path}"})
            continue
        inventory = inventory_by_video.get(record["source_video_id"])
        if inventory is None:
            failures.append({"question_id": record["question_id"], "reason": "missing_ffprobe_inventory"})
            continue
        try:
            decord_frames = decord_length(path)
            plan = real_time_bin_plan(
                float(record["analyzed_start_seconds"]),
                float(record["analyzed_end_seconds"]),
                float(inventory.fps),
                policy,
                decord_length=decord_frames,
                ffprobe_frame_count=int(inventory.num_frames),
            )
            validate_temporal_plan(plan, max_frames=128, decord_length=decord_frames)
        except Exception as exc:
            failures.append({"question_id": record["question_id"], "reason": str(exc)})
            continue
        indices = [int(index) for item in plan for index in item["source_frame_indices"]]
        timestamps = [float(ts) for item in plan for ts in item["source_timestamps"]]
        full_coverage = (
            abs(float(plan[0]["bin_start_seconds"]) - float(plan[0]["effective_analyzed_start_seconds"])) <= 1e-6
            and abs(float(plan[-1]["bin_end_seconds"]) - float(plan[-1]["effective_analyzed_end_seconds"])) <= 1e-6
        )
        summaries.append(
            {
                "question_id": record["question_id"],
                "source_video_id": record["source_video_id"],
                "participant_id": record["participant_id"],
                "category": record["category"],
                "duration_group": record["duration_group"],
                "split": record.get("split"),
                "analyzed_duration_seconds": float(record["analyzed_duration_seconds"]),
                "effective_analyzed_start_seconds": float(plan[0]["effective_analyzed_start_seconds"]),
                "effective_analyzed_end_seconds": float(plan[-1]["effective_analyzed_end_seconds"]),
                "num_bins": len(plan),
                "num_frames": len(indices),
                "target_delta_t_seconds": float(policy.delta_t_seconds or 0.0),
                "effective_seconds_per_bin": float(plan[0]["effective_seconds_per_bin"]),
                "min_bin_clipped": bool(plan[0]["min_bin_clipped"]),
                "max_bin_clipped": bool(plan[0]["max_bin_clipped"]),
                "first_sampled_timestamp_seconds": timestamps[0],
                "last_sampled_timestamp_seconds": timestamps[-1],
                "decord_frame_count": decord_frames,
                "ffprobe_frame_count": int(inventory.num_frames),
                "analyzed_end_adjustment": plan[0].get("analyzed_end_adjustment"),
                "full_coverage": full_coverage,
                "out_of_bounds": any(index < 0 or index >= decord_frames for index in indices),
                "frame_bin_mapping": [
                    {
                        "sample_position": position,
                        "analysis_bin": int(item["analysis_bin"]),
                        "source_frame_index": int(source_frame_index),
                        "timestamp_seconds": float(timestamp),
                        "bin_start_seconds": float(item["bin_start_seconds"]),
                        "bin_end_seconds": float(item["bin_end_seconds"]),
                    }
                    for item in plan
                    for position, (source_frame_index, timestamp) in enumerate(
                        zip(item["source_frame_indices"], item["source_timestamps"]),
                        start=int(item["analysis_bin"]) * int(policy.frames_per_bin),
                    )
                ],
            }
        )
    bin_counts = [item["num_bins"] for item in summaries]
    frame_counts = [item["num_frames"] for item in summaries]
    seconds_per_bin = [item["effective_seconds_per_bin"] for item in summaries]
    sorted_by_duration = sorted(summaries, key=lambda item: item["analyzed_duration_seconds"])
    shortest_gaze = next((item for item in sorted_by_duration if item["category"] == "gaze"), None)
    median_example = sorted_by_duration[len(sorted_by_duration) // 2] if sorted_by_duration else None
    longest_example = sorted_by_duration[-1] if sorted_by_duration else None
    collapsed_categories = sorted(
        category
        for category in {item["category"] for item in summaries}
        if {item["num_bins"] for item in summaries if item["category"] == category} == {1}
    )
    return {
        "num_records": len(records),
        "num_audited": len(summaries),
        "failures": failures,
        "bin_count_histogram": dict(sorted(Counter(bin_counts).items())),
        "min_median_max_bins": _min_median_max(bin_counts),
        "min_median_max_frames": _min_median_max(frame_counts),
        "counts_by_category_and_analyzed_duration_tertile": {
            f"{category}:{group}": count
            for (category, group), count in sorted(
                Counter((item["category"], item["duration_group"]) for item in summaries).items()
            )
        },
        "effective_seconds_per_bin": _min_median_max(seconds_per_bin),
        "min_bins_clipped_count": sum(1 for item in summaries if item["min_bin_clipped"]),
        "max_bins_clipped_count": sum(1 for item in summaries if item["max_bin_clipped"]),
        "full_coverage_failures": [item["question_id"] for item in summaries if not item["full_coverage"]],
        "out_of_bounds_failures": [item["question_id"] for item in summaries if item["out_of_bounds"]],
        "estimated_inference_workload": {
            "total_frames": sum(frame_counts),
            "max_frames_per_example": max(frame_counts) if frame_counts else 0,
            "total_temporal_bins": sum(bin_counts),
            "max_temporal_bins_per_example": max(bin_counts) if bin_counts else 0,
        },
        "shortest_gaze_sampling_summary": shortest_gaze,
        "median_duration_sampling_summary": median_example,
        "longest_duration_sampling_summary": longest_example,
        "collapsed_one_bin_categories": collapsed_categories,
        "per_example": summaries,
    }


def _min_median_max(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def main() -> None:
    args = parse_args()
    primary, additional, _old_mismatches, cohort_exclusions = load_or_build_frozen_records(args)
    thresholds = duration_tertiles(float(record["analyzed_duration_seconds"]) for record in primary)
    primary_v3 = rewrite_records_for_v3(primary, thresholds)
    additional_v3 = rewrite_records_for_v3(additional, thresholds) if additional else []
    primary_v3, split_strata = recompute_v3_split(primary_v3, seed=args.seed, dev_fraction=args.dev_fraction)
    mismatches = recompute_v3_mismatched_queries(primary_v3, seed=args.seed)
    dev_durations = [
        float(record["analyzed_duration_seconds"])
        for record in primary_v3
        if record.get("split") == "dev"
    ]
    if not dev_durations:
        raise ValueError("No development examples are available to freeze the adaptive sampling policy.")
    policy = primary_policy_from_development_durations(dev_durations)
    inventory, inventory_exclusions = inventory_local_mp4s(args.mp4_dir)
    inventory_by_video = {record.video_id: record for record in inventory}
    sampling_policy = {
        "development_duration_median_seconds": percentile_50(dev_durations),
        "target_delta_t_seconds": policy.delta_t_seconds,
        "delta_t_seconds": policy.delta_t_seconds,
        "frames_per_bin": policy.frames_per_bin,
        "min_bins": policy.min_bins,
        "max_bins": policy.max_bins,
        "max_frames": policy.max_bins * policy.frames_per_bin,
        "policy": policy.to_json(),
    }
    summary = {
        "experiment": "experiment1_v3_adaptive_temporal_attention",
        "seed": args.seed,
        "frozen_primary_manifest": str(args.frozen_primary_manifest),
        "frozen_question_video_ids_preserved": True,
        "invalid_prior_pilot_note": "outputs/experiment1_v2/runs/baseline is invalid and non-final.",
        "duration_tertile_thresholds": thresholds,
        "duration_tertile_basis": "analyzed_duration_seconds",
        "realtime_sampling_policy": sampling_policy,
        "split_recomputed_after_analyzed_duration_grouping": True,
        "mismatches_recomputed_after_analyzed_duration_grouping": True,
        "target_delta_t_computed_after_corrected_split": True,
        "primary_manifest_count": len(primary_v3),
        "additional_question_count": len(additional_v3),
        "inventory_complete_mp4s": len(inventory),
        "primary_summary": summarize_records(primary_v3),
        "split_strata": split_strata,
    }
    assert_v3_protocol_invariants(primary, primary_v3, mismatches, split_strata, summary)
    audit = audit_sampling(primary_v3, inventory_by_video, args.mp4_dir, policy)
    summary["sampling_audit"] = {key: value for key, value in audit.items() if key != "per_example"}
    if audit["failures"]:
        summary["audit_status"] = "failed"
    elif audit["full_coverage_failures"] or audit["out_of_bounds_failures"] or audit["collapsed_one_bin_categories"]:
        summary["audit_status"] = "failed"
    else:
        summary["audit_status"] = "passed"

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "duration_inventory.jsonl", [record.to_json() for record in inventory])
    write_jsonl(output_dir / "primary_manifest.jsonl", primary_v3)
    write_jsonl(output_dir / "additional_questions.jsonl", additional_v3)
    write_json(output_dir / "mismatched_queries.json", mismatches)
    write_jsonl(output_dir / "exclusions.jsonl", inventory_exclusions + cohort_exclusions)
    write_jsonl(output_dir / "sampling_audit_per_example.jsonl", audit["per_example"])
    write_json(output_dir / "sampling_audit.json", audit)
    write_json(output_dir / "split_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["audit_status"] != "passed":
        raise SystemExit(1)


def percentile_50(values: list[float]) -> float:
    return float(statistics.median(values))


if __name__ == "__main__":
    main()
