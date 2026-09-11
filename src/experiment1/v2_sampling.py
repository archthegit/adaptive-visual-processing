from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class TemporalSamplingPolicy:
    name: str
    delta_t_seconds: float | None
    max_bins: int
    frames_per_bin: int
    fixed_num_frames: int | None = None
    fixed_num_bins: int | None = None
    min_bins: int = 8

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(float(value) for value in values if value > 0 and math.isfinite(float(value)))
    if not ordered:
        raise ValueError("Cannot compute percentile of an empty positive-duration set.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    position = (len(ordered) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def primary_policy_from_development_durations(
    development_durations: Iterable[float],
    target_bins_at_median: int = 16,
    frames_per_bin: int = 2,
    max_bins: int = 64,
    min_bins: int = 8,
) -> TemporalSamplingPolicy:
    if target_bins_at_median <= 0:
        raise ValueError("target_bins_at_median must be positive.")
    if frames_per_bin <= 0:
        raise ValueError("frames_per_bin must be positive.")
    if min_bins <= 0 or max_bins < min_bins:
        raise ValueError("Require 0 < min_bins <= max_bins.")
    median = percentile(development_durations, 0.5)
    delta_t = float(median / target_bins_at_median)
    return TemporalSamplingPolicy(
        name="adaptive_full_coverage_median_development",
        delta_t_seconds=max(1e-6, delta_t),
        max_bins=max_bins,
        frames_per_bin=frames_per_bin,
        fixed_num_frames=None,
        fixed_num_bins=None,
        min_bins=min_bins,
    )


def robustness_policy() -> TemporalSamplingPolicy:
    return TemporalSamplingPolicy(
        name="uniform_128_frames_16_relative_bins",
        delta_t_seconds=None,
        max_bins=16,
        frames_per_bin=8,
        fixed_num_frames=128,
        fixed_num_bins=16,
        min_bins=16,
    )


def cross_model_8_frame_policy() -> TemporalSamplingPolicy:
    return TemporalSamplingPolicy(
        name="cross_model_8_bins_1_center_frame",
        delta_t_seconds=None,
        max_bins=8,
        frames_per_bin=1,
        fixed_num_frames=8,
        fixed_num_bins=8,
        min_bins=8,
    )


def chronological_indices_without_duplicates_when_possible(
    start_frame: int,
    end_frame_exclusive: int,
    count: int,
) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if end_frame_exclusive <= start_frame:
        end_frame_exclusive = start_frame + 1
    available = end_frame_exclusive - start_frame
    sample_count = min(count, available)
    if sample_count == 1:
        return [start_frame for _ in range(count)]
    stop = end_frame_exclusive - 1
    step = (stop - start_frame) / float(sample_count - 1)
    indices = [int(round(start_frame + step * idx)) for idx in range(sample_count)]
    deduped: list[int] = []
    seen: set[int] = set()
    for index in indices:
        bounded = min(stop, max(start_frame, index))
        while bounded in seen and bounded < stop:
            bounded += 1
        while bounded in seen and bounded > start_frame:
            bounded -= 1
        if bounded not in seen:
            deduped.append(bounded)
            seen.add(bounded)
    deduped = sorted(deduped)
    while len(deduped) < count:
        deduped.append(deduped[-1])
    return deduped


def real_time_bin_plan(
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    policy: TemporalSamplingPolicy,
    decord_length: int | None = None,
    ffprobe_frame_count: int | None = None,
) -> list[dict[str, Any]]:
    if policy.delta_t_seconds is None:
        raise ValueError("real_time_bin_plan requires a policy with delta_t_seconds.")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds.")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    if policy.frames_per_bin <= 0:
        raise ValueError("frames_per_bin must be positive.")
    if policy.max_bins < policy.min_bins:
        raise ValueError("max_bins must be >= min_bins.")
    authoritative_frame_count = int(decord_length) if decord_length is not None else None
    if authoritative_frame_count is not None and authoritative_frame_count <= 0:
        raise ValueError("decord_length must be positive when provided.")
    effective_start = float(start_seconds)
    effective_end = float(end_seconds)
    analyzed_end_adjustment = None
    if authoritative_frame_count is not None:
        decodable_end = authoritative_frame_count / float(source_fps)
        if effective_end > decodable_end:
            analyzed_end_adjustment = {
                "original_analyzed_end_seconds": effective_end,
                "effective_analyzed_end_seconds": decodable_end,
                "reason": "annotated_end_exceeds_decord_boundary",
            }
            effective_end = decodable_end
        if effective_start >= effective_end:
            raise ValueError(
                "Effective analyzed interval is empty after decodable-boundary adjustment: "
                f"start={effective_start}, end={effective_end}, decord_length={authoritative_frame_count}, fps={source_fps}."
            )
    duration = effective_end - effective_start
    desired_bins = int(math.ceil(duration / float(policy.delta_t_seconds)))
    num_bins = min(policy.max_bins, max(policy.min_bins, desired_bins))
    clipped_to_min = num_bins != desired_bins and desired_bins < policy.min_bins
    clipped_to_max = num_bins != desired_bins and desired_bins > policy.max_bins
    effective_seconds_per_bin = duration / float(num_bins)
    plans: list[dict[str, Any]] = []
    for bin_index in range(num_bins):
        bin_start = effective_start + bin_index * effective_seconds_per_bin
        bin_end = effective_end if bin_index == num_bins - 1 else effective_start + (bin_index + 1) * effective_seconds_per_bin
        start_frame = int(math.ceil(bin_start * source_fps))
        end_frame = max(start_frame + 1, int(math.ceil(bin_end * source_fps)))
        if authoritative_frame_count is not None:
            if start_frame >= authoritative_frame_count:
                raise ValueError(
                    f"Bin {bin_index} starts outside decodable frames: start_frame={start_frame}, "
                    f"decord_length={authoritative_frame_count}."
                )
            end_frame = min(end_frame, authoritative_frame_count)
        frame_indices = chronological_indices_without_duplicates_when_possible(
            start_frame,
            end_frame,
            policy.frames_per_bin,
        )
        if authoritative_frame_count is not None:
            out_of_bounds = [idx for idx in frame_indices if idx < 0 or idx >= authoritative_frame_count]
            if out_of_bounds:
                raise ValueError(
                    f"Sampled frame indices outside Decord length for bin {bin_index}: "
                    f"indices={out_of_bounds}, decord_length={authoritative_frame_count}."
                )
        plans.append(
            {
                "analysis_bin": bin_index,
                "bin_start_seconds": bin_start,
                "bin_end_seconds": bin_end,
                "source_frame_indices": frame_indices,
                "source_timestamps": [index / source_fps for index in frame_indices],
                "frame_to_bin": {str(index): bin_index for index in frame_indices},
                "sample_position_to_bin": {
                    str(bin_index * policy.frames_per_bin + offset): bin_index
                    for offset in range(len(frame_indices))
                },
                "original_temporal_position": bin_index,
                "presented_temporal_position": bin_index,
                "target_delta_t_seconds": float(policy.delta_t_seconds),
                "effective_seconds_per_bin": effective_seconds_per_bin,
                "desired_bins": desired_bins,
                "num_bins": num_bins,
                "min_bin_clipped": clipped_to_min,
                "max_bin_clipped": clipped_to_max,
                "effective_analyzed_start_seconds": effective_start,
                "effective_analyzed_end_seconds": effective_end,
                "decord_frame_count": authoritative_frame_count,
                "ffprobe_frame_count": ffprobe_frame_count,
                "analyzed_end_adjustment": analyzed_end_adjustment,
            }
        )
    validate_temporal_plan(plans, max_frames=policy.max_bins * policy.frames_per_bin, decord_length=authoritative_frame_count)
    return plans


def fixed_budget_bin_plan(
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    policy: TemporalSamplingPolicy | None = None,
    decord_length: int | None = None,
    ffprobe_frame_count: int | None = None,
) -> list[dict[str, Any]]:
    policy = policy or robustness_policy()
    if policy.fixed_num_frames is None or policy.fixed_num_bins is None:
        raise ValueError("fixed_budget_bin_plan requires fixed_num_frames and fixed_num_bins.")
    if policy.fixed_num_frames != policy.fixed_num_bins * policy.frames_per_bin:
        raise ValueError("fixed_num_frames must equal fixed_num_bins * frames_per_bin.")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    authoritative_frame_count = int(decord_length) if decord_length is not None else None
    if authoritative_frame_count is not None and authoritative_frame_count <= 0:
        raise ValueError("decord_length must be positive when provided.")
    effective_start = float(start_seconds)
    effective_end = float(end_seconds)
    analyzed_end_adjustment = None
    if authoritative_frame_count is not None:
        decodable_end = authoritative_frame_count / float(source_fps)
        if effective_end > decodable_end:
            analyzed_end_adjustment = {
                "original_analyzed_end_seconds": effective_end,
                "effective_analyzed_end_seconds": decodable_end,
                "reason": "annotated_end_exceeds_decord_boundary",
            }
            effective_end = decodable_end
        if effective_start >= effective_end:
            raise ValueError(
                "Effective analyzed interval is empty after decodable-boundary adjustment: "
                f"start={effective_start}, end={effective_end}, decord_length={authoritative_frame_count}, fps={source_fps}."
            )
    start_frame = int(math.ceil(effective_start * source_fps))
    end_frame = int(math.floor(effective_end * source_fps)) + 1
    if authoritative_frame_count is not None:
        end_frame = min(end_frame, authoritative_frame_count)
    all_indices = chronological_indices_without_duplicates_when_possible(
        start_frame,
        end_frame,
        policy.fixed_num_frames,
    )
    if authoritative_frame_count is not None:
        out_of_bounds = [idx for idx in all_indices if idx < 0 or idx >= authoritative_frame_count]
        if out_of_bounds:
            raise ValueError(
                f"Sampled fixed-budget frame indices outside Decord length: "
                f"indices={out_of_bounds}, decord_length={authoritative_frame_count}."
            )
    plans: list[dict[str, Any]] = []
    for bin_index in range(policy.fixed_num_bins):
        start = bin_index * policy.frames_per_bin
        end = min(len(all_indices), start + policy.frames_per_bin)
        frame_indices = all_indices[start:end]
        plans.append(
            {
                "analysis_bin": bin_index,
                "bin_start_seconds": effective_start + (effective_end - effective_start) * bin_index / policy.fixed_num_bins,
                "bin_end_seconds": effective_start + (effective_end - effective_start) * (bin_index + 1) / policy.fixed_num_bins,
                "source_frame_indices": frame_indices,
                "source_timestamps": [index / source_fps for index in frame_indices],
                "frame_to_bin": {str(index): bin_index for index in frame_indices},
                "sample_position_to_bin": {
                    str(bin_index * policy.frames_per_bin + offset): bin_index
                    for offset in range(len(frame_indices))
                },
                "original_temporal_position": bin_index,
                "presented_temporal_position": bin_index,
                "target_delta_t_seconds": None,
                "effective_seconds_per_bin": (effective_end - effective_start) / float(policy.fixed_num_bins),
                "desired_bins": policy.fixed_num_bins,
                "num_bins": policy.fixed_num_bins,
                "min_bin_clipped": False,
                "max_bin_clipped": False,
                "effective_analyzed_start_seconds": effective_start,
                "effective_analyzed_end_seconds": effective_end,
                "decord_frame_count": authoritative_frame_count,
                "ffprobe_frame_count": ffprobe_frame_count,
                "analyzed_end_adjustment": analyzed_end_adjustment,
            }
        )
    validate_temporal_plan(plans, max_frames=policy.fixed_num_frames, decord_length=authoritative_frame_count)
    return plans


def cross_model_center_frame_bin_plan(
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    policy: TemporalSamplingPolicy | None = None,
    decord_length: int | None = None,
    ffprobe_frame_count: int | None = None,
) -> list[dict[str, Any]]:
    policy = policy or cross_model_8_frame_policy()
    if policy.fixed_num_frames != 8 or policy.fixed_num_bins != 8 or policy.frames_per_bin != 1:
        raise ValueError("cross_model_center_frame_bin_plan requires exactly 8 bins and one frame per bin.")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    authoritative_frame_count = int(decord_length) if decord_length is not None else None
    if authoritative_frame_count is not None and authoritative_frame_count <= 0:
        raise ValueError("decord_length must be positive when provided.")
    effective_start = float(start_seconds)
    effective_end = float(end_seconds)
    if effective_end <= effective_start:
        raise ValueError("end_seconds must be greater than start_seconds.")
    analyzed_end_adjustment = None
    if authoritative_frame_count is not None:
        decodable_end = authoritative_frame_count / float(source_fps)
        if effective_end > decodable_end:
            analyzed_end_adjustment = {
                "original_analyzed_end_seconds": effective_end,
                "effective_analyzed_end_seconds": decodable_end,
                "reason": "annotated_end_exceeds_decord_boundary",
            }
            effective_end = decodable_end
        if effective_start >= effective_end:
            raise ValueError(
                "Effective analyzed interval is empty after decodable-boundary adjustment: "
                f"start={effective_start}, end={effective_end}, decord_length={authoritative_frame_count}, fps={source_fps}."
            )
    duration = effective_end - effective_start
    seconds_per_bin = duration / float(policy.fixed_num_bins)
    plans: list[dict[str, Any]] = []
    sampled_indices: list[int] = []
    for bin_index in range(policy.fixed_num_bins):
        bin_start = effective_start + bin_index * seconds_per_bin
        bin_end = effective_end if bin_index == policy.fixed_num_bins - 1 else effective_start + (bin_index + 1) * seconds_per_bin
        center_timestamp = (bin_start + bin_end) / 2.0
        frame_index = int(round(center_timestamp * source_fps))
        if authoritative_frame_count is not None:
            frame_index = min(authoritative_frame_count - 1, max(0, frame_index))
        sampled_indices.append(frame_index)
        plans.append(
            {
                "analysis_bin": bin_index,
                "bin_start_seconds": bin_start,
                "bin_end_seconds": bin_end,
                "source_frame_indices": [frame_index],
                "source_timestamps": [frame_index / source_fps],
                "frame_to_bin": {str(frame_index): bin_index},
                "sample_position_to_bin": {str(bin_index): bin_index},
                "original_temporal_position": bin_index,
                "presented_temporal_position": bin_index,
                "target_delta_t_seconds": None,
                "effective_seconds_per_bin": seconds_per_bin,
                "desired_bins": policy.fixed_num_bins,
                "num_bins": policy.fixed_num_bins,
                "min_bin_clipped": False,
                "max_bin_clipped": False,
                "effective_analyzed_start_seconds": effective_start,
                "effective_analyzed_end_seconds": effective_end,
                "decord_frame_count": authoritative_frame_count,
                "ffprobe_frame_count": ffprobe_frame_count,
                "analyzed_end_adjustment": analyzed_end_adjustment,
                "center_timestamp_seconds": center_timestamp,
            }
        )
    if len(set(sampled_indices)) != len(sampled_indices):
        raise ValueError(
            "Cross-model 8-frame sampling requires distinct center frames; "
            f"sampled indices were {sampled_indices}."
        )
    validate_temporal_plan(plans, max_frames=8, decord_length=authoritative_frame_count)
    return plans


def validate_temporal_plan(
    plan: list[dict[str, Any]],
    max_frames: int = 128,
    decord_length: int | None = None,
) -> None:
    if not plan:
        raise ValueError("Temporal plan is empty.")
    num_bins = len(plan)
    if not 8 <= num_bins <= 64:
        raise ValueError(f"Temporal plan has {num_bins} bins; expected 8..64.")
    frame_count = sum(len(item.get("source_frame_indices") or []) for item in plan)
    if frame_count > max_frames:
        raise ValueError(f"Temporal plan has {frame_count} frames; maximum is {max_frames}.")
    first_start = float(plan[0]["bin_start_seconds"])
    effective_start = float(plan[0].get("effective_analyzed_start_seconds", first_start))
    if abs(first_start - effective_start) > 1e-6:
        raise ValueError("First bin does not begin at the effective analyzed start.")
    last_end = float(plan[-1]["bin_end_seconds"])
    effective_end = float(plan[-1].get("effective_analyzed_end_seconds", last_end))
    if abs(last_end - effective_end) > 1e-6:
        raise ValueError("Last bin does not end at the effective analyzed end.")
    timestamps = [float(ts) for item in plan for ts in item.get("source_timestamps", [])]
    if timestamps != sorted(timestamps):
        raise ValueError("Sampled timestamps are not monotonic.")
    seen_positions: set[int] = set()
    expected_position = 0
    for item in plan:
        mapping = item.get("sample_position_to_bin") or {}
        if len(mapping) != len(item.get("source_frame_indices") or []):
            raise ValueError("Every sampled frame position must map to exactly one bin.")
        for position_text, bin_index in mapping.items():
            position = int(position_text)
            if position in seen_positions:
                raise ValueError(f"Sample position {position} maps to more than one bin.")
            if position != expected_position:
                raise ValueError(f"Expected sample position {expected_position}, got {position}.")
            if int(bin_index) != int(item["analysis_bin"]):
                raise ValueError("Sample position maps to the wrong analysis bin.")
            seen_positions.add(position)
            expected_position += 1
        if decord_length is not None:
            out_of_bounds = [
                int(index)
                for index in item.get("source_frame_indices") or []
                if int(index) < 0 or int(index) >= int(decord_length)
            ]
            if out_of_bounds:
                raise ValueError(f"Sampled frame index outside Decord length: {out_of_bounds}.")


def reverse_presented_order(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = len(plan)
    reversed_plan: list[dict[str, Any]] = []
    for presented_position, item in enumerate(reversed(plan)):
        updated = dict(item)
        updated["presented_temporal_position"] = presented_position
        updated["original_temporal_position"] = int(item["analysis_bin"])
        updated["reversal_maps_to_original_bin"] = count - 1 - presented_position
        reversed_plan.append(updated)
    return reversed_plan


def repeated_frame_plan(plan: list[dict[str, Any]], source_frame_index: int | None = None) -> list[dict[str, Any]]:
    if not plan:
        return []
    if source_frame_index is None:
        first_frames = plan[0].get("source_frame_indices") or []
        if not first_frames:
            raise ValueError("Cannot build repeated-frame control from an empty first bin.")
        source_frame_index = int(first_frames[0])
    repeated: list[dict[str, Any]] = []
    for item in plan:
        updated = dict(item)
        frame_count = len(item.get("source_frame_indices") or [])
        updated["source_frame_indices"] = [source_frame_index for _ in range(frame_count)]
        updated["repeated_source_frame_index"] = source_frame_index
        repeated.append(updated)
    return repeated
