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
    target_bins_at_p95: int = 64,
    frames_per_bin: int = 2,
    max_bins: int = 64,
) -> TemporalSamplingPolicy:
    if target_bins_at_p95 <= 0:
        raise ValueError("target_bins_at_p95 must be positive.")
    if frames_per_bin <= 0:
        raise ValueError("frames_per_bin must be positive.")
    p95 = percentile(development_durations, 0.95)
    delta_t = float(math.ceil(p95 / target_bins_at_p95))
    return TemporalSamplingPolicy(
        name="real_time_p95_development",
        delta_t_seconds=max(1.0, delta_t),
        max_bins=max_bins,
        frames_per_bin=frames_per_bin,
        fixed_num_frames=None,
        fixed_num_bins=None,
    )


def robustness_policy() -> TemporalSamplingPolicy:
    return TemporalSamplingPolicy(
        name="uniform_128_frames_16_relative_bins",
        delta_t_seconds=None,
        max_bins=16,
        frames_per_bin=8,
        fixed_num_frames=128,
        fixed_num_bins=16,
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
) -> list[dict[str, Any]]:
    if policy.delta_t_seconds is None:
        raise ValueError("real_time_bin_plan requires a policy with delta_t_seconds.")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds.")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive.")
    duration = end_seconds - start_seconds
    num_bins = min(policy.max_bins, max(1, int(math.ceil(duration / policy.delta_t_seconds))))
    plans: list[dict[str, Any]] = []
    for bin_index in range(num_bins):
        bin_start = start_seconds + bin_index * policy.delta_t_seconds
        bin_end = min(end_seconds, start_seconds + (bin_index + 1) * policy.delta_t_seconds)
        start_frame = int(math.ceil(bin_start * source_fps))
        end_frame = max(start_frame + 1, int(math.ceil(bin_end * source_fps)))
        frame_indices = chronological_indices_without_duplicates_when_possible(
            start_frame,
            end_frame,
            policy.frames_per_bin,
        )
        plans.append(
            {
                "analysis_bin": bin_index,
                "bin_start_seconds": bin_start,
                "bin_end_seconds": bin_end,
                "source_frame_indices": frame_indices,
                "source_timestamps": [index / source_fps for index in frame_indices],
                "frame_to_bin": {str(index): bin_index for index in frame_indices},
                "original_temporal_position": bin_index,
                "presented_temporal_position": bin_index,
            }
        )
    return plans


def fixed_budget_bin_plan(
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    policy: TemporalSamplingPolicy | None = None,
) -> list[dict[str, Any]]:
    policy = policy or robustness_policy()
    if policy.fixed_num_frames is None or policy.fixed_num_bins is None:
        raise ValueError("fixed_budget_bin_plan requires fixed_num_frames and fixed_num_bins.")
    if policy.fixed_num_frames != policy.fixed_num_bins * policy.frames_per_bin:
        raise ValueError("fixed_num_frames must equal fixed_num_bins * frames_per_bin.")
    start_frame = int(math.ceil(start_seconds * source_fps))
    end_frame = int(math.floor(end_seconds * source_fps)) + 1
    all_indices = chronological_indices_without_duplicates_when_possible(
        start_frame,
        end_frame,
        policy.fixed_num_frames,
    )
    plans: list[dict[str, Any]] = []
    for bin_index in range(policy.fixed_num_bins):
        start = bin_index * policy.frames_per_bin
        end = min(len(all_indices), start + policy.frames_per_bin)
        frame_indices = all_indices[start:end]
        plans.append(
            {
                "analysis_bin": bin_index,
                "bin_start_seconds": start_seconds + (end_seconds - start_seconds) * bin_index / policy.fixed_num_bins,
                "bin_end_seconds": start_seconds + (end_seconds - start_seconds) * (bin_index + 1) / policy.fixed_num_bins,
                "source_frame_indices": frame_indices,
                "source_timestamps": [index / source_fps for index in frame_indices],
                "frame_to_bin": {str(index): bin_index for index in frame_indices},
                "original_temporal_position": bin_index,
                "presented_temporal_position": bin_index,
            }
        )
    return plans


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
