import pytest

from src.experiment1.v2_sampling import (
    TemporalSamplingPolicy,
    chronological_indices_without_duplicates_when_possible,
    cross_model_center_frame_bin_plan,
    fixed_budget_bin_plan,
    primary_policy_from_development_durations,
    real_time_bin_plan,
    repeated_frame_plan,
    reverse_presented_order,
    robustness_policy,
    validate_temporal_plan,
)


def test_primary_policy_uses_development_median_for_target_delta_t():
    policy = primary_policy_from_development_durations([10, 20, 30, 640])
    assert policy.name == "adaptive_full_coverage_median_development"
    assert policy.delta_t_seconds == 25.0 / 16.0
    assert policy.frames_per_bin == 2
    assert policy.min_bins == 8
    assert policy.max_bins == 64


def test_real_time_bin_plan_samples_two_chronological_frames_per_bin():
    policy = primary_policy_from_development_durations([64, 64, 64])
    plan = real_time_bin_plan(0.0, 32.0, source_fps=10.0, policy=policy, decord_length=400)
    assert len(plan) == 8
    assert [item["source_frame_indices"] for item in plan[:2]] == [[0, 39], [40, 79]]
    assert all(len(item["source_frame_indices"]) == 2 for item in plan)
    assert plan[0]["source_timestamps"] == [0.0, 3.9]
    assert plan[0]["bin_start_seconds"] == 0.0
    assert plan[-1]["bin_end_seconds"] == 32.0
    validate_temporal_plan(plan, decord_length=400)


def test_real_time_bin_plan_clips_short_examples_to_eight_bins():
    policy = primary_policy_from_development_durations([160, 160, 160])
    plan = real_time_bin_plan(10.0, 14.0, source_fps=10.0, policy=policy, decord_length=200)
    assert len(plan) == 8
    assert sum(len(item["source_frame_indices"]) for item in plan) == 16
    assert plan[0]["min_bin_clipped"] is True
    assert plan[0]["max_bin_clipped"] is False
    assert plan[0]["bin_start_seconds"] == 10.0
    assert plan[-1]["bin_end_seconds"] == 14.0
    timestamps = [ts for item in plan for ts in item["source_timestamps"]]
    assert timestamps == sorted(timestamps)


def test_real_time_bin_plan_caps_long_examples_but_covers_full_interval():
    policy = TemporalSamplingPolicy(
        name="adaptive_full_coverage_median_development",
        delta_t_seconds=1.0,
        max_bins=64,
        frames_per_bin=2,
        min_bins=8,
    )
    plan = real_time_bin_plan(0.0, 100.0, source_fps=10.0, policy=policy, decord_length=1200)
    assert len(plan) == 64
    assert sum(len(item["source_frame_indices"]) for item in plan) == 128
    assert plan[0]["max_bin_clipped"] is True
    assert plan[0]["effective_seconds_per_bin"] == 100.0 / 64.0
    assert plan[0]["bin_start_seconds"] == 0.0
    assert plan[-1]["bin_end_seconds"] == 100.0
    validate_temporal_plan(plan, decord_length=1200)


def test_real_time_bin_plan_adjusts_end_to_decord_boundary_before_sampling():
    policy = TemporalSamplingPolicy(
        name="adaptive_full_coverage_median_development",
        delta_t_seconds=1.0,
        max_bins=64,
        frames_per_bin=2,
        min_bins=8,
    )
    plan = real_time_bin_plan(0.0, 11.0, source_fps=10.0, policy=policy, decord_length=100, ffprobe_frame_count=110)
    indices = [idx for item in plan for idx in item["source_frame_indices"]]
    assert max(indices) < 100
    assert plan[-1]["bin_end_seconds"] == 10.0
    assert plan[0]["analyzed_end_adjustment"]["original_analyzed_end_seconds"] == 11.0
    assert plan[0]["ffprobe_frame_count"] == 110
    validate_temporal_plan(plan, decord_length=100)


def test_chronological_indices_duplicate_only_when_required():
    assert chronological_indices_without_duplicates_when_possible(0, 10, 4) == [0, 3, 6, 9]
    assert chronological_indices_without_duplicates_when_possible(5, 6, 2) == [5, 5]


def test_fixed_budget_policy_uses_128_frames_and_16_bins():
    plan = fixed_budget_bin_plan(0.0, 128.0, source_fps=2.0, policy=robustness_policy())
    assert len(plan) == 16
    assert sum(len(item["source_frame_indices"]) for item in plan) == 128
    assert all(len(item["source_frame_indices"]) == 8 for item in plan)
    flattened = [idx for item in plan for idx in item["source_frame_indices"]]
    assert flattened == sorted(flattened)
    assert len(flattened) == len(set(flattened))


def test_fixed_budget_plan_adjusts_end_to_decord_boundary():
    plan = fixed_budget_bin_plan(
        0.0,
        11.0,
        source_fps=10.0,
        policy=robustness_policy(),
        decord_length=100,
        ffprobe_frame_count=110,
    )
    flattened = [idx for item in plan for idx in item["source_frame_indices"]]
    assert len(plan) == 16
    assert len(flattened) == 128
    assert max(flattened) < 100
    assert plan[-1]["bin_end_seconds"] == 10.0
    assert plan[0]["analyzed_end_adjustment"]["effective_analyzed_end_seconds"] == 10.0


def test_cross_model_policy_uses_eight_center_frames_with_full_coverage():
    plan = cross_model_center_frame_bin_plan(10.0, 18.0, source_fps=10.0, decord_length=200)
    assert len(plan) == 8
    assert sum(len(item["source_frame_indices"]) for item in plan) == 8
    assert [item["source_frame_indices"][0] for item in plan] == [105, 115, 125, 135, 145, 155, 165, 175]
    assert len({item["source_frame_indices"][0] for item in plan}) == 8
    assert plan[0]["bin_start_seconds"] == 10.0
    assert plan[-1]["bin_end_seconds"] == 18.0
    assert plan[0]["effective_seconds_per_bin"] == 1.0
    validate_temporal_plan(plan, max_frames=8, decord_length=200)


def test_cross_model_policy_rejects_duplicate_center_frames():
    with pytest.raises(ValueError, match="distinct center frames"):
        cross_model_center_frame_bin_plan(0.0, 0.1, source_fps=10.0, decord_length=10)


def test_reverse_and_repeated_frame_controls_preserve_positions():
    policy = primary_policy_from_development_durations([64, 64, 64])
    plan = real_time_bin_plan(0.0, 16.0, source_fps=10.0, policy=policy, decord_length=200)
    reversed_plan = reverse_presented_order(plan)
    assert [item["analysis_bin"] for item in reversed_plan] == list(reversed(range(8)))
    assert [item["presented_temporal_position"] for item in reversed_plan] == list(range(8))
    assert [item["reversal_maps_to_original_bin"] for item in reversed_plan] == list(reversed(range(8)))
    repeated = repeated_frame_plan(plan, source_frame_index=7)
    assert [item["presented_temporal_position"] for item in repeated] == list(range(8))
    assert all(set(item["source_frame_indices"]) == {7} for item in repeated)


def test_fixed_budget_rejects_invalid_policy():
    policy = robustness_policy()
    bad = type(policy)(
        name=policy.name,
        delta_t_seconds=policy.delta_t_seconds,
        max_bins=policy.max_bins,
        frames_per_bin=7,
        fixed_num_frames=128,
        fixed_num_bins=16,
    )
    with pytest.raises(ValueError, match="fixed_num_frames"):
        fixed_budget_bin_plan(0.0, 1.0, 30.0, bad)
