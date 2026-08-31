import pytest

from src.experiment1.v2_sampling import (
    chronological_indices_without_duplicates_when_possible,
    fixed_budget_bin_plan,
    primary_policy_from_development_durations,
    real_time_bin_plan,
    repeated_frame_plan,
    reverse_presented_order,
    robustness_policy,
)


def test_primary_policy_uses_development_p95_for_delta_t():
    policy = primary_policy_from_development_durations([10, 20, 30, 640])
    assert policy.name == "real_time_p95_development"
    assert policy.delta_t_seconds == 9.0
    assert policy.frames_per_bin == 2
    assert policy.max_bins == 64


def test_real_time_bin_plan_samples_two_chronological_frames_per_bin():
    policy = primary_policy_from_development_durations([64, 64, 64])
    plan = real_time_bin_plan(0.0, 3.0, source_fps=10.0, policy=policy)
    assert len(plan) == 3
    assert [item["source_frame_indices"] for item in plan] == [[0, 9], [10, 19], [20, 29]]
    assert all(len(item["source_frame_indices"]) == 2 for item in plan)
    assert plan[0]["source_timestamps"] == [0.0, 0.9]


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


def test_reverse_and_repeated_frame_controls_preserve_positions():
    policy = primary_policy_from_development_durations([64, 64, 64])
    plan = real_time_bin_plan(0.0, 4.0, source_fps=10.0, policy=policy)
    reversed_plan = reverse_presented_order(plan)
    assert [item["analysis_bin"] for item in reversed_plan] == [3, 2, 1, 0]
    assert [item["presented_temporal_position"] for item in reversed_plan] == [0, 1, 2, 3]
    assert [item["reversal_maps_to_original_bin"] for item in reversed_plan] == [3, 2, 1, 0]
    repeated = repeated_frame_plan(plan, source_frame_index=7)
    assert [item["presented_temporal_position"] for item in repeated] == [0, 1, 2, 3]
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
