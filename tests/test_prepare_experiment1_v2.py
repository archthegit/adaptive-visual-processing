from scripts.prepare_experiment1_v2 import qwen_run_command


def test_qwen_run_command_exposes_sampling_mode_condition_and_max_tokens():
    command = qwen_run_command(
        "primary.jsonl",
        "runs/baseline_fixed_budget",
        "/questions",
        "/mp4",
        "baseline_fixed_budget",
        sampling_mode="fixed_budget",
        max_new_tokens=32,
    )

    assert "--sampling-mode fixed_budget" in command
    assert "--condition baseline_fixed_budget" in command
    assert "--max-new-tokens 32" in command
    assert "--allow-7b-inference" in command


def test_realtime_qwen_run_command_includes_frozen_sampling_policy():
    command = qwen_run_command(
        "primary.jsonl",
        "runs/baseline",
        "/questions",
        "/mp4",
        "baseline",
        sampling_policy_json="outputs/experiment1_v2/split_summary.json",
    )

    assert "--sampling-mode realtime" in command
    assert "--sampling-policy-json outputs/experiment1_v2/split_summary.json" in command


def test_fixed_budget_qwen_run_command_does_not_require_sampling_policy():
    command = qwen_run_command(
        "interventions/mask_top20_fixed_budget.jsonl",
        "runs/mask_top20_fixed_budget",
        "/questions",
        "/mp4",
        "mask_top20_fixed_budget",
        sampling_mode="fixed_budget",
    )

    assert "--sampling-mode fixed_budget" in command
    assert "--sampling-policy-json" not in command
