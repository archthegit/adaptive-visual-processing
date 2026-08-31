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
