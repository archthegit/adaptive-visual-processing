import sys

import pytest

from scripts.run_experiment1 import frames_per_input, parse_args


def test_run_experiment1_exposes_max_new_tokens(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment1.py",
            "--manifest",
            "manifest.jsonl",
            "--max-new-tokens",
            "24",
            "--attention-extraction",
            "reduced_sdpa",
        ],
    )
    args = parse_args()
    assert args.max_new_tokens == 24
    assert args.attention_extraction == "reduced_sdpa"
    assert args.frame_budget_mode == "total"


def test_frames_per_input_splits_total_budget_deterministically():
    assert frames_per_input(8, 2, "total") == [4, 4]
    assert frames_per_input(5, 2, "total") == [3, 2]
    assert frames_per_input(4, 1, "total") == [4]


def test_frames_per_input_supports_legacy_per_input_mode():
    assert frames_per_input(4, 2, "per-input") == [4, 4]


def test_frames_per_input_rejects_too_small_total_budget():
    with pytest.raises(ValueError, match="smaller than the 3 visual inputs"):
        frames_per_input(2, 3, "total")
