import sys

from scripts.run_experiment1 import parse_args


def test_run_experiment1_exposes_max_new_tokens(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_experiment1.py", "--manifest", "manifest.jsonl", "--max-new-tokens", "24"])
    args = parse_args()
    assert args.max_new_tokens == 24

