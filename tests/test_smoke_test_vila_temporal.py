from types import SimpleNamespace

import pytest

from scripts import smoke_test_vila_temporal


def _args(tmp_path):
    return SimpleNamespace(
        questions_dir="questions",
        mp4_dir="mp4",
        manifest="manifest.jsonl",
        output_root=str(tmp_path),
        checkpoint="Efficient-Large-Model/Llama-3-VILA1.5-8B",
        resolution_config="medium",
        max_new_tokens=16,
    )


def test_vila_smoke_raises_recorded_runner_failure(tmp_path, monkeypatch):
    def fake_run(cmd, check):
        output_dir = tmp_path / "bins8"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "records.jsonl").write_text(
            '{"question_id":"gaze_interaction_anticipation_537","status":"failed","error":"real VILA tokenizer error"}\n'
        )

    monkeypatch.setattr(smoke_test_vila_temporal.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="real VILA tokenizer error"):
        smoke_test_vila_temporal.run_example(
            _args(tmp_path),
            {
                "question_id": "gaze_interaction_anticipation_537",
            },
        )


def test_vila_smoke_uses_last_matching_record_for_resume(tmp_path, monkeypatch):
    def fake_run(cmd, check):
        output_dir = tmp_path / "bins8"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "records.jsonl").write_text(
            "\n".join(
                [
                    '{"question_id":"gaze_interaction_anticipation_537","status":"failed","error":"old failure"}',
                    '{"question_id":"gaze_interaction_anticipation_537","status":"complete","artifact":"ok.json"}',
                ]
            )
            + "\n"
        )

    monkeypatch.setattr(smoke_test_vila_temporal.subprocess, "run", fake_run)

    path = smoke_test_vila_temporal.run_example(
        _args(tmp_path),
        {
            "question_id": "gaze_interaction_anticipation_537",
        },
    )

    assert path == tmp_path / "bins8" / "gaze_interaction_anticipation_537.json"
