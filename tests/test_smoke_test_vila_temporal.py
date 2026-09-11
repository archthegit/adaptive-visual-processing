from types import SimpleNamespace

import pytest

from scripts import smoke_test_vila_temporal


def test_vila_smoke_raises_recorded_runner_failure(tmp_path, monkeypatch):
    def fake_run(cmd, check):
        output_dir = tmp_path / "bins8"
        output_dir.mkdir(parents=True)
        (output_dir / "records.jsonl").write_text(
            '{"question_id":"gaze_interaction_anticipation_537","status":"failed","error":"real VILA tokenizer error"}\n'
        )

    monkeypatch.setattr(smoke_test_vila_temporal.subprocess, "run", fake_run)
    args = SimpleNamespace(
        questions_dir="questions",
        mp4_dir="mp4",
        manifest="manifest.jsonl",
        output_root=str(tmp_path),
        checkpoint="Efficient-Large-Model/Llama-3-VILA1.5-8B",
        resolution_config="medium",
        max_new_tokens=16,
    )

    with pytest.raises(RuntimeError, match="real VILA tokenizer error"):
        smoke_test_vila_temporal.run_example(
            args,
            {
                "question_id": "gaze_interaction_anticipation_537",
            },
        )
