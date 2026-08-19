import json
from pathlib import Path

from scripts.create_temporal_intervention_manifest import main as intervention_main
from src.io import append_jsonl, write_json


def _artifact(question_id, scores):
    return {
        "question_id": question_id,
        "question_type": "gaze_interaction_anticipation",
        "category": "gaze",
        "question": "q?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
        "correct_answer": "A",
        "video_clip": [{"video_id": "v1"}],
        "temporal_relevance": {
            "by_layer": [
                {"normalized_temporal_bin_scores": [0.25, 0.25, 0.25, 0.25]},
                {"normalized_temporal_bin_scores": scores},
            ]
        },
    }


def test_create_temporal_intervention_manifest_selects_top_bins(tmp_path, monkeypatch):
    artifact_path = tmp_path / "q1.json"
    write_json(artifact_path, _artifact("q1", [0.1, 0.4, 0.2, 0.3]))
    append_jsonl(
        tmp_path / "records.jsonl",
        {"question_id": "q1", "status": "complete", "artifact": str(artifact_path)},
    )
    output = tmp_path / "intervention.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_temporal_intervention_manifest.py",
            "--baseline-output-dir",
            str(tmp_path),
            "--output-jsonl",
            str(output),
            "--strategy",
            "top",
            "--intervention-type",
            "decoder_direct_access",
            "--removal-fraction",
            "0.5",
            "--ranking-layer",
            "-1",
            "--seed",
            "7",
        ],
    )

    intervention_main()

    record = json.loads(output.read_text().strip())
    assert record["decoder_direct_access_mask_temporal_bins"] == [1, 3]
    assert "pre_encoder_mask_temporal_bins" not in record
    assert record["intervention"]["ranking_layer"] == 1
    assert record["intervention"]["removal_fraction"] == 0.5
    assert record["intervention"]["seed"] == 7


def test_create_temporal_intervention_manifest_random_is_seeded(tmp_path, monkeypatch):
    artifact_path = tmp_path / "q1.json"
    write_json(artifact_path, _artifact("q1", [0.1, 0.4, 0.2, 0.3]))
    append_jsonl(
        tmp_path / "records.jsonl",
        {"question_id": "q1", "status": "complete", "artifact": str(artifact_path)},
    )
    outputs = []
    for index in range(2):
        output = tmp_path / f"random_{index}.jsonl"
        monkeypatch.setattr(
            "sys.argv",
            [
                "create_temporal_intervention_manifest.py",
                "--baseline-output-dir",
                str(tmp_path),
                "--output-jsonl",
                str(output),
                "--strategy",
                "random",
                "--intervention-type",
                "pre_encoder",
                "--removal-fraction",
                "0.25",
                "--seed",
                "11",
            ],
        )
        intervention_main()
        outputs.append(json.loads(output.read_text().strip()))

    assert outputs[0]["pre_encoder_mask_temporal_bins"] == outputs[1]["pre_encoder_mask_temporal_bins"]
    assert outputs[0]["intervention"]["strategy"] == "random"
