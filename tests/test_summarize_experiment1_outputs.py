import json

from scripts.summarize_experiment1_outputs import accuracy_summary, layer_fusion_summary, load_artifacts


def test_summarize_experiment1_outputs_loads_artifacts_and_layer_stats(tmp_path):
    artifact = {
        "question_id": "q1",
        "category": "gaze",
        "relevance": {
            "absolute_visual_mass_by_layer": [0.2, 0.4],
            "normalized_frame_scores": [[0.7, 0.3], [0.4, 0.6]],
            "normalized_frame_scores_by_input": [
                [[0.45, 0.05], [0.5, 0.0]],
                [[0.9, 0.0], [0.1, 0.0]],
            ],
            "concentration_by_layer": [
                {"normalized_entropy": 0.8},
                {"normalized_entropy": 0.9},
            ],
        },
    }
    artifact_path = tmp_path / "q1.json"
    artifact_path.write_text(json.dumps(artifact))
    records = [
        {
            "question_id": "q1",
            "category": "gaze",
            "status": "complete",
            "artifact": str(artifact_path),
            "correct": True,
            "num_visual_tokens": 4,
        },
        {
            "question_id": "q2",
            "category": "ingredient",
            "status": "failed",
            "error": "missing video",
        },
    ]

    assert load_artifacts(records) == [artifact]
    accuracy = accuracy_summary(records)
    assert accuracy["completed"] == 1
    assert accuracy["failed"] == 1
    assert accuracy["status_counts"] == {"complete": 1, "failed": 1}
    assert accuracy["accuracy"] == 1.0
    fusion = layer_fusion_summary([artifact])
    assert fusion["num_layers"] == 2
    assert fusion["peak_overall_top1_layer"]["layer"] == 1
    assert fusion["lowest_overall_entropy_layer"]["layer"] == 1
    assert fusion["overall"][1]["mean_absolute_visual_mass"] == 0.4
