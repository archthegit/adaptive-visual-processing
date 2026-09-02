import json

from src.experiment1.v2_reference_layer import (
    MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS,
    score_reference_layers,
    select_reference_layer,
    write_frozen_reference_layer,
)
from src.io import write_jsonl


def _artifact(path, temporal_scores, masses):
    path.write_text(
        json.dumps(
            {
                "question_id": path.stem,
                "temporal_relevance": {
                    "normalized_temporal_bin_scores": temporal_scores,
                    "absolute_question_to_visual_attention_mass": masses,
                },
            }
        )
    )


def _manifest(tmp_path):
    path = tmp_path / "primary.jsonl"
    write_jsonl(
        path,
        [
            {"question_id": "q1", "split": "dev"},
            {"question_id": "q2", "split": "test"},
        ],
    )
    return path


def test_reference_layer_selection_prioritizes_mismatch_separation(tmp_path):
    baseline = tmp_path / "baseline"
    mismatched = tmp_path / "mismatched"
    baseline.mkdir()
    mismatched.mkdir()
    _artifact(
        baseline / "q1.json",
        [[0.5, 0.5], [0.9, 0.1], [0.55, 0.45]],
        [0.1, 0.2, 0.9],
    )
    _artifact(
        mismatched / "q1.json",
        [[0.5, 0.5], [0.1, 0.9], [0.50, 0.50]],
        [0.1, 0.2, 0.9],
    )

    scores = score_reference_layers([{"question_id": "q1", "split": "dev"}], baseline, mismatched)
    selected = select_reference_layer(scores)

    assert selected.layer == 1
    assert selected.mean_correct_mismatch_jsd > scores[0].mean_correct_mismatch_jsd
    assert all(score.passes_absolute_visual_mass_threshold for score in scores)


def test_reference_layer_selection_excludes_low_absolute_visual_mass_layers():
    from src.experiment1.v2_reference_layer import ReferenceLayerScore

    scores = [
        ReferenceLayerScore(0, 3, 0.9, 0.01, 0.9, False),
        ReferenceLayerScore(1, 3, 0.2, 0.10, 0.7, True),
    ]

    selected = select_reference_layer(scores, min_absolute_visual_mass=0.05)

    assert selected.layer == 1


def test_reference_layer_selection_ties_end_at_shallower_layer():
    from src.experiment1.v2_reference_layer import ReferenceLayerScore

    scores = [
        ReferenceLayerScore(5, 3, 0.3, 0.10, 0.7, True),
        ReferenceLayerScore(2, 3, 0.3, 0.10, 0.7, True),
    ]

    selected = select_reference_layer(scores, min_absolute_visual_mass=0.05)

    assert selected.layer == 2


def test_write_frozen_reference_layer_uses_dev_records_only(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    manifest = _manifest(tmp_path)
    _artifact(baseline / "q1.json", [[0.5, 0.5], [0.8, 0.2]], [0.1, 0.2])
    _artifact(baseline / "q2.json", [[0.1, 0.9], [0.1, 0.9]], [10.0, 10.0])

    output = tmp_path / "frozen.json"
    payload = write_frozen_reference_layer(manifest, baseline, output)

    assert output.exists()
    assert payload["selected_layer"] == 1
    assert payload["minimum_absolute_visual_mass_threshold"] == MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS
    assert payload["selected"]["num_examples"] == 1
    assert payload["annotation_alignment_available"] is False
