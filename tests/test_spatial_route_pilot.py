from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_spatial_route_pilot import analyze
from scripts.create_spatial_route_pilot import manifest_rows
from src.experiment1.route_reuse import LayerRouteMask
from src.experiment1.spatial import (
    build_spatial_relevance_from_token_scores,
    spatial_route_spec_from_baseline_artifact,
    validate_spatial_route_spec,
)
from src.experiment1.token_layout import TokenLayout, VisualTokenCell


def _layout(frames: int = 4, h: int = 2, w: int = 2) -> TokenLayout:
    cells = []
    visual_index = 0
    for temporal in range(frames):
        for y in range(h):
            for x in range(w):
                cells.append(
                    VisualTokenCell(
                        token_index=visual_index + 1,
                        visual_index=visual_index,
                        modality="video",
                        input_index=0,
                        temporal_index=temporal,
                        spatial_y=y,
                        spatial_x=x,
                        grid_t=frames,
                        grid_h=h,
                        grid_w=w,
                    )
                )
                visual_index += 1
    return TokenLayout(
        question_token_indices=(30, 31),
        prompt_token_indices=tuple(range(32)),
        visual_token_indices=tuple(range(1, visual_index + 1)),
        visual_cells=tuple(cells),
        visual_grid_metadata={"video_grid_thw": [[frames, h, w]], "spatial_merge_size": 1},
        query_scope="question",
    )


def _scores(layers: int = 28, tokens: int = 16) -> np.ndarray:
    scores = np.ones((layers, tokens), dtype=np.float64)
    for layer in range(layers):
        scores[layer, layer % tokens] = 10.0
        scores[layer, (layer + 3) % tokens] = 7.0
    return scores


def _artifact(qid: str = "q00", index: int = 0) -> dict:
    layout = _layout()
    scores = _scores(tokens=len(layout.visual_token_indices))
    spatial = build_spatial_relevance_from_token_scores(scores, layout, "unit")
    return {
        "question_id": qid,
        "model_backend": "qwen",
        "model_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "question": f"Question {qid}?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
        "predicted_idx": index % 5,
        "correct": index % 3 == 0,
        "video_clip": [{"video_id": f"P{index % 4:02d}-video-{index}", "participant_id": f"P{index % 4:02d}"}],
        "sampled_frame_indices": [list(range(8))],
        "sampled_timestamps": [[float(value) for value in range(8)]],
        "frame_bin_mappings": [[{"sample_position": value, "analysis_bin": value} for value in range(8)]],
        "sampling_metadata": [{"mode": "cross_model_8"}],
        "token_layout": {
            "visual_token_indices": list(layout.visual_token_indices),
            "visual_token_cells": [
                {
                    "token_index": cell.token_index,
                    "visual_index": cell.visual_index,
                    "modality": cell.modality,
                    "input_index": cell.input_index,
                    "temporal_bin": cell.temporal_index,
                    "spatial_row": cell.spatial_y,
                    "spatial_col": cell.spatial_x,
                    "grid_t": cell.grid_t,
                    "grid_h": cell.grid_h,
                    "grid_w": cell.grid_w,
                }
                for cell in layout.visual_cells
            ],
        },
        "spatial_relevance": spatial,
        "answer_choice_scores": {
            "correct_choice_log_probability": -2.0 - index * 0.01,
            "correct_vs_best_incorrect_margin": -0.4 + index * 0.01,
        },
    }


def _write_run(path: Path, artifacts: dict[str, dict]) -> None:
    path.mkdir(parents=True)
    records = []
    for qid, artifact in artifacts.items():
        artifact_path = path / f"{qid}.json"
        artifact_path.write_text(json.dumps(artifact))
        records.append({"question_id": qid, "status": "complete", "artifact": artifact_path.name})
    (path / "records.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")


def _write_manifest(path: Path, question_ids: list[str]) -> None:
    rows = [
        {
            "question_id": qid,
            "source_video_id": f"P{index % 4:02d}-video-{index}",
            "participant_id": f"P{index % 4:02d}",
            "category": "gaze",
            "question_type": "gaze_synthetic",
        }
        for index, qid in enumerate(question_ids)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _condition_artifact(base: dict, route: dict, delta: float) -> dict:
    artifact = dict(base)
    artifact["spatial_route"] = route
    artifact["metadata"] = {"spatial_route": route}
    artifact["run_config"] = {"git_commit": "exec"}
    artifact["answer_choice_scores"] = {"correct_choice_log_probability": 99.0, "correct_vs_best_incorrect_margin": 99.0}
    artifact["intervention_answer_choice_scores"] = {
        "correct_choice_log_probability": base["answer_choice_scores"]["correct_choice_log_probability"] + delta,
        "correct_vs_best_incorrect_margin": base["answer_choice_scores"]["correct_vs_best_incorrect_margin"] + delta,
    }
    return artifact


def test_spatial_score_serialization_and_per_frame_normalization():
    layout = _layout()
    relevance = build_spatial_relevance_from_token_scores(_scores(tokens=16), layout, "unit")
    assert relevance["schema_version"] == "spatial_relevance_v1"
    assert len(relevance["visual_token_mappings"]) == 16
    assert len(relevance["normalized_token_scores"]) == 28
    for layer_frames in relevance["normalized_spatial_distribution_by_temporal_frame"]:
        assert len(layer_frames) == 4
        for distribution in layer_frames:
            assert np.isclose(sum(distribution), 1.0)
    np.testing.assert_allclose(
        relevance["normalized_global_spatial_distribution"],
        relevance["normalized_token_scores"],
    )


def test_spatial_routes_have_exact_per_frame_budgets_and_preserve_frames():
    artifact = _artifact()
    specs = {
        condition: spatial_route_spec_from_baseline_artifact(
            artifact,
            condition=condition,
            baseline_artifact="baseline/q00.json",
            seed=123,
            git_commit="manifest",
        )
        for condition in ("spatial_route_gap4_top50", "spatial_random_gap4_top50", "spatial_uniform_gap4_top50")
    }
    signatures = set()
    for spec in specs.values():
        validate_spatial_route_spec(spec)
        assert spec["temporal_frames_preserved"] == [0, 1, 2, 3]
        assert set(spec["per_frame_retained_visual_tokens"].values()) == {2}
        for route in spec["layer_routes"].values():
            assert route["actual_retained_visual_token_fraction"] == 0.5
            for tokens in route["selected_visual_tokens_by_temporal_frame"].values():
                assert len(tokens) == 2
        signatures.add(json.dumps({
            "frames": spec["temporal_frames_preserved"],
            "budgets": spec["per_frame_retained_visual_tokens"],
            "layers": sorted(spec["layer_routes"]),
        }, sort_keys=True))
    assert len(signatures) == 1


def test_spatial_random_is_deterministic_and_uniform_has_spatial_coverage():
    artifact = _artifact()
    left = spatial_route_spec_from_baseline_artifact(artifact, condition="spatial_random_gap4_top50", baseline_artifact="b", seed=7)
    right = spatial_route_spec_from_baseline_artifact(artifact, condition="spatial_random_gap4_top50", baseline_artifact="b", seed=7)
    assert left["anchor_routes"] == right["anchor_routes"]
    uniform = spatial_route_spec_from_baseline_artifact(artifact, condition="spatial_uniform_gap4_top50", baseline_artifact="b", seed=7)
    for route in uniform["anchor_routes"].values():
        for tokens in route["selected_visual_tokens_by_temporal_frame"].values():
            assert len(tokens) == 2
            # 2x2 grid checkerboard selection should not choose adjacent row-major tokens.
            assert abs(tokens[0] - tokens[1]) > 1


def test_spatial_route_causality_and_mask_preserves_text_and_visual_attention():
    torch = pytest.importorskip("torch")
    artifact = _artifact()
    spec = spatial_route_spec_from_baseline_artifact(artifact, condition="spatial_route_gap4_top50", baseline_artifact="b", seed=7)
    assert spec["anchor_layers"] == [8, 12, 16, 20, 24]
    assert sorted(int(layer) for layer in spec["layer_routes"]) == [9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27]
    assert all(int(route["source_anchor_layer"]) < int(layer) for layer, route in spec["layer_routes"].items())

    from src.experiment1 import qwen_reduced_attention as reduced
    from src.experiment1.qwen_reduced_attention import _route_reuse_block_mask

    class Module:
        layer_idx = 9

    route_mask = LayerRouteMask.from_route_spec(spec, None, tuple(range(1, 17)))
    previous = reduced._ACTIVE_ROUTE_REUSE_MASK
    reduced._ACTIVE_ROUTE_REUSE_MASK = route_mask
    try:
        query = torch.zeros((1, 1, 2, 4))
        key_states = torch.zeros((1, 1, 20, 4))
        mask = _route_reuse_block_mask(Module(), query, key_states)
        visual_query = torch.zeros((1, 1, 2, 4))
        visual_key_states = torch.zeros((1, 1, 16, 4))
        visual_mask = _route_reuse_block_mask(Module(), visual_query, visual_key_states)
    finally:
        reduced._ACTIVE_ROUTE_REUSE_MASK = previous

    assert mask is not None
    blocked = torch.finfo(mask.dtype).min
    route = spec["layer_routes"]["9"]
    text_rows = mask[:, :, :, :]
    assert torch.all(text_rows[:, :, :, route["blocked_visual_token_indices"]] == blocked)
    assert torch.all(text_rows[:, :, :, route["allowed_visual_token_indices"]] == 0)
    assert torch.all(text_rows[:, :, :, [0, 17, 18, 19]] == 0)
    assert visual_mask is None


def test_create_spatial_manifest_and_analyzer_validate_ids_and_budgets(tmp_path: Path):
    qids = [f"q{index:02d}" for index in range(15)]
    baseline = {qid: _artifact(qid, index) for index, qid in enumerate(qids)}
    baseline_dir = tmp_path / "baseline"
    _write_run(baseline_dir, baseline)
    manifest = tmp_path / "dev.jsonl"
    _write_manifest(manifest, qids)
    dev_records = [json.loads(line) for line in manifest.read_text().splitlines()]

    routes = {
        "adaptive": manifest_rows(baseline_dir=baseline_dir, dev_records=dev_records, condition="spatial_route_gap4_top50", seed=3, git_commit="manifest"),
        "random": manifest_rows(baseline_dir=baseline_dir, dev_records=dev_records, condition="spatial_random_gap4_top50", seed=3, git_commit="manifest"),
        "uniform": manifest_rows(baseline_dir=baseline_dir, dev_records=dev_records, condition="spatial_uniform_gap4_top50", seed=3, git_commit="manifest"),
    }
    dirs = {}
    deltas = {"adaptive": 0.3, "random": 0.1, "uniform": 0.0}
    for label, rows in routes.items():
        path = tmp_path / label
        artifacts = {
            row["question_id"]: _condition_artifact(baseline[row["question_id"]], row["spatial_route"], deltas[label])
            for row in rows
        }
        _write_run(path, artifacts)
        dirs[label] = path

    summary = analyze(
        baseline_dir=baseline_dir,
        condition_dirs=dirs,
        dev_manifest=manifest,
        output_dir=tmp_path / "analysis",
        bootstrap_samples=10,
        seed=5,
    )
    assert summary["metrics"]["development_gate"]["status"] == "PROMISING"
    assert summary["validation"]["conditions"]["adaptive"]["num_examples"] == 15

    (dirs["random"] / "q14.json").unlink()
    records = [json.loads(line) for line in (dirs["random"] / "records.jsonl").read_text().splitlines()]
    records = [row for row in records if row["question_id"] != "q14"]
    (dirs["random"] / "records.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(RuntimeError, match="expected exactly dev IDs"):
        analyze(
            baseline_dir=baseline_dir,
            condition_dirs=dirs,
            dev_manifest=manifest,
            output_dir=tmp_path / "analysis2",
            bootstrap_samples=10,
            seed=5,
        )
