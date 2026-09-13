from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_route_reuse_pilot_manifests import manifest_rows_for_model
from src.experiment1.route_reuse import (
    route_layers_from_anchor_bins,
    route_mask_summary,
    route_spec_from_baseline_artifact,
)
from src.experiment1.token_layout import TokenLayout, VisualTokenCell


def _baseline_artifact(question_id: str = "q1", layers: int = 32) -> dict:
    scores = []
    for layer in range(layers):
        row = [0.02] * 8
        row[layer % 8] = 0.30
        row[(layer + 1) % 8] = 0.24
        row[(layer + 2) % 8] = 0.18
        row[(layer + 3) % 8] = 0.12
        total = sum(row)
        scores.append([value / total for value in row])
    return {
        "question_id": question_id,
        "temporal_relevance": {"normalized_temporal_bin_scores": scores},
    }


def _layout() -> TokenLayout:
    cells = tuple(
        VisualTokenCell(
            token_index=index,
            visual_index=index,
            modality="video",
            input_index=0,
            temporal_index=index,
            spatial_y=0,
            spatial_x=0,
            grid_t=8,
            grid_h=1,
            grid_w=1,
        )
        for index in range(8)
    )
    return TokenLayout(
        question_token_indices=(8, 9),
        prompt_token_indices=tuple(range(10)),
        visual_token_indices=tuple(range(8)),
        visual_cells=cells,
        visual_grid_metadata={},
        query_scope="question",
    )


def test_qwen_and_vila_layer_counts_use_frozen_anchor_schedule():
    qwen_spec = route_spec_from_baseline_artifact(
        _baseline_artifact(layers=28),
        model="qwen",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="qwen.json",
        seed=7,
        git_commit="abc",
    )
    vila_spec = route_spec_from_baseline_artifact(
        _baseline_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )

    assert qwen_spec["anchor_layers"] == [8, 12, 16, 20, 24]
    assert sorted(int(layer) for layer in qwen_spec["layer_routes"]) == list(range(9, 12)) + list(range(13, 16)) + list(range(17, 20)) + list(range(21, 24)) + list(range(25, 28))
    assert vila_spec["anchor_layers"] == [8, 12, 16, 20, 24, 28]
    assert sorted(int(layer) for layer in vila_spec["layer_routes"] if int(layer) >= 29) == [29, 30, 31]
    assert all(len(route["selected_bins"]) == 4 for route in vila_spec["layer_routes"].values())


def test_routed_layers_use_preceding_anchor_selection_and_dense_layers_are_absent():
    spec = route_spec_from_baseline_artifact(
        _baseline_artifact(layers=32),
        model="vila",
        condition="route_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )

    routed = {int(layer): route for layer, route in spec["layer_routes"].items()}
    assert all(layer not in routed for layer in range(0, 9))
    assert all(anchor not in routed for anchor in spec["anchor_layers"])
    for anchor in spec["anchor_layers"]:
        expected = routed[anchor + 1]["selected_bins"]
        for layer in range(anchor + 1, min(anchor + 4, 32)):
            assert routed[layer]["source_anchor_layer"] == anchor
            assert routed[layer]["selected_bins"] == expected


def test_random_and_uniform_controls_have_identical_budgets():
    random_spec = route_spec_from_baseline_artifact(
        _baseline_artifact(layers=32),
        model="vila",
        condition="random_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )
    uniform_spec = route_spec_from_baseline_artifact(
        _baseline_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )

    assert all(len(route["selected_bins"]) == 4 for route in random_spec["layer_routes"].values())
    assert all(route["selected_bins"] == [0, 2, 5, 7] for route in uniform_spec["layer_routes"].values())
    assert {
        layer: len(route["omitted_bins"])
        for layer, route in random_spec["layer_routes"].items()
    } == {
        layer: len(route["omitted_bins"])
        for layer, route in uniform_spec["layer_routes"].items()
    }


def test_qwen_route_mask_preserves_text_text_and_visual_visual_attention():
    torch = pytest.importorskip("torch")
    from src.experiment1 import qwen_reduced_attention as reduced
    from src.experiment1.qwen_reduced_attention import _route_reuse_block_mask

    class Module:
        layer_idx = 9

    spec = {
        "layer_routes": {
            "9": {
                "source_anchor_layer": 8,
                "selected_bins": [0, 1, 2, 4],
                "omitted_bins": [3, 5, 6, 7],
            }
        }
    }
    layout = _layout()
    route_mask = reduced.LayerRouteMask.from_route_spec(
        spec,
        {index: (index,) for index in range(8)},
        layout.visual_token_indices,
    )
    old = reduced._ACTIVE_ROUTE_REUSE_MASK
    reduced._ACTIVE_ROUTE_REUSE_MASK = route_mask
    try:
        query = torch.zeros(1, 1, 10, 4)
        key_states = torch.zeros(1, 1, 10, 4)
        mask = _route_reuse_block_mask(Module(), query, key_states)
    finally:
        reduced._ACTIVE_ROUTE_REUSE_MASK = old

    assert mask is not None
    blocked = torch.finfo(query.dtype).min
    text_rows = [8, 9]
    visual_rows = list(range(8))
    assert torch.all(mask[:, :, text_rows, 3] == blocked)
    assert torch.all(mask[:, :, text_rows, [8, 9]] == 0)
    assert torch.all(mask[:, :, visual_rows, :] == 0)


def test_full_retention_route_is_numerically_equivalent_to_baseline():
    torch = pytest.importorskip("torch")
    from src.experiment1 import qwen_reduced_attention as reduced

    class Module:
        num_key_value_groups = 1
        training = False
        layer_idx = 9

    query = torch.tensor([[[[1.0, 0.5], [0.5, 1.0]]]])
    key = torch.eye(4, 2).reshape(1, 1, 4, 2)
    value = torch.eye(4).reshape(1, 1, 4, 4)
    baseline, _ = reduced.qwen_relevance_masked_eager_forward(Module(), query, key, value, None, scaling=1.0)
    old = reduced._ACTIVE_ROUTE_REUSE_MASK
    reduced._ACTIVE_ROUTE_REUSE_MASK = reduced.LayerRouteMask.from_route_spec(
        {"layer_routes": {"9": {"source_anchor_layer": 8, "selected_bins": list(range(8)), "omitted_bins": []}}},
        {index: () for index in range(8)},
        (0, 1),
    )
    try:
        routed, _ = reduced.qwen_relevance_masked_eager_forward(Module(), query, key, value, None, scaling=1.0)
    finally:
        reduced._ACTIVE_ROUTE_REUSE_MASK = old
    torch.testing.assert_close(routed, baseline)


def test_manifest_generator_is_dev_only(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    artifact_path = baseline_dir / "q-dev.json"
    artifact_path.write_text(json.dumps(_baseline_artifact("q-dev", layers=28)))
    (baseline_dir / "records.jsonl").write_text(
        json.dumps({"question_id": "q-dev", "status": "complete", "artifact": str(artifact_path)}) + "\n"
        + json.dumps({"question_id": "q-test", "status": "complete", "artifact": str(artifact_path)}) + "\n"
    )
    dev_records = [
        {
            "question_id": "q-dev",
            "category": "fine_grained",
            "question_type": "fine_grained_action_recognition",
            "source_video_id": "P01-video",
        }
    ]

    rows = manifest_rows_for_model(
        model="qwen",
        baseline_dir=baseline_dir,
        dev_records=dev_records,
        condition="uniform_reuse_gap4_top50",
        seed=7,
        git_commit="abc",
    )

    assert [row["question_id"] for row in rows] == ["q-dev"]
    assert rows[0]["route_reuse"]["condition"] == "uniform_reuse_gap4_top50"


def test_route_mask_summary_counts_allowed_and_blocked_tokens():
    layer_routes = route_layers_from_anchor_bins({8: [0, 1, 2, 3]}, 12)
    spec = {
        "condition": "route_reuse_gap4_top50",
        "retention_ratio": 0.5,
        "anchor_layers": [8],
        "layer_routes": {str(layer): route for layer, route in layer_routes.items()},
        "baseline_artifact": "base.json",
        "seed": 7,
        "git_commit": "abc",
    }
    summary = route_mask_summary(spec, {index: (index * 10, index * 10 + 1) for index in range(8)})

    assert summary["layer_routes"]["9"]["num_allowed_visual_tokens"] == 8
    assert summary["layer_routes"]["9"]["num_blocked_visual_tokens"] == 8
