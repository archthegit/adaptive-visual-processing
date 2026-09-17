from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.create_route_reuse_pilot_manifests import manifest_rows_for_model
from src.experiment1.route_reuse import (
    assert_matched_condition_budgets,
    native_routing_units_from_artifact,
    route_mask_summary,
    route_spec_from_baseline_artifact,
)
from src.experiment1.qwen_execution import decoder_generation_requires_masked_context


def _scores(layers: int = 32) -> list[list[float]]:
    output = []
    for layer in range(layers):
        row = [0.02] * 8
        row[layer % 8] = 0.30
        row[(layer + 1) % 8] = 0.24
        row[(layer + 2) % 8] = 0.18
        row[(layer + 3) % 8] = 0.12
        total = sum(row)
        output.append([value / total for value in row])
    return output


def _qwen_artifact(question_id: str = "q1", layers: int = 28) -> dict:
    cells = []
    token = 0
    for native_t in range(4):
        for spatial in range(2):
            cells.append(
                {
                    "token_index": token,
                    "visual_index": token,
                    "modality": "video",
                    "temporal_bin": native_t,
                    "grid_t": 4,
                    "analysis_bin": None,
                }
            )
            token += 1
    return {
        "question_id": question_id,
        "temporal_relevance": {"normalized_temporal_bin_scores": _scores(layers)},
        "token_layout": {
            "visual_token_cells": cells,
            "visual_token_indices": list(range(8)),
        },
        "sampled_frame_indices": [tuple(range(8))],
        "frame_bin_mappings": [
            [
                {"sample_position": index, "analysis_bin": index}
                for index in range(8)
            ]
        ],
    }


def _vila_artifact(question_id: str = "q1", layers: int = 32) -> dict:
    return {
        "question_id": question_id,
        "temporal_relevance": {"normalized_temporal_bin_scores": _scores(layers)},
        "token_layout": {
            "visual_token_cells": [
                {
                    "token_index": index,
                    "visual_index": index,
                    "modality": "video",
                    "analysis_bin": index,
                }
                for index in range(8)
            ],
            "visual_token_indices": list(range(8)),
        },
        "sampled_frame_indices": [tuple(range(8))],
        "frame_bin_mappings": [
            [
                {"sample_position": index, "analysis_bin": index}
                for index in range(8)
            ]
        ],
    }


def test_qwen_native_units_can_span_two_analysis_bins():
    units = native_routing_units_from_artifact(_qwen_artifact(), "qwen")

    assert len(units) == 4
    assert [unit["analysis_bins"] for unit in units] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert all(unit["num_visual_tokens"] == 2 for unit in units)


def test_qwen_and_vila_layer_counts_use_frozen_anchor_schedule():
    qwen_spec = route_spec_from_baseline_artifact(
        _qwen_artifact(layers=28),
        model="qwen",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="qwen.json",
        seed=7,
        git_commit="abc",
    )
    vila_spec = route_spec_from_baseline_artifact(
        _vila_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )

    assert qwen_spec["anchor_layers"] == [8, 12, 16, 20, 24]
    assert qwen_spec["routing_unit_type"] == "qwen_native_temporal_cell"
    assert qwen_spec["retained_native_units"] == 2
    assert sorted(int(layer) for layer in qwen_spec["layer_routes"]) == list(range(9, 12)) + list(range(13, 16)) + list(range(17, 20)) + list(range(21, 24)) + list(range(25, 28))
    assert vila_spec["anchor_layers"] == [8, 12, 16, 20, 24, 28]
    assert vila_spec["routing_unit_type"] == "vila_frame_bin"
    assert vila_spec["retained_native_units"] == 4
    assert sorted(int(layer) for layer in vila_spec["layer_routes"] if int(layer) >= 29) == [29, 30, 31]


def test_qwen_gap2_and_gap3_anchor_schedules_are_exact():
    gap2 = route_spec_from_baseline_artifact(
        _qwen_artifact(layers=28),
        model="qwen",
        condition="route_reuse_gap2_top50",
        baseline_artifact="gap2.json",
        seed=7,
        git_commit="abc",
    )
    gap3 = route_spec_from_baseline_artifact(
        _qwen_artifact(layers=28),
        model="qwen",
        condition="route_reuse_gap3_top50",
        baseline_artifact="gap3.json",
        seed=7,
        git_commit="abc",
    )

    assert gap2["anchor_layers"] == [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    assert sorted(int(layer) for layer in gap2["layer_routes"]) == [9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    assert all(route["source_anchor_layer"] == int(layer) - 1 for layer, route in gap2["layer_routes"].items())
    assert gap3["anchor_layers"] == [8, 11, 14, 17, 20, 23, 26]
    assert sorted(int(layer) for layer in gap3["layer_routes"]) == [9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27]
    assert gap3["layer_routes"]["27"]["source_anchor_layer"] == 26


def test_gap2_gap3_gap4_differ_only_in_refresh_frequency():
    specs = {
        condition: route_spec_from_baseline_artifact(
            _qwen_artifact(layers=28),
            model="qwen",
            condition=condition,
            baseline_artifact=f"{condition}.json",
            seed=7,
            git_commit="abc",
        )
        for condition in ("route_reuse_gap2_top50", "route_reuse_gap3_top50", "route_reuse_gap4_top50")
    }

    for spec in specs.values():
        assert spec["retention_ratio"] == 0.5
        assert spec["retained_native_units"] == 2
        assert spec["routing_unit_type"] == "qwen_native_temporal_cell"
        assert {route["actual_retained_visual_token_fraction"] for route in spec["anchor_routes"].values()} == {0.5}
    assert specs["route_reuse_gap2_top50"]["refresh_gap"] == 2
    assert specs["route_reuse_gap3_top50"]["refresh_gap"] == 3
    assert specs["route_reuse_gap4_top50"]["refresh_gap"] == 4


def test_routed_layers_use_preceding_anchor_selection_and_dense_layers_are_absent():
    spec = route_spec_from_baseline_artifact(
        _vila_artifact(layers=32),
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
        expected = routed[anchor + 1]["selected_native_unit_ids"]
        for layer in range(anchor + 1, min(anchor + 4, 32)):
            assert routed[layer]["source_anchor_layer"] == anchor
            assert routed[layer]["selected_native_unit_ids"] == expected


def test_uniform_random_top_conditions_have_equal_native_token_budgets():
    specs = [
        route_spec_from_baseline_artifact(
            _qwen_artifact(layers=28),
            model="qwen",
            condition=condition,
            baseline_artifact=f"{condition}.json",
            seed=7,
            git_commit="abc",
        )
        for condition in ("route_reuse_gap4_top50", "random_reuse_gap4_top50", "uniform_reuse_gap4_top50")
    ]

    assert_matched_condition_budgets(specs)
    fractions = {
        route["actual_retained_visual_token_fraction"]
        for spec in specs
        for route in spec["anchor_routes"].values()
    }
    assert fractions == {0.5}


def test_route_mask_summary_records_native_units_and_token_counts():
    spec = route_spec_from_baseline_artifact(
        _vila_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="base.json",
        seed=7,
        git_commit="abc",
    )
    summary = route_mask_summary(spec)

    route = summary["layer_routes"]["9"]
    assert route["selected_native_unit_ids"] == [0, 2, 5, 7]
    assert route["num_allowed_visual_tokens"] == 4
    assert route["num_blocked_visual_tokens"] == 4
    assert route["actual_retained_visual_token_fraction"] == 0.5


def test_route_mask_preserves_text_text_and_visual_visual_attention():
    torch = pytest.importorskip("torch")
    from src.experiment1 import qwen_reduced_attention as reduced
    from src.experiment1.qwen_reduced_attention import _route_reuse_block_mask

    class Module:
        layer_idx = 9

    spec = route_spec_from_baseline_artifact(
        _vila_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )
    route_mask = reduced.LayerRouteMask.from_route_spec(spec, None, tuple(range(8)))
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
    text_rows = mask[:, :, [8, 9], :]
    assert torch.all(text_rows[:, :, :, [1, 3, 4, 6]] == blocked)
    assert torch.all(text_rows[:, :, :, [8, 9]] == 0)
    assert torch.all(mask[:, :, list(range(8)), :] == 0)


def test_qwen_generation_route_only_requires_masked_context():
    assert decoder_generation_requires_masked_context("none", (), {"layer_routes": {"9": {}}})
    assert decoder_generation_requires_masked_context(None, (1,), None)
    assert decoder_generation_requires_masked_context(8, (), None)
    assert not decoder_generation_requires_masked_context("none", (), None)


def test_qwen_generation_q_len_one_with_kv_cache_routes_text_row():
    torch = pytest.importorskip("torch")
    from src.experiment1 import qwen_reduced_attention as reduced
    from src.experiment1.qwen_reduced_attention import _route_reuse_block_mask

    class Module:
        layer_idx = 9

    spec = route_spec_from_baseline_artifact(
        _vila_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )
    old = reduced._ACTIVE_ROUTE_REUSE_MASK
    reduced._ACTIVE_ROUTE_REUSE_MASK = reduced.LayerRouteMask.from_route_spec(spec, None, tuple(range(8)))
    try:
        query = torch.zeros(1, 1, 1, 4)
        key_states = torch.zeros(1, 1, 11, 4)
        mask = _route_reuse_block_mask(Module(), query, key_states)
    finally:
        reduced._ACTIVE_ROUTE_REUSE_MASK = old
    assert mask is not None
    assert mask.shape == (1, 1, 1, 11)
    assert torch.all(mask[:, :, :, [1, 3, 4, 6]] == torch.finfo(query.dtype).min)


def test_vila_prefill_and_generation_masks_preserve_causality():
    torch = pytest.importorskip("torch")
    from src.experiment1 import vila_execution as vila

    spec = route_spec_from_baseline_artifact(
        _vila_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )
    old_mask = vila._ACTIVE_VILA_ROUTE_MASK
    old_layer = vila._ACTIVE_VILA_LAYER
    vila._ACTIVE_VILA_ROUTE_MASK = vila.LayerRouteMask.from_route_spec(spec, None, tuple(range(8)))
    vila._ACTIVE_VILA_LAYER = 9
    try:
        query = torch.zeros(1, 1, 2, 4)
        key = torch.zeros(1, 1, 10, 4)
        mask, is_causal = vila._combine_vila_attention_mask(None, query, key, True)
        assert mask is not None
        assert is_causal is False
        assert mask.shape == (1, 1, 2, 10)
        assert torch.all(mask[:, :, :, [1, 3, 4, 6]] == torch.finfo(query.dtype).min)
        query_gen = torch.zeros(1, 1, 1, 4)
        key_gen = torch.zeros(1, 1, 11, 4)
        gen_mask, gen_is_causal = vila._combine_vila_attention_mask(None, query_gen, key_gen, True)
        assert gen_mask is not None
        assert gen_is_causal is False
        assert gen_mask.shape == (1, 1, 1, 11)
        assert torch.all(gen_mask[:, :, :, [1, 3, 4, 6]] == torch.finfo(query.dtype).min)
    finally:
        vila._ACTIVE_VILA_ROUTE_MASK = old_mask
        vila._ACTIVE_VILA_LAYER = old_layer


def test_full_retention_is_numerically_equivalent_to_dense_for_both_backends():
    torch = pytest.importorskip("torch")
    from src.experiment1 import qwen_reduced_attention as reduced
    from src.experiment1 import vila_execution as vila

    class Module:
        num_key_value_groups = 1
        training = False
        layer_idx = 9

    query = torch.tensor([[[[1.0, 0.5], [0.5, 1.0]]]])
    key = torch.eye(4, 2).reshape(1, 1, 4, 2)
    value = torch.eye(4).reshape(1, 1, 4, 4)
    baseline, _ = reduced.qwen_relevance_masked_eager_forward(Module(), query, key, value, None, scaling=1.0)
    old_qwen = reduced._ACTIVE_ROUTE_REUSE_MASK
    full_spec = {
        "layer_routes": {
            "9": {
                "blocked_visual_token_indices": [],
                "allowed_visual_token_indices": [0, 1, 2, 3],
            }
        }
    }
    reduced._ACTIVE_ROUTE_REUSE_MASK = reduced.LayerRouteMask.from_route_spec(full_spec, None, (0, 1, 2, 3))
    try:
        routed, _ = reduced.qwen_relevance_masked_eager_forward(Module(), query, key, value, None, scaling=1.0)
    finally:
        reduced._ACTIVE_ROUTE_REUSE_MASK = old_qwen
    torch.testing.assert_close(routed, baseline)

    old_vila_mask = vila._ACTIVE_VILA_ROUTE_MASK
    old_vila_layer = vila._ACTIVE_VILA_LAYER
    vila._ACTIVE_VILA_ROUTE_MASK = vila.LayerRouteMask.from_route_spec(full_spec, None, (0, 1, 2, 3))
    vila._ACTIVE_VILA_LAYER = 9
    try:
        mask, is_causal = vila._combine_vila_attention_mask(None, query, key, False)
    finally:
        vila._ACTIVE_VILA_ROUTE_MASK = old_vila_mask
        vila._ACTIVE_VILA_LAYER = old_vila_layer
    assert mask is None
    assert is_causal is False


def test_fifty_percent_retention_removes_half_video_key_columns():
    qwen_spec = route_spec_from_baseline_artifact(
        _qwen_artifact(layers=28),
        model="qwen",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="qwen.json",
        seed=7,
        git_commit="abc",
    )
    vila_spec = route_spec_from_baseline_artifact(
        _vila_artifact(layers=32),
        model="vila",
        condition="uniform_reuse_gap4_top50",
        baseline_artifact="vila.json",
        seed=7,
        git_commit="abc",
    )

    assert qwen_spec["anchor_routes"]["8"]["actual_retained_visual_token_fraction"] == 0.5
    assert vila_spec["anchor_routes"]["8"]["actual_retained_visual_token_fraction"] == 0.5


def test_manifest_generator_is_dev_only_and_records_native_routes(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    artifact_path = baseline_dir / "q-dev.json"
    artifact_path.write_text(json.dumps(_qwen_artifact("q-dev", layers=28)))
    test_artifact_path = baseline_dir / "q-test.json"
    test_artifact_path.write_text(json.dumps(_qwen_artifact("q-test", layers=28)))
    (baseline_dir / "records.jsonl").write_text(
        json.dumps({"question_id": "q-dev", "status": "complete", "artifact": str(artifact_path)}) + "\n"
        + json.dumps({"question_id": "q-test", "status": "complete", "artifact": str(test_artifact_path)}) + "\n"
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
    assert rows[0]["route_reuse"]["type"] == "baseline_derived_causal_route_replay"
    assert rows[0]["route_reuse"]["native_routing_units"]
