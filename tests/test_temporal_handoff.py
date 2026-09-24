from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.experiment1.temporal_handoff import (
    TemporalHandoffConfig,
    aggregate_analysis_scores_to_native_cells,
    build_additive_causal_mask,
    build_compaction_plan,
    compact_hidden_states_and_positions,
    condition_from_baseline_scores,
    qwen_decoder_stack,
    qwen_multimodal_decoder_inputs,
    run_custom_decoder_prefill,
    select_random_temporal_regions,
    select_top_temporal_regions,
    temporal_regions_from_layout,
)


torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class FakeCell:
    token_index: int
    visual_index: int
    temporal_index: int
    spatial_y: int
    spatial_x: int
    input_index: int = 0


def fake_layout(num_temporal_regions: int = 4, spatial_tokens_per_region: int = 3):
    cells = []
    token_index = 2
    visual_index = 0
    for temporal_index in range(num_temporal_regions):
        for spatial in range(spatial_tokens_per_region):
            cells.append(
                FakeCell(
                    token_index=token_index,
                    visual_index=visual_index,
                    temporal_index=temporal_index,
                    spatial_y=0,
                    spatial_x=spatial,
                )
            )
            token_index += 1
            visual_index += 1
    # Text tokens exist before and after the visual block; the last two are the question rows.
    return SimpleNamespace(
        visual_cells=tuple(cells),
        question_token_indices=(token_index, token_index + 1),
        visual_token_indices=tuple(cell.token_index for cell in cells),
    )


def native_mapping_artifact(
    analysis_to_frames: dict[int, list[int]],
    native_to_frames: dict[int, list[int]],
    expected_native_count: int | None = None,
) -> dict:
    frame_bin_mapping = []
    for analysis_bin, frames in sorted(analysis_to_frames.items()):
        for frame in frames:
            frame_bin_mapping.append(
                {
                    "analysis_bin": int(analysis_bin),
                    "source_frame_index": int(frame),
                }
            )
    cells = []
    visual_index = 0
    for native, frames in sorted(native_to_frames.items()):
        for spatial in range(2):
            cells.append(
                {
                    "token_index": 10 + visual_index,
                    "visual_index": visual_index,
                    "temporal_bin": int(native),
                    "sampled_frame_indices": [int(frame) for frame in frames],
                    "spatial_row": 0,
                    "spatial_col": spatial,
                }
            )
            visual_index += 1
    token_layout = {
        "visual_token_cells": cells,
        "visual_grid_metadata": {},
    }
    if expected_native_count is not None:
        token_layout["visual_grid_metadata"]["video_grid_thw"] = [[expected_native_count, 52, 52]]
    return {
        "frame_bin_mappings": [frame_bin_mapping],
        "token_layout": token_layout,
    }


class IdentityLayer(torch.nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        assert attention_mask is not None
        assert attention_mask.shape[-1] == hidden_states.shape[1]
        assert position_embeddings is not None
        return (hidden_states + 0.01,)


class FakeRotary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, position_ids):
        self.calls.append((tuple(hidden_states.shape), tuple(position_ids.shape)))
        return (
            torch.zeros_like(hidden_states),
            torch.ones_like(hidden_states),
        )


def test_temporal_regions_from_layout_group_visual_tokens_by_native_cell():
    layout = fake_layout()
    regions = temporal_regions_from_layout(layout)
    assert set(regions) == {0, 1, 2, 3}
    assert regions[0] == (2, 3, 4)
    assert regions[3] == (11, 12, 13)


def test_select_top_and_random_temporal_regions_are_deterministic():
    scores = [
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.4, 0.2, 0.3],
    ]
    assert select_top_temporal_regions(scores, handoff_layer=1, retain_count=2) == (1, 3)
    first = select_random_temporal_regions(4, 2, seed=7, question_id="q")
    second = select_random_temporal_regions(4, 2, seed=7, question_id="q")
    assert first == second
    assert len(first) == 2


def test_compaction_plan_physically_replaces_removed_regions_with_memory_tokens():
    layout = fake_layout()
    plan = build_compaction_plan(
        layout,
        sequence_length=16,
        retained_temporal_regions=(1, 3),
        memory_tokens_per_region=2,
        condition="handoff_mean",
    )
    assert plan.original_sequence_length == 16
    # Remove 6 visual tokens and insert 4 memory tokens.
    assert plan.compacted_sequence_length == 14
    assert plan.retained_temporal_regions == (1, 3)
    assert plan.handed_off_temporal_regions == (0, 2)
    assert len(plan.memory_token_positions) == 4
    assert set(plan.removed_visual_token_positions) == {2, 3, 4, 8, 9, 10}
    retained_old = {5, 6, 7, 11, 12, 13}
    retained_new_old = {plan.new_to_old[index] for index in plan.retained_visual_token_positions}
    assert retained_new_old == retained_old


def test_compaction_preserves_text_order_and_memory_sources():
    layout = fake_layout()
    plan = build_compaction_plan(
        layout,
        sequence_length=16,
        retained_temporal_regions=(1, 3),
        memory_tokens_per_region=2,
        condition="handoff_mean",
    )
    hidden = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    position_ids = torch.arange(16).view(1, 16)
    compacted, compacted_pos = compact_hidden_states_and_positions(hidden, position_ids, plan)
    assert compacted.shape[1] == 14
    assert compacted_pos.shape[-1] == 14
    # Text tokens 0, 1, 14, 15 remain in canonical order.
    old_positions = [entry.old_position for entry in plan.output_entries if entry.kind == "original"]
    assert [position for position in old_positions if position in {0, 1, 14, 15}] == [0, 1, 14, 15]
    # First removed region [2,3,4] becomes two means: mean([2]) and mean([3,4]).
    memory_values = [
        float(compacted[0, idx, 0])
        for idx, entry in enumerate(plan.output_entries)
        if entry.kind == "memory" and entry.memory_region == 0
    ]
    assert memory_values == [2.0, 3.5]


def test_causal_mask_blocks_future_attention_only():
    mask = build_additive_causal_mask(5, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 5, 5)
    assert torch.all(mask[0, 0].tril() == 0)
    assert mask[0, 0, 0, 4] < -1e20


def test_custom_dense_prefill_preserves_sequence_length_and_matches_manual_loop():
    layout = fake_layout()
    layers = torch.nn.ModuleList([IdentityLayer() for _ in range(4)])
    rotary = FakeRotary()
    hidden = torch.zeros((1, 16, 6), dtype=torch.float32)
    position_ids = torch.arange(16).view(1, 1, 16).repeat(3, 1, 1)
    config = TemporalHandoffConfig(condition="dense_custom", handoff_layer=2, memory_tokens_per_region=0)
    result = run_custom_decoder_prefill(
        layers=layers,
        hidden_states=hidden.clone(),
        position_ids=position_ids,
        layout=layout,
        config=config,
        lm_head=None,
        norm=None,
        num_attention_heads=2,
        head_dim=3,
        rotary_emb=rotary,
        layer_types=("full_attention",) * 4,
    )
    assert result.logits.shape == (1, 1, 6)
    assert torch.allclose(result.logits, torch.full((1, 1, 6), 0.04))
    assert result.compaction_plan is None
    assert all(layer.sequence_length_in == 16 and layer.sequence_length_out == 16 for layer in result.instrumentation)
    assert rotary.calls == [((1, 16, 6), (3, 1, 16))]
    assert result.rotary_embedding_computations == 1
    assert result.instrumentation_metadata()["rotary_embedding_computations"] == 1


def test_custom_handoff_shortens_sequence_and_reduces_later_flops():
    layout = fake_layout()
    layers = torch.nn.ModuleList([IdentityLayer() for _ in range(6)])
    rotary = FakeRotary()
    hidden = torch.zeros((1, 16, 8), dtype=torch.float32)
    position_ids = torch.arange(16).view(1, 1, 16).repeat(3, 1, 1)
    config = TemporalHandoffConfig(
        condition="handoff_mean",
        handoff_layer=2,
        retained_temporal_regions=(1, 3),
        memory_tokens_per_region=2,
    )
    result = run_custom_decoder_prefill(
        layers=layers,
        hidden_states=hidden,
        position_ids=position_ids,
        layout=layout,
        config=config,
        lm_head=None,
        norm=None,
        num_attention_heads=2,
        head_dim=4,
        rotary_emb=rotary,
        layer_types=("full_attention",) * 6,
    )
    lengths = [(layer.sequence_length_in, layer.sequence_length_out) for layer in result.instrumentation]
    assert lengths[2] == (16, 14)
    assert lengths[3][0] == 14
    assert result.compaction_plan is not None
    assert result.final_memory_token_indices == result.compaction_plan.memory_token_positions
    assert result.instrumentation[3].estimated_qk_flops < result.instrumentation[2].estimated_qk_flops
    assert torch.isfinite(result.logits).all()
    assert rotary.calls[0] == ((1, 16, 8), (3, 1, 16))
    assert rotary.calls[1] == ((1, 14, 8), (3, 1, 14))
    assert len(rotary.calls) == 2
    assert result.rotary_embedding_computations == 2
    assert result.instrumentation_metadata()["rotary_embedding_computations"] == 2


def test_hard_evict_and_handoff_share_retained_regions_but_different_memory_budget():
    scores = [[0.25, 0.25, 0.25, 0.25] for _ in range(9)]
    scores[8] = [0.1, 0.5, 0.3, 0.1]
    handoff = condition_from_baseline_scores(
        condition="handoff_mean",
        baseline_temporal_scores=scores,
        handoff_layer=8,
        retain_count=2,
        num_regions=4,
        memory_tokens_per_region=2,
        seed=1,
        question_id="q",
    )
    hard = condition_from_baseline_scores(
        condition="hard_evict",
        baseline_temporal_scores=scores,
        handoff_layer=8,
        retain_count=2,
        num_regions=4,
        memory_tokens_per_region=2,
        seed=1,
        question_id="q",
    )
    assert handoff.retained_temporal_regions == hard.retained_temporal_regions == (1, 2)
    assert handoff.memory_tokens_per_region == 2
    assert hard.memory_tokens_per_region == 0


def test_random_and_adaptive_handoff_have_identical_token_budgets():
    layout = fake_layout()
    adaptive = build_compaction_plan(
        layout,
        sequence_length=16,
        retained_temporal_regions=(1, 2),
        memory_tokens_per_region=2,
        condition="handoff_mean",
    )
    random_plan = build_compaction_plan(
        layout,
        sequence_length=16,
        retained_temporal_regions=(0, 3),
        memory_tokens_per_region=2,
        condition="random_handoff",
    )
    assert adaptive.compacted_sequence_length == random_plan.compacted_sequence_length
    assert len(adaptive.retained_visual_token_positions) == len(random_plan.retained_visual_token_positions)
    assert len(adaptive.memory_token_positions) == len(random_plan.memory_token_positions)


def test_aggregate_analysis_scores_to_native_cells_real_eight_to_four_structure():
    artifact = native_mapping_artifact(
        analysis_to_frames={
            0: [11712],
            1: [13339],
            2: [14965],
            3: [16591],
            4: [18217],
            5: [19844],
            6: [21470],
            7: [23096],
        },
        native_to_frames={
            0: [11712, 13339],
            1: [14965, 16591],
            2: [18217, 19844],
            3: [21470, 23096],
        },
        expected_native_count=4,
    )
    scores = [[0.02, 0.08, 0.1, 0.2, 0.05, 0.15, 0.12, 0.28]]
    result = aggregate_analysis_scores_to_native_cells(scores, artifact)
    assert result.analysis_bin_count == 8
    assert result.native_temporal_cell_count == 4
    assert result.analysis_bin_to_native_cell == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
    assert result.native_cell_to_analysis_bins == {0: (0, 1), 1: (2, 3), 2: (4, 5), 3: (6, 7)}
    assert result.aggregation_method == "sum"
    assert result.native_temporal_scores == ((0.1, 0.30000000000000004, 0.2, 0.4),)
    assert sum(result.native_temporal_scores[0]) == pytest.approx(1.0)


def test_aggregate_analysis_scores_to_native_cells_handles_nonuniform_group_sizes():
    artifact = native_mapping_artifact(
        analysis_to_frames={
            0: [10],
            1: [11],
            2: [12],
            3: [13],
            4: [14],
        },
        native_to_frames={
            0: [10, 11, 12],
            1: [13],
            2: [14],
        },
        expected_native_count=3,
    )
    result = aggregate_analysis_scores_to_native_cells([[0.1, 0.2, 0.3, 0.15, 0.25]], artifact)
    assert result.analysis_bin_to_native_cell == {0: 0, 1: 0, 2: 0, 3: 1, 4: 2}
    assert result.native_temporal_scores == ((0.6000000000000001, 0.15, 0.25),)


def test_aggregate_analysis_scores_to_native_cells_detects_missing_frame_mapping():
    artifact = native_mapping_artifact(
        analysis_to_frames={0: [1], 1: [2]},
        native_to_frames={0: [1], 1: [2, 3]},
        expected_native_count=2,
    )
    with pytest.raises(ValueError, match="missing from frame_bin_mappings"):
        aggregate_analysis_scores_to_native_cells([[0.5, 0.5]], artifact)


def test_aggregate_analysis_scores_to_native_cells_detects_duplicate_frame_ownership():
    artifact = native_mapping_artifact(
        analysis_to_frames={0: [1], 1: [2]},
        native_to_frames={0: [1, 2], 1: [2]},
        expected_native_count=2,
    )
    with pytest.raises(ValueError, match="multiple native temporal cells"):
        aggregate_analysis_scores_to_native_cells([[0.5, 0.5]], artifact)


def test_aggregate_analysis_scores_to_native_cells_detects_uncovered_analysis_bin():
    artifact = native_mapping_artifact(
        analysis_to_frames={0: [1], 1: [2]},
        native_to_frames={0: [1]},
        expected_native_count=1,
    )
    with pytest.raises(ValueError, match="no native temporal-cell owner"):
        aggregate_analysis_scores_to_native_cells([[0.5, 0.5]], artifact)


def test_aggregate_analysis_scores_to_native_cells_detects_native_count_mismatch():
    artifact = native_mapping_artifact(
        analysis_to_frames={0: [1], 1: [2]},
        native_to_frames={0: [1], 1: [2]},
        expected_native_count=3,
    )
    with pytest.raises(ValueError, match="video_grid_thw"):
        aggregate_analysis_scores_to_native_cells([[0.5, 0.5]], artifact)


def test_aggregate_analysis_scores_to_native_cells_no_regression_when_bins_already_native():
    artifact = native_mapping_artifact(
        analysis_to_frames={0: [1], 1: [2], 2: [3]},
        native_to_frames={0: [1], 1: [2], 2: [3]},
        expected_native_count=3,
    )
    result = aggregate_analysis_scores_to_native_cells([[0.2, 0.3, 0.5]], artifact)
    assert result.analysis_bin_to_native_cell == {0: 0, 1: 1, 2: 2}
    assert result.native_temporal_scores == ((0.2, 0.3, 0.5),)


class FakeFeatureOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=91, video_token_id=92)
        self.language_model = SimpleNamespace(
            layers=torch.nn.ModuleList([IdentityLayer() for _ in range(2)]),
            norm=torch.nn.Identity(),
            rotary_emb=FakeRotary(),
            config=SimpleNamespace(
                layer_types=("full_attention", "full_attention"),
                num_attention_heads=2,
                hidden_size=4,
                head_dim=2,
                sliding_window=None,
            ),
        )
        self.emb = torch.nn.Embedding(100, 4)
        self.video_feature_calls = 0
        self.position_calls = 0

    def get_input_embeddings(self):
        return self.emb

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        self.video_feature_calls += 1
        return FakeFeatureOutput((torch.full((2, 4), 7.0),))

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
        return image_mask, video_mask

    def compute_3d_position_ids(
        self,
        input_ids,
        image_grid_thw,
        video_grid_thw,
        inputs_embeds,
        attention_mask,
        past_key_values,
        second_per_grid_ts=None,
        mm_token_type_ids=None,
    ):
        self.position_calls += 1
        assert mm_token_type_ids is not None
        assert past_key_values is None
        return torch.arange(input_ids.shape[1]).view(1, 1, -1).repeat(3, 1, 1)


class FakeConditional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeCore()
        self.config = SimpleNamespace(text_config=self.model.language_model.config)
        self.lm_head = torch.nn.Linear(4, 10, bias=False)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


def test_qwen_multimodal_inputs_use_pooler_output_placeholder_mask_and_compute_3d_positions():
    model = FakeConditional()
    inputs = {
        "input_ids": torch.tensor([[1, 92, 92, 2]]),
        "pixel_values_videos": torch.zeros((1, 3, 2, 2)),
        "video_grid_thw": torch.tensor([[2, 1, 1]]),
        "mm_token_type_ids": torch.tensor([[0, 2, 2, 0]], dtype=torch.int32),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "second_per_grid_ts": torch.tensor([0.5]),
    }
    embeds, position_ids = qwen_multimodal_decoder_inputs(model, inputs)
    assert model.model.video_feature_calls == 1
    assert model.model.position_calls == 1
    assert embeds.shape == (1, 4, 4)
    assert torch.all(embeds[0, 1:3] == 7.0)
    assert position_ids.shape == (3, 1, 4)


def test_qwen_decoder_stack_exposes_language_model_rotary_and_layer_types():
    model = FakeConditional()
    stack = qwen_decoder_stack(model)
    assert len(stack["layers"]) == 2
    assert stack["rotary_emb"] is model.model.language_model.rotary_emb
    assert stack["layer_types"] == ("full_attention", "full_attention")


def test_decoder_layer_without_position_embeddings_fails_loudly():
    class BadLayer(torch.nn.Module):
        def forward(self, hidden_states, attention_mask=None):
            return hidden_states

    layout = fake_layout()
    config = TemporalHandoffConfig(condition="dense_custom", handoff_layer=0, memory_tokens_per_region=0)
    with pytest.raises(RuntimeError, match="position_embeddings"):
        run_custom_decoder_prefill(
            layers=[BadLayer()],
            hidden_states=torch.zeros((1, 16, 4)),
            position_ids=torch.arange(16).view(1, 1, 16).repeat(3, 1, 1),
            layout=layout,
            config=config,
            lm_head=None,
            norm=None,
            num_attention_heads=2,
            head_dim=2,
            rotary_emb=FakeRotary(),
            layer_types=("full_attention",),
        )


def test_custom_loop_returns_final_token_only_logits_with_lm_head():
    layout = fake_layout()
    hidden = torch.zeros((1, 16, 4))
    position_ids = torch.arange(16).view(1, 1, 16).repeat(3, 1, 1)
    lm_head = torch.nn.Linear(4, 11, bias=False)
    result = run_custom_decoder_prefill(
        layers=[IdentityLayer()],
        hidden_states=hidden,
        position_ids=position_ids,
        layout=layout,
        config=TemporalHandoffConfig(condition="dense_custom", handoff_layer=0, memory_tokens_per_region=0),
        lm_head=lm_head,
        norm=None,
        num_attention_heads=2,
        head_dim=2,
        rotary_emb=FakeRotary(),
        layer_types=("full_attention",),
    )
    assert result.logits.shape == (1, 1, 11)
    assert result.final_token_hidden_state.shape == (1, 1, 4)
