from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.experiment1.temporal_handoff import (
    TemporalHandoffConfig,
    build_additive_causal_mask,
    build_compaction_plan,
    compact_hidden_states_and_positions,
    condition_from_baseline_scores,
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


class IdentityLayer(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False, output_attentions=False):
        assert attention_mask is not None
        assert attention_mask.shape[-1] == hidden_states.shape[1]
        return (hidden_states + 0.01,)


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
    hidden = torch.zeros((1, 16, 6), dtype=torch.float32)
    position_ids = torch.arange(16).view(1, 16)
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
    )
    assert result.logits.shape == hidden.shape
    assert torch.allclose(result.logits, torch.full_like(hidden, 0.04))
    assert result.compaction_plan is None
    assert all(layer.sequence_length_in == 16 and layer.sequence_length_out == 16 for layer in result.instrumentation)


def test_custom_handoff_shortens_sequence_and_reduces_later_flops():
    layout = fake_layout()
    layers = torch.nn.ModuleList([IdentityLayer() for _ in range(6)])
    hidden = torch.zeros((1, 16, 8), dtype=torch.float32)
    position_ids = torch.arange(16).view(1, 16)
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
    )
    lengths = [(layer.sequence_length_in, layer.sequence_length_out) for layer in result.instrumentation]
    assert lengths[2] == (16, 14)
    assert lengths[3][0] == 14
    assert result.compaction_plan is not None
    assert result.final_memory_token_indices == result.compaction_plan.memory_token_positions
    assert result.instrumentation[3].estimated_qk_flops < result.instrumentation[2].estimated_qk_flops
    assert torch.isfinite(result.logits).all()


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

