import numpy as np

from src.experiment1.temporal import (
    bins_to_attention_mass,
    build_temporal_relevance_from_token_scores,
    normalized_temporal_entropy,
    pool_token_scores_to_temporal_bins,
    spearman_rank_correlation,
    temporal_rank_order,
    topk_overlap,
)
from src.experiment1.token_layout import TokenLayout, VisualTokenCell
from src.frame_sampling import FrameBatch


def _layout():
    cells = (
        VisualTokenCell(2, 0, "video", 0, 0, 0, 0, 2, 1, 2),
        VisualTokenCell(3, 1, "video", 0, 0, 0, 1, 2, 1, 2),
        VisualTokenCell(4, 2, "video", 0, 1, 0, 0, 2, 1, 2),
        VisualTokenCell(5, 3, "video", 0, 1, 0, 1, 2, 1, 2),
    )
    return TokenLayout(
        question_token_indices=(6, 7),
        prompt_token_indices=tuple(range(8)),
        visual_token_indices=(2, 3, 4, 5),
        visual_cells=cells,
        visual_grid_metadata={"video_grid_thw": [[2, 1, 4]], "spatial_merge_size": 2},
        query_scope="question",
    )


def test_temporal_pooling_spatially_sums_tokens_by_qwen_temporal_bin():
    scores = np.array([[1.0, 2.0, 4.0, 8.0], [0.0, 1.0, 1.0, 0.0]])
    pooled = pool_token_scores_to_temporal_bins(scores, _layout())
    np.testing.assert_allclose(pooled, [[3.0, 12.0], [1.0, 1.0]])


def test_entropy_and_bins_to_80_mass():
    assert normalized_temporal_entropy([1.0, 0.0, 0.0, 0.0]) == 0.0
    assert round(normalized_temporal_entropy([1.0, 1.0, 1.0, 1.0]), 6) == 1.0
    assert bins_to_attention_mass([0.7, 0.1, 0.1, 0.1]) == (2, 0.5)


def test_rank_correlation_and_topk_overlap():
    final = temporal_rank_order([0.1, 0.8, 0.05, 0.05])
    same = temporal_rank_order([0.2, 0.7, 0.05, 0.05])
    reverse = tuple(reversed(final))
    assert spearman_rank_correlation(same, final) == 1.0
    assert spearman_rank_correlation(reverse, final) == -1.0
    assert topk_overlap(same, final, 2) == (2, 1.0)


def test_build_temporal_relevance_records_bin_metadata_and_layer_metrics():
    batch = FrameBatch(
        frames=np.zeros((4, 2, 2, 3), dtype=np.uint8),
        frame_indices=(10, 20, 30, 40),
        timestamps=(1.0, 2.0, 3.0, 4.0),
        video_path=None,
        metadata={"input_modality": "video"},
    )
    relevance = build_temporal_relevance_from_token_scores(
        np.array([[1.0, 1.0, 4.0, 4.0], [5.0, 5.0, 1.0, 1.0]]),
        _layout(),
        [batch],
        "unit",
        topk=1,
    ).to_json_dict()
    assert relevance["metadata"]["num_temporal_bins"] == 2
    assert relevance["temporal_bins"][0]["sampled_frame_indices"] == [10, 20]
    assert relevance["layer_metrics"][0]["temporal_bin_rank_order"] == (1, 0)
    assert relevance["layer_metrics"][1]["spearman_with_final_layer_ordering"] == 1.0
