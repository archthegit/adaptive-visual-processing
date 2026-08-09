import numpy as np

from src.experiment1.masking import (
    VisionAccessMaskSpec,
    build_text_to_visual_block_mask,
    cutoff_alias_to_layer,
    mask_blocks_text_to_visual,
)


def test_layer_specific_mask_blocks_only_after_cutoff():
    spec = VisionAccessMaskSpec(
        through_layer=1,
        num_layers=4,
        question_token_indices=(3,),
        text_token_indices=(3, 4),
        visual_token_indices=(0, 1),
    )
    mask = build_text_to_visual_block_mask(spec, seq_len=5)
    assert np.all(mask[0] == 0)
    assert np.all(mask[1] == 0)
    assert mask_blocks_text_to_visual(mask, 2, (3, 4), (0, 1))
    assert mask[2, 3, 4] == 0


def test_cutoff_aliases_are_configurable():
    assert cutoff_alias_to_layer("none", 32) is None
    assert cutoff_alias_to_layer("early", 32) == 7
    assert cutoff_alias_to_layer("middle", 32) == 15
    assert cutoff_alias_to_layer("late", 32) == 23
    assert cutoff_alias_to_layer("5", 32) == 5
