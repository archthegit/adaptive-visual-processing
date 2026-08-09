from src.experiment1.token_layout import (
    build_token_layout,
    cells_from_grid_specs,
    cells_from_video_grid,
    derive_question_token_indices,
    token_indices_for_char_span,
    visual_indices_from_ids,
    map_unexpanded_to_expanded_indices,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded


class OffsetFallbackTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [500 if char == "V" else ord(char) for char in text]
        if return_offsets_mapping:
            return {
                "input_ids": ids,
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }
        if text == "What?":
            return {"input_ids": [999]}
        return {"input_ids": ids}


def test_question_indices_are_derived_from_tokenized_prompt_not_hardcoded():
    prompt = "User: video What is the object? Answer:"
    question = "What is the object?"
    indices = derive_question_token_indices(FakeTokenizer(), prompt, question)
    assert indices == tuple(range(prompt.index(question), prompt.index(question) + len(question)))


def test_visual_indices_can_use_mm_token_types_or_special_ids():
    assert visual_indices_from_ids([1, 99, 99, 2], video_token_id=99) == (1, 2)
    assert visual_indices_from_ids([1, 5, 6, 7], mm_token_type_ids=[0, 2, 2, 0]) == (1, 2)


def test_unexpanded_to_expanded_mapping_skips_repeated_visual_tokens():
    mapping = map_unexpanded_to_expanded_indices(
        [ord("A"), 500, ord("Q"), ord("?")],
        [ord("A"), 500, 500, 500, ord("Q"), ord("?")],
        {500},
    )
    assert mapping == {0: 0, 2: 4, 3: 5}


def test_token_indices_for_char_span_uses_overlapping_offsets():
    prompt = "User: video What? Answer:"
    start = prompt.index("What?")
    indices = token_indices_for_char_span(FakeTokenizer(), prompt, start, start + len("What?"))
    assert indices == tuple(range(start, start + len("What?")))


def test_video_grid_maps_visual_tokens_to_temporal_spatial_cells():
    cells = cells_from_video_grid([10, 11, 12, 13, 14, 15, 16, 17], [(2, 4, 4)], 2, [0.5])
    assert len(cells) == 8
    assert cells[0].token_index == 10
    assert (cells[0].temporal_index, cells[0].spatial_y, cells[0].spatial_x) == (0, 0, 0)
    assert (cells[-1].temporal_index, cells[-1].spatial_y, cells[-1].spatial_x) == (1, 1, 1)
    assert cells[-1].timestamp == 0.5


def test_mixed_image_video_grid_specs_preserve_input_order():
    cells = cells_from_grid_specs(
        [10, 11, 12],
        [
            {"input_index": 0, "modality": "video", "grid": (1, 2, 2), "seconds_per_grid": 0.5},
            {"input_index": 1, "modality": "image", "grid": (1, 2, 2), "seconds_per_grid": None},
            {"input_index": 2, "modality": "video", "grid": (1, 2, 2), "seconds_per_grid": 1.0},
        ],
        spatial_merge_size=2,
    )
    assert [(cell.modality, cell.input_index, cell.token_index) for cell in cells] == [
        ("video", 0, 10),
        ("image", 1, 11),
        ("video", 2, 12),
    ]


def test_build_token_layout_combines_question_visual_and_grid_metadata():
    prompt = "User: VV What? Answer:"
    layout = build_token_layout(
        input_ids=[ord("U"), 500, 500, ord(" "), ord("W"), ord("h"), ord("a"), ord("t"), ord("?")],
        tokenizer=FakeTokenizer(),
        rendered_prompt=prompt,
        question_text="What?",
        video_grid_thw=[(1, 2, 4)],
        spatial_merge_size=2,
        video_token_id=500,
        second_per_grid_ts=[1.0],
    )
    assert layout.question_token_indices == (4, 5, 6, 7, 8)
    assert layout.visual_token_indices == (1, 2)
    assert layout.num_visual_tokens == 2


def test_build_token_layout_falls_back_from_unexpanded_prompt_to_expanded_input_ids():
    prompt = "User: V What? Answer:"
    layout = build_token_layout(
        input_ids=[
            ord("U"),
            ord("s"),
            ord("e"),
            ord("r"),
            ord(":"),
            ord(" "),
            500,
            500,
            500,
            ord(" "),
            ord("W"),
            ord("h"),
            ord("a"),
            ord("t"),
            ord("?"),
        ],
        tokenizer=FakeTokenizer(),
        rendered_prompt=prompt,
        question_text="What?",
        video_grid_thw=[(1, 2, 6)],
        spatial_merge_size=2,
        video_token_id=500,
        second_per_grid_ts=[1.0],
    )
    assert layout.question_token_indices == (10, 11, 12, 13, 14)
    assert layout.visual_token_indices == (6, 7, 8)


def test_build_token_layout_handles_mixed_image_video_grids():
    prompt = "User: VIV What? Answer:"
    layout = build_token_layout(
        input_ids=[
            ord("U"),
            500,
            600,
            500,
            ord(" "),
            ord("W"),
            ord("h"),
            ord("a"),
            ord("t"),
            ord("?"),
        ],
        tokenizer=FakeTokenizer(),
        rendered_prompt=prompt,
        question_text="What?",
        video_grid_thw=[(1, 2, 2), (1, 2, 2)],
        image_grid_thw=[(1, 2, 2)],
        visual_input_modalities=["video", "image", "video"],
        spatial_merge_size=2,
        video_token_id=500,
        image_token_id=600,
        second_per_grid_ts=[0.5, 1.0],
    )
    assert [(cell.modality, cell.input_index) for cell in layout.visual_cells] == [
        ("video", 0),
        ("image", 1),
        ("video", 2),
    ]
    assert layout.visual_grid_metadata["visual_input_modalities"] == ["video", "image", "video"]


def test_build_token_layout_falls_back_to_offsets_when_token_subsequence_fails():
    prompt = "User: V What? Answer:"
    layout = build_token_layout(
        input_ids=[
            ord("U"),
            ord("s"),
            ord("e"),
            ord("r"),
            ord(":"),
            ord(" "),
            500,
            500,
            500,
            ord(" "),
            ord("W"),
            ord("h"),
            ord("a"),
            ord("t"),
            ord("?"),
        ],
        tokenizer=OffsetFallbackTokenizer(),
        rendered_prompt=prompt,
        question_text="What?",
        video_grid_thw=[(1, 2, 6)],
        spatial_merge_size=2,
        video_token_id=500,
        second_per_grid_ts=[1.0],
    )
    assert layout.question_token_indices == (10, 11, 12, 13, 14)
