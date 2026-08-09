from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class VisualTokenCell:
    token_index: int
    visual_index: int
    modality: str
    input_index: int
    temporal_index: int
    spatial_y: int
    spatial_x: int
    grid_t: int
    grid_h: int
    grid_w: int
    seconds_per_grid: float | None = None
    timestamp: float | None = None


@dataclass(frozen=True)
class TokenLayout:
    question_token_indices: tuple[int, ...]
    prompt_token_indices: tuple[int, ...]
    visual_token_indices: tuple[int, ...]
    visual_cells: tuple[VisualTokenCell, ...]
    visual_grid_metadata: dict[str, Any]
    query_scope: str

    @property
    def num_visual_tokens(self) -> int:
        return len(self.visual_token_indices)


def _to_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, ...]:
    if not subsequence:
        return tuple()
    n = len(subsequence)
    for start in range(0, len(sequence) - n + 1):
        if list(sequence[start : start + n]) == list(subsequence):
            return tuple(range(start, start + n))
    return tuple()


def map_unexpanded_to_expanded_indices(
    unexpanded_ids: Sequence[int],
    expanded_ids: Sequence[int],
    visual_token_ids: set[int],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    expanded_pos = 0
    expanded = list(expanded_ids)
    for unexpanded_pos, token_id in enumerate(unexpanded_ids):
        if token_id in visual_token_ids:
            while expanded_pos < len(expanded) and expanded[expanded_pos] in visual_token_ids:
                expanded_pos += 1
            continue
        while expanded_pos < len(expanded) and expanded[expanded_pos] != token_id:
            expanded_pos += 1
        if expanded_pos >= len(expanded):
            continue
        mapping[unexpanded_pos] = expanded_pos
        expanded_pos += 1
    return mapping


def token_indices_for_char_span(
    tokenizer: Any,
    text: str,
    start: int,
    end: int,
) -> tuple[int, ...]:
    if start < 0 or end <= start:
        return tuple()
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        return tuple()
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return tuple()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    indices: list[int] = []
    for index, offset in enumerate(offsets):
        if offset is None:
            continue
        token_start, token_end = int(offset[0]), int(offset[1])
        if token_end <= token_start:
            continue
        if token_start < end and token_end > start:
            indices.append(index)
    return tuple(indices)


def derive_question_token_indices(
    tokenizer: Any,
    rendered_prompt: str | None,
    question_text: str,
    query_scope: str = "question",
    user_prompt_text: str | None = None,
    input_ids: Sequence[int] | None = None,
    visual_token_ids: set[int] | None = None,
) -> tuple[int, ...]:
    all_ids = _to_list(input_ids) if input_ids is not None else _to_list(tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"])
    if query_scope == "full_user_prompt":
        scoped = user_prompt_text or rendered_prompt
    elif query_scope == "question":
        scoped = question_text
    else:
        raise ValueError("query_scope must be 'question' or 'full_user_prompt'.")
    scoped_ids = _to_list(tokenizer(scoped, add_special_tokens=False)["input_ids"])
    direct = find_subsequence(all_ids, scoped_ids)
    if direct:
        return direct

    if input_ids is None or rendered_prompt is None:
        return tuple()
    unexpanded_ids = _to_list(tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"])
    unexpanded_span = find_subsequence(unexpanded_ids, scoped_ids)
    if not unexpanded_span:
        char_start = rendered_prompt.find(scoped)
        unexpanded_span = token_indices_for_char_span(
            tokenizer,
            rendered_prompt,
            char_start,
            char_start + len(scoped) if char_start >= 0 else -1,
        )
    if not unexpanded_span:
        return tuple()
    index_map = map_unexpanded_to_expanded_indices(unexpanded_ids, all_ids, visual_token_ids or set())
    mapped = [index_map[idx] for idx in unexpanded_span if idx in index_map]
    return tuple(mapped) if len(mapped) == len(unexpanded_span) else tuple()


def visual_indices_from_ids(
    input_ids: Sequence[int],
    video_token_id: int | None = None,
    image_token_id: int | None = None,
    mm_token_type_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    ids = _to_list(input_ids)
    if mm_token_type_ids is not None:
        types = _to_list(mm_token_type_ids)
        return tuple(idx for idx, token_type in enumerate(types) if token_type in {1, 2})
    visual_ids = {token_id for token_id in (video_token_id, image_token_id) if token_id is not None}
    return tuple(idx for idx, token_id in enumerate(ids) if token_id in visual_ids)


def cells_from_grid_specs(
    visual_token_indices: Sequence[int],
    grid_specs: Iterable[dict[str, Any]],
    spatial_merge_size: int,
) -> tuple[VisualTokenCell, ...]:
    cells: list[VisualTokenCell] = []
    cursor = 0
    for spec in grid_specs:
        input_index = int(spec["input_index"])
        modality = str(spec["modality"])
        grid = spec["grid"]
        t, h, w = [int(x) for x in grid]
        if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
            raise ValueError(f"Grid {(t, h, w)} is not divisible by spatial_merge_size={spatial_merge_size}.")
        grid_h = h // spatial_merge_size
        grid_w = w // spatial_merge_size
        expected = t * grid_h * grid_w
        interval = spec.get("seconds_per_grid")
        interval = float(interval) if interval is not None else None
        for temporal_index in range(t):
            for spatial_y in range(grid_h):
                for spatial_x in range(grid_w):
                    if cursor >= len(visual_token_indices):
                        raise ValueError("Not enough visual token indices for video_grid_thw metadata.")
                    timestamp = temporal_index * interval if interval is not None else None
                    cells.append(
                        VisualTokenCell(
                            token_index=int(visual_token_indices[cursor]),
                            visual_index=cursor,
                            modality=modality,
                            input_index=input_index,
                            temporal_index=temporal_index,
                            spatial_y=spatial_y,
                            spatial_x=spatial_x,
                            grid_t=t,
                            grid_h=grid_h,
                            grid_w=grid_w,
                            seconds_per_grid=interval,
                            timestamp=timestamp,
                        )
                    )
                    cursor += 1
        if expected == 0:
            raise ValueError(f"Empty visual grid for input {input_index}.")
    if cursor != len(visual_token_indices):
        raise ValueError(
            f"Visual token count mismatch: consumed {cursor}, got {len(visual_token_indices)} indices."
        )
    return tuple(cells)


def cells_from_video_grid(
    visual_token_indices: Sequence[int],
    video_grid_thw: Iterable[Sequence[int]],
    spatial_merge_size: int,
    second_per_grid_ts: Sequence[float] | None = None,
) -> tuple[VisualTokenCell, ...]:
    seconds = list(second_per_grid_ts or [])
    specs = [
        {
            "input_index": input_index,
            "modality": "video",
            "grid": grid,
            "seconds_per_grid": seconds[input_index] if input_index < len(seconds) else None,
        }
        for input_index, grid in enumerate(video_grid_thw)
    ]
    return cells_from_grid_specs(visual_token_indices, specs, spatial_merge_size)


def grid_specs_for_visual_inputs(
    visual_input_modalities: Sequence[str],
    video_grid_thw: Iterable[Sequence[int]],
    image_grid_thw: Iterable[Sequence[int]] | None = None,
    second_per_grid_ts: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    video_grids = list(video_grid_thw)
    image_grids = list(image_grid_thw or [])
    seconds = list(second_per_grid_ts or [])
    video_cursor = 0
    image_cursor = 0
    specs: list[dict[str, Any]] = []
    for input_index, modality in enumerate(visual_input_modalities):
        if modality == "video":
            if video_cursor >= len(video_grids):
                raise ValueError("visual_input_modalities requested more video grids than Qwen returned.")
            specs.append(
                {
                    "input_index": input_index,
                    "modality": "video",
                    "grid": video_grids[video_cursor],
                    "seconds_per_grid": seconds[video_cursor] if video_cursor < len(seconds) else None,
                }
            )
            video_cursor += 1
        elif modality == "image":
            if image_cursor >= len(image_grids):
                raise ValueError("visual_input_modalities requested more image grids than Qwen returned.")
            specs.append(
                {
                    "input_index": input_index,
                    "modality": "image",
                    "grid": image_grids[image_cursor],
                    "seconds_per_grid": None,
                }
            )
            image_cursor += 1
        else:
            raise ValueError(f"Unsupported visual modality {modality!r}.")
    if video_cursor != len(video_grids) or image_cursor != len(image_grids):
        raise ValueError("Qwen returned visual grids that were not matched to visual_input_modalities.")
    return specs


def build_token_layout(
    input_ids: Sequence[int],
    tokenizer: Any,
    rendered_prompt: str,
    question_text: str,
    video_grid_thw: Iterable[Sequence[int]],
    spatial_merge_size: int,
    image_grid_thw: Iterable[Sequence[int]] | None = None,
    visual_input_modalities: Sequence[str] | None = None,
    video_token_id: int | None = None,
    image_token_id: int | None = None,
    mm_token_type_ids: Sequence[int] | None = None,
    second_per_grid_ts: Sequence[float] | None = None,
    query_scope: str = "question",
    user_prompt_text: str | None = None,
) -> TokenLayout:
    ids = _to_list(input_ids)
    visual_indices = visual_indices_from_ids(ids, video_token_id, image_token_id, mm_token_type_ids)
    question_indices = derive_question_token_indices(
        tokenizer,
        rendered_prompt,
        question_text,
        query_scope=query_scope,
        user_prompt_text=user_prompt_text,
        input_ids=ids,
        visual_token_ids={token_id for token_id in (video_token_id, image_token_id) if token_id is not None},
    )
    if not question_indices:
        raise ValueError(
            "Could not derive question token indices from tokenizer/prompt. "
            f"query_scope={query_scope!r}, question_text={question_text!r}, "
            f"input_tokens={len(ids)}, visual_tokens={len(visual_indices)}."
        )
    video_grids = list(video_grid_thw)
    image_grids = list(image_grid_thw or [])
    modalities = tuple(visual_input_modalities or ["video"] * len(video_grids))
    grid_specs = grid_specs_for_visual_inputs(modalities, video_grids, image_grids, second_per_grid_ts)
    cells = cells_from_grid_specs(visual_indices, grid_specs, spatial_merge_size)
    return TokenLayout(
        question_token_indices=question_indices,
        prompt_token_indices=tuple(range(len(ids))),
        visual_token_indices=visual_indices,
        visual_cells=cells,
        visual_grid_metadata={
            "video_grid_thw": [list(map(int, grid)) for grid in video_grids],
            "image_grid_thw": [list(map(int, grid)) for grid in image_grids],
            "visual_input_modalities": list(modalities),
            "spatial_merge_size": spatial_merge_size,
            "second_per_grid_ts": list(second_per_grid_ts or []),
        },
        query_scope=query_scope,
    )
