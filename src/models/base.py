from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..dataset import VQAExample, seconds_from_time_str
from ..frame_sampling import FrameBatch


@dataclass(frozen=True)
class ModelPrediction:
    raw_response: str
    predicted_idx: int
    correct_idx: int
    correct: bool
    frame_indices: tuple[tuple[int, ...], ...]
    timestamps: tuple[tuple[float, ...], ...]
    metadata: dict[str, Any]


def parse_choice_response(response: str, n_choices: int = 5) -> int:
    """Mirror the official benchmark: first capital A-Z maps to an index."""
    match = re.search(r"[A-Z]", response)
    if match is None:
        return -1
    idx = ord(match.group(0)) - ord("A")
    return idx if 0 <= idx < n_choices else -1


def _format_seconds(seconds: float) -> str:
    if seconds >= 3600:
        return time.strftime("%H:%M:%S", time.gmtime(seconds))
    return time.strftime("%M:%S", time.gmtime(seconds))


def parse_question_tags(
    text: str,
    example: VQAExample,
    temporal_divisor: float = 1.0,
    dataset_orig_res: int = 1408,
) -> str:
    segment_by_key = {segment.input_key: segment for segment in example.inputs}

    def time_repl(match: re.Match[str]) -> str:
        seconds = seconds_from_time_str(match.group(1))
        input_key = match.group(2)
        segment = segment_by_key.get(input_key)
        if segment is not None and segment.start_seconds is not None:
            seconds -= segment.start_seconds
        return _format_seconds(seconds / temporal_divisor)

    def bbox_repl(match: re.Match[str]) -> str:
        coords = [float(match.group(i)) for i in range(1, 5)]
        scaled = [int(coord / dataset_orig_res * 1000) for coord in coords]
        return f"({', '.join(str(coord) for coord in scaled)})"

    text = re.sub(r"<TIME\s+([\d:.]+)\s+(.+?)>", time_repl, text)
    return re.sub(r"<BBOX\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*>", bbox_repl, text)


def format_multiple_choice_prompt(example: VQAExample) -> str:
    text = f"Question: {example.question}. Answers: "
    for idx, choice in enumerate(example.choices):
        text += f"({chr(ord('A') + idx)}) {choice}. "
    return parse_question_tags(text + "Correct: ", example)


class BaseVQAModel(Protocol):
    model_name: str

    def predict(self, example: VQAExample, frame_batches: list[FrameBatch]) -> ModelPrediction:
        ...
