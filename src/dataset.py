from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VideoSegment:
    input_key: str
    video_id: str
    participant_id: str
    start_seconds: float | None
    end_seconds: float | None
    image_time_seconds: float | None
    raw: dict[str, Any]

    @property
    def is_image(self) -> bool:
        return self.image_time_seconds is not None

    def path_under(self, mp4_dir: str | Path) -> Path:
        return Path(mp4_dir) / self.participant_id / f"{self.video_id}.mp4"


@dataclass(frozen=True)
class VQAExample:
    question_id: str
    question_type: str
    question: str
    choices: tuple[str, ...]
    correct_idx: int
    inputs: tuple[VideoSegment, ...]
    raw: dict[str, Any]

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(segment.video_id for segment in self.inputs)


def seconds_from_time_str(value: str) -> float:
    """Parse HD-EPIC time strings such as ``00:01:15.539``.

    Some public VQA text tags use non-zero-padded seconds, e.g. ``00:03:1.8``.
    """
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS.sss time string, got {value!r}.")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def infer_question_type(question_id: str, annotation_file: Path | None = None) -> str:
    if annotation_file is not None:
        return annotation_file.stem
    return question_id.rsplit("_", 1)[0]


def parse_video_segment(input_key: str, raw: dict[str, Any]) -> VideoSegment:
    if "id" not in raw:
        raise ValueError(f"Input {input_key!r} is missing required field 'id'.")

    video_id = str(raw["id"])
    participant_id = video_id.split("-")[0]
    image_time = seconds_from_time_str(raw["time"]) if "time" in raw else None
    start = seconds_from_time_str(raw["start_time"]) if "start_time" in raw else None
    end = seconds_from_time_str(raw["end_time"]) if "end_time" in raw else None

    if image_time is not None:
        start = image_time
        end = image_time + 1.0

    return VideoSegment(
        input_key=input_key,
        video_id=video_id,
        participant_id=participant_id,
        start_seconds=start,
        end_seconds=end,
        image_time_seconds=image_time,
        raw=dict(raw),
    )


def parse_vqa_example(
    question_id: str, raw: dict[str, Any], annotation_file: Path | None = None
) -> VQAExample:
    required = ("inputs", "question", "choices", "correct_idx")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Question {question_id!r} is missing fields: {missing}")

    choices = tuple(str(choice) for choice in raw["choices"])
    if len(choices) != 5:
        raise ValueError(f"Question {question_id!r} has {len(choices)} choices, expected 5.")

    correct_idx = int(raw["correct_idx"])
    if not 0 <= correct_idx < len(choices):
        raise ValueError(f"Question {question_id!r} has invalid correct_idx {correct_idx}.")

    inputs = tuple(
        parse_video_segment(input_key, input_raw)
        for input_key, input_raw in raw["inputs"].items()
    )
    if not inputs:
        raise ValueError(f"Question {question_id!r} has no visual inputs.")

    return VQAExample(
        question_id=question_id,
        question_type=infer_question_type(question_id, annotation_file),
        question=str(raw["question"]),
        choices=choices,
        correct_idx=correct_idx,
        inputs=inputs,
        raw=dict(raw),
    )


class HDEpicVQADataset:
    def __init__(self, questions_dir: str | Path, question_files: Iterable[str] | None = None):
        self.questions_dir = Path(questions_dir)
        self.question_files = self._resolve_question_files(question_files)
        self.examples = self._load_examples()

    def _resolve_question_files(self, question_files: Iterable[str] | None) -> list[Path]:
        if question_files is None:
            files = sorted(self.questions_dir.glob("*.json"))
        else:
            files = []
            for name in question_files:
                path = Path(name)
                if not path.suffix:
                    path = path.with_suffix(".json")
                if not path.is_absolute():
                    path = self.questions_dir / path
                files.append(path)
        if not files:
            raise FileNotFoundError(f"No VQA JSON files found in {self.questions_dir}.")
        return files

    def _load_examples(self) -> list[VQAExample]:
        examples: list[VQAExample] = []
        for path in self.question_files:
            with path.open("r") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError(f"Expected object at top level of {path}.")
            for question_id, raw in data.items():
                examples.append(parse_vqa_example(question_id, raw, path))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> VQAExample:
        return self.examples[index]

    def iter_limit(self, limit: int | None = None) -> Iterable[VQAExample]:
        yield from self.examples if limit is None else self.examples[:limit]
