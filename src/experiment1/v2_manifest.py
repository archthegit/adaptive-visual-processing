from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from src.dataset import HDEpicVQADataset, VQAExample

from .manifest import EXPERIMENT1_CATEGORIES, infer_experiment1_category
from .temporal_splits import bounded_duration_seconds


FFProbeRunner = Callable[[list[str]], str]


@dataclass(frozen=True)
class VideoInventoryRecord:
    video_id: str
    participant_id: str
    path: str
    size_bytes: int
    duration_seconds: float
    fps: float
    num_frames: int
    width: int
    height: int
    complete: bool = True

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrimaryCandidate:
    example: VQAExample
    category: str
    source_video_id: str
    participant_id: str
    analyzed_start_seconds: float
    analyzed_end_seconds: float
    analyzed_duration_seconds: float
    is_unbounded_full_video: bool
    duration_group: str

    @property
    def question_id(self) -> str:
        return self.example.question_id


@dataclass(frozen=True)
class Experiment1V2Config:
    seed: int = 20260830
    dev_fraction: float = 0.2
    max_primary_per_source_video: int = 1
    min_test_per_category: int = 1
    mismatch_token_length_weight: float = 1.0


def parse_frame_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denom = float(denominator)
        return 0.0 if denom == 0 else float(numerator) / denom
    return float(value)


def parse_ffprobe_stream(video_id: str, path: Path, ffprobe_json: str) -> VideoInventoryRecord:
    data = json.loads(ffprobe_json)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe returned no video stream for {path}.")
    stream = streams[0]
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    fps = parse_frame_rate(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"))
    if duration <= 0:
        raise ValueError(f"ffprobe returned non-positive duration for {path}: {duration}")
    if fps <= 0:
        raise ValueError(f"ffprobe returned non-positive fps for {path}: {fps}")
    raw_frames = stream.get("nb_frames")
    num_frames = int(raw_frames) if raw_frames not in {None, "N/A", ""} else max(1, int(round(duration * fps)))
    if num_frames <= 0:
        raise ValueError(f"ffprobe returned non-positive frame count for {path}: {num_frames}")
    return VideoInventoryRecord(
        video_id=video_id,
        participant_id=video_id.split("-")[0],
        path=str(path),
        size_bytes=int(path.stat().st_size),
        duration_seconds=duration,
        fps=fps,
        num_frames=num_frames,
        width=int(stream["width"]),
        height=int(stream["height"]),
        complete=True,
    )


def ffprobe_video(path: Path, runner: FFProbeRunner | None = None) -> str:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    if runner is not None:
        return runner(command)
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout


def inventory_local_mp4s(mp4_dir: str | Path, runner: FFProbeRunner | None = None) -> tuple[list[VideoInventoryRecord], list[dict[str, Any]]]:
    root = Path(mp4_dir)
    inventory: list[VideoInventoryRecord] = []
    exclusions: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.mp4")):
        video_id = path.stem
        part_path = path.with_suffix(path.suffix + ".part")
        try:
            record = parse_ffprobe_stream(video_id, path, ffprobe_video(path, runner=runner))
        except Exception as exc:
            exclusions.append({"kind": "video", "video_id": video_id, "path": str(path), "reason": f"ffprobe_failed: {exc}"})
            continue
        if part_path.exists():
            exclusions.append(
                {
                    "kind": "video",
                    "video_id": video_id,
                    "path": str(part_path),
                    "reason": "incomplete_part_file_present_ignored",
                }
            )
        inventory.append(record)
    for path in sorted(root.glob("*/*.mp4.part")):
        video_id = path.name[: -len(".mp4.part")]
        exclusions.append({"kind": "video", "video_id": video_id, "path": str(path), "reason": "incomplete_part_file"})
    return inventory, exclusions


def duration_tertiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values if value > 0 and math.isfinite(value))
    if len(ordered) < 3:
        raise ValueError("Need at least three positive durations to compute tertiles.")

    def percentile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {"short_medium": percentile(1.0 / 3.0), "medium_long": percentile(2.0 / 3.0)}


def assign_duration_group(duration: float, thresholds: dict[str, float]) -> str:
    if duration <= thresholds["short_medium"]:
        return "short"
    if duration <= thresholds["medium_long"]:
        return "medium"
    return "long"


def analyzed_bounds(example: VQAExample, inventory: dict[str, VideoInventoryRecord]) -> tuple[float, float, float, bool]:
    segment = example.inputs[0]
    video = inventory[segment.video_id]
    start = 0.0 if segment.start_seconds is None else max(0.0, float(segment.start_seconds))
    end = video.duration_seconds if segment.end_seconds is None else min(video.duration_seconds, float(segment.end_seconds))
    if end <= start:
        end = min(video.duration_seconds, start + 1.0)
    return start, end, max(0.0, end - start), segment.start_seconds is None or segment.end_seconds is None


def question_token_length(example: VQAExample) -> int:
    return len(example.question.split())


def eligible_primary_candidates(
    examples: list[VQAExample],
    inventory_records: list[VideoInventoryRecord],
    thresholds: dict[str, float],
) -> tuple[list[PrimaryCandidate], list[dict[str, Any]]]:
    inventory = {record.video_id: record for record in inventory_records if record.complete}
    candidates: list[PrimaryCandidate] = []
    exclusions: list[dict[str, Any]] = []
    for example in examples:
        category = infer_experiment1_category(example.question_type)
        if category not in EXPERIMENT1_CATEGORIES:
            exclusions.append({"kind": "question", "question_id": example.question_id, "reason": "category_not_in_experiment1"})
            continue
        if len(example.inputs) != 1:
            exclusions.append({"kind": "question", "question_id": example.question_id, "reason": "not_exactly_one_visual_input"})
            continue
        segment = example.inputs[0]
        if segment.is_image:
            exclusions.append({"kind": "question", "question_id": example.question_id, "reason": "image_input"})
            continue
        if segment.video_id not in inventory:
            exclusions.append(
                {
                    "kind": "question",
                    "question_id": example.question_id,
                    "source_video_id": segment.video_id,
                    "reason": "source_video_not_available",
                }
            )
            continue
        start, end, duration, unbounded = analyzed_bounds(example, inventory)
        if duration <= 0:
            exclusions.append({"kind": "question", "question_id": example.question_id, "reason": "non_positive_duration"})
            continue
        candidates.append(
            PrimaryCandidate(
                example=example,
                category=category,
                source_video_id=segment.video_id,
                participant_id=segment.participant_id,
                analyzed_start_seconds=start,
                analyzed_end_seconds=end,
                analyzed_duration_seconds=duration,
                is_unbounded_full_video=unbounded,
                duration_group=assign_duration_group(duration, thresholds),
            )
        )
    return candidates, exclusions


def choose_primary_per_video(candidates: list[PrimaryCandidate], seed: int) -> tuple[list[PrimaryCandidate], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_video: dict[str, list[PrimaryCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_video[candidate.source_video_id].append(candidate)
    selected: list[PrimaryCandidate] = []
    additional: list[dict[str, Any]] = []
    for video_id, items in sorted(by_video.items()):
        items.sort(
            key=lambda item: (
                item.category,
                item.duration_group,
                item.example.question_type,
                abs(question_token_length(item.example) - 12),
                rng.random(),
                item.question_id,
            )
        )
        selected.append(items[0])
        for item in items[1:]:
            additional_record = primary_manifest_record(item)
            additional_record["primary_question_id"] = items[0].question_id
            additional.append(additional_record)
    return selected, additional


def split_primary_videos(
    primaries: list[PrimaryCandidate],
    config: Experiment1V2Config,
) -> tuple[list[PrimaryCandidate], list[PrimaryCandidate]]:
    if not 0.0 < config.dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1.")
    rng = random.Random(config.seed)
    strata: dict[tuple[str, str], list[PrimaryCandidate]] = defaultdict(list)
    for item in primaries:
        strata[(item.category, item.duration_group)].append(item)
    dev: list[PrimaryCandidate] = []
    test: list[PrimaryCandidate] = []
    for key, items in sorted(strata.items()):
        items = sorted(items, key=lambda item: (item.participant_id, item.source_video_id, rng.random()))
        if len(items) == 1:
            test.extend(items)
            continue
        dev_count = max(1, int(round(len(items) * config.dev_fraction)))
        dev_count = min(dev_count, len(items) - 1)
        dev.extend(items[:dev_count])
        test.extend(items[dev_count:])
    test_counts = Counter(item.category for item in test)
    missing = [cat for cat in EXPERIMENT1_CATEGORIES if test_counts.get(cat, 0) < config.min_test_per_category]
    if missing:
        raise ValueError(f"Test split lacks required category coverage: {missing}")
    return sorted(dev, key=lambda item: item.question_id), sorted(test, key=lambda item: item.question_id)


def build_mismatched_queries(primaries: list[PrimaryCandidate], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[PrimaryCandidate]] = defaultdict(list)
    for item in primaries:
        by_stratum[(item.category, item.duration_group)].append(item)
    mapping: dict[str, dict[str, Any]] = {}
    for key, items in sorted(by_stratum.items()):
        if len(items) < 2:
            raise ValueError(f"Cannot derange mismatched queries for stratum {key}: only {len(items)} example(s).")
        ordered = sorted(items, key=lambda item: (question_token_length(item.example), item.source_video_id, item.question_id))
        donors = ordered[1:] + ordered[:1]
        for item, donor in zip(ordered, donors):
            if item.source_video_id == donor.source_video_id:
                alternatives = [alt for alt in ordered if alt.source_video_id != item.source_video_id]
                if not alternatives:
                    raise ValueError(f"No different-video mismatch exists for {item.question_id}.")
                donor = min(alternatives, key=lambda alt: (abs(question_token_length(alt.example) - question_token_length(item.example)), rng.random()))
            mapping[item.question_id] = {
                "mismatched_question_id": donor.question_id,
                "mismatched_source_video_id": donor.source_video_id,
                "category": item.category,
                "duration_group": item.duration_group,
                "question": donor.example.question,
                "choices": list(donor.example.choices),
                "correct_idx": donor.example.correct_idx,
                "token_length_difference": abs(question_token_length(donor.example) - question_token_length(item.example)),
            }
    return {"seed": seed, "mismatches": mapping}


def primary_manifest_record(candidate: PrimaryCandidate, split: str | None = None) -> dict[str, Any]:
    example = candidate.example
    segment = example.inputs[0]
    record = {
        "question_id": example.question_id,
        "question_type": example.question_type,
        "category": candidate.category,
        "question": example.question,
        "choices": list(example.choices),
        "correct_idx": example.correct_idx,
        "correct_answer": example.choices[example.correct_idx],
        "source_video_id": candidate.source_video_id,
        "participant_id": candidate.participant_id,
        "input_key": segment.input_key,
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "analyzed_start_seconds": candidate.analyzed_start_seconds,
        "analyzed_end_seconds": candidate.analyzed_end_seconds,
        "analyzed_duration_seconds": candidate.analyzed_duration_seconds,
        "duration_group": candidate.duration_group,
        "is_unbounded_full_video": candidate.is_unbounded_full_video,
        "video_clip": [
            {
                "input_key": segment.input_key,
                "video_id": segment.video_id,
                "participant_id": segment.participant_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "image_time_seconds": segment.image_time_seconds,
                "is_unbounded_full_video": candidate.is_unbounded_full_video,
                "raw": segment.raw,
            }
        ],
        "raw_metadata": {key: value for key, value in example.raw.items() if key in {"others", "stat", "metadata"}},
        "experiment": "experiment1_v2_temporal_attention",
    }
    if split is not None:
        record["split"] = split
    return record


def assert_v2_manifest_invariants(
    primary_records: list[dict[str, Any]],
    mismatches: dict[str, Any],
    inventory: list[VideoInventoryRecord],
) -> None:
    available = {record.video_id for record in inventory if record.complete}
    question_ids = [record["question_id"] for record in primary_records]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Primary manifest contains duplicate question IDs.")
    videos = [record["source_video_id"] for record in primary_records]
    if len(videos) != len(set(videos)):
        raise ValueError("Primary manifest contains duplicate source videos.")
    missing = sorted(set(videos) - available)
    if missing:
        raise ValueError(f"Primary manifest references unavailable videos: {missing}")
    split_by_video: dict[str, str] = {}
    for record in primary_records:
        previous = split_by_video.setdefault(record["source_video_id"], record["split"])
        if previous != record["split"]:
            raise ValueError(f"Source video crosses splits: {record['source_video_id']}")
    mismatch_map = mismatches.get("mismatches", {})
    for record in primary_records:
        mismatch = mismatch_map.get(record["question_id"])
        if mismatch is None:
            raise ValueError(f"Missing mismatch for {record['question_id']}.")
        if mismatch["mismatched_source_video_id"] == record["source_video_id"]:
            raise ValueError(f"Mismatch uses same source video for {record['question_id']}.")


def build_split_summary(
    config: Experiment1V2Config,
    inventory: list[VideoInventoryRecord],
    candidates: list[PrimaryCandidate],
    primary_records: list[dict[str, Any]],
    thresholds: dict[str, float],
    additional_questions: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_split = defaultdict(list)
    for record in primary_records:
        by_split[record["split"]].append(record)

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "num_examples": len(records),
            "num_source_videos": len({record["source_video_id"] for record in records}),
            "by_category": dict(sorted(Counter(record["category"] for record in records).items())),
            "by_duration_group": dict(sorted(Counter(record["duration_group"] for record in records).items())),
            "by_question_type": dict(sorted(Counter(record["question_type"] for record in records).items())),
            "by_participant": dict(sorted(Counter(record["participant_id"] for record in records).items())),
        }

    return {
        "experiment": "experiment1_v2_temporal_attention",
        "seed": config.seed,
        "dev_fraction": config.dev_fraction,
        "max_primary_per_source_video": config.max_primary_per_source_video,
        "duration_tertile_thresholds": thresholds,
        "inventory_complete_mp4s": len(inventory),
        "eligible_question_count_before_primary_video_cap": len(candidates),
        "primary_manifest_count": len(primary_records),
        "additional_question_count": len(additional_questions),
        "exclusion_count": len(exclusions),
        "splits": {name: summarize(records) for name, records in sorted(by_split.items())},
    }


def build_experiment1_v2_manifests(
    questions_dir: str | Path,
    mp4_dir: str | Path,
    config: Experiment1V2Config,
    ffprobe_runner: FFProbeRunner | None = None,
) -> dict[str, Any]:
    dataset = HDEpicVQADataset(questions_dir)
    inventory, video_exclusions = inventory_local_mp4s(mp4_dir, runner=ffprobe_runner)
    if not inventory:
        raise ValueError(f"No complete MP4 files were inventoried under {mp4_dir}.")
    inventory_map = {record.video_id: record for record in inventory}
    preliminary_durations = []
    for example in dataset.examples:
        if len(example.inputs) == 1 and not example.inputs[0].is_image and example.inputs[0].video_id in inventory_map:
            start, end, duration, _unbounded = analyzed_bounds(example, inventory_map)
            if end > start:
                preliminary_durations.append(duration)
    thresholds = duration_tertiles(preliminary_durations)
    candidates, question_exclusions = eligible_primary_candidates(dataset.examples, inventory, thresholds)
    primaries, additional_questions = choose_primary_per_video(candidates, seed=config.seed)
    dev, test = split_primary_videos(primaries, config)
    records = [primary_manifest_record(item, split="dev") for item in dev]
    records.extend(primary_manifest_record(item, split="test") for item in test)
    records.sort(key=lambda record: (record["split"], record["category"], record["duration_group"], record["question_id"]))
    mismatches = build_mismatched_queries(dev + test, seed=config.seed)
    exclusions = video_exclusions + question_exclusions
    assert_v2_manifest_invariants(records, mismatches, inventory)
    summary = build_split_summary(config, inventory, candidates, records, thresholds, additional_questions, exclusions)
    if summary["primary_manifest_count"] != len(records):
        raise ValueError("Summary count disagrees with primary manifest length.")
    return {
        "duration_inventory": [record.to_json() for record in inventory],
        "primary_manifest": records,
        "additional_questions": additional_questions,
        "mismatched_queries": mismatches,
        "split_summary": summary,
        "exclusions": exclusions,
    }
