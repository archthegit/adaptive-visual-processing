from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.io import write_json

from .v2_metrics import jensen_shannon_divergence, top_fraction_mass


MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS = 0.05


@dataclass(frozen=True)
class ReferenceLayerScore:
    layer: int
    num_examples: int
    mean_correct_mismatch_jsd: float
    mean_absolute_visual_mass: float
    mean_top20_mass: float
    passes_absolute_visual_mass_threshold: bool


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def temporal_layers(artifact: dict[str, Any]) -> list[list[float]]:
    scores = (artifact.get("temporal_relevance") or {}).get("normalized_temporal_bin_scores")
    if not scores:
        raise ValueError(f"Artifact {artifact.get('question_id')} lacks normalized temporal scores.")
    return [[float(value) for value in layer] for layer in scores]


def absolute_visual_mass_by_layer(artifact: dict[str, Any]) -> list[float]:
    mass = (artifact.get("temporal_relevance") or {}).get("absolute_question_to_visual_attention_mass")
    if not mass:
        raise ValueError(f"Artifact {artifact.get('question_id')} lacks absolute visual mass.")
    return [float(value) for value in mass]


def common_dev_records(primary_manifest: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in primary_manifest if record.get("split") == "dev"]


def score_reference_layers(
    dev_records: list[dict[str, Any]],
    baseline_output_dir: str | Path,
    mismatched_output_dir: str | Path | None = None,
    min_absolute_visual_mass: float = MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS,
) -> list[ReferenceLayerScore]:
    if not dev_records:
        raise ValueError("Reference-layer selection requires at least one development example.")
    layer_jsd: dict[int, list[float]] = {}
    layer_mass: dict[int, list[float]] = {}
    layer_top20: dict[int, list[float]] = {}
    for record in dev_records:
        question_id = record["question_id"]
        baseline = load_artifact(Path(baseline_output_dir) / f"{question_id}.json")
        baseline_layers = temporal_layers(baseline)
        baseline_mass = absolute_visual_mass_by_layer(baseline)
        mismatch_layers = None
        if mismatched_output_dir is not None and (Path(mismatched_output_dir) / f"{question_id}.json").exists():
            mismatch_layers = temporal_layers(load_artifact(Path(mismatched_output_dir) / f"{question_id}.json"))
        for layer, values in enumerate(baseline_layers):
            layer_mass.setdefault(layer, []).append(baseline_mass[layer])
            layer_top20.setdefault(layer, []).append(top_fraction_mass(values, 0.2))
            if mismatch_layers is not None:
                layer_jsd.setdefault(layer, []).append(jensen_shannon_divergence(values, mismatch_layers[layer]))
            else:
                layer_jsd.setdefault(layer, []).append(0.0)
    scores = []
    for layer in sorted(layer_mass):
        mean_jsd = float(np.mean(layer_jsd[layer]))
        mean_mass = float(np.mean(layer_mass[layer]))
        mean_top20 = float(np.mean(layer_top20[layer]))
        scores.append(
            ReferenceLayerScore(
                layer=layer,
                num_examples=len(layer_mass[layer]),
                mean_correct_mismatch_jsd=mean_jsd,
                mean_absolute_visual_mass=mean_mass,
                mean_top20_mass=mean_top20,
                passes_absolute_visual_mass_threshold=mean_mass >= min_absolute_visual_mass,
            )
        )
    return scores


def select_reference_layer(
    scores: list[ReferenceLayerScore],
    min_absolute_visual_mass: float = MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS,
) -> ReferenceLayerScore:
    if not scores:
        raise ValueError("No reference-layer scores available.")
    eligible = [score for score in scores if score.mean_absolute_visual_mass >= min_absolute_visual_mass]
    if not eligible:
        raise ValueError(
            "No decoder layer passed the frozen minimum absolute visual mass threshold "
            f"({min_absolute_visual_mass})."
        )
    return max(
        eligible,
        key=lambda item: (
            item.mean_correct_mismatch_jsd,
            item.mean_absolute_visual_mass,
            item.mean_top20_mass,
            -item.layer,
        ),
    )


def write_frozen_reference_layer(
    primary_manifest_path: str | Path,
    baseline_output_dir: str | Path,
    output_json: str | Path,
    mismatched_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    primary = load_manifest_records(primary_manifest_path)
    dev = common_dev_records(primary)
    scores = score_reference_layers(
        dev,
        baseline_output_dir,
        mismatched_output_dir=mismatched_output_dir,
        min_absolute_visual_mass=MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS,
    )
    selected = select_reference_layer(scores, min_absolute_visual_mass=MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS)
    payload = {
        "selected_layer": selected.layer,
        "minimum_absolute_visual_mass_threshold": MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS,
        "selection_rule": [
            "annotation temporal alignment is unavailable, so no ground-truth temporal evidence term is used",
            f"exclude decoder layers with mean absolute visual mass below {MIN_REFERENCE_LAYER_ABSOLUTE_VISUAL_MASS}",
            "maximize correct-query versus mismatched-query temporal Jensen-Shannon divergence",
            "break ties by higher mean absolute visual mass",
            "then higher top-20% temporal mass",
            "then shallower decoder layer index",
        ],
        "annotation_alignment_available": False,
        "scores": [asdict(score) for score in scores],
        "selected": asdict(selected),
    }
    write_json(output_json, payload)
    return payload
