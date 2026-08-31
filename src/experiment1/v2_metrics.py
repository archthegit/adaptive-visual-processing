from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


def normalize_distribution(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D distribution.")
    if np.any(arr < 0):
        raise ValueError("Distributions must be non-negative.")
    total = float(arr.sum())
    if total <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / total


def normalized_entropy(values: Sequence[float]) -> float:
    probs = normalize_distribution(values)
    if probs.size <= 1 or float(probs.sum()) <= 0:
        return 0.0
    nonzero = probs[probs > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / math.log(probs.size))


def gini_coefficient(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D vector.")
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini is defined here for non-negative values.")
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    sorted_values = np.sort(arr)
    n = arr.size
    weighted_sum = float(np.sum((np.arange(1, n + 1) * sorted_values)))
    return float((2.0 * weighted_sum) / (n * total) - (n + 1.0) / n)


def top_fraction_mass(values: Sequence[float], fraction: float = 0.2) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    probs = normalize_distribution(values)
    if probs.size == 0:
        return 0.0
    k = max(1, int(math.ceil(probs.size * fraction)))
    return float(np.sort(probs)[::-1][:k].sum())


def uniform_top_fraction_expectation(num_bins: int, fraction: float = 0.2) -> float:
    if num_bins <= 0:
        return 0.0
    k = max(1, int(math.ceil(num_bins * fraction)))
    return float(k / num_bins)


def bins_to_mass(values: Sequence[float], mass: float = 0.8) -> int:
    if not 0.0 < mass <= 1.0:
        raise ValueError("mass must be in (0, 1].")
    probs = normalize_distribution(values)
    if probs.size == 0 or float(probs.sum()) <= 0:
        return 0
    cumulative = np.cumsum(np.sort(probs)[::-1])
    return int(np.searchsorted(cumulative, mass, side="left") + 1)


def rank_order(values: Sequence[float]) -> tuple[int, ...]:
    arr = np.asarray(values, dtype=np.float64)
    return tuple(int(index) for index in np.lexsort((np.arange(arr.size), -arr)))


def _rank_positions(order: Sequence[int]) -> np.ndarray:
    ranks = np.zeros(len(order), dtype=np.float64)
    for rank, index in enumerate(order):
        ranks[int(index)] = float(rank)
    return ranks


def spearman_from_scores(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have the same length.")
    if len(left) <= 1:
        return 1.0
    left_ranks = _rank_positions(rank_order(left))
    right_ranks = _rank_positions(rank_order(right))
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    denom = float(np.sqrt((left_ranks * left_ranks).sum() * (right_ranks * right_ranks).sum()))
    return float((left_ranks * right_ranks).sum() / denom) if denom > 0 else 0.0


def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = normalize_distribution(left)
    q = normalize_distribution(right)
    if p.shape != q.shape:
        raise ValueError("JSD inputs must have the same shape.")
    if float(p.sum()) <= 0 and float(q.sum()) <= 0:
        return 0.0
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def top_fraction_jaccard(left: Sequence[float], right: Sequence[float], fraction: float = 0.2) -> float:
    if len(left) != len(right):
        raise ValueError("Jaccard inputs must have the same length.")
    if len(left) == 0:
        return 0.0
    k = max(1, int(math.ceil(len(left) * fraction)))
    left_set = set(rank_order(left)[:k])
    right_set = set(rank_order(right)[:k])
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 0.0


def temporal_distribution_metrics(values: Sequence[float], previous: Sequence[float] | None = None) -> dict[str, Any]:
    probs = normalize_distribution(values)
    top_bin = int(np.argmax(probs)) if probs.size else None
    return {
        "normalized_entropy": normalized_entropy(probs),
        "top_20pct_bin_mass": top_fraction_mass(probs, 0.2),
        "uniform_top_20pct_expectation": uniform_top_fraction_expectation(int(probs.size), 0.2),
        "gini": gini_coefficient(probs),
        "bins_to_80pct_mass": bins_to_mass(probs, 0.8),
        "first_bin_mass": float(probs[0]) if probs.size else 0.0,
        "last_bin_mass": float(probs[-1]) if probs.size else 0.0,
        "top_bin": top_bin,
        "top_bin_relative_temporal_position": None if top_bin is None or probs.size <= 1 else float(top_bin / (probs.size - 1)),
        "rank_order": list(rank_order(probs)),
        "layer_to_previous_jsd": None if previous is None else jensen_shannon_divergence(previous, probs),
        "layer_to_previous_spearman": None if previous is None else spearman_from_scores(previous, probs),
    }


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def lag_similarity_profile(representations: Sequence[Sequence[float]]) -> list[dict[str, float | int]]:
    reps = np.asarray(representations, dtype=np.float64)
    if reps.ndim != 2:
        raise ValueError("Representations must have shape [bins, dim].")
    output = []
    for lag in range(1, reps.shape[0]):
        sims = [cosine_similarity(reps[idx], reps[idx + lag]) for idx in range(reps.shape[0] - lag)]
        output.append({"lag": lag, "mean_cosine_similarity": float(np.mean(sims)) if sims else 0.0})
    return output


def adjacent_nonadjacent_far_similarity(representations: Sequence[Sequence[float]]) -> dict[str, float | None]:
    reps = np.asarray(representations, dtype=np.float64)
    if reps.ndim != 2:
        raise ValueError("Representations must have shape [bins, dim].")
    adjacent = []
    non_adjacent = []
    far = []
    max_lag = max(1, reps.shape[0] // 2)
    for left in range(reps.shape[0]):
        for right in range(left + 1, reps.shape[0]):
            sim = cosine_similarity(reps[left], reps[right])
            lag = right - left
            if lag == 1:
                adjacent.append(sim)
            else:
                non_adjacent.append(sim)
            if lag >= max_lag:
                far.append(sim)
    return {
        "adjacent_cosine_similarity": float(np.mean(adjacent)) if adjacent else None,
        "non_adjacent_cosine_similarity": float(np.mean(non_adjacent)) if non_adjacent else None,
        "far_bin_cosine_similarity": float(np.mean(far)) if far else None,
    }


def effective_rank(representations: Sequence[Sequence[float]]) -> float:
    reps = np.asarray(representations, dtype=np.float64)
    if reps.ndim != 2 or min(reps.shape) == 0:
        return 0.0
    centered = reps - reps.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if float(singular_values.sum()) <= 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    nonzero = probs[probs > 0]
    return float(np.exp(-(nonzero * np.log(nonzero)).sum()))


def paired_differences(records: Sequence[dict[str, Any]], value_key: str, pair_key: str = "source_video_id", condition_key: str = "condition") -> dict[str, Any]:
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        by_pair[str(record[pair_key])][str(record[condition_key])] = float(record[value_key])
    conditions = sorted({condition for values in by_pair.values() for condition in values})
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly two paired conditions, got {conditions}.")
    left, right = conditions
    diffs = [values[right] - values[left] for values in by_pair.values() if left in values and right in values]
    if not diffs:
        raise ValueError("No complete pairs found.")
    return {
        "left_condition": left,
        "right_condition": right,
        "num_pairs": len(diffs),
        "mean_difference": float(np.mean(diffs)),
        "differences": diffs,
    }


def paired_permutation_pvalue(differences: Sequence[float], replicates: int = 10000, seed: int = 20260830) -> float:
    diffs = [float(value) for value in differences if math.isfinite(float(value))]
    if not diffs:
        raise ValueError("No differences available for permutation test.")
    observed = abs(float(np.mean(diffs)))
    rng = random.Random(seed)
    extreme = 0
    for _rep in range(replicates):
        signed = [value if rng.random() < 0.5 else -value for value in diffs]
        if abs(float(np.mean(signed))) >= observed:
            extreme += 1
    return float((extreme + 1) / (replicates + 1))


def paired_effect_size(differences: Sequence[float]) -> float:
    diffs = np.asarray([float(value) for value in differences if math.isfinite(float(value))], dtype=np.float64)
    if diffs.size == 0:
        raise ValueError("No differences available for effect size.")
    std = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    return float(diffs.mean() / std) if std > 0 else 0.0


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    values = [float(value) for value in pvalues]
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: values[idx])
    adjusted = [1.0] * n
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, values[idx] * n / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_ci_clustered_by_video(
    records: Sequence[dict[str, Any]],
    value_key: str,
    participant_key: str = "participant_id",
    video_key: str = "source_video_id",
    replicates: int = 10000,
    seed: int = 20260830,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive.")
    by_participant: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_participant[str(record[participant_key])][str(record[video_key])].append(float(record[value_key]))
    participants = sorted(by_participant)
    if not participants:
        raise ValueError("No records available for bootstrap.")
    rng = random.Random(seed)
    estimates = []
    for _rep in range(replicates):
        values = []
        for _ in participants:
            participant = rng.choice(participants)
            videos = sorted(by_participant[participant])
            for _ in videos:
                video = rng.choice(videos)
                values.extend(by_participant[participant][video])
        estimates.append(float(np.mean(values)))
    lower, upper = np.percentile(np.asarray(estimates), [2.5, 97.5])
    observed_values = [float(record[value_key]) for record in records]
    return {
        "mean": float(np.mean(observed_values)),
        "ci95": [float(lower), float(upper)],
        "replicates": int(replicates),
        "seed": int(seed),
    }
