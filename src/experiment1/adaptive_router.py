from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROUTER_SCHEMA_VERSION = "adaptive_temporal_compaction_router_v1"


@dataclass(frozen=True)
class RouterDataset:
    rows: list[dict[str, Any]]
    feature_names: list[str]
    features: np.ndarray
    safe: np.ndarray
    logp_delta: np.ndarray
    margin_delta: np.ndarray
    flop_reduction: np.ndarray


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_manifest(path: str | Path, split: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in read_jsonl(path):
        row = dict(record)
        row.setdefault("split", split)
        output[str(row["question_id"])] = row
    return output


def assert_source_disjoint(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> None:
    left_sources = {str(row.get("source_video_id")) for row in left.values()}
    right_sources = {str(row.get("source_video_id")) for row in right.values()}
    overlap = sorted(left_sources & right_sources)
    if overlap:
        raise ValueError(f"Manifest source-video overlap is not allowed: {overlap[:10]}")


def assert_all_sources_disjoint(manifests: dict[str, dict[str, dict[str, Any]]]) -> None:
    splits = sorted(manifests)
    for idx, left_name in enumerate(splits):
        for right_name in splits[idx + 1 :]:
            try:
                assert_source_disjoint(manifests[left_name], manifests[right_name])
            except ValueError as exc:
                raise ValueError(f"{left_name}/{right_name} {exc}") from exc


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "min", "max")}
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_min": float(np.min(flat)),
        f"{prefix}_max": float(np.max(flat)),
    }


def _vector_stats(values: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    if values.size == 0:
        raise ValueError(f"{prefix} values are empty.")
    return {
        f"{prefix}_mean": np.mean(values, axis=0),
        f"{prefix}_std": np.std(values, axis=0),
        f"{prefix}_min": np.min(values, axis=0),
        f"{prefix}_max": np.max(values, axis=0),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * max(float(np.linalg.norm(b)), 1e-12)
    return np.sum(a * b[None, :], axis=-1) / np.maximum(denom, 1e-12)


def route_feature_dict(feature_payload: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    cell_ids = [int(item) for item in feature_payload["native_temporal_cell_ids"]]
    cell_to_row = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    retained = [int(item) for item in action["retained_cell_ids"]]
    if len(set(retained)) != len(retained):
        raise ValueError(f"Duplicate retained cells for {action.get('action_id')}.")
    if any(cell not in cell_to_row for cell in retained):
        raise ValueError(f"Invalid retained cells for {action.get('action_id')}: {retained}")
    dropped = [cell for cell in cell_ids if cell not in set(retained)]
    if not dropped:
        raise ValueError("Router training expects at least one dropped cell.")

    cell_mean = _tensor_to_numpy(feature_payload["cell_mean_residual"])
    question = _tensor_to_numpy(feature_payload["question_mean_residual"]).reshape(-1)
    norms = np.asarray(feature_payload["cell_residual_norm"], dtype=np.float64)
    dispersion = np.asarray(feature_payload["cell_residual_dispersion"], dtype=np.float64)
    token_counts = np.asarray(feature_payload["token_count_per_cell"], dtype=np.float64)
    retained_rows = np.asarray([cell_to_row[cell] for cell in retained], dtype=np.int64)
    dropped_rows = np.asarray([cell_to_row[cell] for cell in dropped], dtype=np.int64)
    retained_vecs = cell_mean[retained_rows]
    dropped_vecs = cell_mean[dropped_rows]
    retained_mean = np.mean(retained_vecs, axis=0)
    dropped_mean = np.mean(dropped_vecs, axis=0)
    similarities = _cosine(cell_mean, question)

    n_cells = len(cell_ids)
    denom = max(1, n_cells - 1)
    retained_pos = np.asarray([cell / denom for cell in retained], dtype=np.float64)
    retained_sorted = sorted(retained)
    gaps = np.diff(retained_sorted) if len(retained_sorted) > 1 else np.asarray([0], dtype=np.float64)
    coverage_span = (max(retained_sorted) - min(retained_sorted) + 1) / float(n_cells)

    vectors = {
        "question": question,
        **_vector_stats(retained_vecs, "retained_cell_residual"),
        **_vector_stats(dropped_vecs, "dropped_cell_residual"),
        "retained_minus_dropped_mean": retained_mean - dropped_mean,
        "abs_retained_minus_dropped_mean": np.abs(retained_mean - dropped_mean),
    }
    scalars: dict[str, float] = {
        "frame_count": float(feature_payload["frame_count"]),
        "compaction_layer": float(action["compaction_layer"]),
        "retention_fraction": float(action["retention_fraction"]),
        "native_temporal_cell_count": float(n_cells),
        "retained_cell_count": float(len(retained)),
        "dropped_cell_count": float(len(dropped)),
        "temporal_coverage_span": float(coverage_span),
        "retained_position_mean": float(np.mean(retained_pos)),
        "retained_position_std": float(np.std(retained_pos)),
        "retained_position_min": float(np.min(retained_pos)),
        "retained_position_max": float(np.max(retained_pos)),
        "retained_contains_first_cell": float(0 in retained),
        "retained_contains_last_cell": float((n_cells - 1) in retained),
        "retained_gap_mean": float(np.mean(gaps)),
        "retained_gap_max": float(np.max(gaps)),
    }
    for prefix, indices in (("retained", retained_rows), ("dropped", dropped_rows)):
        scalars.update(_stats(norms[indices], f"{prefix}_residual_norm"))
        scalars.update(_stats(dispersion[indices], f"{prefix}_residual_dispersion"))
        scalars.update(_stats(token_counts[indices], f"{prefix}_token_count"))
        scalars.update(_stats(similarities[indices], f"{prefix}_question_similarity"))
    scalars.update(_stats(norms[retained_rows] - np.mean(norms[dropped_rows]), "retained_norm_minus_dropped_mean"))
    scalars.update(_stats(similarities[retained_rows] - np.mean(similarities[dropped_rows]), "retained_similarity_minus_dropped_mean"))

    return {"vectors": vectors, "scalars": scalars}


def flatten_feature_dict(features: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    for name, vector in sorted(features["vectors"].items()):
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
        for idx, value in enumerate(arr):
            names.append(f"{name}_{idx}")
            values.append(float(value))
    for name, value in sorted(features["scalars"].items()):
        names.append(name)
        values.append(float(value))
    return names, np.asarray(values, dtype=np.float64)


def _artifact_paths(label_dir: Path) -> Iterable[Path]:
    yield from sorted((label_dir / "artifacts").glob("*/frames_*/*.json"))


def build_router_dataset(
    *,
    label_dir: str | Path,
    train_manifest: str | Path,
    development_manifest: str | Path,
    test_manifest: str | Path | None = None,
    allow_test: bool = False,
    output_dir: str | Path,
    git_commit: str,
) -> RouterDataset:
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    if test_manifest is not None and not allow_test:
        raise RuntimeError("Pass allow_test=True only for final frozen test-dataset construction.")
    manifests = {
        "train": load_manifest(train_manifest, "train"),
        "development": load_manifest(development_manifest, "development"),
    }
    if test_manifest is not None:
        manifests["test"] = load_manifest(test_manifest, "test")
    assert_all_sources_disjoint(manifests)
    manifest_by_qid = {qid: row for split_rows in manifests.values() for qid, row in split_rows.items()}
    split_by_qid = {qid: split for split, split_rows in manifests.items() for qid in split_rows}

    dense_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []
    for path in _artifact_paths(label_dir):
        artifact = json.loads(path.read_text())
        if artifact.get("status") != "complete":
            continue
        qid = str(artifact.get("question_id"))
        if qid not in manifest_by_qid:
            continue
        if artifact.get("condition") == "dense_custom":
            dense_by_key[(qid, int(artifact["frame_count"]))] = artifact
        elif artifact.get("condition") == "hard_evict":
            artifact["_artifact_path"] = str(path)
            actions.append(artifact)

    rows: list[dict[str, Any]] = []
    feature_arrays: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for action in sorted(actions, key=lambda row: (row["question_id"], int(row["frame_count"]), row["action_id"])):
        qid = str(action["question_id"])
        frame_count = int(action["frame_count"])
        dense = dense_by_key.get((qid, frame_count))
        if dense is None:
            raise ValueError(f"Missing dense artifact for {qid}/frames={frame_count}.")
        if int(action["native_temporal_cell_count"]) != int(dense["native_temporal_cell_count"]):
            raise ValueError(f"Native-cell count mismatch for {qid}/{action['action_id']}.")
        layer = int(action["compaction_layer"])
        feature_meta = {int(item["layer"]): item for item in dense.get("router_features", [])}.get(layer)
        if feature_meta is None:
            raise ValueError(f"Missing dense feature tensor for {qid}/frames={frame_count}/layer={layer}.")
        feature_path = Path(feature_meta["feature_file"])
        if not feature_path.exists() and not feature_path.is_absolute():
            feature_path = label_dir / feature_path
        payload = _torch_load(feature_path)
        if str(payload.get("frame_count")) != str(frame_count) or int(payload.get("layer")) != layer:
            raise ValueError(f"Feature tensor metadata mismatch for {feature_path}.")
        if len(payload.get("native_temporal_cell_ids", [])) != int(action["native_temporal_cell_count"]):
            raise ValueError(f"Feature native-cell count mismatch for {feature_path}.")
        feature_dict = route_feature_dict(payload, action)
        names, vector = flatten_feature_dict(feature_dict)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names changed across rows; hidden dimensions must be consistent.")
        if not np.isfinite(vector).all():
            raise ValueError(f"Nonfinite features for {qid}/{action['action_id']}.")
        logp_delta = float(action["delta_correct_choice_log_probability_from_dense"])
        margin_delta = float(action["delta_answer_margin_from_dense"])
        flop = float(action["paired_flop_reduction_from_dense"])
        if not all(math.isfinite(value) for value in (logp_delta, margin_delta, flop)):
            raise ValueError(f"Nonfinite labels for {qid}/{action['action_id']}.")
        safe = bool(logp_delta >= -0.10 and not bool(action.get("prediction_changed")))
        condition_correct = bool(action.get("correct", action.get("condition_correct", False)))
        dense_correct = bool(dense.get("correct", dense.get("dense_correct", False)))
        manifest = manifest_by_qid[qid]
        row = {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "question_id": qid,
            "source_video_id": manifest.get("source_video_id", action.get("source_video_id")),
            "split": split_by_qid[qid],
            "category": manifest.get("category"),
            "question_type": manifest.get("question_type"),
            "frame_count": frame_count,
            "compaction_layer": layer,
            "retention_fraction": float(action["retention_fraction"]),
            "native_temporal_cell_count": int(action["native_temporal_cell_count"]),
            "retained_cell_ids": list(action["retained_cell_ids"]),
            "route_family": action.get("route_family"),
            "action_id": action.get("action_id"),
            "safe": safe,
            "delta_correct_choice_log_probability_from_dense": logp_delta,
            "delta_answer_margin_from_dense": margin_delta,
            "condition_correct": condition_correct,
            "dense_correct": dense_correct,
            "accuracy_delta_from_dense": float(int(condition_correct) - int(dense_correct)),
            "prediction_changed": bool(action.get("prediction_changed")),
            "paired_flop_reduction_from_dense": flop,
            "artifact_path": action["_artifact_path"],
            "feature_file": str(feature_path),
        }
        rows.append(row)
        feature_arrays.append(vector)

    if not rows:
        raise ValueError("No completed router action rows found.")
    X = np.vstack(feature_arrays)
    y_safe = np.asarray([int(row["safe"]) for row in rows], dtype=np.int64)
    y_logp = np.asarray([float(row["delta_correct_choice_log_probability_from_dense"]) for row in rows], dtype=np.float64)
    y_margin = np.asarray([float(row["delta_answer_margin_from_dense"]) for row in rows], dtype=np.float64)
    flops = np.asarray([float(row["paired_flop_reduction_from_dense"]) for row in rows], dtype=np.float64)
    dataset = RouterDataset(rows, feature_names or [], X, y_safe, y_logp, y_margin, flops)
    write_router_dataset(dataset, output_dir, label_dir=label_dir, git_commit=git_commit)
    return dataset


def write_router_dataset(dataset: RouterDataset, output_dir: str | Path, *, label_dir: Path, git_commit: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "router_rows.jsonl", dataset.rows)
    np.savez_compressed(
        output_dir / "router_features.npz",
        X=dataset.features,
        y_safe=dataset.safe,
        y_logp_delta=dataset.logp_delta,
        y_margin_delta=dataset.margin_delta,
        flop_reduction=dataset.flop_reduction,
        question_id=np.asarray([row["question_id"] for row in dataset.rows], dtype=object),
        split=np.asarray([row["split"] for row in dataset.rows], dtype=object),
    )
    write_json(
        output_dir / "metadata.json",
        {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "git_commit": git_commit,
            "label_dir": str(label_dir),
            "num_rows": len(dataset.rows),
            "num_features": len(dataset.feature_names),
            "feature_names": dataset.feature_names,
            "rows_by_split": dict(Counter(row["split"] for row in dataset.rows)),
            "safe_rate_by_split": {
                split: float(np.mean([row["safe"] for row in dataset.rows if row["split"] == split]))
                for split in sorted({row["split"] for row in dataset.rows})
            },
            "safety_label": "delta_correct_choice_log_probability_from_dense >= -0.10 and prediction_changed == false",
        },
    )


def load_router_dataset(dataset_dir: str | Path) -> RouterDataset:
    dataset_dir = Path(dataset_dir)
    rows = read_jsonl(dataset_dir / "router_rows.jsonl")
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    arrays = np.load(dataset_dir / "router_features.npz", allow_pickle=True)
    return RouterDataset(
        rows=rows,
        feature_names=list(metadata["feature_names"]),
        features=np.asarray(arrays["X"], dtype=np.float64),
        safe=np.asarray(arrays["y_safe"], dtype=np.int64),
        logp_delta=np.asarray(arrays["y_logp_delta"], dtype=np.float64),
        margin_delta=np.asarray(arrays["y_margin_delta"], dtype=np.float64),
        flop_reduction=np.asarray(arrays["flop_reduction"], dtype=np.float64),
    )


def bootstrap_ci(values: Sequence[float], *, samples: int = 10000, seed: int = 1) -> dict[str, Any]:
    clean = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "median": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(clean[rng.integers(0, clean.size, size=clean.size)])) for _ in range(samples)]
    return {
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n": int(clean.size),
    }


def train_router_models(dataset_dir: str | Path, output_dir: str | Path, *, seed: int, git_commit: str) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to train adaptive router models.") from exc

    dataset = load_router_dataset(dataset_dir)
    train_idx = np.asarray([row["split"] == "train" for row in dataset.rows], dtype=bool)
    if not train_idx.any():
        raise ValueError("No train rows available.")
    X = dataset.features[train_idx]
    y = dataset.safe[train_idx]
    if len(set(y.tolist())) < 2:
        raise ValueError("Safety classifier requires both safe and unsafe training examples.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    min_class = min(Counter(y.tolist()).values())
    if min_class >= 3:
        logistic = CalibratedClassifierCV(base_logistic, cv=min(3, min_class), method="sigmoid")
    else:
        logistic = base_logistic
    models = {
        "logistic_regression": logistic,
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed, early_stopping=False),
        ),
        "gradient_boosted_tree": HistGradientBoostingClassifier(random_state=seed, max_iter=100),
    }
    regressors = {
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)),
        "gradient_boosted_tree": HistGradientBoostingRegressor(random_state=seed, max_iter=100),
    }
    model_index = {
        "schema_version": ROUTER_SCHEMA_VERSION,
        "git_commit": git_commit,
        "dataset_dir": str(dataset_dir),
        "seed": seed,
        "safety_models": {},
        "quality_delta_regressors": {},
        "primary_safety_model": "logistic_regression",
        "primary_regressor": "ridge",
    }
    for name, model in models.items():
        model.fit(X, y)
        path = output_dir / f"safety_{name}.joblib"
        joblib.dump(model, path)
        model_index["safety_models"][name] = str(path)
    for name, model in regressors.items():
        model.fit(X, dataset.logp_delta[train_idx])
        path = output_dir / f"logp_delta_{name}.joblib"
        joblib.dump(model, path)
        model_index["quality_delta_regressors"][name] = str(path)
    write_json(output_dir / "model_index.json", model_index)
    return model_index


def _predict_safe_probability(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.ones(X.shape[0]) * float(model.classes_[0] == 1)
        class_to_col = {int(cls): idx for idx, cls in enumerate(model.classes_)}
        return proba[:, class_to_col[1]]
    return np.asarray(model.predict(X), dtype=np.float64)


def select_fixed_policy_training_only(rows: Sequence[dict[str, Any]]) -> tuple[int, float]:
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("Cannot select fixed baseline without training rows.")
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        grouped[(int(row["compaction_layer"]), float(row["retention_fraction"]))].append(row)
    scored = []
    for key, items in grouped.items():
        unsafe = np.mean([not bool(row["safe"]) for row in items])
        flop = np.mean([float(row["paired_flop_reduction_from_dense"]) for row in items])
        scored.append((unsafe <= 0.10, -unsafe, flop, -key[0], -key[1], key))
    return max(scored)[-1]


def _choose_row(candidates: list[dict[str, Any]], model_scores: dict[str, np.ndarray] | None = None, threshold: float = 0.10) -> dict[str, Any] | None:
    if model_scores is None:
        safe = [row for row in candidates if row["safe"]]
        return max(safe, key=lambda row: (float(row["paired_flop_reduction_from_dense"]), float(row["delta_correct_choice_log_probability_from_dense"])), default=None)
    safe_indices = [idx for idx, prob in enumerate(model_scores["unsafe_probability"]) if float(prob) <= threshold]
    if not safe_indices:
        return None
    return max(
        (candidates[idx] for idx in safe_indices),
        key=lambda row: (float(row["paired_flop_reduction_from_dense"]), float(row.get("_predicted_logp_delta", 0.0))),
    )


def evaluate_router(
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    allow_test: bool = False,
    unsafe_probability_threshold: float = 0.10,
    bootstrap_samples: int = 10000,
    seed: int = 1,
) -> dict[str, Any]:
    if split == "test" and not allow_test:
        raise RuntimeError("Pass --allow-test to evaluate the frozen router on test.")
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn/joblib to evaluate adaptive router models.") from exc
    dataset = load_router_dataset(dataset_dir)
    model_index = json.loads((Path(model_dir) / "model_index.json").read_text())
    safety_model = joblib.load(model_index["safety_models"][model_index["primary_safety_model"]])
    regressor = joblib.load(model_index["quality_delta_regressors"][model_index["primary_regressor"]])
    safe_prob = _predict_safe_probability(safety_model, dataset.features)
    pred_logp = np.asarray(regressor.predict(dataset.features), dtype=np.float64)
    rows = [dict(row, _row_index=idx, _predicted_logp_delta=float(pred_logp[idx])) for idx, row in enumerate(dataset.rows)]
    fixed_layer, fixed_fraction = select_fixed_policy_training_only(rows)

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_question[row["question_id"]].append(row)
    decisions: list[dict[str, Any]] = []
    for qid, candidates in sorted(by_question.items()):
        for baseline in ("dense", "fixed_training_policy", "uniform", "deterministic_random", "learned_router", "oracle_safe"):
            chosen: dict[str, Any] | None
            if baseline == "dense":
                chosen = None
            elif baseline == "fixed_training_policy":
                subset = [row for row in candidates if int(row["compaction_layer"]) == fixed_layer and float(row["retention_fraction"]) == fixed_fraction]
                chosen = max(subset, key=lambda row: float(row["paired_flop_reduction_from_dense"]), default=None)
            elif baseline == "uniform":
                subset = [row for row in candidates if row.get("route_family") == "uniform_temporal_coverage"]
                chosen = max(subset, key=lambda row: float(row["paired_flop_reduction_from_dense"]), default=None)
            elif baseline == "deterministic_random":
                subset = [row for row in candidates if row.get("route_family") == "deterministic_random_subset"]
                chosen = sorted(subset, key=lambda row: str(row["action_id"]))[0] if subset else None
            elif baseline == "learned_router":
                indices = [int(row["_row_index"]) for row in candidates]
                scores = {
                    "unsafe_probability": 1.0 - safe_prob[indices],
                }
                chosen = _choose_row(candidates, scores, threshold=unsafe_probability_threshold)
            else:
                chosen = _choose_row(candidates)
            if chosen is None:
                template = candidates[0]
                decisions.append({
                    "question_id": qid,
                    "source_video_id": template.get("source_video_id"),
                    "split": split,
                    "baseline": baseline,
                    "compacted": False,
                    "delta_logp": 0.0,
                    "delta_margin": 0.0,
                    "accuracy_delta_from_dense": 0.0,
                    "prediction_changed": False,
                    "safe": True,
                    "flop_reduction": 0.0,
                    "frame_count": None,
                    "category": template.get("category"),
                })
            else:
                decisions.append({
                    "question_id": qid,
                    "source_video_id": chosen.get("source_video_id"),
                    "split": split,
                    "baseline": baseline,
                    "compacted": True,
                    "action_id": chosen.get("action_id"),
                    "delta_logp": float(chosen["delta_correct_choice_log_probability_from_dense"]),
                    "delta_margin": float(chosen["delta_answer_margin_from_dense"]),
                    "accuracy_delta_from_dense": float(chosen.get("accuracy_delta_from_dense", 0.0)),
                    "prediction_changed": bool(chosen["prediction_changed"]),
                    "safe": bool(chosen["safe"]),
                    "flop_reduction": float(chosen["paired_flop_reduction_from_dense"]),
                    "frame_count": int(chosen["frame_count"]),
                    "compaction_layer": int(chosen["compaction_layer"]),
                    "retention_fraction": float(chosen["retention_fraction"]),
                    "category": chosen.get("category"),
                })

    summary = summarize_router_decisions(decisions, bootstrap_samples=bootstrap_samples, seed=seed)
    output_dir = Path(output_dir)
    write_jsonl(output_dir / "decisions.jsonl", decisions)
    write_json(output_dir / "summary.json", {
        "schema_version": ROUTER_SCHEMA_VERSION,
        "split": split,
        "unsafe_probability_threshold": unsafe_probability_threshold,
        "fixed_policy_selected_on_train_only": {"compaction_layer": fixed_layer, "retention_fraction": fixed_fraction},
        "metrics": summary,
    })
    write_csv(output_dir / "decisions.csv", decisions)
    return summary


def summarize_router_decisions(decisions: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for baseline in sorted({row["baseline"] for row in decisions}):
        rows = [row for row in decisions if row["baseline"] == baseline]
        output[baseline] = {
            "num_examples": len(rows),
            "coverage": float(np.mean([bool(row["compacted"]) for row in rows])) if rows else None,
            "unsafe_selection_rate": float(np.mean([not bool(row["safe"]) for row in rows])) if rows else None,
            "prediction_flip_rate": float(np.mean([bool(row["prediction_changed"]) for row in rows])) if rows else None,
            "mean_logp_delta": bootstrap_ci([row["delta_logp"] for row in rows], samples=bootstrap_samples, seed=seed),
            "mean_margin_delta": bootstrap_ci([row["delta_margin"] for row in rows], samples=bootstrap_samples, seed=seed + 1),
            "mean_flop_reduction": bootstrap_ci([row["flop_reduction"] for row in rows], samples=bootstrap_samples, seed=seed + 2),
            "accuracy_difference_from_dense": float(np.mean([float(row.get("accuracy_delta_from_dense", 0.0)) for row in rows])) if rows else None,
            "by_frame_count": _group_metric(rows, "frame_count"),
            "by_category": _group_metric(rows, "category"),
            "by_selected_layer": _group_metric(rows, "compaction_layer"),
            "by_retention_fraction": _group_metric(rows, "retention_fraction"),
        }
    if "oracle_safe" in output and "learned_router" in output:
        output["oracle_gap_learned_minus_oracle_flop_reduction"] = (
            output["learned_router"]["mean_flop_reduction"]["mean"] - output["oracle_safe"]["mean_flop_reduction"]["mean"]
        )
    return output


def _group_metric(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            grouped[str(value)].append(row)
    return {
        value: {
            "n": len(items),
            "mean_logp_delta": float(np.mean([row["delta_logp"] for row in items])),
            "mean_flop_reduction": float(np.mean([row["flop_reduction"] for row in items])),
            "unsafe_selection_rate": float(np.mean([not bool(row["safe"]) for row in items])),
        }
        for value, items in sorted(grouped.items())
    }


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
