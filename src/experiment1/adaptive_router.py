from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROUTER_SCHEMA_VERSION = "adaptive_temporal_compaction_router_v2"
SAFETY_DEFINITION = "delta_correct_choice_log_probability_from_dense >= -0.10 and prediction_changed == false"
DEFAULT_LAYER_ORDER = [4, 8, 12, 16, 20]


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_manifest(path: str | Path, split: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in read_jsonl(path):
        row = dict(record)
        row.setdefault("split", split)
        output[str(row["question_id"])] = row
    return output


def assert_all_sources_disjoint(manifests: dict[str, dict[str, dict[str, Any]]]) -> None:
    split_names = sorted(manifests)
    for left_idx, left_name in enumerate(split_names):
        left = {str(row.get("source_video_id")) for row in manifests[left_name].values()}
        for right_name in split_names[left_idx + 1 :]:
            right = {str(row.get("source_video_id")) for row in manifests[right_name].values()}
            overlap = sorted(left & right)
            if overlap:
                raise ValueError(f"{left_name}/{right_name} source-video overlap is not allowed: {overlap[:10]}")


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "min", "max")}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * max(float(np.linalg.norm(b)), 1e-12)
    return np.sum(a * b[None, :], axis=-1) / np.maximum(denom, 1e-12)


def route_feature_dict(feature_payload: dict[str, Any], action: dict[str, Any]) -> dict[str, float]:
    """Build compact permutation-aware route features.

    The representation intentionally avoids serializing hidden-size vectors per
    action. It uses only pooled norms, dispersions, similarities, contrasts, and
    route geometry, so the complete action grid remains memory-safe.
    """

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
    norms = np.asarray(feature_payload["cell_residual_norm"], dtype=np.float32)
    dispersion = np.asarray(feature_payload["cell_residual_dispersion"], dtype=np.float32)
    token_counts = np.asarray(feature_payload["token_count_per_cell"], dtype=np.float32)
    if not np.isfinite(cell_mean).all() or not np.isfinite(question).all():
        raise ValueError("Nonfinite router feature tensor.")
    if len(cell_ids) != cell_mean.shape[0] or len(cell_ids) != len(norms) or len(cell_ids) != len(dispersion):
        raise ValueError("Router feature cell dimensions are inconsistent.")

    retained_rows = np.asarray([cell_to_row[cell] for cell in retained], dtype=np.int64)
    dropped_rows = np.asarray([cell_to_row[cell] for cell in dropped], dtype=np.int64)
    similarities = _cosine(cell_mean, question).astype(np.float32)
    retained_mean = np.mean(cell_mean[retained_rows], axis=0)
    dropped_mean = np.mean(cell_mean[dropped_rows], axis=0)
    contrast = retained_mean - dropped_mean

    n_cells = len(cell_ids)
    denom = max(1, n_cells - 1)
    retained_pos = np.asarray([cell / denom for cell in retained], dtype=np.float32)
    retained_sorted = sorted(retained)
    gaps = np.diff(retained_sorted).astype(np.float32) if len(retained_sorted) > 1 else np.asarray([0.0], dtype=np.float32)

    features: dict[str, float] = {
        "question_norm": float(np.linalg.norm(question)),
        "question_mean": float(np.mean(question)),
        "question_std": float(np.std(question)),
        "question_min": float(np.min(question)),
        "question_max": float(np.max(question)),
        "retained_minus_dropped_residual_norm": float(np.linalg.norm(contrast)),
        "retained_minus_dropped_residual_mean": float(np.mean(contrast)),
        "retained_minus_dropped_residual_std": float(np.std(contrast)),
        "abs_retained_minus_dropped_residual_mean": float(np.mean(np.abs(contrast))),
        "frame_count": float(feature_payload["frame_count"]),
        "compaction_layer": float(action["compaction_layer"]),
        "retention_fraction": float(action["retention_fraction"]),
        "native_temporal_cell_count": float(n_cells),
        "retained_cell_count": float(len(retained)),
        "dropped_cell_count": float(len(dropped)),
        "temporal_coverage_span": float((max(retained_sorted) - min(retained_sorted) + 1) / n_cells),
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
        features.update(_stats(norms[indices], f"{prefix}_residual_norm"))
        features.update(_stats(dispersion[indices], f"{prefix}_residual_dispersion"))
        features.update(_stats(token_counts[indices], f"{prefix}_token_count"))
        features.update(_stats(similarities[indices], f"{prefix}_question_similarity"))
    features.update(_stats(norms[retained_rows] - float(np.mean(norms[dropped_rows])), "retained_norm_minus_dropped_mean"))
    features.update(_stats(similarities[retained_rows] - float(np.mean(similarities[dropped_rows])), "retained_similarity_minus_dropped_mean"))
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Nonfinite compact router features.")
    return features


def flatten_feature_dict(features: dict[str, float]) -> tuple[list[str], np.ndarray]:
    names = sorted(features)
    values = np.asarray([features[name] for name in names], dtype=np.float32)
    return names, values


def _artifact_paths(label_dir: Path) -> Iterable[Path]:
    yield from sorted((label_dir / "artifacts").glob("*/frames_*/*.json"))


def _load_split_artifacts(label_dir: Path, split: str, manifest_by_qid: dict[str, dict[str, Any]]) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    dense_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []
    for path in _artifact_paths(label_dir):
        artifact = json.loads(path.read_text())
        if artifact.get("status") != "complete":
            continue
        qid = str(artifact.get("question_id"))
        if qid not in manifest_by_qid:
            continue
        artifact_split = str(artifact.get("split", split))
        if artifact_split != split:
            raise ValueError(f"Artifact {path} has split={artifact_split}; expected {split}.")
        artifact["_artifact_path"] = str(path)
        if artifact.get("condition") == "dense_custom":
            dense_by_key[(qid, int(artifact["frame_count"]))] = artifact
        elif artifact.get("condition") == "hard_evict":
            actions.append(artifact)
    return dense_by_key, actions


def build_router_dataset(
    *,
    train_label_dir: str | Path | None = None,
    development_label_dir: str | Path | None = None,
    test_label_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
    train_manifest: str | Path,
    development_manifest: str | Path,
    test_manifest: str | Path | None = None,
    allow_test: bool = False,
    output_dir: str | Path,
    git_commit: str,
) -> RouterDataset:
    if label_dir is not None:
        train_label_dir = train_label_dir or label_dir
        development_label_dir = development_label_dir or label_dir
    if train_label_dir is None or development_label_dir is None:
        raise ValueError("Both train_label_dir and development_label_dir are required.")
    if test_label_dir is not None and (test_manifest is None or not allow_test):
        raise RuntimeError("Pass both test_manifest and allow_test=True only for final frozen test-dataset construction.")

    manifests = {
        "train": load_manifest(train_manifest, "train"),
        "development": load_manifest(development_manifest, "development"),
    }
    manifest_paths = {"train": str(train_manifest), "development": str(development_manifest)}
    label_dirs = {"train": Path(train_label_dir), "development": Path(development_label_dir)}
    if test_label_dir is not None and test_manifest is not None:
        manifests["test"] = load_manifest(test_manifest, "test")
        manifest_paths["test"] = str(test_manifest)
        label_dirs["test"] = Path(test_label_dir)
    assert_all_sources_disjoint(manifests)

    manifest_by_qid = {qid: row for split_rows in manifests.values() for qid, row in split_rows.items()}
    split_by_qid = {qid: split for split, split_rows in manifests.items() for qid in split_rows}
    dense_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []
    split_label_dirs: dict[str, str] = {}
    for split, split_dir in label_dirs.items():
        split_dense, split_actions = _load_split_artifacts(split_dir, split, manifests[split])
        dense_by_key.update(split_dense)
        actions.extend(split_actions)
        split_label_dirs[split] = str(split_dir)

    rows: list[dict[str, Any]] = []
    feature_arrays: list[np.ndarray] = []
    feature_names: list[str] | None = None
    feature_cache: dict[Path, dict[str, Any]] = {}
    for action in sorted(actions, key=lambda row: (row["question_id"], int(row["frame_count"]), int(row["compaction_layer"]), str(row["action_id"]))):
        qid = str(action["question_id"])
        split = split_by_qid[qid]
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
            feature_path = Path(split_label_dirs[split]) / feature_path
        if feature_path not in feature_cache:
            feature_cache[feature_path] = _torch_load(feature_path)
        payload = feature_cache[feature_path]
        if int(payload.get("frame_count")) != frame_count or int(payload.get("layer")) != layer:
            raise ValueError(f"Feature tensor metadata mismatch for {feature_path}.")
        if len(payload.get("native_temporal_cell_ids", [])) != int(action["native_temporal_cell_count"]):
            raise ValueError(f"Feature native-cell count mismatch for {feature_path}.")
        names, vector = flatten_feature_dict(route_feature_dict(payload, action))
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names changed across rows.")
        logp_delta = float(action["delta_correct_choice_log_probability_from_dense"])
        margin_delta = float(action["delta_answer_margin_from_dense"])
        flop = float(action["paired_flop_reduction_from_dense"])
        if not all(math.isfinite(value) for value in (logp_delta, margin_delta, flop)) or not np.isfinite(vector).all():
            raise ValueError(f"Nonfinite input or label for {qid}/{action['action_id']}.")
        manifest = manifest_by_qid[qid]
        condition_correct = bool(action.get("correct", action.get("condition_correct", False)))
        dense_correct = bool(dense.get("correct", dense.get("dense_correct", False)))
        safe = bool(logp_delta >= -0.10 and not bool(action.get("prediction_changed")))
        rows.append(
            {
                "schema_version": ROUTER_SCHEMA_VERSION,
                "question_id": qid,
                "source_video_id": manifest.get("source_video_id", action.get("source_video_id")),
                "split": split,
                "category": manifest.get("category"),
                "question_type": manifest.get("question_type"),
                "frame_count": frame_count,
                "compaction_layer": layer,
                "retention_fraction": float(action["retention_fraction"]),
                "native_temporal_cell_count": int(action["native_temporal_cell_count"]),
                "retained_cell_ids": [int(cell) for cell in action["retained_cell_ids"]],
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
        )
        feature_arrays.append(vector)

    if not rows:
        raise ValueError("No completed router action rows found.")
    dataset = RouterDataset(
        rows=rows,
        feature_names=feature_names or [],
        features=np.vstack(feature_arrays).astype(np.float32, copy=False),
        safe=np.asarray([int(row["safe"]) for row in rows], dtype=np.int64),
        logp_delta=np.asarray([float(row["delta_correct_choice_log_probability_from_dense"]) for row in rows], dtype=np.float32),
        margin_delta=np.asarray([float(row["delta_answer_margin_from_dense"]) for row in rows], dtype=np.float32),
        flop_reduction=np.asarray([float(row["paired_flop_reduction_from_dense"]) for row in rows], dtype=np.float32),
    )
    write_router_dataset(
        dataset,
        output_dir,
        split_label_dirs=split_label_dirs,
        manifest_paths=manifest_paths,
        git_commit=git_commit,
    )
    return dataset


def write_router_dataset(
    dataset: RouterDataset,
    output_dir: str | Path,
    *,
    split_label_dirs: dict[str, str],
    manifest_paths: dict[str, str],
    git_commit: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "router_rows.jsonl", dataset.rows)
    np.savez_compressed(
        output_dir / "router_features.npz",
        X=dataset.features.astype(np.float32, copy=False),
        y_safe=dataset.safe,
        y_logp_delta=dataset.logp_delta.astype(np.float32, copy=False),
        y_margin_delta=dataset.margin_delta.astype(np.float32, copy=False),
        flop_reduction=dataset.flop_reduction.astype(np.float32, copy=False),
        question_id=np.asarray([row["question_id"] for row in dataset.rows], dtype=object),
        frame_count=np.asarray([row["frame_count"] for row in dataset.rows], dtype=np.int32),
        split=np.asarray([row["split"] for row in dataset.rows], dtype=object),
    )
    feature_schema_hash = stable_hash_payload(dataset.feature_names)
    write_json(
        output_dir / "metadata.json",
        {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "git_commit": git_commit,
            "split_label_dirs": split_label_dirs,
            "manifest_paths": manifest_paths,
            "manifest_hashes": {split: sha256_file(path) for split, path in manifest_paths.items()},
            "num_rows": len(dataset.rows),
            "num_features": len(dataset.feature_names),
            "feature_dtype": "float32",
            "feature_names": dataset.feature_names,
            "feature_schema_hash": feature_schema_hash,
            "estimated_ram_for_158000_actions_bytes": int(158000 * max(1, len(dataset.feature_names)) * 4),
            "rows_by_split": dict(Counter(row["split"] for row in dataset.rows)),
            "rows_by_question_frame": len({(row["question_id"], row["frame_count"]) for row in dataset.rows}),
            "safe_rate_by_split": {
                split: float(np.mean([row["safe"] for row in dataset.rows if row["split"] == split]))
                for split in sorted({row["split"] for row in dataset.rows})
            },
            "safety_label": SAFETY_DEFINITION,
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
        features=np.asarray(arrays["X"], dtype=np.float32),
        safe=np.asarray(arrays["y_safe"], dtype=np.int64),
        logp_delta=np.asarray(arrays["y_logp_delta"], dtype=np.float32),
        margin_delta=np.asarray(arrays["y_margin_delta"], dtype=np.float32),
        flop_reduction=np.asarray(arrays["flop_reduction"], dtype=np.float32),
    )


def question_frame_weights(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    counts = Counter((row["question_id"], int(row["frame_count"])) for row in rows)
    return np.asarray([1.0 / counts[(row["question_id"], int(row["frame_count"]))] for row in rows], dtype=np.float32)


def train_router_models(dataset_dir: str | Path, output_dir: str | Path, *, seed: int, git_commit: str) -> dict[str, Any]:
    try:
        import joblib
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
    train_rows = [row for row in dataset.rows if row["split"] == "train"]
    if len(set(y.tolist())) < 2:
        raise ValueError("Safety classifier requires both safe and unsafe training examples.")
    sample_weight = question_frame_weights(train_rows)
    groups = [row.get("source_video_id") or row["question_id"] for row in train_rows]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed, early_stopping=False),
        ),
        "gradient_boosted_tree": HistGradientBoostingClassifier(random_state=seed, max_iter=100, class_weight="balanced"),
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
        "grouping_unit": "source_video_id",
        "train_groups": sorted(set(map(str, groups))),
        "sample_weighting": "each question_id/frame_count receives equal total weight",
        "safety_models": {},
        "quality_delta_regressors": {},
    }
    for name, model in models.items():
        fit_kwargs = {}
        if name == "logistic_regression":
            fit_kwargs["logisticregression__sample_weight"] = sample_weight
        elif name == "gradient_boosted_tree":
            fit_kwargs["sample_weight"] = sample_weight
        try:
            model.fit(X, y, **fit_kwargs)
        except TypeError:
            model.fit(X, y)
        path = output_dir / f"safety_{name}.joblib"
        joblib.dump(model, path)
        model_index["safety_models"][name] = str(path)
    for name, model in regressors.items():
        fit_kwargs = {}
        if name == "ridge":
            fit_kwargs["ridge__sample_weight"] = sample_weight
        elif name == "gradient_boosted_tree":
            fit_kwargs["sample_weight"] = sample_weight
        try:
            model.fit(X, dataset.logp_delta[train_idx], **fit_kwargs)
        except TypeError:
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
            return np.ones(X.shape[0], dtype=np.float32) * float(model.classes_[0] == 1)
        class_to_col = {int(cls): idx for idx, cls in enumerate(model.classes_)}
        return np.asarray(proba[:, class_to_col[1]], dtype=np.float32)
    return np.asarray(model.predict(X), dtype=np.float32)


def load_model_predictions(dataset: RouterDataset, model_dir: str | Path, safety_model_name: str, regressor_name: str) -> tuple[np.ndarray, np.ndarray]:
    import joblib

    model_index = json.loads((Path(model_dir) / "model_index.json").read_text())
    safety_model = joblib.load(model_index["safety_models"][safety_model_name])
    regressor = joblib.load(model_index["quality_delta_regressors"][regressor_name])
    return _predict_safe_probability(safety_model, dataset.features), np.asarray(regressor.predict(dataset.features), dtype=np.float32)


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
        logp = np.mean([float(row["delta_correct_choice_log_probability_from_dense"]) for row in items])
        scored.append((unsafe <= 0.10, -unsafe, flop, logp, -key[0], -key[1], key))
    return max(scored)[-1]


def deterministic_route_row(candidates: Sequence[dict[str, Any]], route_family: str) -> dict[str, Any] | None:
    subset = [row for row in candidates if row.get("route_family") == route_family]
    return sorted(subset, key=lambda row: str(row["action_id"]))[0] if subset else None


def choose_predicted_row(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    eligible = [row for row in rows if float(row["_unsafe_probability"]) <= threshold]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["paired_flop_reduction_from_dense"]),
            float(row.get("_predicted_logp_delta", 0.0)),
            -str(row["action_id"]).__len__(),
            str(row["action_id"]),
        ),
    )


def choose_oracle_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if bool(row["safe"])]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["paired_flop_reduction_from_dense"]),
            float(row["delta_correct_choice_log_probability_from_dense"]),
            str(row["action_id"]),
        ),
    )


def causal_policy_decision(candidates: Sequence[dict[str, Any]], *, threshold: float, layer_order: Sequence[int], oracle: bool = False) -> dict[str, Any] | None:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_layer[int(row["compaction_layer"])].append(row)
    for layer in layer_order:
        rows = by_layer.get(int(layer), [])
        chosen = choose_oracle_row(rows) if oracle else choose_predicted_row(rows, threshold)
        if chosen is not None:
            return chosen
    return None


def decision_from_choice(
    *,
    baseline: str,
    split: str,
    question_id: str,
    frame_count: int,
    template: dict[str, Any],
    chosen: dict[str, Any] | None,
) -> dict[str, Any]:
    if chosen is None:
        return {
            "question_id": question_id,
            "source_video_id": template.get("source_video_id"),
            "split": split,
            "baseline": baseline,
            "compacted": False,
            "delta_logp": 0.0,
            "delta_margin": 0.0,
            "accuracy_delta_from_dense": 0.0,
            "prediction_changed": False,
            "safe": True,
            "unsafe": False,
            "flop_reduction": 0.0,
            "frame_count": frame_count,
            "category": template.get("category"),
            "selected_layer": None,
            "retention_fraction": None,
        }
    return {
        "question_id": question_id,
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
        "unsafe": not bool(chosen["safe"]),
        "flop_reduction": float(chosen["paired_flop_reduction_from_dense"]),
        "frame_count": int(chosen["frame_count"]),
        "selected_layer": int(chosen["compaction_layer"]),
        "retention_fraction": float(chosen["retention_fraction"]),
        "category": chosen.get("category"),
        "route_family": chosen.get("route_family"),
        "retained_cell_ids": chosen.get("retained_cell_ids"),
    }


def make_decisions(
    dataset: RouterDataset,
    *,
    split: str,
    safety_probability: np.ndarray | None,
    predicted_logp: np.ndarray | None,
    threshold: float,
    layer_order: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset.rows):
        enriched = dict(row, _row_index=idx)
        if safety_probability is not None:
            enriched["_unsafe_probability"] = 1.0 - float(safety_probability[idx])
        if predicted_logp is not None:
            enriched["_predicted_logp_delta"] = float(predicted_logp[idx])
        rows.append(enriched)
    fixed_layer, fixed_fraction = select_fixed_policy_training_only(rows)
    by_qf: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_qf[(str(row["question_id"]), int(row["frame_count"]))].append(row)
    decisions: list[dict[str, Any]] = []
    for (qid, frame_count), candidates in sorted(by_qf.items()):
        template = candidates[0]
        fixed_candidates = [
            row for row in candidates if int(row["compaction_layer"]) == fixed_layer and float(row["retention_fraction"]) == fixed_fraction
        ]
        baselines = {
            "dense": None,
            "fixed_uniform": deterministic_route_row(fixed_candidates, "uniform_temporal_coverage"),
            "fixed_random": deterministic_route_row(fixed_candidates, "deterministic_random_subset"),
            "causal_learned_router": causal_policy_decision(candidates, threshold=threshold, layer_order=layer_order, oracle=False),
            "causal_oracle": causal_policy_decision(candidates, threshold=threshold, layer_order=layer_order, oracle=True),
            "global_oracle_upper_bound": choose_oracle_row(candidates),
        }
        for baseline, chosen in baselines.items():
            decisions.append(
                decision_from_choice(
                    baseline=baseline,
                    split=split,
                    question_id=qid,
                    frame_count=frame_count,
                    template=template,
                    chosen=chosen,
                )
            )
    return decisions


def bootstrap_ci(values: Sequence[float], *, samples: int = 10000, seed: int = 1) -> dict[str, Any]:
    clean = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float32)
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


def cluster_bootstrap_ci(rows: Sequence[dict[str, Any]], metric: str, *, samples: int = 10000, seed: int = 1) -> dict[str, Any]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[str(row["question_id"])].append(row)
    qids = sorted(by_question)
    if not qids:
        return {"mean": None, "median": None, "ci95": [None, None], "n_questions": 0, "n_decisions": 0}
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float32)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        sampled_rows = [row for qid in sampled for row in by_question[qid]]
        estimates.append(float(np.mean([float(row[metric]) for row in sampled_rows])))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n_questions": len(qids),
        "n_decisions": len(rows),
    }


def summarize_router_decisions(decisions: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for baseline in sorted({row["baseline"] for row in decisions}):
        rows = [row for row in decisions if row["baseline"] == baseline]
        output[baseline] = {
            "num_question_frame_decisions": len(rows),
            "num_questions": len({row["question_id"] for row in rows}),
            "coverage": float(np.mean([bool(row["compacted"]) for row in rows])) if rows else None,
            "dense_fallback_rate": float(np.mean([not bool(row["compacted"]) for row in rows])) if rows else None,
            "unsafe_selection_rate": float(np.mean([bool(row["unsafe"]) for row in rows])) if rows else None,
            "prediction_flip_rate": float(np.mean([bool(row["prediction_changed"]) for row in rows])) if rows else None,
            "mean_logp_delta": cluster_bootstrap_ci(rows, "delta_logp", samples=bootstrap_samples, seed=seed),
            "mean_margin_delta": cluster_bootstrap_ci(rows, "delta_margin", samples=bootstrap_samples, seed=seed + 1),
            "mean_flop_reduction": cluster_bootstrap_ci(rows, "flop_reduction", samples=bootstrap_samples, seed=seed + 2),
            "accuracy_difference_from_dense": float(np.mean([float(row.get("accuracy_delta_from_dense", 0.0)) for row in rows])) if rows else None,
            "selected_layer_distribution": dict(Counter(str(row.get("selected_layer")) for row in rows)),
            "by_frame_count": _group_metric(rows, "frame_count"),
            "by_category": _group_metric(rows, "category"),
            "by_selected_layer": _group_metric(rows, "selected_layer"),
            "by_retention_fraction": _group_metric(rows, "retention_fraction"),
        }
    if "global_oracle_upper_bound" in output and "causal_learned_router" in output:
        output["oracle_gap_learned_minus_global_oracle_flop_reduction"] = (
            output["causal_learned_router"]["mean_flop_reduction"]["mean"]
            - output["global_oracle_upper_bound"]["mean_flop_reduction"]["mean"]
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
            "unsafe_selection_rate": float(np.mean([bool(row["unsafe"]) for row in items])),
        }
        for value, items in sorted(grouped.items())
    }


def calibrate_router_policy(
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    unsafe_rate_bound: float = 0.10,
    layer_order: Sequence[int] = DEFAULT_LAYER_ORDER,
    bootstrap_samples: int = 10000,
    seed: int = 1,
    git_commit: str,
) -> dict[str, Any]:
    dataset = load_router_dataset(dataset_dir)
    model_index = json.loads((Path(model_dir) / "model_index.json").read_text())
    candidates: list[dict[str, Any]] = []
    thresholds = [round(float(value), 3) for value in np.linspace(0.0, 1.0, 51)]
    for safety_name in sorted(model_index["safety_models"]):
        for regressor_name in sorted(model_index["quality_delta_regressors"]):
            safe_probability, predicted_logp = load_model_predictions(dataset, model_dir, safety_name, regressor_name)
            for threshold in thresholds:
                decisions = make_decisions(
                    dataset,
                    split="development",
                    safety_probability=safe_probability,
                    predicted_logp=predicted_logp,
                    threshold=threshold,
                    layer_order=layer_order,
                )
                learned = [row for row in decisions if row["baseline"] == "causal_learned_router"]
                unsafe = float(np.mean([bool(row["unsafe"]) for row in learned])) if learned else 1.0
                flop = float(np.mean([float(row["flop_reduction"]) for row in learned])) if learned else 0.0
                logp = float(np.mean([float(row["delta_logp"]) for row in learned])) if learned else 0.0
                candidates.append(
                    {
                        "safety_model": safety_name,
                        "quality_regressor": regressor_name,
                        "unsafe_probability_threshold": threshold,
                        "development_unsafe_selection_rate": unsafe,
                        "development_mean_flop_reduction": flop,
                        "development_mean_logp_delta": logp,
                        "eligible": unsafe <= unsafe_rate_bound,
                    }
                )
    eligible = [row for row in candidates if row["eligible"]]
    selected = max(eligible or candidates, key=lambda row: (row["eligible"], row["development_mean_flop_reduction"], row["development_mean_logp_delta"]))
    safe_probability, predicted_logp = load_model_predictions(dataset, model_dir, selected["safety_model"], selected["quality_regressor"])
    decisions = make_decisions(
        dataset,
        split="development",
        safety_probability=safe_probability,
        predicted_logp=predicted_logp,
        threshold=float(selected["unsafe_probability_threshold"]),
        layer_order=layer_order,
    )
    summary = summarize_router_decisions(decisions, bootstrap_samples=bootstrap_samples, seed=seed)
    metadata = json.loads((Path(dataset_dir) / "metadata.json").read_text())
    model_hashes = {name: sha256_file(path) for name, path in model_index["safety_models"].items()}
    model_hashes.update({name: sha256_file(path) for name, path in model_index["quality_delta_regressors"].items()})
    policy = {
        "schema_version": ROUTER_SCHEMA_VERSION,
        "git_commit": git_commit,
        "selected_safety_model": selected["safety_model"],
        "selected_quality_regressor": selected["quality_regressor"],
        "model_hashes": model_hashes,
        "feature_schema_hash": metadata["feature_schema_hash"],
        "unsafe_probability_threshold": selected["unsafe_probability_threshold"],
        "layer_order": list(map(int, layer_order)),
        "safety_definition": SAFETY_DEFINITION,
        "development_objective": f"maximize mean FLOP reduction subject to unsafe_selection_rate <= {unsafe_rate_bound}",
        "unsafe_rate_bound": unsafe_rate_bound,
        "achieved_development_metrics": summary,
        "train_manifest_hash": metadata["manifest_hashes"].get("train"),
        "development_manifest_hash": metadata["manifest_hashes"].get("development"),
        "candidate_grid": candidates,
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "frozen_policy.json", policy)
    write_jsonl(output_dir / "development_decisions.jsonl", decisions)
    write_csv(output_dir / "development_decisions.csv", decisions)
    return policy


def evaluate_router(
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    frozen_policy: str | Path,
    allow_test: bool = False,
    bootstrap_samples: int = 10000,
    seed: int = 1,
) -> dict[str, Any]:
    if split == "test" and not allow_test:
        raise RuntimeError("Pass --allow-test to evaluate the frozen router on test.")
    policy = json.loads(Path(frozen_policy).read_text())
    if split == "test" and "unsafe_probability_threshold" not in policy:
        raise RuntimeError("Test evaluation requires frozen_policy.json.")
    dataset = load_router_dataset(dataset_dir)
    metadata = json.loads((Path(dataset_dir) / "metadata.json").read_text())
    if metadata["feature_schema_hash"] != policy["feature_schema_hash"]:
        raise ValueError("Dataset feature schema does not match frozen policy.")
    safe_probability, predicted_logp = load_model_predictions(
        dataset,
        model_dir,
        policy["selected_safety_model"],
        policy["selected_quality_regressor"],
    )
    decisions = make_decisions(
        dataset,
        split=split,
        safety_probability=safe_probability,
        predicted_logp=predicted_logp,
        threshold=float(policy["unsafe_probability_threshold"]),
        layer_order=[int(layer) for layer in policy["layer_order"]],
    )
    summary = summarize_router_decisions(decisions, bootstrap_samples=bootstrap_samples, seed=seed)
    output_dir = Path(output_dir)
    write_jsonl(output_dir / "decisions.jsonl", decisions)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "split": split,
            "frozen_policy": str(frozen_policy),
            "unsafe_probability_threshold": policy["unsafe_probability_threshold"],
            "metrics": summary,
        },
    )
    write_csv(output_dir / "decisions.csv", decisions)
    return summary
