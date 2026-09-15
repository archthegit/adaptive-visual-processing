from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.analyze_route_reuse_failure_modes as failure
from scripts.analyze_route_reuse_failure_modes import (
    analyze,
    choose_recommendation,
    hypothesis_summary,
    normalized_entropy,
    route_layer_metrics,
    spearman_correlation,
    stable_route_harm,
)
from scripts.analyze_route_reuse_pilot import CONDITIONS, EXPECTED_ROUTED_LAYERS


def _distribution(layer: int, selected_bins: set[int], diffuse: bool = False) -> list[float]:
    if diffuse:
        return [0.125] * 8
    values = [0.04] * 8
    for item in selected_bins:
        values[item] += 0.18
    values[(layer + 3) % 8] += 0.06
    total = sum(values)
    return [value / total for value in values]


def _scores(selected_bins: set[int], diffuse: bool = False) -> list[list[float]]:
    return [_distribution(layer, selected_bins, diffuse=diffuse) for layer in range(28)]


def _route(condition: str, selected_units: tuple[int, int] = (0, 1), commit: str = "manifest456") -> dict:
    omitted_units = tuple(unit for unit in range(4) if unit not in selected_units)
    selected_bins = sorted({2 * unit + offset for unit in selected_units for offset in (0, 1)})
    omitted_bins = sorted({2 * unit + offset for unit in omitted_units for offset in (0, 1)})
    layer_routes = {}
    for layer in EXPECTED_ROUTED_LAYERS:
        layer_routes[str(layer)] = {
            "source_anchor_layer": max(anchor for anchor in (8, 12, 16, 20, 24) if anchor < layer),
            "selected_native_unit_ids": list(selected_units),
            "omitted_native_unit_ids": list(omitted_units),
            "selected_analysis_bins": selected_bins,
            "omitted_analysis_bins": omitted_bins,
            "allowed_visual_token_indices": selected_bins,
            "blocked_visual_token_indices": omitted_bins,
            "num_allowed_visual_tokens": 4,
            "num_blocked_visual_tokens": 4,
            "actual_retained_visual_token_fraction": 0.5,
        }
    return {
        "type": "baseline_derived_causal_route_replay",
        "condition": condition,
        "routing_unit_type": "qwen_native_temporal_cell",
        "native_routing_units": [
            {
                "unit_id": unit,
                "analysis_bins": [2 * unit, 2 * unit + 1],
                "visual_token_indices": [2 * unit, 2 * unit + 1],
            }
            for unit in range(4)
        ],
        "retained_native_units": 2,
        "mean_actual_retained_visual_token_fraction": 0.5,
        "layer_routes": layer_routes,
        "git_commit": commit,
    }


def _base_artifact(question_id: str, index: int, selected_units: tuple[int, int] = (0, 1), diffuse: bool = False) -> dict:
    selected_bins = {2 * unit + offset for unit in selected_units for offset in (0, 1)}
    category = ("fine_grained", "gaze", "ingredient", "object_motion")[index % 4]
    return {
        "question_id": question_id,
        "model_backend": "qwen",
        "model_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "question_type": f"{category}_synthetic",
        "category": category,
        "question": f"What happens in example {question_id}?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct_idx": 0,
        "predicted_idx": index % 5,
        "correct": index % 3 == 0,
        "video_clip": [{"video_id": f"P{index % 7:02d}-video-{question_id}", "participant_id": f"P{index % 7:02d}"}],
        "sampled_frame_indices": [list(range(8))],
        "sampled_timestamps": [[float(value) for value in range(8)]],
        "frame_bin_mappings": [[{"sample_position": value, "analysis_bin": value} for value in range(8)]],
        "sampling_metadata": [{"mode": "cross_model_8", "frames_per_bin": 1}],
        "temporal_relevance": {"normalized_temporal_bin_scores": _scores(selected_bins, diffuse=diffuse)},
        "answer_choice_scores": {
            "correct_choice_log_probability": -2.0 - 0.01 * index,
            "correct_vs_best_incorrect_margin": -0.5 + 0.01 * index,
        },
    }


def _routed_artifact(
    question_id: str,
    index: int,
    condition: str,
    delta: float,
    selected_units: tuple[int, int] = (0, 1),
) -> dict:
    artifact = _base_artifact(question_id, index, selected_units=selected_units)
    artifact["route_reuse"] = _route(condition, selected_units=selected_units)
    artifact["metadata"] = {"route_reuse": {"type": "baseline_derived_causal_route_replay", "condition": condition}}
    artifact["run_config"] = {"git_commit": "exec123"}
    artifact["answer_choice_scores"] = {
        "correct_choice_log_probability": 99.0,
        "correct_vs_best_incorrect_margin": 99.0,
    }
    artifact["intervention_answer_choice_scores"] = {
        "correct_choice_log_probability": -2.0 - 0.01 * index + delta,
        "correct_vs_best_incorrect_margin": -0.5 + 0.01 * index + delta,
    }
    artifact["predicted_idx"] = (index + (delta < 0)) % 5
    artifact["correct"] = artifact["predicted_idx"] == artifact["correct_idx"]
    return artifact


def _write_run(output_dir: Path, artifacts: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for question_id, artifact in artifacts.items():
        artifact_path = output_dir / f"{question_id}.json"
        artifact_path.write_text(json.dumps(artifact))
        records.append({"question_id": question_id, "status": "complete", "artifact": artifact_path.name})
    (output_dir / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _write_manifest(path: Path, question_ids: list[str], start_index: int = 0) -> None:
    rows = []
    for offset, qid in enumerate(question_ids):
        index = start_index + offset
        category = ("fine_grained", "gaze", "ingredient", "object_motion")[index % 4]
        rows.append(
            {
                "question_id": qid,
                "source_video_id": f"P{index % 7:02d}-video-{qid}",
                "participant_id": f"P{index % 7:02d}",
                "category": category,
                "question_type": f"{category}_synthetic",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _fixtures(tmp_path: Path):
    dev_ids = [f"dev-{index:02d}" for index in range(15)]
    heldout_ids = [f"held-{index:02d}" for index in range(56)]
    baseline_dir = tmp_path / "baseline"
    all_ids = dev_ids + heldout_ids
    baseline = {}
    for index, qid in enumerate(all_ids):
        selected_units = (0, 1) if index % 2 == 0 else (0, 3)
        baseline[qid] = _base_artifact(qid, index, selected_units=selected_units, diffuse=index % 5 == 0)
    _write_run(baseline_dir, baseline)
    dev_manifest = tmp_path / "dev.jsonl"
    heldout_manifest = tmp_path / "heldout.jsonl"
    _write_manifest(dev_manifest, dev_ids, 0)
    _write_manifest(heldout_manifest, heldout_ids, len(dev_ids))

    condition_dirs = {}
    for cohort, ids, start, deltas in (
        ("dev", dev_ids, 0, {"adaptive": -0.10, "random": -0.20, "uniform": -0.15}),
        ("heldout", heldout_ids, len(dev_ids), {"adaptive": -0.25, "random": -0.10, "uniform": -0.12}),
    ):
        condition_dirs[cohort] = {}
        for label, condition in CONDITIONS.items():
            path = tmp_path / f"{cohort}_{label}"
            artifacts = {}
            for offset, qid in enumerate(ids):
                index = start + offset
                selected_units = (0, 1) if index % 2 == 0 else (0, 3)
                artifacts[qid] = _routed_artifact(qid, index, condition, deltas[label], selected_units=selected_units)
            _write_run(path, artifacts)
            key = "route_reuse" if label == "adaptive" else label
            condition_dirs[cohort][key] = path
    return baseline_dir, condition_dirs, dev_manifest, heldout_manifest


def _stub_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(failure, "save_scatter", lambda *_args, **_kwargs: Path(_args[3]).write_bytes(b"png"))
    monkeypatch.setattr(failure, "save_bar", lambda *_args, **_kwargs: Path(_args[2]).write_bytes(b"png"))
    monkeypatch.setattr(failure, "save_route_stability_by_anchor_distance", lambda _rows, output: Path(output).write_bytes(b"png"))
    monkeypatch.setattr(failure, "save_adaptive_controls", lambda _rows, output: Path(output).write_bytes(b"png"))


def test_failure_mode_analyzer_writes_dev_and_heldout_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline_dir, dirs, dev_manifest, heldout_manifest = _fixtures(tmp_path)
    _stub_plots(monkeypatch)
    output_dir = tmp_path / "analysis"

    summary = analyze(
        baseline_dir=baseline_dir,
        dev_condition_dirs=dirs["dev"],
        heldout_condition_dirs=dirs["heldout"],
        dev_manifest=dev_manifest,
        heldout_manifest=heldout_manifest,
        output_dir=output_dir,
        bootstrap_samples=10,
        seed=7,
    )

    assert summary["validation"]["cohorts"]["development"]["num_examples"] == 15
    assert summary["validation"]["cohorts"]["heldout"]["num_examples"] == 56
    assert summary["cohorts"]["development"]["num_examples"] == 15
    assert summary["cohorts"]["heldout"]["num_examples"] == 56
    assert summary["cohorts"]["development"]["num_layer_rows"] == 225
    assert summary["cohorts"]["heldout"]["num_layer_rows"] == 840
    assert summary["recommendation"] in {
        "variable-budget routing",
        "coverage-constrained routing",
        "shorter/dynamic refresh",
        "compression instead of deletion",
        "diagnosis inconclusive; run a targeted discriminating pilot",
    }
    for filename in (
        "per_example.csv",
        "per_layer.csv",
        "summary.json",
        "report.md",
        "harm_vs_entropy.png",
        "harm_vs_retained_mass.png",
        "harm_by_temporal_coverage.png",
        "route_stability_by_anchor_distance.png",
        "adaptive_vs_controls.png",
    ):
        assert (output_dir / filename).is_file()
    with (output_dir / "per_example.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["cohort"] for row in rows} == {"development", "heldout"}
    assert len(rows) == 71


def test_failure_mode_analyzer_rejects_manifest_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline_dir, dirs, dev_manifest, heldout_manifest = _fixtures(tmp_path)
    _stub_plots(monkeypatch)
    heldout_rows = [json.loads(line) for line in heldout_manifest.read_text().splitlines()]
    dev_first = json.loads(dev_manifest.read_text().splitlines()[0])
    heldout_rows[0] = dev_first
    heldout_manifest.write_text("\n".join(json.dumps(row) for row in heldout_rows) + "\n")

    with pytest.raises(RuntimeError, match="overlap"):
        analyze(
            baseline_dir=baseline_dir,
            dev_condition_dirs=dirs["dev"],
            heldout_condition_dirs=dirs["heldout"],
            dev_manifest=dev_manifest,
            heldout_manifest=heldout_manifest,
            output_dir=tmp_path / "analysis",
            bootstrap_samples=5,
            seed=7,
        )


def _hypothesis_rows() -> list[dict]:
    rows = []
    for index in range(8):
        rows.append(
            {
                "participant_id": f"P{index % 3}",
                "mean_anchor_native_temporal_entropy": 0.2 + 0.1 * index,
                "mean_native_retained_target_layer_mass": 0.9 - 0.08 * index,
                "fraction_adjacent_selections": 1.0 if index >= 4 else 0.0,
                "mean_normalized_temporal_coverage_span": 0.5 if index >= 4 else 1.0,
                "mean_distance_from_anchor_descriptive_only": 2.0,
                "mean_native_source_target_spearman": 0.9 - 0.1 * index,
                "adaptive_minus_dense_correct_answer_log_probability": 0.1 - 0.1 * index,
                "adaptive_minus_dense_answer_margin": 0.05 - 0.08 * index,
            }
        )
    return rows


def test_failure_hypotheses_cover_entropy_mass_coverage_distance_and_hard_deletion():
    rows = _hypothesis_rows()
    summary = hypothesis_summary(rows, samples=10, seed=3)
    logp = summary["adaptive_minus_dense_correct_answer_log_probability"]

    assert logp["H1_entropy"]["spearman"] < 0
    assert logp["H1_retained_mass"]["spearman"] > 0
    assert logp["H2_adjacent_selection"]["median_split"]["high_minus_low"] < 0
    assert logp["H2_temporal_coverage"]["spearman"] > 0
    assert "H3_anchor_distance" not in logp
    assert logp["H3_route_agreement"]["spearman"] > 0

    stable = stable_route_harm(
        [
            {
                **row,
                "mean_anchor_native_temporal_entropy": 0.05 if index == 0 else 0.5 + index * 0.05,
                "mean_native_retained_target_layer_mass": 0.95 if index == 0 else 0.5 + index * 0.02,
                "mean_normalized_temporal_coverage_span": 1.0 if index == 0 else 0.5 + index * 0.02,
                "mean_native_source_target_spearman": 0.95 if index == 0 else 0.4 + index * 0.02,
                "adaptive_minus_dense_correct_answer_log_probability": -0.2,
                "adaptive_minus_dense_answer_margin": -0.1,
            }
            for index, row in enumerate(rows[:5])
        ]
    )
    assert stable["num_stable_examples"] > 0
    assert stable["mean_logp_delta"] < 0


def test_native_oracle_differs_from_top_four_analysis_bins_and_entropy_uses_log4():
    baseline = _base_artifact("q-native", 0, selected_units=(0, 1))
    route = _route(CONDITIONS["adaptive"], selected_units=(0, 1))
    # Layer 8 source keeps units 0/1, but layer 9 target has four large analysis
    # bins split across all native units. The top four analysis bins are not the
    # same object as the top two aggregated native units.
    scores = [[0.125] * 8 for _ in range(28)]
    scores[8] = [0.30, 0.20, 0.18, 0.12, 0.06, 0.05, 0.05, 0.04]
    scores[9] = [0.22, 0.01, 0.21, 0.01, 0.20, 0.19, 0.08, 0.08]
    baseline["temporal_relevance"]["normalized_temporal_bin_scores"] = scores
    adaptive = _routed_artifact("q-native", 0, CONDITIONS["adaptive"], -0.1, selected_units=(0, 1))
    adaptive["route_reuse"] = route

    row = route_layer_metrics(
        "q-native",
        "development",
        {"question_id": "q-native", "participant_id": "P0", "source_video_id": "P0-v", "category": "gaze", "question_type": "gaze_synthetic"},
        baseline,
        adaptive,
    )[0]

    assert row["target_layer"] == 9
    assert row["descriptive_eight_bin_top4_oracle_mass"] == pytest.approx(0.82)
    assert row["native_oracle_top50_target_layer_mass"] == pytest.approx(0.39 + 0.23)
    assert row["native_retained_target_layer_mass"] == pytest.approx(0.45)
    assert row["native_source_target_top2_jaccard"] == pytest.approx(1 / 3)
    assert normalized_entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(1.0)


def test_constant_anchor_distance_does_not_drive_recommendation_and_point_sign_is_inconclusive():
    summary = {
        "cohorts": {
            "development": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [-0.2, 0.8]}
                    },
                    "H4_stable_high_mass_well_covered_routes": {"num_stable_examples": 0},
                }
            },
            "heldout": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [-0.2, 0.8]}
                    },
                    "H4_stable_high_mass_well_covered_routes": {"num_stable_examples": 0},
                }
            },
        }
    }
    assert "H3_anchor_distance" not in summary["cohorts"]["heldout"]["hypotheses"]["adaptive_minus_dense_correct_answer_log_probability"]
    assert choose_recommendation(summary) == "diagnosis inconclusive; run a targeted discriminating pilot"


def test_recommendation_requires_replicated_direction_and_heldout_ci_excluding_zero():
    base_summary = {
        "cohorts": {
            "development": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [0.1, 0.7]}
                    },
                    "adaptive_minus_dense_answer_margin": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [0.1, 0.7]}
                    },
                    "H4_stable_high_mass_well_covered_routes": {"num_stable_examples": 0},
                }
            },
            "heldout": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [-0.1, 0.7]}
                    },
                    "adaptive_minus_dense_answer_margin": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [-0.1, 0.7]}
                    },
                    "H4_stable_high_mass_well_covered_routes": {"num_stable_examples": 0},
                }
            },
        }
    }
    assert choose_recommendation(base_summary) == "diagnosis inconclusive; run a targeted discriminating pilot"
    base_summary["cohorts"]["heldout"]["hypotheses"]["adaptive_minus_dense_correct_answer_log_probability"]["H2_temporal_coverage"]["spearman_ci95"] = [0.05, 0.7]
    assert choose_recommendation(base_summary) == "coverage-constrained routing"


def test_recommendation_requires_margin_directional_consistency():
    summary = {
        "cohorts": {
            "development": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [0.1, 0.7]}
                    },
                    "adaptive_minus_dense_answer_margin": {
                        "H2_temporal_coverage": {"spearman": -0.4, "spearman_ci95": [-0.7, -0.1]}
                    },
                    "H4_stable_high_mass_well_covered_routes": {"num_stable_examples": 0},
                }
            },
            "heldout": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [0.1, 0.7]}
                    },
                    "adaptive_minus_dense_answer_margin": {
                        "H2_temporal_coverage": {"spearman": 0.4, "spearman_ci95": [0.1, 0.7]}
                    },
                    "H4_stable_high_mass_well_covered_routes": {"num_stable_examples": 0},
                }
            },
        }
    }
    assert choose_recommendation(summary) == "diagnosis inconclusive; run a targeted discriminating pilot"


def test_h4_recommendation_requires_margin_ci_and_minimum_heldout_subgroup():
    summary = {
        "cohorts": {
            "development": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {},
                    "adaptive_minus_dense_answer_margin": {},
                    "H4_stable_high_mass_well_covered_routes": {
                        "num_stable_examples": 6,
                        "mean_logp_delta": -0.2,
                        "logp_ci95": [-0.3, -0.1],
                        "mean_margin_delta": -0.2,
                        "margin_ci95": [-0.3, -0.1],
                    },
                }
            },
            "heldout": {
                "hypotheses": {
                    "adaptive_minus_dense_correct_answer_log_probability": {},
                    "adaptive_minus_dense_answer_margin": {},
                    "H4_stable_high_mass_well_covered_routes": {
                        "num_stable_examples": 9,
                        "mean_logp_delta": -0.2,
                        "logp_ci95": [-0.3, -0.1],
                        "mean_margin_delta": -0.2,
                        "margin_ci95": [-0.3, -0.1],
                    },
                }
            },
        }
    }
    assert choose_recommendation(summary) == "diagnosis inconclusive; run a targeted discriminating pilot"
    summary["cohorts"]["heldout"]["hypotheses"]["H4_stable_high_mass_well_covered_routes"]["num_stable_examples"] = 10
    assert choose_recommendation(summary) == "compression instead of deletion"


def test_tie_correct_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    left = [0.4, 0.4, 0.1, 0.1]
    right = [0.2, 0.3, 0.3, 0.1]
    expected = float(scipy_stats.spearmanr(left, right).statistic)

    assert spearman_correlation(left, right) == pytest.approx(expected)


def test_malformed_native_routes_are_rejected():
    baseline = _base_artifact("q-bad", 0)
    adaptive = _routed_artifact("q-bad", 0, CONDITIONS["adaptive"], -0.1)
    manifest = {"question_id": "q-bad", "participant_id": "P0", "source_video_id": "P0-v", "category": "gaze", "question_type": "gaze_synthetic"}

    missing = json.loads(json.dumps(adaptive))
    missing["route_reuse"]["native_routing_units"][0].pop("analysis_bins")
    with pytest.raises(RuntimeError, match="missing explicit analysis_bins"):
        route_layer_metrics("q-bad", "development", manifest, baseline, missing)

    overlapping = json.loads(json.dumps(adaptive))
    overlapping["route_reuse"]["native_routing_units"][1]["analysis_bins"] = [1, 2]
    with pytest.raises(RuntimeError, match="overlaps analysis bins"):
        route_layer_metrics("q-bad", "development", manifest, baseline, overlapping)

    incomplete = json.loads(json.dumps(adaptive))
    incomplete["route_reuse"]["native_routing_units"][3]["analysis_bins"] = [6]
    with pytest.raises(RuntimeError, match="cover analysis bins 0..7"):
        route_layer_metrics("q-bad", "development", manifest, baseline, incomplete)


def test_validate_condition_artifacts_requires_shared_execution_and_manifest_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline_dir, dirs, dev_manifest, heldout_manifest = _fixtures(tmp_path)
    _stub_plots(monkeypatch)
    random_records = {}
    for index, line in enumerate((dirs["dev"]["random"] / "records.jsonl").read_text().splitlines()):
        record = json.loads(line)
        artifact_path = dirs["dev"]["random"] / record["artifact"]
        artifact = json.loads(artifact_path.read_text())
        artifact["run_config"]["git_commit"] = f"exec-different-{index}"
        artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(RuntimeError, match="share exactly one execution commit"):
        analyze(
            baseline_dir=baseline_dir,
            dev_condition_dirs=dirs["dev"],
            heldout_condition_dirs=dirs["heldout"],
            dev_manifest=dev_manifest,
            heldout_manifest=heldout_manifest,
            output_dir=tmp_path / "analysis",
            bootstrap_samples=5,
            seed=7,
        )

    # Restore execution commits, then break route-manifest commit sharing.
    for line in (dirs["dev"]["random"] / "records.jsonl").read_text().splitlines():
        record = json.loads(line)
        artifact_path = dirs["dev"]["random"] / record["artifact"]
        artifact = json.loads(artifact_path.read_text())
        artifact["run_config"]["git_commit"] = "exec123"
        artifact["route_reuse"]["git_commit"] = "manifest-other"
        artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(RuntimeError, match="share exactly one route-manifest commit"):
        analyze(
            baseline_dir=baseline_dir,
            dev_condition_dirs=dirs["dev"],
            heldout_condition_dirs=dirs["heldout"],
            dev_manifest=dev_manifest,
            heldout_manifest=heldout_manifest,
            output_dir=tmp_path / "analysis2",
            bootstrap_samples=5,
            seed=7,
        )
