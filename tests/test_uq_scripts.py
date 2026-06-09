"""N4: script-level output-schema + invariant tests for the smoke-only VESP-UQ drivers.

These scripts were previously exercised only by the CI smoke step (no pytest assertions on their
JSON keys / CSV headers / row counts), so their output contracts could drift silently. Each test
runs the driver's pure entry point on a tiny synthetic config in ``tmp_path`` and locks the schema
plus a couple of invariants (e.g. ``n_flagged <= n_trajectories``).

Covered here: ``run_force_error_benchmark``, ``compare_risk_baselines``, ``run_calibration_audit``.
``run_propagation`` (the Monte Carlo demo) writes no files -- nothing to lock -- and its numerical
core is covered by ``tests/test_uq_propagation.py``; it gets only an import-safety guard below.
The artifact/manifest contract itself is covered by ``tests/test_uq_run_artifacts.py``.
"""

from __future__ import annotations

import json

import scripts.compare_risk_baselines as crb
import scripts.run_calibration_audit as ca
import scripts.run_force_error_benchmark as feb
from vesp.uq.benchmarking import METRIC_KEYS


def _tiny_config():
    """Tiny synthetic config (fast fit + small trajectory ensemble) shared by the script tests."""

    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "data": {"type": "synthetic", "n": 240, "noise_std": 1.0e-4, "train_fraction": 0.7},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {
            "regularization": {"method": "lcurve"},
            "noise_model": "heteroscedastic",
            "conformal": {"enabled": True, "alpha": 0.10, "mode": "norm"},
            "audit": {"enabled": True, "audit_fraction": 0.1, "min_audit": 2, "seed": 0},
            "risk": {"scoring": "supervisor_rel", "low_altitude_radius": 1.15},
            "screening": {"n_orbits": 16, "n_points": 18, "rerun_fraction": 0.25},
        },
        "_config_path": "scripts_test.yaml",
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- force-error benchmark
def test_force_error_benchmark_schema_and_csv(tmp_path):
    feb.run_and_write(_tiny_config(), out_dir=tmp_path, scoring=None, rerun_fraction=0.25)

    data = _read_json(tmp_path / "force_error_benchmark.json")
    expected = {
        "benchmark", "claim", "is_position_error_benchmark", "scoring",
        "true_force_error_aggregator", "true_error_mode", "n_trajectories", "rerun_fraction",
        "spearman_force_risk_vs_true_force_error", "capture_rate", "precision", "lift_over_random",
        "mean_true_force_error_flagged", "mean_true_force_error_accepted",
        "force_error_ratio_flagged_to_accepted",
    }
    assert expected <= set(data)
    assert data["is_position_error_benchmark"] is False  # force-error, never position-error
    assert data["n_trajectories"] == 16
    assert "_scores" not in data  # the per-row scores are written to CSV, not the JSON
    assert data["_provenance"]["tool"] == "run_force_error_benchmark"

    lines = (tmp_path / "force_error_scores.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == ["trajectory_id", "force_risk", "true_force_error", "flagged"]
    assert len(lines) - 1 == data["n_trajectories"]
    flagged_col = [int(row.split(",")[3]) for row in lines[1:]]
    assert set(flagged_col) <= {0, 1}
    assert sum(flagged_col) <= data["n_trajectories"]  # flagged subset of all trajectories


# --------------------------------------------------------------------------- baseline comparison
def test_baseline_comparison_schema_and_csv(tmp_path):
    config = _tiny_config()
    payload = crb.run_baseline_comparison(config, rerun_fraction=0.25)
    crb.write_outputs(payload, tmp_path, config=config)

    expected = {
        "config_dataset", "n_trajectories", "trajectory_source", "true_force_error_source",
        "true_force_error_aggregator", "rerun_fraction", "uncertainty_scoring", "supervisor_scoring",
        "baselines", "best_by_spearman", "best_by_lift",
    }
    assert expected <= set(payload)
    # the five always-present baselines (domain_support is off in this config)
    assert {"random", "min_altitude", "low_altitude_exposure", "uncertainty_only", "supervisor"} <= set(
        payload["baselines"]
    )
    for name, metrics in payload["baselines"].items():
        assert set(METRIC_KEYS) <= set(metrics), name
        assert metrics["n_flagged"] <= metrics["n_trajectories"]  # flagged subset of trajectories
    assert payload["best_by_spearman"] in payload["baselines"]
    assert payload["best_by_lift"] in payload["baselines"]

    data = _read_json(tmp_path / "baseline_comparison.json")
    assert data["_provenance"]["tool"] == "compare_risk_baselines"
    lines = (tmp_path / "baseline_comparison.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == ["baseline", *METRIC_KEYS]
    assert len(lines) - 1 == len(payload["baselines"])  # one row per baseline


# --------------------------------------------------------------------------- calibration + audit
def test_calibration_audit_schema_and_invariants(tmp_path):
    ca.run_and_write(_tiny_config(), out_dir=tmp_path)

    data = _read_json(tmp_path / "calibration_audit.json")
    assert {"config_path", "error_basis", "scope_note", "fit", "conformal", "screening", "audit"} <= set(data)
    assert data["error_basis"] == "true_force_model_error"
    assert {"alpha", "mode", "coverage"} <= set(data["conformal"])
    assert {"target_coverage", "coverage_before", "coverage_after"} <= set(data["conformal"]["coverage"])
    screening = data["screening"]
    assert {"scoring", "n_trajectories", "n_flagged"} <= set(screening)
    assert 0 <= screening["n_flagged"] <= screening["n_trajectories"]  # flagged subset of trajectories
    assert "_sentinel_rows" not in data  # sentinel rows are written to CSV, not the JSON
    assert data["_provenance"]["tool"] == "run_calibration_audit"

    lines = (tmp_path / "sentinel_audit.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == [
        "trajectory_id", "risk_score", "true_force_error", "is_high_force_error", "flagged"
    ]
    # sentinel rows are drawn from the ACCEPTED (not-flagged) set, so their flagged column is 0
    for row in lines[1:]:
        assert row.split(",")[-1] == "0"


# --------------------------------------------------------------------------- MC demo: import safety
def test_run_propagation_importable():
    # No file output to schema-test; its numerical core is in tests/test_uq_propagation.py. Guard
    # against import-time breakage (e.g. a bad import surfacing only when the smoke demo is run).
    import scripts.run_propagation as rp

    assert callable(rp.main)
