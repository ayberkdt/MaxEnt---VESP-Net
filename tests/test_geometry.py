"""Tests for the E8 source-geometry shootout."""

from __future__ import annotations

from pathlib import Path

from vesp.feasibility.experiments.geometry import geometry_report
from vesp.feasibility.experiments.runner import expand_trials, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def _row(name, test_low, cancel=1.0, sigma=1.0, eff_n=100.0, sel_lambda=1.0e-3):
    return {
        "run_name": name,
        "test_low_acceleration_rmse": test_low,
        "low_altitude_acceleration_rmse": test_low,
        "low_to_high_error_ratio": 10.0,
        "relative_acceleration_rmse": 0.1,
        "shell_cancellation_ratio": cancel,
        "sigma_l2": sigma,
        "effective_source_count": eff_n,
        "selected_lambda_l2": sel_lambda,
    }


def test_geometry_config_expands_into_families():
    cfg = load_experiment_config(ROOT / "configs" / "experiments" / "synthetic_geometry_shootout.yaml")
    trials = expand_trials(cfg)
    names = {t.name for t in trials}
    assert {"single_086", "multi_baseline", "surface_dense", "multi_resolution"} <= names
    # each trial overrides geometry and uses auto lambda
    for t in trials:
        assert t.config["model"]["type"] == "multishell"
        assert t.config["loss"]["lambda_l2"] == "auto"
        assert isinstance(t.config["model"]["shell_alphas"], list)


def test_geometry_report_ranks_by_low_altitude_error():
    rows = [
        _row("multi_baseline", 0.20),
        _row("surface_dense", 0.08, cancel=1.2),
        _row("deep_only", 0.35),
    ]
    md, ranking = geometry_report(rows)
    # ranked ascending by low-altitude error -> surface_dense first
    assert ranking[0]["run_name"] == "surface_dense"
    assert "Best low-altitude geometry: `surface_dense`" in md
    # baseline comparison present and recognizes the improvement
    assert "reduces" in md


def test_geometry_report_handles_no_improvement():
    rows = [
        _row("multi_baseline", 0.20),
        _row("surface_dense", 0.199),  # essentially no change
    ]
    md, ranking = geometry_report(rows)
    assert ranking[0]["run_name"] == "surface_dense"
    # within ~2% -> should flag "does not materially help" / band-limit ceiling
    assert "band-limit" in md or "does **not**" in md
