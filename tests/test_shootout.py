"""Tests for the E7 regularizer shootout (L2 vs entropy at matched error)."""

from __future__ import annotations

from pathlib import Path

from vesp.feasibility.experiments.runner import expand_trials, load_experiment_config
from vesp.feasibility.experiments.shootout import shootout_report

ROOT = Path(__file__).resolve().parents[1]


def _row(name, solver, err, cancel, sigma, top5, eff_n, mode=""):
    return {
        "run_name": name,
        "solver": solver,
        "entropy_mode": mode,
        "relative_acceleration_rmse": err,
        "shell_cancellation_ratio": cancel,
        "sigma_l2": sigma,
        "top_5pct_source_contribution": top5,
        "effective_source_count": eff_n,
    }


def test_shootout_config_expands_into_two_families():
    cfg = load_experiment_config(ROOT / "configs" / "experiments" / "synthetic_regularizer_shootout.yaml")
    trials = expand_trials(cfg)
    solvers = {t.config["solver"]["type"] for t in trials}
    assert solvers == {"ridge", "maxent"}
    assert sum(1 for t in trials if t.config["solver"]["type"] == "ridge") == 6
    assert sum(1 for t in trials if t.config["solver"]["type"] == "maxent") == 5
    # maxent trials carry the constrained mode
    for t in trials:
        if t.config["solver"]["type"] == "maxent":
            assert t.config["maxent"]["mode"] == "constrained"


def test_shootout_report_aligns_by_error_and_picks_winner():
    rows = [
        _row("ridge_a", "ridge", 0.05, 10.0, 100.0, 0.50, 100),
        _row("ridge_b", "ridge", 0.10, 2.0, 10.0, 0.30, 200),
        _row("ridge_c", "ridge", 0.15, 1.0, 2.0, 0.25, 250),
        _row("maxent_x", "maxent", 0.10, 5.0, 50.0, 0.40, 150, mode="positive_negative"),
    ]
    md, matched, tally = shootout_report(rows)
    assert len(matched) == 1
    e = matched[0]
    # ridge interpolated at err=0.10 is exactly ridge_b
    assert abs(e["ridge_at_err_shell_cancellation_ratio"] - 2.0) < 1e-9
    assert abs(e["ridge_at_err_effective_source_count"] - 200.0) < 1e-9
    # maxent worse on cancellation (5 > 2, lower is better) and on eff_N (150 < 200, higher is better)
    assert e["winner_shell_cancellation_ratio"] == "ridge"
    assert e["winner_effective_source_count"] == "ridge"
    assert tally["shell_cancellation_ratio"]["ridge"] == 1
    assert "Verdict" in md


def test_shootout_report_detects_entropy_win_on_concentration():
    # construct a case where maxent is LESS concentrated than ridge at matched error
    rows = [
        _row("ridge_a", "ridge", 0.05, 1.0, 5.0, 0.60, 100),
        _row("ridge_b", "ridge", 0.10, 1.0, 5.0, 0.55, 110),
        _row("maxent_x", "maxent", 0.10, 1.0, 5.0, 0.30, 200, mode="abs"),
    ]
    _md, matched, tally = shootout_report(rows)
    e = matched[0]
    assert e["winner_top_5pct_source_contribution"] == "maxent"  # 0.30 < 0.55
    assert e["winner_effective_source_count"] == "maxent"  # 200 > 110
    assert tally["top_5pct_source_contribution"]["maxent"] == 1
