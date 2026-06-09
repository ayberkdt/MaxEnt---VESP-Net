"""Smoke test for the force-error ranking benchmark (the core VESP-UQ claim)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_force_error_benchmark import force_error_benchmark  # noqa: E402


def _tiny_config():
    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "data": {"type": "synthetic", "n": 240, "noise_std": 1.0e-4, "train_fraction": 0.7},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {
            "risk": {"scoring": "supervisor_rel", "low_altitude_radius": 1.15},
            "screening": {"n_orbits": 30, "n_points": 24, "true_error_aggregator": "p95"},
        },
    }


def test_force_error_benchmark_is_force_not_position():
    n = force_error_benchmark(_tiny_config(), scoring="supervisor_rel_p95", rerun_fraction=0.2)
    assert n["is_position_error_benchmark"] is False
    assert n["benchmark"] == "force_error_ranking"
    assert n["true_error_mode"].startswith("nn_oracle")
    assert n["n_trajectories"] == 30
    # the score record is one per trajectory and carries the true FORCE error (not position error)
    assert len(n["_scores"]) == 30
    assert n["spearman_force_risk_vs_true_force_error"] is not None
