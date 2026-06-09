"""Tests for the exploratory online force-model correction (N6).

Pins the **operator consistency** of the correction (it must equal the posterior-mean force-error
field used everywhere else, honoring sign/eps), the integrator's basic behaviour, and the benchmark's
accuracy-improves / cost-increases / output-schema contract. Nothing here is a position-accuracy
claim -- the correction is the ridge posterior mean.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import scripts.run_force_correction_benchmark as fcb
from vesp.common.artifacts import compute_file_sha256
from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples
from vesp.uq.correction import CorrectedForceField, integrate_trajectory


def _fitted_plugin(*, acceleration_sign: float = 1.0, eps: float = 0.0, seed: int = 0):
    samples = make_synthetic_uq_samples(n=300, noise_std=1.0e-4, seed=1)
    src = make_shell_sources([0.75, 0.9], [24, 32], dtype=torch.float64)
    plugin = VESPUQPlugin(
        src, reg_method="lcurve", noise_model="heteroscedastic",
        acceleration_sign=acceleration_sign, eps=eps, seed=seed,
    )
    plugin.fit_error(samples.positions, samples.error)
    return plugin


def _zero(x):
    return torch.zeros_like(x)


@pytest.mark.parametrize(("sign", "eps"), [(1.0, 0.0), (-1.0, 1.0e-3)])
def test_correction_equals_posterior_mean_error(sign, eps):
    # The correction must be exactly the posterior-mean force-error field (same operator convention
    # as predict_uncertainty / the MC + STM propagators), honoring the fitted sign and softening.
    plugin = _fitted_plugin(acceleration_sign=sign, eps=eps)
    field = CorrectedForceField(plugin, surrogate_accel_fn=_zero)
    x = torch.tensor([[1.1, 0.0, 0.0], [0.2, 1.3, 0.1], [0.5, 0.5, 1.0]], dtype=torch.float64)
    assert torch.allclose(field.correction(x), plugin.predict_uncertainty(x).mean_error, atol=1.0e-12)


def test_correction_single_input_matches_batched_row():
    plugin = _fitted_plugin()
    field = CorrectedForceField(plugin, surrogate_accel_fn=_zero)
    x = torch.tensor([[1.1, 0.0, 0.0], [0.3, 1.2, 0.0]], dtype=torch.float64)
    single = field.correction(x[0])
    assert single.shape == (3,)
    assert torch.allclose(single, field.correction(x)[0], atol=1.0e-12)


def test_call_is_surrogate_plus_correction():
    plugin = _fitted_plugin()

    def surrogate(x):
        return torch.ones_like(x)  # arbitrary surrogate field

    field = CorrectedForceField(plugin, surrogate_accel_fn=surrogate)
    x = torch.tensor([[1.15, 0.0, 0.0], [0.4, 1.1, 0.2]], dtype=torch.float64)
    assert torch.allclose(field(x), surrogate(x) + field.correction(x), atol=1.0e-12)


def test_requires_fitted_plugin():
    src = make_shell_sources([0.8], [16], dtype=torch.float64)
    plugin = VESPUQPlugin(src, seed=0)  # not fitted
    with pytest.raises(RuntimeError):
        CorrectedForceField(plugin, surrogate_accel_fn=_zero)


def test_integrate_trajectory_zero_accel_is_linear_motion():
    # zero acceleration -> constant velocity: r(t) = r0 + v0 t
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    times, states = integrate_trajectory(lambda r: torch.zeros(3, dtype=torch.float64),
                                          y0, dt=0.1, duration=2.0, output_dt=0.5)
    assert states.shape == (times.shape[0], 6)
    assert np.allclose(states[:, 0], 1.0)  # x unchanged (vx = 0)
    assert states[-1, 1] == pytest.approx(0.5 * times[-1], abs=1.0e-9)  # y = vy * t


def test_integrate_trajectory_determinism_and_bad_state():
    def accel(r):  # point mass
        return -r / (torch.dot(r, r) * torch.sqrt(torch.dot(r, r)))

    y0 = np.array([1.2, 0.0, 0.0, 0.0, np.sqrt(1.0 / 1.2), 0.0])
    _, a = integrate_trajectory(accel, y0, dt=0.05, duration=3.0, output_dt=0.5)
    _, b = integrate_trajectory(accel, y0, dt=0.05, duration=3.0, output_dt=0.5)
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        integrate_trajectory(accel, np.zeros(5), dt=0.1, duration=1.0, output_dt=0.5)


def _tiny_config():
    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "data": {"type": "synthetic", "n": 300, "n_truth_sources": 16, "noise_std": 1.0e-4},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {"regularization": {"method": "lcurve"}, "noise_model": "heteroscedastic"},
        "_config_path": "correction_test.yaml",
    }


_PARAMS = {"mu": 1.0, "r_initial": 1.1, "duration": 4.0, "dt": 0.1, "output_dt": 0.5}


def test_benchmark_improves_accuracy_and_costs_more():
    result = fcb.run_force_correction_benchmark(_tiny_config(), _PARAMS, cost_reps=5)
    s, c = result["summary"], result["cost"]
    # the correction reduces the integrated position error (corrected closer to the reference)...
    assert s["final_corrected_position_error"] < s["final_surrogate_position_error"]
    assert s["final_improvement_factor"] > 1.0
    # ...at a higher per-RHS cost (it evaluates the full equivalent-source field every call)
    assert c["per_rhs_cost_ratio"] > 1.0
    assert len(result["surrogate_position_error"]) == result["n_steps"]
    assert len(result["corrected_position_error"]) == result["n_steps"]


def test_benchmark_writes_artifacts(tmp_path):
    fcb.run_and_write(_tiny_config(), _PARAMS, out_dir=tmp_path, cost_reps=5)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool"] == "run_force_correction_benchmark"
    for name in ("force_correction_benchmark.json", "force_correction_benchmark.md", "force_correction_errors.csv"):
        assert name in manifest["artifacts"]
        assert manifest["artifacts"][name]["sha256"] == compute_file_sha256(tmp_path / name)

    data = json.loads((tmp_path / "force_correction_benchmark.json").read_text(encoding="utf-8"))
    assert data["_provenance"]["tool"] == "run_force_correction_benchmark"
    assert {"summary", "cost", "times", "surrogate_position_error", "corrected_position_error"} <= set(data)

    lines = (tmp_path / "force_correction_errors.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == ["time", "surrogate_position_error", "corrected_position_error"]
    assert len(lines) - 1 == data["n_steps"]
