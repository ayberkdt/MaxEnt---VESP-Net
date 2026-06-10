"""Equivalence tests for query-chunked prediction and batched ensemble scoring.

The batched ``score_ensemble`` path and the ``query_chunk_size`` chunking are pure
amortizations: every per-point / per-trajectory number must match the sequential,
single-block computation to floating-point noise. These tests pin that contract.
"""

from __future__ import annotations

import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin
from vesp.uq.ensemble import generate_orbit_ensemble

RTOL = 1.0e-9
ATOL = 1.0e-15


def _query_shell(n: int, r_lo: float, r_hi: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)).unsqueeze(-1)
    return dirs * radii


def _fitted_plugin(**kw) -> VESPUQPlugin:
    sources = make_shell_sources([0.8], 48, dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(
        sources.n_sources, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    positions = _query_shell(400, 1.05, 1.6, seed=1)
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error = (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)
    defaults = dict(
        reg_method="fixed",
        lambda_l2=1.0e-8,
        noise_model="heteroscedastic",
        val_fraction=0.25,
        risk_scoring="supervisor_rel",
        domain_support=True,
        seed=0,
    )
    defaults.update(kw)
    plugin = VESPUQPlugin(sources, **defaults)
    plugin.fit_error(positions, error)
    return plugin


def _assert_close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    assert torch.allclose(a, b, rtol=RTOL, atol=ATOL), f"{what} differs (max abs diff {(a - b).abs().max()})"


# ---------------------------------------------------------------- query chunking
def test_predict_uncertainty_chunked_matches_single_block():
    plugin = _fitted_plugin()
    queries = _query_shell(257, 1.05, 1.6, seed=7)  # deliberately not divisible by the chunk
    full = plugin.predict_uncertainty(queries)
    plugin.query_chunk_size = 64
    chunked = plugin.predict_uncertainty(queries)
    for name in (
        "radius",
        "mean_error",
        "std_components",
        "sigma",
        "epistemic_sigma",
        "mean_error_magnitude",
        "expected_error",
        "epistemic_fraction",
        "risk_score",
    ):
        _assert_close(getattr(chunked, name), getattr(full, name), f"predict_uncertainty.{name}")


@pytest.mark.parametrize("mode", ["exact", "diagonal", "lowrank"])
def test_predict_covariance_chunked_matches_single_block(mode):
    plugin = _fitted_plugin(covariance_mode=mode, lowrank_rank=16)
    queries = _query_shell(131, 1.05, 1.6, seed=9)
    full = plugin.predict_covariance_3x3(queries)
    plugin.query_chunk_size = 32
    chunked = plugin.predict_covariance_3x3(queries)
    for name in ("mean_error", "covariance", "std_components", "sigma"):
        _assert_close(getattr(chunked, name), getattr(full, name), f"predict_covariance.{name}")


def test_evaluate_calibration_chunked_matches_single_block():
    plugin = _fitted_plugin()
    held = _query_shell(180, 1.05, 1.6, seed=11)
    err = 1.0e-3 * torch.randn(180, 3, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    full = plugin.evaluate_calibration(held, err)
    plugin.query_chunk_size = 50
    chunked = plugin.evaluate_calibration(held, err)
    assert full.keys() == chunked.keys()
    for band, metrics in full.items():
        if not isinstance(metrics, dict):
            assert chunked[band] == pytest.approx(metrics, rel=1.0e-9)
            continue
        for key, value in metrics.items():
            assert chunked[band][key] == pytest.approx(value, rel=1.0e-9, abs=1.0e-15), f"{band}.{key}"


def test_invalid_query_chunk_size_rejected():
    sources = make_shell_sources([0.8], 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="query_chunk_size"):
        VESPUQPlugin(sources, query_chunk_size=0)


def test_from_config_reads_query_chunk_size():
    cfg = {
        "model": {"type": "single", "shell_alpha": 0.8, "n_source": 16},
        "uq": {"query_chunk_size": 123},
    }
    assert VESPUQPlugin.from_config(cfg).query_chunk_size == 123
    cfg["uq"]["query_chunk_size"] = None
    assert VESPUQPlugin.from_config(cfg).query_chunk_size is None


# ---------------------------------------------------------------- batched ensemble scoring
@pytest.mark.parametrize("use_weights", [False, True])
def test_score_ensemble_matches_sequential_score_trajectory(use_weights):
    plugin = _fitted_plugin()
    plugin.query_chunk_size = 512  # force multi-chunk batching across the ensemble
    ens = generate_orbit_ensemble(n_orbits=12, n_points=33, seed=5, dtype=torch.float64)
    trajectories = ens.trajectories
    if use_weights:
        weights = [torch.linalg.norm(t, dim=-1) ** 2 for t in trajectories]
        weights[3] = None  # mixed None entries must keep that trajectory uniform
    else:
        weights = None

    batched = plugin.score_ensemble(trajectories, weights=weights)
    sequential = [
        plugin.score_trajectory(t, weights=None if weights is None else weights[i])
        for i, t in enumerate(trajectories)
    ]
    assert len(batched) == len(sequential) == len(trajectories)
    for got, want in zip(batched, sequential, strict=True):
        for key, value in want.to_dict().items():
            assert got.to_dict()[key] == pytest.approx(value, rel=1.0e-9, abs=1.0e-15, nan_ok=True), key


def test_score_ensemble_empty_and_mismatched_weights():
    plugin = _fitted_plugin()
    assert plugin.score_ensemble([]) == []
    traj = _query_shell(8, 1.1, 1.5, seed=21)
    with pytest.raises(ValueError, match="one weight vector per trajectory"):
        plugin.score_ensemble([traj, traj], weights=[None])


def test_score_ensemble_rejects_bad_trajectory_shape():
    plugin = _fitted_plugin()
    with pytest.raises(ValueError, match="positions"):
        plugin.score_ensemble([torch.zeros(4, 2)])
