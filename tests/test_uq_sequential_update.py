"""Tests for the exact sequential posterior update (VESPUQPlugin.update_error).

The binding contract: with the SAME Tikhonov weight and noise floor, updating the fitted
posterior with new rows equals the batch fit on the concatenated data -- exactly (to floating
point), not approximately. Everything else (noise/altitude recalibration, domain-support
extension, provenance bookkeeping, persistence) is layered on top of that contract.
"""

from __future__ import annotations

import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.extensions.probabilistic import LinearGaussianPosterior
from vesp.uq import VESPUQPlugin


def _query_shell(n: int, r_lo: float, r_hi: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)).unsqueeze(-1)
    return dirs * radii


def _flatten(err: torch.Tensor) -> torch.Tensor:
    return torch.cat([err[:, 0], err[:, 1], err[:, 2]])


def _make_world(seed: int = 3):
    sources = make_shell_sources([0.8], 48, dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(
        sources.n_sources, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
    )

    def field(positions: torch.Tensor) -> torch.Tensor:
        A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
        return (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)

    return sources, field


def _fit_plugin(sources, field, *, noise_model="homoscedastic", lam=1.0e-6):
    pos1 = _query_shell(250, 1.10, 1.6, seed=1)
    val = _query_shell(120, 1.10, 1.6, seed=2)
    plugin = VESPUQPlugin(
        sources, reg_method="fixed", lambda_l2=lam, noise_model=noise_model, seed=0
    )
    plugin.fit_error(pos1, field(pos1), val_positions=val, val_error=field(val))
    return plugin, pos1, val


def test_update_equals_batch_fit_exactly():
    sources, field = _make_world()
    plugin, pos1, _ = _fit_plugin(sources, field)
    lam = plugin.posterior.lambda_l2
    noise_var = plugin.posterior.noise_var

    pos2 = _query_shell(150, 1.05, 1.5, seed=7)
    plugin.update_error(pos2, field(pos2))

    pos_cat = torch.cat([pos1, pos2], dim=0)
    op_cat = build_acceleration_operator(pos_cat, sources, eps=0.0, sign=1.0)
    expected = LinearGaussianPosterior.fit(
        op_cat, _flatten(field(pos_cat)), lambda_l2=lam, noise_var=noise_var
    )
    assert plugin.posterior.noise_var == pytest.approx(noise_var)  # no val -> noise floor fixed
    assert torch.allclose(plugin.posterior.mean, expected.mean, rtol=1.0e-8, atol=1.0e-12)
    assert torch.allclose(plugin.posterior.cov, expected.cov, rtol=1.0e-7, atol=1.0e-14)


def test_two_sequential_updates_equal_one_batch():
    sources, field = _make_world()
    plugin, pos1, _ = _fit_plugin(sources, field)
    lam, noise_var = plugin.posterior.lambda_l2, plugin.posterior.noise_var

    pos2 = _query_shell(90, 1.05, 1.5, seed=8)
    pos3 = _query_shell(70, 1.20, 1.7, seed=9)
    plugin.update_error(pos2, field(pos2))
    plugin.update_error(pos3, field(pos3))

    pos_cat = torch.cat([pos1, pos2, pos3], dim=0)
    op_cat = build_acceleration_operator(pos_cat, sources, eps=0.0, sign=1.0)
    expected = LinearGaussianPosterior.fit(
        op_cat, _flatten(field(pos_cat)), lambda_l2=lam, noise_var=noise_var
    )
    assert torch.allclose(plugin.posterior.mean, expected.mean, rtol=1.0e-8, atol=1.0e-12)
    assert torch.allclose(plugin.posterior.cov, expected.cov, rtol=1.0e-7, atol=1.0e-14)
    assert plugin.fit_info["n_updates"] == 2
    assert plugin.fit_info["n_train"] == pos_cat.shape[0]


def test_update_predictions_match_batch_posterior():
    sources, field = _make_world()
    plugin, pos1, _ = _fit_plugin(sources, field)
    pos2 = _query_shell(100, 1.05, 1.5, seed=10)
    plugin.update_error(pos2, field(pos2))

    queries = _query_shell(40, 1.1, 1.5, seed=11)
    pred = plugin.predict_uncertainty(queries)
    op_q = build_acceleration_operator(queries, sources, eps=0.0, sign=1.0)
    expected_mean = (op_q @ plugin.posterior.mean).reshape(3, queries.shape[0]).transpose(0, 1)
    assert torch.allclose(pred.mean_error, expected_mean, rtol=1.0e-10, atol=0.0)
    assert bool((pred.sigma > 0).all())


def test_update_recalibrates_noise_and_altitude_law_on_fresh_val():
    sources, field = _make_world()
    plugin, _, _ = _fit_plugin(sources, field, noise_model="heteroscedastic")
    g = torch.Generator().manual_seed(20)

    pos2 = _query_shell(120, 1.05, 1.5, seed=12)
    fresh_val = _query_shell(150, 1.05, 1.6, seed=13)
    fresh_val_err = field(fresh_val) + 5.0e-4 * torch.randn(150, 3, generator=g, dtype=torch.float64)
    plugin.update_error(pos2, field(pos2), val_positions=fresh_val, val_error=fresh_val_err)

    # the new noise floor is exactly the held-out mean-squared residual of the updated mean
    op_val = build_acceleration_operator(fresh_val, sources, eps=0.0, sign=1.0)
    resid = op_val @ plugin.posterior.mean - _flatten(fresh_val_err)
    assert plugin.posterior.noise_var == pytest.approx(float(torch.mean(resid * resid)), rel=1.0e-10)
    assert plugin.altitude_noise is not None
    assert plugin.altitude_noise.b >= 0.0
    assert plugin.fit_info["n_val"] == 150
    assert "altitude_noise_b" in plugin.fit_info


def test_update_extends_domain_support_geometry():
    sources, field = _make_world()
    plugin, pos1, _ = _fit_plugin(sources, field)
    plugin.domain_support = True
    n_before = int(plugin.train_positions.shape[0])

    # new low-altitude band that the original fit never saw
    pos2 = _query_shell(120, 1.02, 1.06, seed=14)
    probe = _query_shell(50, 1.02, 1.05, seed=15)
    score_before = plugin.domain_support_score(probe).mean()
    plugin.update_error(pos2, field(pos2))
    score_after = plugin.domain_support_score(probe).mean()

    assert int(plugin.train_positions.shape[0]) == n_before + 120
    assert float(score_after) < float(score_before), "new samples must become calibration support"


def test_update_guards():
    sources, field = _make_world()
    fresh = VESPUQPlugin(sources, reg_method="fixed", lambda_l2=1.0e-6, noise_model="homoscedastic")
    pos = _query_shell(10, 1.1, 1.5, seed=16)
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.update_error(pos, field(pos))

    plugin, _, _ = _fit_plugin(sources, field)
    with pytest.raises(ValueError, match="same"):
        plugin.update_error(pos, field(pos)[:5])
    with pytest.raises(ValueError, match="both val_positions and val_error"):
        plugin.update_error(pos, field(pos), val_positions=pos)
    with pytest.raises(ValueError, match="at least one"):
        plugin.update_error(torch.zeros(0, 3, dtype=torch.float64), torch.zeros(0, 3, dtype=torch.float64))


def test_updated_plugin_round_trips_through_save_load(tmp_path):
    sources, field = _make_world()
    plugin, _, _ = _fit_plugin(sources, field)
    pos2 = _query_shell(80, 1.05, 1.5, seed=17)
    plugin.update_error(pos2, field(pos2))

    path = tmp_path / "updated.pt"
    plugin.save(path)
    loaded = VESPUQPlugin.load(path)
    queries = _query_shell(30, 1.1, 1.5, seed=18)
    a, b = plugin.predict_uncertainty(queries), loaded.predict_uncertainty(queries)
    assert torch.allclose(a.sigma, b.sigma, rtol=1.0e-12, atol=0.0)
    assert torch.allclose(a.mean_error, b.mean_error, rtol=1.0e-12, atol=0.0)
    assert loaded.fit_info["n_updates"] == 1
