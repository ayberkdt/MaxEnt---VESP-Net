"""Round-trip tests for fitted-plugin persistence (VESPUQPlugin.save / .load).

A loaded plugin must predict, score, and screen identically to the plugin that was saved --
persistence exists so operational consumers (screening, CorrectedForceField, propagators) can
reuse a calibrated layer without refitting.
"""

from __future__ import annotations

import pytest
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin
from vesp.uq.correction import CorrectedForceField
from vesp.uq.plugin import PLUGIN_STATE_VERSION


def _query_shell(n: int, r_lo: float, r_hi: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)).unsqueeze(-1)
    return dirs * radii


def _fitted_plugin() -> VESPUQPlugin:
    sources = make_shell_sources([0.75, 0.9], [24, 32], dtype=torch.float64)
    sigma_true = 0.02 * torch.randn(
        sources.n_sources, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    positions = _query_shell(300, 1.05, 1.6, seed=1)
    A = build_acceleration_operator(positions, sources, eps=0.0, sign=1.0)
    error = (A @ sigma_true).reshape(3, positions.shape[0]).transpose(0, 1)
    plugin = VESPUQPlugin(
        sources,
        reg_method="fixed",
        lambda_l2=1.0e-8,
        noise_model="heteroscedastic",
        val_fraction=0.25,
        risk_scoring="supervisor_rel",
        domain_support=True,
        seed=0,
    )
    plugin.fit_error(positions, error)
    return plugin


def test_save_load_round_trip_predicts_identically(tmp_path):
    plugin = _fitted_plugin()
    path = tmp_path / "plugin.pt"
    plugin.save(path)
    assert path.exists()

    loaded = VESPUQPlugin.load(path)
    queries = _query_shell(64, 1.05, 1.6, seed=7)

    a = plugin.predict_uncertainty(queries)
    b = loaded.predict_uncertainty(queries)
    for name in ("mean_error", "sigma", "epistemic_sigma", "expected_error", "risk_score"):
        assert torch.allclose(getattr(a, name), getattr(b, name), rtol=1.0e-12, atol=0.0), name

    ca = plugin.predict_covariance_3x3(queries)
    cb = loaded.predict_covariance_3x3(queries)
    assert torch.allclose(ca.covariance, cb.covariance, rtol=1.0e-12, atol=0.0)

    # domain support state (train geometry + cached scales) must survive the round trip
    da = plugin.domain_support_score(queries)
    db = loaded.domain_support_score(queries)
    assert torch.allclose(da, db, rtol=1.0e-12, atol=0.0)

    sa = plugin.score_trajectory(queries)
    sb = loaded.score_trajectory(queries)
    assert sb.risk_score == pytest.approx(sa.risk_score, rel=1.0e-12)
    assert sb.scoring == sa.scoring

    # fit provenance and options ride along
    assert loaded.fit_info == plugin.fit_info
    assert loaded.risk_scoring == plugin.risk_scoring
    assert loaded.noise_model == plugin.noise_model
    assert loaded.altitude_noise is not None
    assert loaded.altitude_noise.log_a == pytest.approx(plugin.altitude_noise.log_a)
    assert loaded.posterior.lambda_l2 == pytest.approx(plugin.posterior.lambda_l2)


def test_loaded_plugin_drives_corrected_force_field(tmp_path):
    plugin = _fitted_plugin()
    path = tmp_path / "plugin.pt"
    plugin.save(path)
    loaded = VESPUQPlugin.load(path)

    def zero_surrogate(x):
        return torch.zeros_like(x)

    field_orig = CorrectedForceField(plugin, surrogate_accel_fn=zero_surrogate)
    field_loaded = CorrectedForceField(loaded, surrogate_accel_fn=zero_surrogate)
    x = _query_shell(16, 1.1, 1.5, seed=9)
    assert torch.allclose(field_orig.correction(x), field_loaded.correction(x), rtol=1.0e-12, atol=0.0)


def test_unfitted_plugin_refuses_to_save(tmp_path):
    sources = make_shell_sources([0.8], 16, dtype=torch.float64)
    plugin = VESPUQPlugin(sources)
    with pytest.raises(RuntimeError, match="not fitted"):
        plugin.save(tmp_path / "nope.pt")


def test_load_rejects_foreign_and_future_payloads(tmp_path):
    plugin = _fitted_plugin()

    bad_format = tmp_path / "foreign.pt"
    torch.save({"format": "something.else", "version": 1}, bad_format)
    with pytest.raises(ValueError, match="format"):
        VESPUQPlugin.load(bad_format)

    state = plugin.state_dict()
    state["version"] = PLUGIN_STATE_VERSION + 1
    future = tmp_path / "future.pt"
    torch.save(state, future)
    with pytest.raises(ValueError, match="version"):
        VESPUQPlugin.load(future)


def test_state_dict_is_weights_only_safe(tmp_path):
    plugin = _fitted_plugin()
    path = tmp_path / "plugin.pt"
    plugin.save(path)
    # the explicit contract: a clean, restricted unpickler must accept the payload
    state = torch.load(path, weights_only=True)
    assert state["format"] == "vesp.uq.plugin"
    rebuilt = VESPUQPlugin.from_state_dict(state)
    assert rebuilt.posterior is not None
