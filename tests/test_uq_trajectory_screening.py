"""Tests for end-to-end trajectory risk screening with a fitted plugin."""

from __future__ import annotations

import pytest
import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin, make_synthetic_uq_samples, run_risk_screening
from vesp.uq.ensemble import nearest_neighbor_error_magnitude


def _fitted_plugin():
    # interior-source error field naturally grows toward low altitude
    s = make_synthetic_uq_samples(n=600, noise_std=5.0e-5, seed=1)
    src = make_shell_sources([0.75, 0.9], [48, 64], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", noise_model="heteroscedastic", seed=0)
    plugin.fit_error(s.positions, s.error)
    return plugin, s


def _circular_orbit(radius: float, n: int = 40, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    theta = torch.linspace(0.0, 2.0 * torch.pi, n + 1, dtype=torch.float64)[:-1]
    plane = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    return radius * (plane @ q.T)


def test_low_altitude_trajectory_scores_higher_risk():
    plugin, _ = _fitted_plugin()
    low = _circular_orbit(1.05, seed=1)
    high = _circular_orbit(1.50, seed=1)
    # legacy sigma modes plus the new expected-error / supervisor modes must all rank low > high
    for scoring in ("low_alt_integral", "combined", "max", "expected", "supervisor", "supervisor_p95"):
        s_low = plugin.score_trajectory(low, scoring=scoring)
        s_high = plugin.score_trajectory(high, scoring=scoring)
        assert s_low.risk_score > s_high.risk_score, scoring


def _shell_cloud(n, r_lo, r_hi, seed):
    g = torch.Generator().manual_seed(seed)
    radii = r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    return dirs, dirs * radii.unsqueeze(1)


def test_domain_support_score_grows_outside_calibration_support():
    # training cloud confined to a radius shell [1.1, 1.5]
    dirs, pos = _shell_cloud(800, 1.1, 1.5, seed=0)
    g = torch.Generator().manual_seed(7)
    err = 1.0e-4 * torch.randn(pos.shape[0], 3, generator=g, dtype=torch.float64)
    src = make_shell_sources([0.8], [32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", seed=0, domain_support=True, val_fraction=0.25)
    plugin.fit_error(pos, err)

    inside = dirs[:64] * 1.30  # mid-shell, along training directions -> well supported
    below = dirs[:64] * 1.02  # below the radius range
    far = dirs[:64] * 3.00  # far outside radially

    s_inside = float(plugin.domain_support_score(inside).mean())
    s_below = float(plugin.domain_support_score(below).mean())
    s_far = float(plugin.domain_support_score(far).mean())

    assert s_inside < s_below
    assert s_below < s_far
    assert s_far > 1.0  # clearly unsupported


def test_domain_support_raises_pointwise_supervisor_risk():
    dirs, pos = _shell_cloud(800, 1.1, 1.5, seed=2)
    g = torch.Generator().manual_seed(9)
    err = 1.0e-4 * torch.randn(pos.shape[0], 3, generator=g, dtype=torch.float64)
    src = make_shell_sources([0.8], [32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", seed=0, domain_support=True, val_fraction=0.25)
    plugin.fit_error(pos, err)

    supported = dirs[:48] * 1.30
    extrapolated = dirs[:48] * 1.02  # below support -> larger domain penalty
    s_in = plugin.score_trajectory(supported, scoring="supervisor")
    s_out = plugin.score_trajectory(extrapolated, scoring="supervisor")
    # the extrapolated pass should carry strictly more domain risk and higher supervisor risk
    assert s_out.max_domain_risk > s_in.max_domain_risk
    assert s_out.risk_score > s_in.risk_score


def test_screening_flags_requested_fraction_and_beats_random():
    plugin, samples = _fitted_plugin()
    radii = torch.linspace(1.04, 1.55, 60, dtype=torch.float64)
    trajectories = [_circular_orbit(float(r), n=36, seed=i) for i, r in enumerate(radii)]
    true_error = torch.tensor(
        [
            float(nearest_neighbor_error_magnitude(t, samples.positions, samples.error).max())
            for t in trajectories
        ],
        dtype=torch.float64,
    )

    result = run_risk_screening(plugin, trajectories, true_error=true_error, rerun_fraction=0.2, scoring="max")
    report = result["risk_screening_report"]
    assert len(result["trajectory_scores"]) == len(trajectories)
    assert 0.15 <= report.rerun_fraction <= 0.27  # ~requested 20%
    # the screen should beat the 0.2 random-capture baseline for the top decile
    assert report.capture_rate > 0.4
    assert report.mean_error_flagged > report.mean_error_accepted
    assert report.spearman_risk_vs_error > 0.3


# ============================ Phase 2: robustness upgrades ============================

def _angular_sector_plugin(seed=0):
    """Fit a plugin whose training support is a narrow angular cone around +x at r~1.3-1.6."""

    g = torch.Generator().manual_seed(seed)
    n = 700
    radii = 1.30 + 0.30 * torch.rand(n, generator=g, dtype=torch.float64)  # [1.30, 1.60]
    perturb = 0.12 * torch.randn(n, 2, generator=g, dtype=torch.float64)  # tight cone around +x
    dirs = torch.stack([torch.ones(n, dtype=torch.float64), perturb[:, 0], perturb[:, 1]], dim=1)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    pos = dirs * radii.unsqueeze(1)
    err = 1.0e-4 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    src = make_shell_sources([0.8], [32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", seed=0, domain_support=True, val_fraction=0.25)
    plugin.fit_error(pos, err)
    return plugin


# ---- P3: decomposed domain-support components ----

def test_domain_support_components_sum_to_total_and_decompose():
    # full-sphere shell [1.30, 1.60]: the radial term is the OOD axis here (angle is supported).
    dirs, pos = _shell_cloud(800, 1.30, 1.60, seed=0)
    g = torch.Generator().manual_seed(7)
    err = 1.0e-4 * torch.randn(pos.shape[0], 3, generator=g, dtype=torch.float64)
    src = make_shell_sources([0.8], [32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", seed=0, domain_support=True, val_fraction=0.25)
    plugin.fit_error(pos, err)

    inside = dirs[:64] * 1.45  # mid-shell -> well supported
    below = dirs[:64] * 1.05  # below the radius range -> radial term active
    far_radial = dirs[:64] * 3.0  # far above the range -> both terms large

    for q in (inside, below, far_radial):
        c = plugin.domain_support_components(q)
        total = c["distance_score"] + c["radius_penalty"] + c["angular_score"]
        assert torch.allclose(total, c["total_score"])  # components sum to the total
        for key in ("distance_score", "radius_penalty", "angular_score", "total_score"):
            assert bool((c[key] >= 0).all())  # all nonnegative

    c_in = plugin.domain_support_components(inside)
    c_below = plugin.domain_support_components(below)
    c_far = plugin.domain_support_components(far_radial)
    # below the radius range -> the radial penalty is active and dominates the distance term
    assert float(c_below["radius_penalty"].mean()) > float(c_below["distance_score"].mean())
    # inside the support -> no radial penalty, low total
    assert float(c_in["radius_penalty"].mean()) == 0.0
    assert float(c_in["total_score"].mean()) < float(c_below["total_score"].mean())
    # far radial -> radial penalty large, total exceeds the just-below case
    assert float(c_far["radius_penalty"].mean()) > float(c_below["radius_penalty"].mean())
    assert float(c_far["total_score"].mean()) > float(c_below["total_score"].mean())


def test_domain_distance_component_captures_angular_ood_at_same_radius():
    # angular-sector training: a same-radius query outside the cone must raise the DISTANCE term
    # while the radial penalty stays zero (altitude is not the OOD signal here).
    plugin = _angular_sector_plugin()
    r = 1.45  # inside the trained radius band [1.30, 1.60]
    c_in = plugin.domain_support_components(torch.tensor([[r, 0.0, 0.0]], dtype=torch.float64))
    c_out = plugin.domain_support_components(torch.tensor([[-r, 0.0, 0.0]], dtype=torch.float64))
    assert float(c_in["radius_penalty"]) == 0.0
    assert float(c_out["radius_penalty"]) == 0.0  # same altitude -> no radial term for either
    assert float(c_out["distance_score"]) > float(c_in["distance_score"])  # distance carries angular OOD


def test_domain_angular_component_optional_and_off_by_default():
    dirs, pos = _shell_cloud(600, 1.30, 1.60, seed=3)
    g = torch.Generator().manual_seed(1)
    err = 1.0e-4 * torch.randn(pos.shape[0], 3, generator=g, dtype=torch.float64)
    src = make_shell_sources([0.8], [32], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", seed=0, domain_support=True, domain_angular_weight=2.0)
    plugin.fit_error(pos, err)
    far_angular = -dirs[:32] * 1.45  # same radius, opposite hemisphere
    c = plugin.domain_support_components(far_angular)
    assert float(c["angular_score"].mean()) > 0.0  # angular term engages when its weight > 0


# ---- P9: non-altitude (angular) out-of-support ----

def test_angular_ood_raises_domain_score_at_same_radius():
    plugin = _angular_sector_plugin()
    r = 1.45  # inside the trained radius band for BOTH queries (altitude is not the OOD signal)
    in_sector = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float64)  # along +x (trained cone)
    out_sector = torch.tensor([[-r, 0.0, 0.0]], dtype=torch.float64)  # -x, same radius, untrained
    s_in = float(plugin.domain_support_score(in_sector))
    s_out = float(plugin.domain_support_score(out_sector))
    assert s_out > s_in
    assert s_out > 1.0  # clearly out of support despite identical altitude


def test_supervisor_flags_angular_ood_trajectory_same_altitude():
    plugin = _angular_sector_plugin()
    theta = torch.linspace(0.0, 2.0 * torch.pi, 24, dtype=torch.float64)[:-1]
    # both rings at radius 1.45 (same altitude); in-cone wobbles around +x, out-cone around -x
    in_ring = torch.stack(
        [1.45 * torch.ones_like(theta), 0.1 * torch.cos(theta), 0.1 * torch.sin(theta)], dim=1
    )
    out_ring = torch.stack(
        [-1.45 * torch.ones_like(theta), 0.1 * torch.cos(theta), 0.1 * torch.sin(theta)], dim=1
    )
    in_ring = 1.45 * in_ring / in_ring.norm(dim=1, keepdim=True)
    out_ring = 1.45 * out_ring / out_ring.norm(dim=1, keepdim=True)
    s_in = plugin.score_trajectory(in_ring, scoring="supervisor")
    s_out = plugin.score_trajectory(out_ring, scoring="supervisor")
    # supervisor (with domain support) flags the out-of-sector ring even at identical altitude;
    # the explicit domain-support term rises despite the unchanged radius.
    assert s_out.max_domain_risk > s_in.max_domain_risk
    assert s_out.risk_score > s_in.risk_score


# ---- P7: plugin-level calibration threshold ----

def test_plugin_calibrate_threshold_then_zero_alarms_on_safe_set():
    plugin, samples = _fitted_plugin()
    # calibrate an absolute pointwise budget at the 99th percentile of held-out expected_error
    held = samples.positions
    thr = plugin.calibrate_risk_threshold(held, quantile=0.99)
    assert thr > 0.0
    # a benign high-altitude ensemble scored in absolute mode should mostly/entirely clear it
    safe = [_circular_orbit(1.55, n=30, seed=i) for i in range(20)]
    result = run_risk_screening(plugin, safe, threshold=thr, scoring="expected_abs_p95")
    report = result["risk_screening_report"]
    assert report.selection_mode == "threshold"
    assert report.n_flagged <= 2  # safe set raises few/zero alarms under the calibrated budget
