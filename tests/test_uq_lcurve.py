"""Tests for the operator-only L-curve regularization helper and the screening ensemble."""

from __future__ import annotations

import pytest
import torch

from vesp.core.regularization import DEFAULT_LAMBDA_GRID, lcurve_lambda
from vesp.uq.ensemble import generate_orbit_ensemble, nearest_neighbor_error_magnitude


def test_lcurve_returns_grid_value_and_monotone_curve():
    torch.manual_seed(0)
    A = torch.randn(120, 20, dtype=torch.float64)
    sigma = torch.randn(20, dtype=torch.float64)
    b = A @ sigma + 0.05 * torch.randn(120, dtype=torch.float64)

    lam, points = lcurve_lambda(A, b)
    assert lam in DEFAULT_LAMBDA_GRID
    assert len(points) == len(DEFAULT_LAMBDA_GRID)
    # residual norm increases and solution norm decreases as lambda grows (Tikhonov monotonicity)
    res = [p["residual_norm"] for p in points]
    sol = [p["solution_norm"] for p in points]
    assert all(res[i] <= res[i + 1] + 1.0e-9 for i in range(len(res) - 1))
    assert all(sol[i] >= sol[i + 1] - 1.0e-9 for i in range(len(sol) - 1))


def test_lcurve_closed_form_matches_direct_ridge_solution_norms():
    torch.manual_seed(1)
    A = torch.randn(80, 15, dtype=torch.float64)
    b = torch.randn(80, dtype=torch.float64)
    _, points = lcurve_lambda(A, b, grid=[1.0e-3, 1.0, 100.0])
    eye = torch.eye(15, dtype=torch.float64)
    for p in points:
        lam = p["lambda_l2"]
        mu = torch.linalg.solve(A.T @ A + lam * eye, A.T @ b)
        assert p["solution_norm"] == pytest.approx(float(torch.linalg.norm(mu)), rel=1e-6)
        assert p["residual_norm"] == pytest.approx(float(torch.linalg.norm(A @ mu - b)), rel=1e-6)


def test_orbit_ensemble_shapes_and_bounds():
    ens = generate_orbit_ensemble(n_orbits=25, n_points=40, r_peri_range=(1.05, 1.30), r_apo_range=(1.30, 1.6), seed=0)
    assert len(ens.trajectories) == 25
    for traj, rp, ra in zip(ens.trajectories, ens.periapsis, ens.apoapsis):
        assert traj.shape == (40, 3)
        r = torch.linalg.norm(traj, dim=-1)
        assert float(ra) >= float(rp)
        # sampled radii stay within [periapsis, apoapsis] up to numerical slack
        assert float(r.min()) >= float(rp) - 1.0e-6
        assert float(r.max()) <= float(ra) + 1.0e-6


def test_nearest_neighbor_error_magnitude_picks_closest():
    ref_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    ref_err = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=torch.float64)  # mags 5, 5
    query = torch.tensor([[0.9, 0.0, 0.0], [0.0, 1.9, 0.0]], dtype=torch.float64)
    mags = nearest_neighbor_error_magnitude(query, ref_pos, ref_err)
    assert torch.allclose(mags, torch.tensor([5.0, 5.0], dtype=torch.float64))
