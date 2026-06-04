import math

import torch

from vesp.data.target_scaling import TargetScales, altitude_row_weights, observation_row_weights


def _positions(radii):
    base = torch.tensor(radii, dtype=torch.float64)
    pos = torch.zeros((len(radii), 3), dtype=torch.float64)
    pos[:, 0] = base  # put all magnitude on x so ||x|| == radius
    return pos


def test_altitude_row_weights_boost_below_threshold():
    positions = _positions([1.05, 1.20, 1.10, 1.50])
    cfg = {"loss": {"altitude_weighting": {"enabled": True, "r_threshold": 1.15, "boost": 4.0}}}
    weights = altitude_row_weights(positions, cfg)
    expected = torch.tensor([math.sqrt(4.0), 1.0, math.sqrt(4.0), 1.0], dtype=torch.float64)
    assert torch.allclose(weights, expected)


def test_altitude_row_weights_disabled_returns_none():
    positions = _positions([1.05, 1.20])
    assert altitude_row_weights(positions, {"loss": {"altitude_weighting": {"enabled": False}}}) is None


def test_same_altitude_weight_across_acceleration_components():
    nq = 4
    scales = TargetScales(normalize_targets=False)
    altitude = torch.tensor([2.0, 1.0, 3.0, 1.0], dtype=torch.float64)
    weights = observation_row_weights(
        n_query=nq,
        include_potential=True,
        include_acceleration=True,
        lambda_potential=1.0,
        lambda_acceleration=1.0,
        scales=scales,
        dtype=torch.float64,
        device="cpu",
        altitude_weights=altitude,
    )
    # blocks are [potential(nq), ax(nq), ay(nq), az(nq)]
    pot = weights[0:nq]
    ax = weights[nq : 2 * nq]
    ay = weights[2 * nq : 3 * nq]
    az = weights[3 * nq : 4 * nq]
    # For each query, potential and all three acceleration rows share the same weight.
    assert torch.allclose(pot, ax)
    assert torch.allclose(ax, ay)
    assert torch.allclose(ay, az)
    assert torch.allclose(ax, altitude)  # lambda=1, no normalization -> just the altitude factor


def test_boost_one_is_noop():
    nq = 3
    scales = TargetScales(normalize_targets=False)
    ones = torch.ones(nq, dtype=torch.float64)
    with_w = observation_row_weights(
        n_query=nq, include_potential=True, include_acceleration=True,
        lambda_potential=0.2, lambda_acceleration=1.0, scales=scales,
        dtype=torch.float64, device="cpu", altitude_weights=ones,
    )
    without = observation_row_weights(
        n_query=nq, include_potential=True, include_acceleration=True,
        lambda_potential=0.2, lambda_acceleration=1.0, scales=scales,
        dtype=torch.float64, device="cpu", altitude_weights=None,
    )
    assert torch.allclose(with_w, without)
