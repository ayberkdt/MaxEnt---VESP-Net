"""Tests for trajectory risk scoring and selective-rerun logic."""

from __future__ import annotations

import math

import pytest
import torch

from vesp.uq.trajectory import RiskScreeningReport, score_sigma_profile, select_reruns


def test_score_sigma_profile_basic_aggregations():
    sigma = torch.tensor([1.0, 3.0, 2.0])
    radius = torch.tensor([1.05, 1.50, 1.20])
    s = score_sigma_profile(sigma, radius, scoring="max", sigma_threshold=2.5, low_altitude_radius=1.15)
    assert s.max_sigma == pytest.approx(3.0)
    assert s.mean_sigma == pytest.approx(2.0)
    # only radius 1.05 is below 1.15 -> integral picks up sigma=1.0
    assert s.low_altitude_sigma_integral == pytest.approx(1.0)
    # one of three points exceeds threshold 2.5
    assert s.time_above_threshold == pytest.approx(1.0 / 3.0)
    assert s.risk_score == pytest.approx(3.0)
    assert s.min_radius == pytest.approx(1.05)


def test_combined_altitude_risk_rewards_low_uncertain_points():
    radius = torch.tensor([1.05, 1.50])
    # same sigma; the low-altitude point gets a far larger altitude weight
    low_heavy = score_sigma_profile(torch.tensor([2.0, 0.0]), radius, scoring="combined")
    high_heavy = score_sigma_profile(torch.tensor([0.0, 2.0]), radius, scoring="combined")
    assert low_heavy.combined_altitude_risk > high_heavy.combined_altitude_risk


def test_score_empty_or_invalid_raises():
    with pytest.raises(ValueError):
        score_sigma_profile(torch.tensor([]), torch.tensor([]))
    with pytest.raises(ValueError):
        score_sigma_profile(torch.tensor([1.0]), torch.tensor([1.0]), scoring="nope")


def test_select_reruns_by_fraction_flags_top_subset():
    risk = torch.arange(100, dtype=torch.float64)
    report = select_reruns(risk, rerun_fraction=0.2)
    assert isinstance(report, RiskScreeningReport)
    assert report.n_flagged == 20
    assert 0.18 <= report.rerun_fraction <= 0.22
    assert min(report.flagged_indices) >= 80


def test_select_reruns_threshold_path():
    risk = torch.tensor([0.1, 0.5, 0.9, 0.95])
    report = select_reruns(risk, threshold=0.6)
    assert report.flagged_indices == [2, 3]


def test_select_reruns_requires_exactly_one_budget():
    risk = torch.arange(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        select_reruns(risk)
    with pytest.raises(ValueError):
        select_reruns(risk, rerun_fraction=0.2, threshold=0.5)


def test_capture_rate_and_spearman_when_risk_ranks_error_perfectly():
    n = 100
    err = torch.arange(n, dtype=torch.float64)
    report = select_reruns(err.clone(), rerun_fraction=0.2, true_error=err, true_error_quantile=0.9)
    # risk == error -> every top-decile high-error trajectory is flagged
    assert report.capture_rate == pytest.approx(1.0)
    assert report.spearman_risk_vs_error == pytest.approx(1.0)
    # 20 flagged, 10 of them truly high -> precision 0.5
    assert report.precision == pytest.approx(0.5)
    assert report.error_ratio_flagged_to_accepted > 1.0


def test_anticorrelated_risk_misses_high_error():
    n = 100
    err = torch.arange(n, dtype=torch.float64)
    risk = torch.flip(err, dims=[0])  # risk inversely ranks error
    report = select_reruns(risk, rerun_fraction=0.2, true_error=err)
    assert report.capture_rate == pytest.approx(0.0)
    assert report.spearman_risk_vs_error == pytest.approx(-1.0)


def test_true_error_length_mismatch_raises():
    risk = torch.arange(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        select_reruns(risk, rerun_fraction=0.2, true_error=torch.arange(5, dtype=torch.float64))


def test_time_above_is_nan_without_threshold():
    s = score_sigma_profile(torch.tensor([1.0, 2.0]), torch.tensor([1.1, 1.2]), scoring="mean")
    assert math.isnan(s.time_above_threshold)
