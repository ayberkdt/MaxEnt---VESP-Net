"""Tests for Stage 3C: exact linear-Gaussian posterior + calibration."""

from __future__ import annotations

from pathlib import Path

import torch

from vesp.common.config import load_config
from vesp.extensions.probabilistic import AltitudeNoiseModel, LinearGaussianPosterior, calibration_metrics
from vesp.training.uncertainty import run_uncertainty_eval

ROOT = Path(__file__).resolve().parents[1]


def test_posterior_mean_equals_ridge_and_cov_is_scaled_inverse():
    torch.manual_seed(0)
    A = torch.randn(50, 12, dtype=torch.float64)
    b = torch.randn(50, dtype=torch.float64)
    lam = 0.3
    post = LinearGaussianPosterior.fit(A, b, lambda_l2=lam, noise_var=1.0)

    gram = A.T @ A + lam * torch.eye(12, dtype=torch.float64)
    ridge = torch.linalg.solve(gram, A.T @ b)
    assert torch.allclose(post.mean, ridge, atol=1.0e-8)
    # cov = noise_var * (A^T A + lambda I)^-1
    assert torch.allclose(post.cov, torch.linalg.inv(gram), atol=1.0e-6)


def test_predict_shapes_and_nonnegative_variance():
    torch.manual_seed(1)
    A = torch.randn(40, 8, dtype=torch.float64)
    b = torch.randn(40, dtype=torch.float64)
    post = LinearGaussianPosterior.fit(A, b, lambda_l2=0.1)
    q = torch.randn(15, 8, dtype=torch.float64)
    pred = post.predict(q)
    assert pred["mean"].shape == (15,)
    assert pred["variance"].shape == (15,)
    assert torch.all(pred["variance"] >= 0)
    assert torch.all(pred["epistemic_variance"] >= 0)
    # total variance includes the aleatoric noise floor
    assert torch.all(pred["variance"] >= pred["epistemic_variance"] - 1.0e-12)


def test_calibration_metrics_on_perfectly_calibrated_gaussian():
    torch.manual_seed(2)
    n = 40000
    target = torch.randn(n, dtype=torch.float64)
    m = calibration_metrics(
        torch.zeros(n, dtype=torch.float64), torch.ones(n, dtype=torch.float64), target
    )
    assert abs(m["picp_68"] - 0.68) < 0.02
    assert abs(m["picp_90"] - 0.90) < 0.02
    assert abs(m["picp_95"] - 0.95) < 0.02
    assert abs(m["z_std"] - 1.0) < 0.05


def test_calibration_detects_overconfidence():
    torch.manual_seed(3)
    n = 20000
    target = torch.randn(n, dtype=torch.float64)
    # predicted std too small by 2x -> overconfident -> coverage below nominal, z_std ~ 2
    m = calibration_metrics(
        torch.zeros(n, dtype=torch.float64), torch.full((n,), 0.5, dtype=torch.float64), target
    )
    assert m["picp_90"] < 0.90
    assert m["z_std"] > 1.5


def test_fit_evidence_recovers_noise_and_calibrates_in_distribution():
    torch.manual_seed(0)
    n_query, n_source = 400, 30
    A = torch.randn(n_query, n_source, dtype=torch.float64)
    sigma_true = 0.5 * torch.randn(n_source, dtype=torch.float64)
    noise = 0.1
    b = A @ sigma_true + noise * torch.randn(n_query, dtype=torch.float64)

    post = LinearGaussianPosterior.fit_evidence(A, b)
    assert post.noise_var > 0.0 and post.lambda_l2 is not None and post.lambda_l2 > 0.0
    # evidence should recover roughly the true noise level on a well-specified problem
    assert 0.05 < post.noise_var ** 0.5 < 0.2

    # in-distribution calibration on a fresh design matrix
    A_test = torch.randn(n_query, n_source, dtype=torch.float64)
    b_test = A_test @ sigma_true + noise * torch.randn(n_query, dtype=torch.float64)
    pred = post.predict(A_test)
    m = calibration_metrics(pred["mean"], pred["std"], b_test)
    assert 0.8 < m["picp_90"] < 1.0
    assert 0.7 < m["z_std"] < 1.4


def test_altitude_noise_model_recovers_known_power_law():
    torch.manual_seed(0)
    n = 8000
    radii = 1.02 + 0.5 * torch.rand(n, dtype=torch.float64)  # ~[1.02, 1.52]
    h = radii - 1.0
    a_true, b_true = 1.0e-4, 1.5
    sigma2 = a_true * h.pow(-b_true)
    residuals = torch.sqrt(sigma2) * torch.randn(n, dtype=torch.float64)
    epistemic = torch.full((n,), 1.0e-9, dtype=torch.float64)

    model = AltitudeNoiseModel.fit(radii, residuals, epistemic, iters=900, lr=0.05)
    assert abs(model.b - b_true) < 0.5
    # variance at a couple of altitudes within a factor of ~2 of truth
    test_r = torch.tensor([1.05, 1.30], dtype=torch.float64)
    recovered = model.variance(test_r)
    truth = a_true * (test_r - 1.0).pow(-b_true)
    assert torch.all(recovered > 0)
    assert torch.allclose(recovered, truth, rtol=0.6)


def test_heteroscedastic_beats_homoscedastic_low_band_calibration():
    torch.manual_seed(1)
    n = 8000
    radii = 1.02 + 0.5 * torch.rand(n, dtype=torch.float64)
    h = radii - 1.0
    sigma2 = 1.0e-4 * h.pow(-1.5)
    target = torch.sqrt(sigma2) * torch.randn(n, dtype=torch.float64)  # zero-mean forecast
    zeros = torch.zeros(n, dtype=torch.float64)
    epistemic = torch.zeros(n, dtype=torch.float64)

    homo_std = torch.full((n,), float(sigma2.mean().sqrt()), dtype=torch.float64)
    model = AltitudeNoiseModel.fit(radii, target, epistemic, iters=900)
    het_std = torch.sqrt(model.variance(radii))

    low = radii < 1.12  # high-variance band; homoscedastic underestimates it
    homo_low = calibration_metrics(zeros[low], homo_std[low], target[low])
    het_low = calibration_metrics(zeros[low], het_std[low], target[low])
    # heteroscedastic low-band 90% coverage is closer to nominal than homoscedastic
    assert abs(het_low["picp_90"] - 0.90) < abs(homo_low["picp_90"] - 0.90)


def test_crps_on_calibrated_unit_gaussian():
    torch.manual_seed(2)
    n = 40000
    target = torch.randn(n, dtype=torch.float64)
    m = calibration_metrics(torch.zeros(n, dtype=torch.float64), torch.ones(n, dtype=torch.float64), target)
    # perfectly calibrated unit Gaussian -> mean CRPS = 1/sqrt(pi) ~ 0.5642
    assert abs(m["crps"] - 0.5642) < 0.02


def test_uncertainty_eval_epistemic_grows_toward_low_altitude():
    cfg = load_config(ROOT / "configs" / "uncertainty" / "uncertainty_synthetic_ood.yaml")
    report = run_uncertainty_eval(cfg)
    bands = report["bands"]
    assert {"val", "test_low", "test_high"} <= set(bands)
    # the posterior must be more uncertain (epistemic) where it extrapolates (low altitude)
    assert bands["test_low"]["mean_epistemic_std"] > bands["test_high"]["mean_epistemic_std"]
    assert report["summary"]["low_high_epistemic_std_ratio"] > 1.0
    # in-distribution calibration should be in a sane range (not wildly off)
    assert 0.6 <= bands["val"]["picp_90"] <= 1.0
