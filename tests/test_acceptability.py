from vesp.training.acceptability import (
    STATUS_GOOD,
    STATUS_REJECT_LOW_ALTITUDE,
    STATUS_REJECT_REGULARIZATION,
    STATUS_REJECT_SOURCE_COLLAPSE,
    classify_run_acceptability,
)


def _clean_diag(**overrides):
    diag = {
        "sigma_l2": 0.01,
        "monopole_leakage": 1.0e-12,
        "dipole_leakage": 1.0e-12,
        "dominant_shell_energy_fraction": 0.5,
        "shell_collapse_flag": False,
        "top_5pct_source_contribution": 0.2,
    }
    diag.update(overrides)
    return diag


def _clean_metrics(**overrides):
    metrics = {"relative_acceleration_rmse": 0.4, "low_to_high_error_ratio": 2.0}
    metrics.update(overrides)
    return metrics


def test_good_run():
    result = classify_run_acceptability(_clean_metrics(), _clean_diag(), {})
    assert result["acceptability_status"] == STATUS_GOOD


def test_source_collapse_detected():
    diag = _clean_diag(dominant_shell_energy_fraction=0.97, shell_collapse_flag=True)
    result = classify_run_acceptability(_clean_metrics(), diag, {})
    assert result["acceptability_status"] == STATUS_REJECT_SOURCE_COLLAPSE


def test_low_altitude_detected():
    metrics = _clean_metrics(low_to_high_error_ratio=650.0)
    result = classify_run_acceptability(metrics, _clean_diag(), {})
    assert result["acceptability_status"] == STATUS_REJECT_LOW_ALTITUDE


def test_sigma_blowup_detected():
    diag = _clean_diag(sigma_l2=27.4)
    result = classify_run_acceptability(_clean_metrics(), diag, {})
    assert result["acceptability_status"] == STATUS_REJECT_REGULARIZATION


def test_numerical_priority_over_others():
    # numerical leakage outranks low-altitude in the single returned status
    diag = _clean_diag(monopole_leakage=1.0e-3)
    metrics = _clean_metrics(low_to_high_error_ratio=650.0)
    result = classify_run_acceptability(metrics, diag, {})
    assert result["acceptability_status"] == "REJECT_NUMERICAL"
    # but all triggered reasons are listed
    assert any("low/high" in r for r in result["acceptability_reasons"])


def test_thresholds_overridable_from_config():
    metrics = _clean_metrics(low_to_high_error_ratio=10.0)
    relaxed = {"acceptance": {"max_low_altitude_rmse_factor": 100.0}}
    assert classify_run_acceptability(metrics, _clean_diag(), relaxed)["acceptability_status"] == STATUS_GOOD
