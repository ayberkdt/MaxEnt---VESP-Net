import json

import numpy as np
import pytest

from vesp.data import real_gravity
from vesp.data.real_gravity import SphericalHarmonicGravityModel


def _tiny_model() -> SphericalHarmonicGravityModel:
    return SphericalHarmonicGravityModel(
        name="tiny",
        reference_radius_km=1738.0,
        gm_km3_s2=4902.80012616,
        degree=2,
        order=2,
        c=np.zeros((3, 3)),
        s=np.zeros((3, 3)),
        normalization_state=1,
        source_path="tiny.tab",
        column_order="degree_order",
    )


def test_real_lunar_dataset_writes_diagnostics_json(tmp_path, monkeypatch):
    model = _tiny_model()
    monkeypatch.setattr(real_gravity, "read_pds_sha", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        real_gravity,
        "random_exterior_points",
        lambda *args, **kwargs: np.array([[1.1, 0.0, 0.0], [0.0, 1.2, 0.0]]),
    )
    monkeypatch.setattr(real_gravity, "residual_potential", lambda *args, **kwargs: np.array([2.0, -4.0]))
    monkeypatch.setattr(
        real_gravity,
        "residual_acceleration_finite_difference",
        lambda *args, **kwargs: np.array([[1738.0, 0.0, 0.0], [0.0, -3476.0, 0.0]]),
    )

    path = real_gravity.build_real_lunar_dataset(
        sha_path=tmp_path / "tiny.tab",
        output_path=tmp_path / "tiny.csv",
        n_query=2,
        radius_min=1.03,
        radius_max=1.60,
        degree_min=2,
        degree_max=2,
        finite_difference_step=1.0e-5,
        acceleration_output="physical",
    )

    diagnostics_path = path.with_suffix(".diagnostics.json")
    assert diagnostics_path.exists()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["n_query"] == 2
    assert diagnostics["radius_min"] == 1.03
    assert diagnostics["radius_max"] == 1.6
    assert diagnostics["degree_min"] == 2
    assert diagnostics["degree_max"] == 2
    assert diagnostics["position_norm_min"] == 1.1
    assert diagnostics["position_norm_max"] == 1.2
    assert diagnostics["potential_rms"] > 0.0
    assert diagnostics["acceleration_rms"] > 0.0
    assert diagnostics["acceleration_output"] == "physical"
    assert diagnostics["acceleration_units"] == "km/s^2"
    assert diagnostics["finite_difference_step"] == 1.0e-5
    assert diagnostics["reference_radius_km"] == 1738.0
    assert diagnostics["gm_km3_s2"] == 4902.80012616


def test_physical_acceleration_diagnostics_reject_zero_rms():
    with pytest.raises(ValueError, match="physical acceleration RMS"):
        real_gravity.residual_dataset_diagnostics(
            positions_normalized=np.array([[1.1, 0.0, 0.0]]),
            potential=np.array([1.0]),
            acceleration=np.zeros((1, 3)),
            model=_tiny_model(),
            degree_min=2,
            degree_max=2,
            radius_min=1.03,
            radius_max=1.60,
            finite_difference_step=1.0e-4,
            acceleration_output="physical",
            acceleration_units="km/s^2",
        )
