"""N5: VESP-UQ <-> ST-LRPS adapter boundary tests.

`vesp.adapters.st_lrps` is vendored, exploratory ST-LRPS wiring that is OUT OF SCOPE for VESP-UQ: it
is not maintained / refactored / tested as part of the calibration layer, and it depends on the
external `lunaris` package, so it is not even importable in a clean VESP-UQ environment (these tests
therefore skip in CI). The single seam VESP-UQ depends on is `load_surrogate_force_model`, used by
`scripts/run_stlrps_propagation.py` as the Monte Carlo base field. See
`docs/VESP_UQ_LIMITATIONS.md` ("ST-LRPS adapter") and `src/vesp/adapters/README.md`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_ARTIFACT_ENV = "VESP_STLRPS_MODEL_DIR"  # set to an ST-LRPS run dir to exercise the load smoke


def test_force_model_seam_importable():
    # importorskip: the adapter depends on the external `lunaris` package, absent in a clean VESP-UQ
    # env (e.g. CI) -> skip rather than fail. Where present, this guards the seam against breakage.
    mod = pytest.importorskip("vesp.adapters.st_lrps.runtime.force_model")
    assert callable(mod.load_surrogate_force_model)


def test_run_stlrps_propagation_script_importable():
    # The script imports the adapter seam at module load; skip if the adapter isn't importable.
    rsp = pytest.importorskip("scripts.run_stlrps_propagation")
    assert callable(rsp.main)


def test_force_model_loads_when_artifact_present():
    mod = pytest.importorskip("vesp.adapters.st_lrps.runtime.force_model")
    model_dir = os.environ.get(_ARTIFACT_ENV)
    if not model_dir or not os.path.isdir(model_dir):
        pytest.skip(f"no ST-LRPS artifact: set {_ARTIFACT_ENV} to a run directory to exercise the seam")

    fm = mod.load_surrogate_force_model(
        model_dir=model_dir,
        device="cpu",
        strict_contract=False,
        allow_legacy_contract=True,
        strict_domain=False,
    )
    # The exact interface scripts/run_stlrps_propagation.py relies on (mu_si, degree_min, residual
    # acceleration in the Moon-fixed frame). This is the only adapter contract VESP-UQ depends on.
    assert isinstance(fm.mu_si, float) and fm.mu_si > 0.0
    assert isinstance(fm.degree_min, int)
    da = np.asarray(fm.predict_residual_accel_fixed(np.array([[1.8e6, 0.0, 0.0]])))
    assert da.shape == (1, 3)
    assert np.isfinite(da).all()
