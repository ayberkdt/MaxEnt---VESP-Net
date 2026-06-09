# `vesp.adapters` — external surrogate adapters (out of VESP-UQ scope)

This directory holds **vendored, exploratory** wiring to external surrogate-gravity packages. It is
**not** part of the VESP-UQ calibration layer's maintained, tested surface.

## ST-LRPS (`st_lrps/`)

`st_lrps/` is the Sobolev-Trained Lunar Residual Potential Surrogate package (~70 files). It is
exploratory and depends on the external `lunaris` package, so it is **not importable in a clean
VESP-UQ environment** (e.g. CI). Treat the whole package as out-of-scope vendored code: VESP-UQ does
not maintain, refactor, or test it, and full adapter coverage is explicitly out of scope.

The **only** VESP-UQ ↔ adapter seam is:

    vesp.adapters.st_lrps.runtime.force_model.load_surrogate_force_model

used by `scripts/run_stlrps_propagation.py` as the Monte Carlo `base_accel_fn` (its returned model
exposes `mu_si`, `degree_min`, and `predict_residual_accel_fixed`). That single seam is the only part
bounded by VESP-UQ tests — `tests/test_stlrps_adapter_boundary.py`: an import-safety guard plus a
skip-guarded artifact-load smoke (set `VESP_STLRPS_MODEL_DIR` to a run directory to exercise it).
Both skip when the adapter, its `lunaris` dependency, or the artifact are absent.

See `docs/VESP_UQ_LIMITATIONS.md` → "ST-LRPS adapter: exploratory wiring, not a validated
integration" for the scope / claims boundary.
