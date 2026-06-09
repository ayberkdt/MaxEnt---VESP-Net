"""Absolute force-risk threshold resolution for VESP-UQ screening.

Resolves the screening threshold (and its provenance) from config, enforcing the scale rule that
a *pointwise* expected-force-error budget is only ever paired with an absolute-scale scoring mode
-- never a relative supervisor score (which is on a per-trajectory-normalized scale).

Sources (``uq.screening.threshold_source``):
- ``manual`` -- use ``uq.screening.threshold`` directly;
- ``pointwise_calibration_quantile`` -- quantile of held-out per-point ``expected_error``
  (absolute scoring only);
- ``trajectory_calibration_quantile`` -- quantile of the *same* trajectory score on a held-out
  calibration ensemble (safe for relative or absolute scoring).
"""

from __future__ import annotations

from vesp.uq.ensemble import generate_orbit_ensemble
from vesp.uq.scoring import is_absolute_scoring, is_relative_scoring

THRESHOLD_SOURCES = ("manual", "pointwise_calibration_quantile", "trajectory_calibration_quantile")


def resolve_threshold(screen_cfg, plugin, held, scoring, *, dtype, seed):
    """Resolve the absolute screening threshold and its provenance from config.

    Returns ``(threshold_or_None, meta)``. ``None`` means fall back to fraction mode. ``meta``
    records the threshold source / quantile / multiplier / calibration scoring + count and an
    optional backward-compatibility note. Enforces that a *pointwise* expected-error budget is
    only paired with absolute-scale scoring (never a relative supervisor score).
    """

    threshold = screen_cfg.get("threshold")
    threshold_quantile = screen_cfg.get("threshold_quantile")
    multiplier = float(screen_cfg.get("threshold_multiplier", 1.0))
    src = screen_cfg.get("threshold_source")

    meta = {
        "threshold_source": None,
        "threshold_quantile": None,
        "threshold_multiplier": multiplier,
        "threshold_calibration_scoring": None,
        "threshold_calibration_n": None,
        "threshold_compatibility_note": None,
    }

    # Backward-compatible inference when threshold_source is omitted.
    if src is None:
        if threshold is not None:
            src = "manual"
        elif threshold_quantile is not None:
            src = "pointwise_calibration_quantile"
            meta["threshold_compatibility_note"] = (
                "legacy syntax: threshold_quantile without threshold_source -> inferred "
                "pointwise_calibration_quantile (requires absolute-like scoring)"
            )
        else:
            return None, meta  # no threshold configured -> fraction mode
    src = str(src).lower()
    if src not in THRESHOLD_SOURCES:
        raise ValueError(f"uq.screening.threshold_source must be one of {THRESHOLD_SOURCES}, got {src!r}")

    if src == "manual":
        if threshold is None:
            raise ValueError("threshold_source=manual requires uq.screening.threshold")
        meta["threshold_source"] = "manual"
        return float(threshold), meta

    if threshold_quantile is None:
        raise ValueError(f"threshold_source={src} requires uq.screening.threshold_quantile")

    if src == "pointwise_calibration_quantile":
        if not is_absolute_scoring(scoring):
            why = (
                " (a relative supervisor score is not on the pointwise expected-error scale; use "
                "trajectory_calibration_quantile instead)"
                if is_relative_scoring(scoring)
                else " (use an expected_abs*/supervisor_abs* score, or trajectory_calibration_quantile)"
            )
            raise ValueError(
                f"threshold_source=pointwise_calibration_quantile needs absolute-like scoring; "
                f"got scoring={scoring!r}{why}"
            )
        thr = plugin.calibrate_pointwise_expected_error_threshold(
            held.positions, quantile=float(threshold_quantile), multiplier=multiplier
        )
        meta.update(
            threshold_source=src,
            threshold_quantile=float(threshold_quantile),
            threshold_calibration_n=int(held.positions.shape[0]),
        )
        return thr, meta

    # trajectory_calibration_quantile: calibrate the SAME trajectory score -> safe for any scoring.
    default_n = min(int(screen_cfg.get("n_orbits", 200)), 200)
    cal = generate_orbit_ensemble(
        n_orbits=int(screen_cfg.get("calibration_n_orbits", default_n)),
        n_points=int(screen_cfg.get("calibration_n_points", int(screen_cfg.get("n_points", 48)))),
        r_peri_range=tuple(screen_cfg.get("calibration_r_peri_range", screen_cfg.get("r_peri_range", (1.02, 1.30)))),
        r_apo_range=tuple(screen_cfg.get("calibration_r_apo_range", screen_cfg.get("r_apo_range", (1.30, 1.60)))),
        seed=int(seed) + 1,  # a held-out calibration ensemble, distinct from the screening one
        dtype=dtype,
    )
    thr = plugin.calibrate_trajectory_risk_threshold(
        cal.trajectories, scoring=scoring, quantile=float(threshold_quantile), multiplier=multiplier
    )
    meta.update(
        threshold_source=src,
        threshold_quantile=float(threshold_quantile),
        threshold_calibration_scoring=scoring,
        threshold_calibration_n=len(cal.trajectories),
    )
    return thr, meta
