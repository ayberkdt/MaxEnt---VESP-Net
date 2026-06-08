# VESP-UQ Risk-Screening Benchmark Results

VESP-UQ is a **post-processing uncertainty / risk-calibration layer** for lunar gravity
residual surrogates. It scores the *expected force-model error* and *out-of-support (OOD) risk*
of a trajectory so the riskiest samples can be sent to a high-fidelity rerun. It is **not** a
deterministic trajectory-accuracy improver, not a spherical-harmonic replacement, not a
density-recovery model, and not an operational covariance propagator.

Because of that scope, it matters *what each benchmark actually tests*. The three questions are
distinct and a result can be strong on one and null on another:

| Benchmark | What it tests | Honest read |
| --- | --- | --- |
| 1. OOD altitude sweep | **force-risk / OOD detection** | flagged passes are genuinely lower-altitude (OOD) |
| 2. 512-orbit, 12 h | **position-error ranking** | force-risk does *not* rank ST-LRPS position error here |
| 3. 512-orbit, 5 d | **position-error ranking** | same null result under 5-day drift |
| 4. Absolute-threshold sweep | **false-alarm behavior** | an absolute budget can flag zero (not forced to flag a fixed %) |

Two scoring families are used below:
- **relative** modes (`supervisor_rel*`): per-trajectory altitude normalization — for *ranking*
  which orbits to rerun first within one ensemble.
- **absolute** modes (`expected_abs*`, `supervisor_abs*`): fixed altitude reference — for a
  *physical budget* that means the same thing across trajectories (zero-alarm screening).

---

## Benchmark 1 — Out-of-Distribution Altitude Sweep  *(tests: force-risk / OOD detection)*

**Setup:** 100 random test trajectories oscillating between ~50 km and ~150 km initial
altitude. The surrogate is calibrated in a higher band, so the lower passes are extrapolation
(OOD) zones. Selection here is the relative top-fraction screen (flags ~10% by construction);
the question is whether that flagged 10% is genuinely the low-altitude / OOD set.

```text
--- RISK SCREENING REPORT ---
Total Trajectories Simulated: 100
Trajectories Flagged (top-fraction): 10 (10.0%)
Risk Threshold Used: 55.673226

--- PHYSICAL READOUT ---
Average altitude of FLAGGED trajectories: 54.4 km
Average altitude of ACCEPTED trajectories: 101.8 km
```

**Honest conclusion:** the flagged set is concentrated at ~54 km vs ~102 km accepted, so the
equivalent-source uncertainty layer *does* surface the low-altitude OOD passes without being
told the calibration bounds. This is an **OOD-detection** result. Note the top-fraction screen
flags 10% *by construction*; the meaningful signal is that the flagged 10% is the low-altitude
tail, not that "10% are risky."

---

## Benchmark 2 — 512-Orbit Validation Suite, 12 h  *(tests: position-error ranking)*

**Setup:** the 512-orbit `test_512` lunar suite, point-mass-integrated and scored against the
true `ST_LRPS_DT60` position error. Relative ranking mode (`supervisor_rel_p95`), top 10%, with
a 100-mask random baseline. Reproduce with `python scripts/analyze_512_orbits.py`.

```text
--- (a) RELATIVE RANKING (scoring=supervisor_rel_p95, top 10%) ---
Total Trajectories: 512
Spearman (force-risk vs ST-LRPS position error): -0.0209
Capture Rate (top-risk catching top-10% error): 11.5%
Precision: 11.5%
Lift over random (capture / rerun fraction): 1.14x
Mean true error flagged: 0.036 km  vs  accepted: 0.035 km  (ratio 1.01x)
Random baseline capture (100 masks): mean=10.4% +/- 4.3%
```

**Honest conclusion:** on this in-distribution suite the VESP-UQ **force-risk does not rank the
ST-LRPS position error** — Spearman ≈ 0, capture (11.5%) sits inside the random baseline
(10.4% ± 4.3%), and flagged/accepted true error is ~1.0×. This is *not* evidence that the
force-risk layer is miscalibrated; it means position error over 12 h is not dominated by
force-model error in this regime, so a force-risk score should not be expected to rank it. The
earlier claim that VESP-UQ "correctly identified there are no risky trajectories" is withdrawn:
the relative screen still flags a top fraction by construction, and these orbits do span a wide
expected-force-error range (see Benchmark 4).

---

## Benchmark 3 — 512-Orbit Validation Suite, 5 d  *(tests: position-error ranking)*

**Setup:** the same in-distribution orbits propagated for 5 days (`test_512_5days`); baseline
position error grows from ~35 m to ~1.5 km purely from chaotic drift. (Numbers below are from
the prior run; the diagnostic and interpretation are identical to Benchmark 2.)

```text
--- 5-DAY DIAGNOSTIC (relative ranking, top 10%) ---
Total Trajectories: 512
Spearman (force-risk vs ST-LRPS position error): -0.0483
Capture Rate: 11.5%   Precision: 11.5%
Mean true error flagged: 1.486 km  vs  accepted: 1.546 km  (ratio 0.96x)
```

**Honest conclusion:** the null ranking result persists over 5 days (Spearman ≈ 0, ratio
≈ 1.0×). The accumulated 5-day error is chaotic-drift dominated, not force-error dominated, so
again force-risk does not — and should not be expected to — rank it. This is a *position-error
ranking* statement, not a verdict on the force-risk/OOD calibration shown in Benchmark 1.

---

## Benchmark 4 — Absolute-Threshold Sweep  *(tests: false-alarm behavior)*

**Setup:** the 512 orbits scored with the cross-trajectory-comparable **absolute force-risk
budget** mode (`expected_abs_p95`, normalized acceleration units), then screened with absolute
budgets. Unlike a fixed top-fraction, an absolute threshold flags *only* orbits exceeding the
budget — possibly zero. Pointwise calibration thresholds (a quantile of held-out per-point
`expected_error`) should be used **only** with absolute-scale scores like this one; relative
supervisor scores (`supervisor_rel*`) require trajectory-level calibration instead, because their
per-trajectory altitude normalization puts them on a different scale.

```text
--- (b) ABSOLUTE PHYSICAL-BUDGET (scoring=expected_abs_p95) ---
Per-orbit expected force-error risk: p50=9.10e-04  p90=1.51e-01  p99=1.51e-01  max=1.51e-01
  budget = 1.52e-01 (above worst orbit) -> 0/512 above -> ZERO ALARMS
  budget = 1.51e-01 (p99 budget)        -> 6/512 above
  budget = 1.51e-01 (p90 budget)        -> 52/512 above
```

**Honest conclusion:** the absolute screen is **not forced to flag a fixed fraction** — a budget
above the worst orbit yields zero alarms, demonstrating the zero-alarm capability. We do **not**
claim these 512 orbits are uniformly benign: the per-orbit expected force error spans ~100×
(p50 ≈ 9e-4 vs max ≈ 1.5e-1; eccentric orbits dip to genuinely low, high-force-error periapsis),
so lower budgets correctly flag the high-force-error tail. The right operational use is to set
the budget from a physical force-error tolerance, not from a data quantile.

---

### What to take away
- VESP-UQ reliably surfaces **low-altitude / OOD** passes (Benchmark 1) and supports
  **zero-alarm absolute screening** with a physical budget (Benchmark 4).
- It does **not** rank long-horizon **ST-LRPS position error** on the in-distribution 512 suite
  (Benchmarks 2–3) — expected, because that error is not force-model-error dominated there.
- None of this asserts that VESP-UQ improves deterministic trajectory accuracy; it is a risk /
  uncertainty supervisor at the acceleration interface.
