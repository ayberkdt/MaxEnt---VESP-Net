# Absolute-Threshold (Zero-Alarm) Screening  *(tests: false-alarm behavior)*

**Setup:** the 512 orbits scored with the cross-trajectory-comparable **absolute force-risk
budget** mode (`expected_abs_p95`, normalized acceleration units), then screened with absolute
budgets. Unlike a fixed top-fraction, an absolute threshold flags *only* orbits exceeding the
budget — possibly zero.

Pointwise calibration thresholds (a quantile of held-out per-point `expected_error`) should be
used **only** with absolute-scale scores like this one; relative supervisor scores
(`supervisor_rel*`) require trajectory-level calibration instead
(`threshold_source: trajectory_calibration_quantile`), because their per-trajectory altitude
normalization puts them on a different scale.

```text
--- ABSOLUTE PHYSICAL-BUDGET (scoring=expected_abs_p95) ---
Per-orbit expected force-error risk: p50=9.10e-04  p90=1.51e-01  p99=1.51e-01  max=1.51e-01
  budget = 1.52e-01 (above worst orbit) -> 0/512 above -> ZERO ALARMS
  budget = 1.51e-01 (p99 budget)        -> 6/512 above
  budget = 1.51e-01 (p90 budget)        -> 52/512 above
```

**Honest conclusion:** the absolute screen is **not forced to flag a fixed fraction** — a budget
above the worst orbit yields zero alarms, demonstrating the zero-alarm capability. We do **not**
claim these 512 orbits are uniformly benign: the per-orbit expected force error spans ~100×
(p50 ≈ 9e-4 vs max ≈ 1.5e-1; eccentric orbits dip to genuinely low, high-force-error periapsis),
so lower budgets correctly flag the high-force-error tail. The right operational use is to set the
budget from a physical force-error tolerance, not from a data quantile.

Reproduce: `python scripts/run_iac_benchmarks.py --config configs/vespuq/vespuq_smoke.yaml`
(writes `absolute_threshold_screening.md`), or set `uq.screening.threshold_source` /
`threshold` / `threshold_quantile` in a config and run `python -m vesp.uq.run`.
