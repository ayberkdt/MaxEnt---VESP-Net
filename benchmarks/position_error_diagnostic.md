# ST-LRPS Position-Error Diagnostic  *(diagnostic only — NOT a VESP-UQ claim)*

This is a **diagnostic comparison**, not a core VESP-UQ benchmark and **not** an ST-LRPS
integration. It asks whether the VESP-UQ *force-risk* score happens to co-rank a particular
surrogate's long-horizon *position* error. Reading ST-LRPS position-error metrics for this
comparison is not the same as a propagation adapter or an integrated surrogate workflow (see
[`../docs/VESP_UQ_LIMITATIONS.md`](../docs/VESP_UQ_LIMITATIONS.md)).

The honest, expected result is a **null** correlation: position error here is not force-model-error
dominated, so a force-risk score should not be expected to rank it.

---

## 512-Orbit suite, 12 h

**Setup:** the 512-orbit `test_512` lunar suite, point-mass-integrated and scored against the true
`ST_LRPS_DT60` position error. Relative ranking mode (`supervisor_rel_p95`), top 10%, with a
100-mask random baseline. Reproduce with `python scripts/analyze_512_orbits.py`.

```text
--- RELATIVE RANKING (scoring=supervisor_rel_p95, top 10%) ---
Total Trajectories: 512
Spearman (force-risk vs ST-LRPS position error): -0.0209
Capture Rate (top-risk catching top-10% error): 11.5%
Precision: 11.5%
Lift over random (capture / rerun fraction): 1.14x
Mean true error flagged: 0.036 km  vs  accepted: 0.035 km  (ratio 1.01x)
Random baseline capture (100 masks): mean=10.4% +/- 4.3%
```

**Honest conclusion:** force-risk **does not rank** the ST-LRPS position error — Spearman ≈ 0,
capture (11.5%) sits inside the random baseline (10.4% ± 4.3%), ratio ≈ 1.0×. This is *not*
evidence the force-risk layer is miscalibrated; position error over 12 h is not force-model-error
dominated, so it should not be expected to rank. (These orbits do span a wide expected-force-error
range — see [`absolute_threshold_screening.md`](absolute_threshold_screening.md).)

## 512-Orbit suite, 5 d

**Setup:** the same in-distribution orbits propagated for 5 days; baseline position error grows
from ~35 m to ~1.5 km purely from chaotic drift. (Numbers from the prior run; same diagnostic.)

```text
--- 5-DAY DIAGNOSTIC (relative ranking, top 10%) ---
Total Trajectories: 512
Spearman (force-risk vs ST-LRPS position error): -0.0483
Capture Rate: 11.5%   Precision: 11.5%
Mean true error flagged: 1.486 km  vs  accepted: 1.546 km  (ratio 0.96x)
```

**Honest conclusion:** the null ranking result persists over 5 days (Spearman ≈ 0, ratio ≈ 1.0×).
The accumulated 5-day error is chaotic-drift dominated, not force-error dominated. This is a
*position-error ranking* statement, not a verdict on the force-risk / OOD calibration in
[`force_ood_detection.md`](force_ood_detection.md).
