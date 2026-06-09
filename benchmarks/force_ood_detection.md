# Force-Risk / OOD Detection  *(the core VESP-UQ claim)*

These benchmarks test what VESP-UQ is *for*: detecting where the surrogate's **force-model error**
is large or out-of-support, so those trajectories can be reran at high fidelity. They do **not**
test long-horizon position-error prediction (see
[`position_error_diagnostic.md`](position_error_diagnostic.md)).

---

## A. Direct force-error ranking benchmark

**Question:** does the VESP-UQ force-risk score rank the surrogate's *true force-model error*
along a trajectory? This is the correct, direct test of the core claim.

Reproduce: `python scripts/run_force_error_benchmark.py --config configs/vespuq/vespuq_smoke.yaml`
(true force error read from held-out residual samples by nearest neighbour, or directly from
surrogate/reference acceleration pairs when an external CSV supplies them; aggregated by p95).

It reports **Spearman(force-risk, true force error)**, capture rate, precision, lift over random,
and the flagged/accepted force-error ratio. A positive Spearman / lift > 1 means the force-risk
score concentrates the surrogate's true force-model error — the value VESP-UQ provides. On the
synthetic smoke field this is clearly positive (Spearman ≈ 0.46), and it is the metric to cite
for the core claim, **not** the position-error diagnostic.

---

## B. Out-of-Distribution altitude sweep

**Setup:** 100 random test trajectories oscillating between ~50 km and ~150 km initial altitude.
The surrogate is calibrated in a higher band, so the lower passes are extrapolation (OOD) zones.
Selection here is the relative top-fraction screen (flags ~10% by construction); the question is
whether that flagged 10% is genuinely the low-altitude / OOD set.

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
equivalent-source uncertainty layer *does* surface the low-altitude OOD passes without being told
the calibration bounds — an **OOD-detection** result. The top-fraction screen flags 10% *by
construction*; the meaningful signal is that the flagged 10% is the low-altitude tail, not that
"10% are risky." `scripts/run_iac_benchmarks.py` also emits a compact `ood_altitude_sweep.md`
showing expected force error growing monotonically toward low altitude.
