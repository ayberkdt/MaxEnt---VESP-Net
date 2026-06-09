# Online Force-Model Correction (accuracy vs cost) for VESP-UQ

This note documents the **exploratory** online force-model correction — the one remaining
headline future-work item in the IAC plan. The VESP-UQ posterior mean is the ridge estimate of the
surrogate's force-model error, so adding it to a surrogate's acceleration inside an integrator RHS
gives

    a_corrected(x) = a_surrogate(x) + mean_error(x),   mean_error(x) = K(x) @ sigma_mean,

where `K(x)` is the equivalent-source acceleration operator (honoring the fitted softening `eps`,
`acceleration_sign`, and source weights). `mean_error(x)` is **exactly** the posterior-mean field
returned by `predict_uncertainty` and used as the nominal field of the MC / STM propagators.

- `src/vesp/uq/correction.py` — `CorrectedForceField` (the RHS hook) and `integrate_trajectory`
  (a single-trajectory RK4 with the same snap / sub-step structure as the propagators).
- `scripts/run_force_correction_benchmark.py` — the benchmark below, writing
  `outputs/correction/{force_correction_benchmark.json, .md, force_correction_errors.csv}` through
  the N1 artifact layer (provenance + `run_manifest.json` with SHA-256 checksums).

```
python scripts/run_force_correction_benchmark.py --config configs/vespuq/vespuq_smoke.yaml
```

> **Scope / honesty.** The posterior mean is a regularized **point estimate**, so this is a
> **force-model** correction: it improves trajectory accuracy only insofar as the surrogate's force
> error is captured by the equivalent-source mean, and it carries **no guaranteed long-horizon
> position-accuracy claim** (force-risk ⊥ position-error on the in-distribution diagnostic). The
> synthetic truth below lies in the equivalent-source span, so the numbers are a **best-case**
> illustration of the mechanism, **not** evidence it transfers to real residuals. See
> [`../docs/VESP_UQ_LIMITATIONS.md`](../docs/VESP_UQ_LIMITATIONS.md).

## Setup

The benchmark builds a synthetic world whose truth force-error field is itself an equivalent-source
field (interior truth sources + strengths, the same generative model as
`make_synthetic_uq_samples`), and integrates three trajectories from one low circular orbit:

| trajectory | RHS acceleration | role |
| --- | --- | --- |
| surrogate | `a_base(x)` | the surrogate — here it omits the residual entirely (worst case) |
| corrected | `a_base(x) + K(x) @ sigma_mean` | surrogate + VESP-UQ posterior-mean correction |
| reference | `a_base(x) + truth_residual(x)` | the truth |

VESP-UQ is fit on samples of `reference - surrogate` (= the truth residual + noise). The **fit**
sources (shells `[0.75, 0.9]`) differ from the **truth** source (shell `0.7`), so the correction is
an equivalent-source *approximation* of the truth field from a different basis — not an inverse
crime; the residual gap below is real.

## Results (smoke config, illustrative)

Two orbits (`duration = 14`, `dt = 0.05`) on `configs/vespuq/vespuq_smoke.yaml`:

| metric | surrogate | corrected |
| --- | ---: | ---: |
| final position error (body radii) | 1.03e+00 | 1.31e-02 |
| per-RHS cost (us / call) | ~32 | ~558 |

- **Accuracy:** the correction cuts the final position error by ~**79x** (reduction ratio ≈ `0.013`).
  Note the surrogate, omitting a *small* force residual, still diverges by ~1 body radius over two
  orbits — a small force-model error integrates into a large position error, which is exactly why
  force error and position error are different quantities.
- **Cost:** the corrected RHS evaluates the full equivalent-source field every call, costing ~**17x**
  the bare surrogate. This is the trade-off the IAC docs warn about: a fast surrogate can lose its
  speed advantage if every RHS call re-evaluates the equivalent-source correction.

## When it helps / when it does not

- It helps to the extent the surrogate's force error is captured by the equivalent-source posterior
  **mean**. Here the truth lies in that span, so the gain is large; for real residuals with
  out-of-span structure the captured fraction — and the accuracy gain — will be smaller.
- It is a **force-model** correction, not a position-error model. A small remaining force error can
  still integrate into a meaningful long-horizon position error, and the force-risk score does not
  predict that position error.
- The cost is per-RHS and grows with the source count; for large source sets the correction can erase
  a surrogate's speed advantage. Benchmark the trade-off on your own configuration.

## What should not be claimed

- No guaranteed long-horizon position-accuracy improvement; report measured accuracy **and** cost.
- No claim that this transfers from the synthetic (in-span) truth to real residuals.
- No deterministic-accuracy claim beyond "the posterior-mean force field reduces the measured
  position error in this setup, at a measured per-RHS cost."
