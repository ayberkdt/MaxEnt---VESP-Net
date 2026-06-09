# Force-Error Covariance Propagation (MC vs STM) for VESP-UQ

This note documents the two ways VESP-UQ propagates its fitted **force-model error posterior** along
a nominal trajectory into a state covariance:

- `src/vesp/uq/propagation.py` — `VESPMonteCarloPropagator`: draws posterior samples of the
  equivalent-source strengths and integrates a batch of perturbed trajectories (Monte Carlo).
- `src/vesp/uq/linear_propagation.py` — `LinearForceErrorCovariancePropagator`: a deterministic,
  sampling-free **linearized (STM)** alternative that propagates the variational sensitivity
  `J(t) = d[x(t)]/d[sigma]` and maps the source posterior covariance into a `6x6` state covariance
  `P(t) = J(t) Sigma_sigma J(t)^T`.
- `scripts/run_linear_propagation.py` — driver that fits, propagates the linearized covariance, and
  writes `outputs/propagation/{linear_propagation.json, .md, linear_propagation_states.csv}` through
  the N1 artifact layer (provenance + `run_manifest.json` with SHA-256 checksums). Parity with the
  Monte Carlo driver `scripts/run_propagation.py`.

```
python scripts/run_linear_propagation.py --config configs/vespuq/vespuq_smoke.yaml
python scripts/run_propagation.py        --config configs/vespuq/vespuq_real_lunar.yaml   # MC sampler
```

> **Scope / honesty.** Everything here is the propagation of a **force-model acceleration-error**
> posterior (`a_reference - a_surrogate`). It is an **exploratory** diagnostic, **not validated**
> operational orbit determination: it does not model measurement processing, realistic process
> noise, or dynamic mismodelling beyond the fitted residual. The VESP-UQ force-risk score is **not**
> a position-error predictor (force-risk ⊥ position-error on the in-distribution 512-orbit
> diagnostic — see [`position_error_diagnostic.md`](position_error_diagnostic.md)), so a small
> propagated `sigma` is **not** a guarantee of position accuracy. See
> [`../docs/VESP_UQ_LIMITATIONS.md`](../docs/VESP_UQ_LIMITATIONS.md).

## What the linearized (STM) propagator computes

Around the nominal trajectory (base field + posterior-mean force-error correction), the variational
equation is

```
Jr' = Jv,    Jv' = G(r) Jr + K(r),
```

where `G(r) = d a_base / d r` is the nominal-dynamics gravity gradient (analytic point-mass tide by
default, or a finite-difference Jacobian of a custom base field) and `K(r) = d a_error / d sigma` is
the equivalent-source acceleration operator block — the **same** sign / softening / weight
convention as the fitted posterior. The covariance starts at zero (`J(0) = 0`) and grows as the
force-error posterior is integrated along the path. This is exactly the linearization of the Monte
Carlo sampler's *static force-error field* model, so the two agree in the small-perturbation regime.

## MC vs STM agreement (drift regime)

In the pure-drift regime (`mu = 0`, `v0 = 0`) the dynamics are linear in the source strengths, so the
STM covariance equals the MC sample covariance up to sampling noise. The automated check is
`tests/test_uq_linear_propagation.py::test_linear_covariance_matches_monte_carlo_drift_regime`
(fit: 400 synthetic samples, shells `[0.75, 0.9]` with `[24, 32]` sources; `y0 = [1.2, 0, 0, 0, 0,
0]`; `duration = 300`, `dt = 30`). Final 3D position `sigma`:

| Propagator | position sigma | rel. error vs STM | wall time* |
| --- | ---: | ---: | ---: |
| STM (deterministic) | 1.38495 | — | ~11 ms |
| MC, N = 500   | 1.33440 | 3.65%  | ~68 ms  |
| MC, N = 2000  | 1.36133 | 1.71%  | ~209 ms |
| MC, N = 8000  | 1.38484 | 0.008% | ~806 ms |

\*Indicative single-machine CPU timings; the point is the scaling, not the absolute values.

The Monte Carlo estimate converges to the STM result as `N` grows (0.008% at `N = 8000`), confirming
they describe the same force-error covariance; the residual gap at small `N` is sampling noise, not
bias.

## Cost trade-off

- **STM** is deterministic and **sampling-free**: one variational integration of a `6 x n_sources`
  Jacobian, cost independent of any sample count, and reproducible bit-for-bit. It gives the smooth
  `6x6` covariance directly (no Monte Carlo noise floor to beat down).
- **MC** cost scales with the sample count `N` (it integrates `N` perturbed trajectories), and its
  covariance carries `O(1/sqrt(N))` sampling noise — reaching STM-level agreement above needed
  `N ≈ 8000` and ~70x the wall time. MC's advantage is that it makes **no linearization assumption**,
  so it remains the cross-check when uncertainty is large or the horizon is long.

Use **STM** for a fast, deterministic covariance in the small-perturbation regime; use **MC** to
validate it (or when the linear assumption is suspect).

## When the linearization is valid / breaks

- Valid when the force-error perturbation stays small relative to the nominal dynamics over the
  horizon (the `P(t) = J Sigma J^T` map is first-order in `sigma`).
- Breaks for large posterior uncertainty or long horizons, where second-order trajectory curvature
  matters — there the STM covariance and the MC sample covariance diverge. Always cross-check against
  the MC sampler before reading anything into a long-horizon STM covariance.

## What should not be claimed

- No validated operational orbit determination or realistic operational state covariance.
- No position-error prediction — a propagated `sigma` is a *force-model-error* dispersion, not a
  position-accuracy guarantee (force-risk ⊥ position-error on the in-distribution diagnostic).
- No deterministic accuracy improvement — the nominal trajectory uses the posterior **mean**, which
  is the ridge point estimate.
- State only that these propagators map the fitted force-error posterior into a (linearized or
  Monte-Carlo-sampled) state covariance, and that the two agree in the small-perturbation regime.
