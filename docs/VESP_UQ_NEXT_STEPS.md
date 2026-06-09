# VESP-UQ — Next Steps Plan

A forward-looking, prioritized plan for the VESP-UQ layer, grounded in the current state of the
repository. This is a working roadmap, **not** a claims policy — the binding policy on what may be
claimed stays in [`SCIENTIFIC_CLAIMS.md`](SCIENTIFIC_CLAIMS.md) and the scope boundaries in
[`VESP_UQ_LIMITATIONS.md`](VESP_UQ_LIMITATIONS.md). Every item below must respect those constraints
(no position-error prediction claim, no validated operational orbit covariance, no invented units).

## Where we are

Recently completed (in the working tree, see commit grouping in N0):

- Conformal force-error calibration + sentinel false-negative audit (`vesp.uq.conformal`,
  `vesp.uq.audit`, `scripts/run_calibration_audit.py`).
- Physical acceleration-budget screening (`vesp.uq.physical_units`, `threshold_source:
  physical_budget`, `scripts/run_physical_budget_screening.py`) + optional conformal-corrected
  threshold.
- Propagation hardening: `vesp.uq.propagation` (MC) sign/eps consistency + scale-relative Cholesky
  jitter fix; new deterministic linearized `vesp.uq.linear_propagation` (STM) covariance.
- Unit-aware external trajectory loading (`vesp.uq.io.trajectory_loader`).
- Doc reconciliation (MC sampler + ST-LRPS adapter framed as *exploratory, not validated*) and a
  minimal CI workflow (`.github/workflows/ci.yml`).
- Test count: 335 → 364.

Open gaps found while surveying the code (evidence in parentheses):

- The newer scripts write bare output files and do **not** use the artifact/manifest +
  checksum system the main run uses (`grep` of `ensure_run_layout`/`write_run_manifest` over
  `scripts/run_calibration_audit.py`, `run_physical_budget_screening.py`,
  `run_force_error_benchmark.py`, `compare_risk_baselines.py` → none).
- No linter / formatter / type-check config anywhere (`pyproject.toml` has no `ruff`/`mypy`/`black`;
  no `.ruff.toml`/`.flake8`).
- `vesp.uq.linear_propagation` has a module + tests but **no driver script** and no benchmark doc,
  unlike the MC sampler (`scripts/run_propagation.py`).
- Several scripts are only exercised by the CI smoke step, with no pytest-level output assertions
  (`run_calibration_audit`, `run_force_error_benchmark`, `compare_risk_baselines`, `run_propagation`).
- The ST-LRPS adapter (`src/vesp/adapters/st_lrps`, 72 `.py` files) has **zero tests** and is
  exploratory wiring.

## Phases (prioritized)

### N0 — Commit the current Tier 1–5 work (prerequisite)

- **Why:** 18 modified + 4 new files are uncommitted; everything below should build on a clean base.
- **Action:** commit in logical groups —
  1. propagation hardening + CI + doc reconciliation,
  2. trajectory unit-awareness,
  3. doc-integrity (benchmarks README, units note, README),
  4. linearized covariance propagation,
  5. conformal-corrected physical budget.
- **Acceptance:** clean working tree; CI green on the pushed branch.
- **Effort:** XS.

### N1 — Reproducibility: route script outputs through the artifact/manifest system — **DONE**

- **Why:** reproducibility is a stated project value, but `run_calibration_audit`,
  `run_physical_budget_screening`, `run_force_error_benchmark`, `compare_risk_baselines` wrote bare
  JSON/MD/CSV with no run manifest, config snapshot, or SHA-256 checksums.
- **Done:** added `vesp.uq.io.run_artifacts.write_run_artifacts` (atomic writes + injected
  `_provenance` per JSON + `run_manifest.json` with config snapshot, seed, environment, and per-file
  SHA-256 + byte size). Routed all four scripts through it and added a `write_run_manifest` to the
  main `vesp.uq.run`. Output filenames preserved. Tests in `tests/test_uq_run_artifacts.py` assert
  the manifest exists and its checksums match the files on disk (369 tests pass).

### N2 — Code quality: lint + format check in CI — **DONE**

- **Why:** stated "code quality" goal; no static analysis existed (a stray placeholder import
  slipped in during recent work and was only caught by hand).
- **Done:** added a `ruff` config to `pyproject.toml` (`select = E, F, I, W, B, UP`,
  `line-length 120`, `target-version py310`; `E402` ignored in `scripts/`+`tests/` for the
  sys.path-before-import pattern; `E501` left to the formatter). The gate is **lint-only** by
  deliberate scope choice — a `[tool.ruff.format]` section is present for local use but the formatter
  is intentionally **not** CI-gated, so this stays behavior-preserving rather than a ~1.5k-line mass
  reformat. Added a dedicated `lint` job to `.github/workflows/ci.yml`
  (`ruff check src/vesp/uq scripts tests`) and pinned `ruff==0.15.16` in the `dev` extra. Fixed the
  surfaced issues (autofixed import order/whitespace/unused imports + manual unused-variable,
  `zip(..., strict=True)`, and loop-closure binding fixes); `ruff check` is clean on the uq surface
  and all three scoped dirs, and the suite still passes (372 tests). `mypy` was left out (optional in
  the plan; can be added non-blocking later).

### N3 — Propagation consolidation: driver + benchmark doc — **DONE**

- **Why:** the linearized STM covariance (`vesp.uq.linear_propagation`) had no driver script or doc,
  and there was no documented MC-vs-STM comparison even though a test already showed they agree in
  the linear regime.
- **Done:**
  - added `scripts/run_linear_propagation.py` (parity with `run_propagation.py`): fits from a config,
    propagates a low circular orbit, and writes nominal states, `6x6` covariances, and
    position/velocity sigma through the N1 artifact layer (`linear_propagation.{json,md}` +
    `linear_propagation_states.csv` + `run_manifest.json`); params come from an optional
    `uq.propagation` config block overridable by CLI flags;
  - added `benchmarks/covariance_propagation.md`: MC-vs-STM agreement (converges to **0.008%** at
    `N = 8000` in the drift regime) and the cost trade-off (STM deterministic / sampling-free, ~70x
    faster; MC scales with sample count and carries `O(1/sqrt(N))` noise), with the **exploratory,
    not validated** framing + the force-risk⊥position-error caveat;
  - added the two propagation rows + reproduce commands to `benchmarks/README.md`.
- **Also:** CI smoke now runs the new script, and a focused test
  (`tests/test_uq_linear_propagation_script.py`) locks the artifact + covariance contract (manifest
  checksums, `6x6` shape, `J(0) = 0`, CSV header/row count).
- **Acceptance:** met — script runs on the smoke config and writes artifacts; the doc states the
  honest scope; CI smoke covers the new script.

### N4 — Script-level test coverage — **DONE**

- **Why:** `run_calibration_audit`, `run_force_error_benchmark`, `compare_risk_baselines` were only
  exercised by the CI smoke step (no pytest assertions on their output schemas), so their JSON/CSV
  contracts could drift silently.
- **Done:** added `tests/test_uq_scripts.py` — one test per artifact-writing script asserting the
  JSON keys, CSV header + row count, and invariants (`flagged ⊆ trajectories`,
  `n_flagged ≤ n_trajectories`, `is_position_error_benchmark` is False). `run_propagation` writes no
  files (nothing to lock) and its MC core is already covered by `tests/test_uq_propagation.py`, so it
  gets only an import-safety guard; `run_linear_propagation` / `run_physical_budget_screening` are
  locked by their own modules, and the artifact/manifest contract by `tests/test_uq_run_artifacts.py`.
- **Acceptance:** met — output schemas locked; suite at 378 tests.

### N5 — ST-LRPS adapter boundary: bound it honestly — **DONE**

- **Why:** `src/vesp/adapters/st_lrps` (~70 files) is exploratory wiring with zero tests; its only
  VESP-UQ touchpoint is `scripts/run_stlrps_propagation.py` (uses the runtime force model as the MC
  base field).
- **Done:** added `tests/test_stlrps_adapter_boundary.py` — an import-safety guard for the seam
  (`load_surrogate_force_model`) and the script, plus a skip-guarded artifact-load smoke
  (`VESP_STLRPS_MODEL_DIR`) asserting the exact interface VESP-UQ depends on (`mu_si`, `degree_min`,
  `predict_residual_accel_fixed`). The adapter depends on the external `lunaris` package (not
  vendored here, not a declared dep), so it is not importable in a clean VESP-UQ environment; the
  tests `importorskip` and therefore **skip in CI** by design (they run where the adapter is
  installed). Documented the boundary in `src/vesp/adapters/README.md` and a new
  `docs/VESP_UQ_LIMITATIONS.md` subsection ("Adapter scope: only the force-model seam is in scope"),
  and reconciled the stale local-vs-orbit-covariance note there now that the N3 propagators exist.
- **Acceptance:** met — the VESP-UQ↔adapter seam has tests; the boundary is documented. (Full adapter
  testing is explicitly out of VESP-UQ scope.)

### N6 — Online force correction (Phase 5) — OPTIONAL, larger research item

- **Why:** the one remaining headline future-work item in the IAC plan: evaluate
  `a_corrected(x) = a_surrogate(x) + mean_error(x)` inside an integrator RHS, and benchmark the
  speed/accuracy trade-off (the docs warn that evaluating the full equivalent-source field every RHS
  call may erode the surrogate's speed advantage).
- **Action:** add an `a_corrected` RHS hook (reusing the same operator/sign/eps convention as the MC
  and STM propagators); a benchmark comparing surrogate vs surrogate+correction trajectories against
  a reference, reporting both accuracy delta **and** per-RHS cost. Frame honestly: the posterior mean
  is the ridge estimate, so this is a *force-model* correction, with no guaranteed long-horizon
  position-accuracy claim; report measured results only.
- **Acceptance:** benchmark runs on a synthetic reference; doc reports accuracy **and** cost with the
  honest caveats; tests cover the RHS hook's operator consistency.
- **Effort:** L. **Risk:** medium (scope + careful claims). Do only if explicitly desired.

## Recommended order

`N0 → ~~N1~~ → ~~N2~~ → ~~N3~~ → ~~N4~~ → ~~N5~~`, with `N6` optional. **N1–N5 are done; only N6
remains, and it is gated.** Rationale: commit first (N0); then the low-risk, high-value
reproducibility/quality items (N1, N2) that harden everything already built; then the propagation
capability as a documented, tested deliverable (N3) with script-level schema tests (N4); N5 bounds
the external ST-LRPS subsystem honestly. N6 is the only item that adds new research scope and should
be a deliberate, separately-approved choice — do it only on explicit request.

## Out of scope (and why)

- Full ST-LRPS adapter coverage/refactor — large vendored subsystem, outside the VESP-UQ
  calibration-layer focus (only its seam matters here; see N5).
- Anything that would claim position-error prediction, validated operational orbit covariance, or
  deterministic accuracy improvement — forbidden by `SCIENTIFIC_CLAIMS.md`.
- The Stage-3 MaxEnt / neural-density extensions — separate track from the force-risk/UQ layer.
