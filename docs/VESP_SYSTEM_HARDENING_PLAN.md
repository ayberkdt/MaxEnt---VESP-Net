# VESP System Hardening Plan

Created: 2026-06-11
Status: **H1-H8 complete; no active hardening item is scheduled in this document.**

This plan is the follow-up to the whole-system audit after the N18-N21 implementation wave. It
focuses on silent failures, brittle integration points, and DRY debt across the maintained
VESP-UQ/common/UI surface, while keeping the legacy feasibility and ST-LRPS adapter trees scoped
to targeted fixes.

## Audit Snapshot

Verified locally:

- Targeted N18-N21 regression suite: `80 passed`.
- UI import-safety suite: included in the regression run; the Mission Console remains heavy-import
  free at shell import time.
- Figure renderer: covers both normal artifact inputs and missing-input placeholder degradation.
- Maintained parse-only check: `src/vesp/uq`, `src/vesp/common`, `src/vesp/ui`, and `scripts` parse
  cleanly as UTF-8 Python source.
- H4-H5 regression group: `69 passed`; offscreen Compare identity smoke included.
- Full-runtime scoped typecheck: `Success: no issues found in 51 source files`.
- Full maintained lint scope: `All checks passed`.
- `git diff --check`: clean apart from Windows LF/CRLF warnings.

Found and fixed during the audit:

- `vesp.uq.figures` crashed when both the MC-vs-STM markdown and fallback linear-propagation CSV
  were missing. It now emits a placeholder figure and manifest entry instead.
- `scripts/build_iac_pack.py` silently fell back to `outputs/vespuq_smoke` when config resolution
  failed. H2 replaced that ambiguity with explicit `--train-run` selection and fail-fast figure
  status handling.
- Follow-up implementation completed H1-H8: source parsing is CI-gated without bytecode writes,
  evidence-pack placeholder figures fail by default, full L60/L90 operational conformal
  validation is reported, the Compare workflow has an offscreen Qt E2E smoke, repeated UI
  controls/helpers are shared, typechecking installs the full runtime dependency graph, manifests
  share one schema, lightweight CLI contracts are pinned, and documentation roles are separated.

Known local verification limits:

- The bundled Python was supplemented with `mypy`, pinned `ruff`, and `hypothesis` for this audit;
  CI remains the reproducible source of the declared `.[dev]` environment.
- `python -m compileall src scripts tests` can still be noisy locally because `scripts/__pycache__`
  writes hit Windows permission errors; use `python scripts/check_source_parse.py src scripts tests
  ui` for the no-bytecode UTF-8/AST gate.

## Priority Work

### H1 - Whole-tree source encoding and compile hygiene - **DONE**

Why: a whole-tree compile pass found a real non-UTF-8 source file in the legacy ST-LRPS adapter.
Even if that path is outside the maintained UQ gate, a source tree that cannot be decoded by Python
is a sharp edge for packaging, search, documentation, and future CI.

Deliverables:

- Normalize `src/vesp/adapters/st_lrps/training/losses.py` to valid UTF-8 without changing code
  behavior.
- Add a parse-only source check that does not write `.pyc` files and therefore avoids local
  `__pycache__` permission noise.
- Decide whether legacy adapter parse checks are warning-only or blocking.

Acceptance:

- All tracked `.py` files decode as UTF-8 (`source-parse-ok: 285 files` locally).
- Parse check is blocking in CI via `scripts/check_source_parse.py`.
- The ST-LRPS `losses.py` docstring encoding issue was normalized to UTF-8 without behavior
  changes.

### H2 - Evidence-pack figure fail-fast policy - **DONE**

Why: placeholder figures are useful for local collection, but CI currently checks only file
existence. A pack can therefore contain valid PNG/PDF files that say "missing data" while the smoke
job still passes.

Deliverables:

- Add `--allow-placeholder-figures` to `scripts/build_iac_pack.py`; default CI behavior should fail
  if any figure entry has `status: missing_data`.
- Add `--train-run` to make the figure source run explicit instead of inferring it only from config.
- Record figure statuses in the top-level `run_manifest.json` config or metadata block.

Acceptance:

- Default pack builds fail on `missing_data` figure statuses.
- Local users can opt into placeholders for partial evidence collection with
  `--allow-placeholder-figures`.
- `--train-run` makes the figure source explicit, and bad config paths no longer silently produce
  smoke figures under another run.

### H3 - Conformal long-run validation and stale-update visibility - **DONE; target partially failed**

Why: at audit time, operational conformal prediction was implemented only on smoke data and the
falsifiable L90/L60 acceptance run was pending. Sequential updates without fresh validation also
kept the old conformal scale without sufficiently visible operational status.

Deliverables:

- Run the real-lunar L90 and L60 benchmark configs with `uq.conformal.apply: true`.
- Write before/after calibration tables into `benchmarks/` and update the model-card/report text.
- Mark conformal calibration as `stale_after_update: true` in `fit_info`/metadata when
  `update_error` is called without fresh validation data.

Acceptance:

- L90 z_std and PICP90 outcomes are reported honestly in
  `benchmarks/vespuq_conformal_validation.md`: only the mid band met both target ranges; low/high
  remain conservative/out of range.
- L60 drift from default behavior is quantified in the same table.
- Updated models expose `fit_info["conformal_stale_after_update"]` when an update does not include
  fresh validation data.

### H4 - Mission Console E2E smoke and UI DRY cleanup - **DONE**

Why: import-safety tests are solid, but the new Compare page still needs an end-to-end GUI smoke.
The UI also repeats model-picker, run-output action, `_fmt`, and JSON-report loading patterns across
Train, Screen, Model, Propagate, and Compare.

Deliverables:

- Add an offscreen Qt smoke test for Compare with two dummy saved models and a held-out CSV.
- Extract shared widgets/helpers:
  - `ModelArtifactPicker`
  - `RunOutputActions`
  - `safe_read_json`
  - a shared numeric `fmt`
- Keep heavy imports lazy after the refactor.

Acceptance:

- Compare identity run renders IoU 1.0 and Spearman 1.0 in an automated test.
- `tests/test_vespuq_ui.py::test_ui_imports_stay_light` remains green.
- Duplicate picker/result-action code is removed from at least three pages.

Done:

- Added `tests/test_vespuq_ui_e2e.py`, which creates two saved dummy models and held-out/trajectory
  CSVs, drives the real `scripts.compare_models` subprocess through an offscreen `ComparePage`,
  and verifies perfect identity agreement in the rendered KPI tiles.
- Added shared `ModelArtifactPicker`, `RunOutputActions`, `safe_read_json`, and `fmt` helpers.
  Model selection is shared by Compare, Screen, Model, and Update; output actions are shared by
  Compare, Screen, Train, and Propagate; report loading/formatting is shared across the UI.
- Added PyQt6 to the CI test environment so the offscreen smoke is exercised instead of skipped.

### H5 - Typecheck gate fidelity - **DONE**

Why: the new mypy CI job is wired, but it installs only `mypy` and the editable package with
`--no-deps`. That keeps CI light, but it can hide import drift behind `ignore_missing_imports`.

Deliverables:

- Decide the intended typecheck dependency set: minimal stubs-only, runtime-light, or full runtime.
- Add a local command in docs, for example `python -m pip install -e .[dev]`.
- Consider scoped per-module mypy strictness for pure utility modules (`vesp.common.units`,
  `vesp.uq.selection`, `vesp.uq.figures`) while keeping torch/PyQt-heavy paths pragmatic.

Acceptance:

- CI typecheck catches local interface drift, not just syntax.
- Developers can reproduce the typecheck command locally.
- No legacy feasibility or ST-LRPS churn is pulled into the gate.

Done:

- Chose the full runtime dependency set for fidelity: the typecheck job installs `.[dev]`, so
  torch, PyQt6, numpy, pandas, and the other declared runtime interfaces are present.
- Kept the mypy file scope limited to `vesp.uq`, `vesp.common`, and `vesp.ui`; legacy feasibility
  and ST-LRPS trees remain outside the blocking gate.
- Documented the matching local install and scoped lint/mypy commands in `README.md`.
- Extended `tests/test_typecheck_config.py` to reject a return to `--no-deps`.
- Fixed 86 surfaced typing errors without widening the gate or touching legacy trees; the scoped
  command is now clean across all 51 maintained source files.

### H6 - Artifact and manifest consolidation - **DONE**

Why: `vesp.common.artifacts.write_run_manifest` and `vesp.uq.io.run_artifacts.write_run_artifacts`
now overlap in provenance/checksum responsibilities, and figure files add another prewritten
artifact mode.

Deliverables:

- Define one shared manifest schema for generated files, prewritten artifact files, consumed inputs,
  and optional status fields.
- Move checksum/status helpers into `vesp.common.artifacts`.
- Keep benchmark/script manifests backward-compatible for existing tests and checked artifacts.

Acceptance:

- Existing run, screen, benchmark, compare, and IAC-pack manifests still validate.
- Prewritten artifacts, including PNG/PDF figures, are represented consistently.
- Missing inputs/artifacts are visible and machine-readable.

Done:

- Added shared `file_manifest_entries`, `build_run_manifest`, and `write_manifest` helpers to
  `vesp.common.artifacts`; both training-oriented and script-oriented writers now use them.
- Standardized every file entry on `path` + `origin` (`generated`, `prewritten`, or `consumed`),
  checksum/byte size for existing files, `missing: true` for absent files, and optional `status`.
- Kept `vesp.uq.io.run_artifacts.write_run_artifacts` as a backward-compatible script API without
  creating the training layout's `checkpoints/` directory.
- IAC PNG/PDF files and their figure manifest now carry `ok`/`missing_data` status directly in
  `run_manifest.json`, while the existing top-level schema version and artifact/input maps remain.
- Verified common/feasibility, run, screen, benchmark, comparison, propagation, correction,
  figure, and IAC-pack contracts with `101 passed`.

### H7 - Script and CLI smoke matrix - **DONE**

Why: many scripts are protected only indirectly by broad CI smoke. A few lightweight pytest-level
tests would catch argument drift and output-contract regressions faster.

Deliverables:

- Add pytest-level smoke tests for:
  - `scripts/render_iac_figures.py`
  - `scripts/build_iac_pack.py --collect-only`
  - `scripts/compare_models.py` output contract
  - `python -m vesp.uq.run --version` and `python -m vesp.uq.screen --version`
- Keep heavy real-lunar runs out of the unit test suite.

Acceptance:

- Script output contracts are pinned without long runtime.
- CI smoke can focus on integration rather than first-line contract discovery.

Done:

- Added `tests/test_script_cli_smoke.py` with real subprocess checks for
  `scripts/render_iac_figures.py`, `scripts/build_iac_pack.py --collect-only --skip-figures`,
  and both `python -m vesp.uq.{run,screen} --version` entry points.
- The renderer smoke pins all five PNG/PDF groups plus `figures_manifest.json`, including
  controlled `missing_data` degradation without fitting a model.
- The collect-only smoke runs in an isolated directory, verifies that no benchmarks launch, and
  pins `EVIDENCE.md`, the run manifest, and zip output.
- Strengthened the existing `scripts.compare_models` subprocess test to pin its JSON top-level
  contract, provenance, generated artifacts, consumed inputs, and manifest origins.
- All H7 subprocess smokes run from source without requiring an editable install; the existing CI
  smoke remains responsible for full train/serve and evidence-generation integration.
- Verified the complete H7 and adjacent artifact/serve regression group with `39 passed`.

### H8 - Documentation and plan hygiene - **DONE**

Why: the current roadmap file is now a history plus live plan. Future work will be easier if
implementation history and active hardening tasks are separated.

Deliverables:

- Keep `docs/VESP_UQ_NEXT_STEPS.md` as the completed/ongoing VESP-UQ roadmap.
- Use this file for system hardening, silent-bug prevention, and DRY cleanup.
- Add a short "current open risks" section to `README.md` or `benchmarks/README.md` only if it
  helps users run the evidence pack correctly.

Acceptance:

- A new contributor can tell which tasks are done, which are pending validation, and which are
  hardening work.
- No stale acceptance criteria remain marked as fully done without the supporting run artifacts.

Done:

- Reframed `VESP_UQ_NEXT_STEPS.md` as completed N1-N21 implementation history plus an explicitly
  unscheduled research backlog; hardening remains in this H-plan.
- Marked N18 as implemented with a partially failed validation target, not as pending work.
- Added evidence-pack open risks and safe invocation guidance to `benchmarks/README.md`.
- Reconciled README claims for exploratory MC/STM propagation, the eight-page Mission Console,
  and the documentation map.
- Added documentation-status regression tests so completed plan states and evidence-pack warnings
  cannot silently drift back.

## DRY Debt Audit Outcome

- Model artifact selection, output actions, numeric formatting, and JSON loading were consolidated
  under H4.
- Artifact manifest writing and prewritten-file status handling were consolidated under H6.
- CSV/markdown parsing in `vesp.uq.figures` remains intentionally local; extract it only if the
  publication-figure surface expands.

## Explicitly Not In This Plan

- Full ST-LRPS adapter refactor.
- Broad reformatting of the legacy feasibility tree.
- Any work that claims position-error prediction, validated operational orbit covariance, or
  deterministic surrogate accuracy improvement.
