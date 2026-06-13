import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from vesp.common.config import load_config
from vesp.uq.figures import FIGURE_STEMS, render_iac_figures
from vesp.uq.io.run_artifacts import write_run_artifacts

EVIDENCE_MD = """# VESP-UQ IAC Evidence Pack

This bundle aggregates the claim-mapped evidence for the VESP-UQ calibration layer, tying each reported table and figure to its reproducible run artifact.

## Claim Map

### 1. Post-Hoc Force-Risk Calibration (Band-Limited)
**Claim**: *The heteroscedastic noise model calibrates every altitude band in-distribution on real lunar data (PICP90 ~0.86-1.0, z_std ~1).* (SCIENTIFIC_CLAIMS.md)
**Evidence**: `calibration_report.md` (shows per-band epistemic/predictive ratios and predictive coverage).

### 2. OOD Altitude Sweep (Force-Risk Detection)
**Claim**: *The linear-Gaussian posterior's epistemic uncertainty grows under low-altitude extrapolation.* (SCIENTIFIC_CLAIMS.md)
**Evidence**: `ood_altitude_sweep.md` (shows explicit low vs high altitude predicted force error and epistemic standard deviation).

### 3. Trajectory Force-Risk Screening
**Claim**: *VESP-UQ detects low-altitude/OOD passes and ranks true force-model error along trajectories.* (VESP_UQ_NEXT_STEPS.md)
**Evidence**: `force_error_benchmark.md` (shows risk Spearman and lift-over-random against true force error).

### 4. Zero-Alarm Absolute-Threshold Screening
**Claim**: *VESP-UQ supports zero-alarm absolute-threshold screening with a physical budget.* (benchmarks/README.md)
**Evidence**: `absolute_threshold_screening.md` (shows that an absolute expected error threshold correctly suppresses alarms when risk is bounded).

### 5. Linear Propagation (STM) Covariance
**Claim**: *Provides a deterministic linearized (STM) `6x6` force-error state covariance along a nominal orbit... exploratory, not validated orbit determination.* (SCIENTIFIC_CLAIMS.md)
**Evidence**: `linear_propagation.md` (from `run_linear_propagation.py`).

### 6. Exploratory Force-Model Correction
**Claim**: *Evaluates whether mean-error correction reduces integrated position error and tracks the per-RHS cost.* (benchmarks/README.md)
**Evidence**: `force_correction_benchmark.md` (from `run_force_correction_benchmark.py`).

### 7. Position-Error Diagnostic (Null Result)
**Claim**: *Does not claim position-error prediction. Force-risk does not rank long-horizon ST-LRPS position error.* (SCIENTIFIC_CLAIMS.md)
**Evidence**: `position_error_diagnostic.md` (skipped or indicates diagnostic only).

### 8. Dynamics-Aware Risk Diagnostic (Null Result, N10)
**Claim**: *Weighting the force-error posterior by linearized trajectory dynamics (STM dispersion) also does NOT rank long-horizon ST-LRPS position error -- reported as an honest exploratory null, never a position-error claim.* (VESP_UQ_NEXT_STEPS.md N10)
**Evidence**: `stm_dispersion_diagnostic.md` (from `benchmark_stm_dispersion.py`; requires the local 512-scenario set).

### 9. Surrogate-Agnosticism Across Error Bands (N11)
**Claim**: *The same layer calibrates a second, disjoint residual band (degree-31..90, a degree-30 truncation surrogate) without retuning; coverage is conservative rather than sharp on the second band.* (benchmarks/vespuq_real_lunar_L90_report.md)
**Evidence**: `vespuq_real_lunar_L90_report.md` (band-vs-band comparison table included).

## Provenance
Every file in this pack is tracked in `run_manifest.json` via SHA-256 checksums, tying it directly to the exact source configurations.

## Publication Figures
The `figures/` directory contains PNG and PDF renderings of the same checked evidence: reliability
by band, sigma-vs-altitude, force-risk ranking, MC-vs-STM agreement, and L60-vs-L90 band comparison.
"""


def _train_run_dir(config_path: str) -> Path:
    cfg = load_config(config_path)
    output = cfg.get("output", {})
    return Path(output.get("output_dir", "outputs")) / str(output.get("run_name", "vesp_run"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assemble the IAC evidence pack.")
    parser.add_argument("--config", default="configs/vespuq/vespuq_smoke.yaml", help="Base config to use for generating evidence if not --collect-only")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing outputs, do not run benchmarks")
    parser.add_argument("--out-dir", default="outputs/iac_pack", help="Output directory for the evidence bundle")
    parser.add_argument("--skip-figures", action="store_true", help="Skip publication figure rendering")
    parser.add_argument("--train-run", help="Training run directory to use for figure inputs (default: resolve from --config)")
    parser.add_argument(
        "--allow-placeholder-figures",
        action="store_true",
        help="Allow figures rendered with missing-data placeholders instead of failing the pack build",
    )
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)


    # 1. Run benchmarks if needed
    if not args.collect_only:
        print(f"Running benchmarks with config: {args.config}...")

        print("  -> run_iac_benchmarks.py")
        subprocess.check_call([sys.executable, "scripts/run_iac_benchmarks.py", "--config", args.config, "--out-dir", "outputs/iac"])

        print("  -> run_linear_propagation.py")
        subprocess.check_call([sys.executable, "scripts/run_linear_propagation.py", "--config", args.config, "--out-dir", "outputs/linear_propagation"])

        print("  -> run_force_correction_benchmark.py")
        subprocess.check_call([sys.executable, "scripts/run_force_correction_benchmark.py", "--config", args.config, "--out-dir", "outputs/correction"])

    print(f"Collecting evidence into {out}...")

    # 2. Collect artifacts
    sources = {
        "calibration_report.md": Path("outputs/iac/calibration_report.md"),
        "force_error_benchmark.md": Path("outputs/iac/force_error_benchmark.md"),
        "ood_altitude_sweep.md": Path("outputs/iac/ood_altitude_sweep.md"),
        "absolute_threshold_screening.md": Path("outputs/iac/absolute_threshold_screening.md"),
        "position_error_diagnostic.md": Path("outputs/iac/position_error_diagnostic.md"),
        "linear_propagation.md": Path("outputs/linear_propagation/linear_propagation.md"),
        "force_correction_benchmark.md": Path("outputs/correction/force_correction_benchmark.md"),
        # Optional evidence (collected when present; needs local data / a real-lunar run):
        "stm_dispersion_diagnostic.md": Path("outputs/stm_dispersion/stm_dispersion_diagnostic.md"),
        "vespuq_real_lunar_L90_report.md": Path("benchmarks/vespuq_real_lunar_L90_report.md"),
    }

    collected = []
    missing = []

    for dest_name, src_path in sources.items():
        if src_path.exists():
            shutil.copy(src_path, out / dest_name)
            collected.append(dest_name)
        else:
            missing.append(str(src_path))

    if missing:
        print(f"Warning: the following evidence files were missing and skipped: {missing}")

    figure_artifacts = {}
    figure_statuses = {}
    figure_artifact_statuses = {}
    if not args.skip_figures:
        print("Rendering publication figures...")
        train_run = Path(args.train_run) if args.train_run else _train_run_dir(args.config)
        figure_manifest = render_iac_figures(
            train_run=train_run,
            iac_dir="outputs/iac",
            linear_dir="outputs/linear_propagation",
            benchmarks_dir="benchmarks",
            out_dir=out / "figures",
        )
        for entry in figure_manifest.get("figures", []):
            for ext in ("png", "pdf"):
                path = Path(entry[ext])
                artifact_name = f"figures/{path.name}"
                figure_artifacts[artifact_name] = path
                figure_artifact_statuses[artifact_name] = entry.get("status", "ok")
        figure_artifacts["figures/figures_manifest.json"] = out / "figures" / "figures_manifest.json"
        missing_figures = [
            entry["name"]
            for entry in figure_manifest.get("figures", [])
            if entry.get("status") == "missing_data"
        ]
        figure_statuses = {
            entry["name"]: entry.get("status", "ok")
            for entry in figure_manifest.get("figures", [])
        }
        figure_artifact_statuses["figures/figures_manifest.json"] = (
            "missing_data" if missing_figures else "ok"
        )
        if missing_figures:
            message = (
                "Rendered placeholder figures for missing data: "
                f"{missing_figures}. Re-run with --allow-placeholder-figures for partial packs."
            )
            if not args.allow_placeholder_figures:
                raise SystemExit(message)
            print(f"Warning: {message}")

    # 3. EVIDENCE.md + manifest via the artifact layer. The collected evidence files were
    # copied above, so checksum them into the manifest as consumed inputs -- every table in
    # the pack traces back to the exact bytes of the run that produced it.
    print("Writing artifact manifest...")
    write_run_artifacts(
        out_dir=out,
        tool="build_iac_pack",
        json_files={},
        text_files={"EVIDENCE.md": EVIDENCE_MD},
        artifact_files=figure_artifacts,
        artifact_statuses=figure_artifact_statuses,
        config={
            "source_config": args.config,
            "collected_files": collected,
            "figure_stems": list(FIGURE_STEMS) if not args.skip_figures else [],
            "figure_statuses": figure_statuses,
            "train_run": str(Path(args.train_run) if args.train_run else _train_run_dir(args.config))
            if not args.skip_figures else None,
        },
        inputs={name: out / name for name in collected},
    )

    # 4. Zip the bundle next to the output directory (honors --out-dir).
    zip_base = out.parent / out.name
    shutil.make_archive(str(zip_base), 'zip', out)

    print(f"IAC Pack assembled at: {out}")
    print(f"IAC Pack archived at: {zip_base}.zip")

if __name__ == "__main__":
    main()
