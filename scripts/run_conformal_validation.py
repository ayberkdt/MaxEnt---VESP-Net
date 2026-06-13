"""Run L60/L90 operational conformal validation and write a benchmark summary."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

from vesp.common.artifacts import atomic_write_json, atomic_write_text
from vesp.common.config import load_config
from vesp.uq.run import run

TARGET_Z_STD = (0.70, 1.30)
TARGET_PICP90 = (0.85, 0.95)
DEFAULT_CASES = (
    ("L60", "configs/vespuq/vespuq_real_lunar.yaml", "benchmarks/vespuq_real_lunar_report.md"),
    ("L90", "configs/vespuq/vespuq_real_lunar_L90.yaml", "benchmarks/vespuq_real_lunar_L90_report.md"),
)


def conformal_config(
    config: dict[str, Any],
    *,
    run_name: str,
    output_root: str | Path,
    n_orbits: int | None = None,
    n_points: int | None = None,
    save_model: bool = False,
) -> dict[str, Any]:
    """Return a config copy with operational conformal prediction enabled."""

    cfg = copy.deepcopy(config)
    uq = cfg.setdefault("uq", {})
    uq.setdefault("conformal", {}).update(
        {
            "apply": True,
            "by_band": True,
            "min_band_n": int(uq.get("conformal", {}).get("min_band_n", 30)),
            "alpha": float(uq.get("conformal", {}).get("alpha", 0.10)),
            "prediction_mode": str(uq.get("conformal", {}).get("prediction_mode", "norm")),
        }
    )
    screening = uq.setdefault("screening", {})
    if n_orbits is not None:
        screening["n_orbits"] = int(n_orbits)
    if n_points is not None:
        screening["n_points"] = int(n_points)
    output = cfg.setdefault("output", {})
    output["output_dir"] = str(output_root)
    output["run_name"] = run_name
    output["save_model"] = bool(save_model)
    return cfg


def band_acceptance(calibration: dict[str, Any]) -> dict[str, Any]:
    """Evaluate z_std/PICP90 acceptance for the low/mid/high bands."""

    out: dict[str, Any] = {}
    for band in ("low", "mid", "high"):
        metrics = calibration.get(band) or {}
        z_std = _float(metrics.get("z_std"))
        picp90 = _float(metrics.get("picp_90"))
        out[band] = {
            "z_std": z_std,
            "picp_90": picp90,
            "z_std_in_range": _in_range(z_std, TARGET_Z_STD),
            "picp90_in_range": _in_range(picp90, TARGET_PICP90),
        }
    out["all_bands_pass"] = all(
        row["z_std_in_range"] and row["picp90_in_range"]
        for row in out.values()
        if isinstance(row, dict)
    )
    return out


def run_case(
    label: str,
    config_path: str | Path,
    *,
    output_root: str | Path,
    run_suffix: str,
    n_orbits: int | None,
    n_points: int | None,
    save_model: bool,
) -> dict[str, Any]:
    base = load_config(config_path)
    base_name = str(base.get("output", {}).get("run_name", f"vespuq_{label.lower()}"))
    run_name = f"{base_name}{run_suffix}"
    cfg = conformal_config(
        base,
        run_name=run_name,
        output_root=output_root,
        n_orbits=n_orbits,
        n_points=n_points,
        save_model=save_model,
    )
    report = run(cfg)
    calibration = report.get("experiment_1_calibration", {})
    return {
        "label": label,
        "config_path": str(config_path),
        "run_dir": str(Path(output_root) / run_name),
        "calibration": {
            band: calibration.get(band)
            for band in ("all", "low", "mid", "high")
            if isinstance(calibration.get(band), dict)
        },
        "conformal": report.get("conformal_calibration") or {},
        "acceptance": band_acceptance(calibration),
        "summary": report.get("summary", {}),
        "runtime": report.get("runtime", {}),
    }


def build_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# VESP-UQ Operational Conformal Validation",
        "",
        "This report reruns the real-lunar L60 and L90 residual-band configs with "
        "`uq.conformal.apply: true` and per-band conformal scaling enabled. It validates the "
        "operational prediction path, not the older audit-only conformal threshold path.",
        "",
        f"Acceptance target: z_std in [{TARGET_Z_STD[0]:.2f}, {TARGET_Z_STD[1]:.2f}], "
        f"PICP90 in [{TARGET_PICP90[0]:.2f}, {TARGET_PICP90[1]:.2f}] for low/mid/high bands.",
        "",
        "| case | band | baseline z_std | conformal z_std | baseline PICP90 | conformal PICP90 | band scale | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in results["cases"]:
        baseline = case.get("baseline_calibration") or {}
        scales = _band_scales(case.get("conformal") or {})
        for band in ("low", "mid", "high"):
            base = baseline.get(band, {})
            after = case.get("calibration", {}).get(band, {}) or {}
            acceptance = case.get("acceptance", {}).get(band, {}) or {}
            passed = acceptance.get("z_std_in_range") and acceptance.get("picp90_in_range")
            lines.append(
                "| {case} | {band} | {base_z} | {after_z} | {base_picp} | {after_picp} | {scale} | {passed} |".format(
                    case=case["label"],
                    band=band,
                    base_z=_fmt(base.get("z_std")),
                    after_z=_fmt(after.get("z_std")),
                    base_picp=_fmt(base.get("picp_90")),
                    after_picp=_fmt(after.get("picp_90")),
                    scale=_fmt(scales.get(band, scales.get("global"))),
                    passed="yes" if passed else "no",
                )
            )
    lines += [
        "",
        "## Run Directories",
        "",
    ]
    for case in results["cases"]:
        lines.append(f"- {case['label']}: `{case['run_dir']}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run operational conformal validation on L60/L90 configs.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for conformal validation runs")
    parser.add_argument("--run-suffix", default="_conformal", help="Suffix appended to each base run_name")
    parser.add_argument("--report", default="benchmarks/vespuq_conformal_validation.md", help="Markdown report path")
    parser.add_argument("--json", default="benchmarks/vespuq_conformal_validation.json", help="JSON summary path")
    parser.add_argument("--n-orbits", type=int, help="Override screening n_orbits for quicker smoke validation")
    parser.add_argument("--n-points", type=int, help="Override screening n_points for quicker smoke validation")
    parser.add_argument("--save-model", action="store_true", help="Persist conformal model artifacts")
    args = parser.parse_args(argv)

    cases = []
    for label, config_path, baseline_path in DEFAULT_CASES:
        print(f"Running conformal validation case {label}: {config_path}")
        case = run_case(
            label,
            config_path,
            output_root=args.output_root,
            run_suffix=args.run_suffix,
            n_orbits=args.n_orbits,
            n_points=args.n_points,
            save_model=args.save_model,
        )
        case["baseline_report"] = baseline_path
        case["baseline_calibration"] = parse_report_calibration(Path(baseline_path))
        cases.append(case)

    results = {
        "schema_version": 1,
        "target_z_std": list(TARGET_Z_STD),
        "target_picp90": list(TARGET_PICP90),
        "cases": cases,
    }
    atomic_write_json(args.json, results)
    atomic_write_text(args.report, build_markdown(results))
    print(f"Wrote conformal validation report: {args.report}")
    return 0


def parse_report_calibration(path: Path) -> dict[str, dict[str, float | None]]:
    """Parse the calibration table from a checked benchmark markdown report."""

    if not path.exists():
        return {}
    rows: dict[str, dict[str, float | None]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not re.match(r"\|\s*(all|low|mid|high)\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        rows[cells[0]] = {
            "mean_radius": _float(cells[1]),
            "rmse": _float(cells[2]),
            "mean_pred_std": _float(cells[3]),
            "mean_epistemic_std": _float(cells[4]),
            "z_std": _float(cells[5]),
            "picp_90": _float(cells[6]),
        }
    return rows


def _band_scales(conformal: dict[str, Any]) -> dict[str, float | None]:
    scales = {"global": _float((conformal.get("global") or {}).get("scale"))}
    for band in conformal.get("bands", []) or []:
        if band.get("used") and band.get("name"):
            scales[str(band["name"])] = _float(band.get("scale"))
    return scales


def _in_range(value: float | None, rng: tuple[float, float]) -> bool:
    return value is not None and rng[0] <= value <= rng[1]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    x = _float(value)
    return "n/a" if x is None else f"{x:.3g}"


if __name__ == "__main__":
    raise SystemExit(main())
