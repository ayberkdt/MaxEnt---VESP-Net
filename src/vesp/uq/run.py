"""Driver for the VESP-UQ IAC experiments: standalone calibration + trajectory risk screening.

    python -m vesp.uq.run --config configs/vespuq_real_lunar.yaml

Runs, on the real lunar residual dataset, the two experiments the plan calls minimal:

  * Experiment 1 -- standalone residual-error calibration (does the layer reduce low-altitude
    overconfidence?): fit the equivalent-source posterior, calibrate altitude-dependent noise,
    report per-band PICP90 / z_std and whether epistemic uncertainty grows toward the surface.
  * Experiment 3 -- trajectory risk screening: score a synthetic orbit ensemble with the
    fitted layer, flag the riskiest subset for high-fidelity rerun, and validate against a
    nearest-neighbour ground-truth error read from held-out real samples.

The dataset's degree-2..60 GRAIL residual IS the error of a degree-60 truncation surrogate, so
``e_a = a_reference - a_surrogate`` is exactly the stored residual (surrogate = the truncated
model); we therefore fit with ``surrogate = 0``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import torch

from vesp.common.artifacts import atomic_write_json, atomic_write_text, ensure_run_layout
from vesp.common.config import get_dtype, load_config
from vesp.common.units import UnitConfig
from vesp.data.dataset import load_csv_dataset
from vesp.uq.ensemble import generate_orbit_ensemble, nearest_neighbor_error_magnitude
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.trajectory import select_reruns


def _split(n: int, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    n_train = int(round(float(train_fraction) * n))
    return perm[:n_train], perm[n_train:]


def run_vespuq(config: dict) -> dict:
    dtype = get_dtype(config)
    data_cfg = config.get("data", {})
    path = data_cfg.get("path")
    if not path:
        raise ValueError("config.data.path is required (a residual-gravity CSV)")
    units = UnitConfig.from_config(config)
    data = load_csv_dataset(path, dtype=dtype, unit_config=units)
    positions, error = data.positions, data.acceleration  # residual == surrogate error

    seed = int(config.get("seed", 0))
    train_idx, held_idx = _split(positions.shape[0], data_cfg.get("train_fraction", 0.7), seed)
    train_pos, train_err = positions[train_idx], error[train_idx]
    held_pos, held_err = positions[held_idx], error[held_idx]

    plugin = VESPUQPlugin.from_config(config)
    plugin.fit(train_pos, torch.zeros_like(train_err), train_err)

    bands = config.get("evaluation", {}).get("altitude_bands")
    calibration = plugin.evaluate_calibration(held_pos, held_err, altitude_bands=bands)

    # ---------------- Experiment 3: trajectory risk screening ----------------
    screen_cfg = config.get("uq", {}).get("screening", {})
    ensemble = generate_orbit_ensemble(
        n_orbits=int(screen_cfg.get("n_orbits", 200)),
        n_points=int(screen_cfg.get("n_points", 48)),
        r_peri_range=tuple(screen_cfg.get("r_peri_range", (1.02, 1.30))),
        r_apo_range=tuple(screen_cfg.get("r_apo_range", (1.30, 1.60))),
        seed=seed,
        dtype=dtype,
    )
    scoring = plugin.risk_scoring
    aggregate = torch.amax if scoring in {"max", "low_alt_integral", "time_above"} else torch.mean

    t0 = time.perf_counter()
    scores = plugin.score_ensemble(ensemble.trajectories)
    score_seconds = time.perf_counter() - t0
    risk_scores = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)

    # nearest-neighbour ground-truth error magnitude along each orbit. The oracle uses the full
    # real sample set (denser -> less NN noise); this is not leakage because the layer's RISK
    # score is the geometric posterior predictive std, never fit to these per-point magnitudes.
    true_error = torch.empty(len(ensemble.trajectories), dtype=torch.float64)
    for i, traj in enumerate(ensemble.trajectories):
        nn = nearest_neighbor_error_magnitude(traj.to(dtype), positions, error)
        true_error[i] = aggregate(nn.to(torch.float64))

    rerun_fraction = float(screen_cfg.get("rerun_fraction", 0.20))
    screening = select_reruns(risk_scores, rerun_fraction=rerun_fraction, true_error=true_error)

    n_traj = len(ensemble.trajectories)
    n_points_total = sum(int(t.shape[0]) for t in ensemble.trajectories)
    report = {
        "dataset": str(path),
        "fit": plugin.fit_info,
        "experiment_1_calibration": calibration,
        "experiment_3_screening": {
            "scoring": scoring,
            "n_trajectories": n_traj,
            "n_output_points_total": n_points_total,
            "screen": screening.to_dict(),
            "runtime": {
                "score_seconds_total": score_seconds,
                "score_ms_per_trajectory": 1.0e3 * score_seconds / max(1, n_traj),
                "score_us_per_output_point": 1.0e6 * score_seconds / max(1, n_points_total),
            },
        },
    }
    report["summary"] = _summary(report)
    return report


def _summary(report: dict) -> dict:
    cal = report["experiment_1_calibration"]
    screen = report["experiment_3_screening"]["screen"]
    out: dict = {}
    if "low_high_epistemic_std_ratio" in cal:
        out["epistemic_grows_at_low_altitude"] = cal["low_high_epistemic_std_ratio"] > 1.0
        out["low_high_epistemic_std_ratio"] = cal["low_high_epistemic_std_ratio"]
    low = cal.get("low", {})
    if "picp_90" in low:
        out["low_band_picp_90"] = low["picp_90"]
        out["low_band_calibrated_90"] = abs(low["picp_90"] - 0.90) <= 0.1
    out["rerun_fraction"] = screen["rerun_fraction"]
    out["capture_rate"] = screen.get("capture_rate")
    out["spearman_risk_vs_error"] = screen.get("spearman_risk_vs_error")
    out["error_ratio_flagged_to_accepted"] = screen.get("error_ratio_flagged_to_accepted")
    # the operational headline: flagged trajectories really do carry the larger reference error
    if screen.get("error_ratio_flagged_to_accepted"):
        out["screen_concentrates_error"] = screen["error_ratio_flagged_to_accepted"] > 1.0
    return out


def _fmt(x, spec: str = ".3g") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def build_report_md(report: dict) -> str:
    fit = report["fit"]
    cal = report["experiment_1_calibration"]
    screen = report["experiment_3_screening"]
    sc = screen["screen"]
    s = report["summary"]
    lines = [
        "# VESP-UQ Report - Equivalent-Source Uncertainty Calibration Layer",
        "",
        f"dataset: `{report['dataset']}`",
        f"sources: {fit['n_sources']}  |  reg: {fit['reg_method']} (lambda_l2={_fmt(fit.get('lambda_l2'))})  "
        f"|  noise_model: {fit['noise_model']}  |  global noise_std={_fmt(fit.get('noise_std'))}",
    ]
    if "altitude_noise_b" in fit:
        lines.append(
            f"altitude noise sigma^2(h)=a*h^(-b): a={_fmt(fit['altitude_noise_a'], '.3e')}, "
            f"b={_fmt(fit['altitude_noise_b'], '.3f')} (h=r-1; larger b = faster growth toward surface)"
        )
    lines += [
        "",
        "## Experiment 1 - Standalone residual-error calibration",
        "",
        "| band | mean_radius | rmse | mean_pred_std | mean_epistemic_std | z_std | picp_68 | picp_90 | picp_95 | nll |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("all", "low", "mid", "high"):
        m = cal.get(name)
        if not m:
            continue
        lines.append(
            f"| {name} | {_fmt(m.get('mean_radius'), '.3f')} | {_fmt(m.get('rmse'), '.3e')} | "
            f"{_fmt(m.get('mean_pred_std'), '.3e')} | {_fmt(m.get('mean_epistemic_std'), '.3e')} | "
            f"{_fmt(m.get('z_std'), '.2f')} | {_fmt(m.get('picp_68'), '.2f')} | {_fmt(m.get('picp_90'), '.2f')} | "
            f"{_fmt(m.get('picp_95'), '.2f')} | {_fmt(m.get('nll'), '.3f')} |"
        )
    if "low_high_epistemic_std_ratio" in cal:
        grows = s.get("epistemic_grows_at_low_altitude")
        lines += [
            "",
            f"- Epistemic uncertainty grows toward low altitude: **{'YES' if grows else 'NO'}** "
            f"(low/high epistemic std ratio = {_fmt(cal['low_high_epistemic_std_ratio'], '.2f')}).",
        ]
    lines += [
        "",
        "## Experiment 3 - Trajectory risk screening",
        "",
        f"- ensemble: {screen['n_trajectories']} orbits, {screen['n_output_points_total']} output points "
        f"(scoring = `{screen['scoring']}`)",
        f"- rerun threshold (risk score): {_fmt(sc['threshold'], '.3e')}  ->  "
        f"flagged {sc['n_flagged']}/{sc['n_trajectories']} ({_fmt(100 * sc['rerun_fraction'], '.1f')}%)",
        f"- capture rate (top-decile true-error orbits flagged): **{_fmt(sc.get('capture_rate'), '.2f')}**  "
        f"| precision: {_fmt(sc.get('precision'), '.2f')}",
        f"- Spearman(risk, true error): {_fmt(sc.get('spearman_risk_vs_error'), '.2f')}",
        f"- mean true error  flagged: {_fmt(sc.get('mean_error_flagged'), '.3e')}  vs  "
        f"accepted: {_fmt(sc.get('mean_error_accepted'), '.3e')}  "
        f"(ratio {_fmt(sc.get('error_ratio_flagged_to_accepted'), '.2f')}x)",
        f"- scoring runtime: {_fmt(screen['runtime']['score_ms_per_trajectory'], '.3f')} ms/trajectory "
        f"({_fmt(screen['runtime']['score_us_per_output_point'], '.2f')} us/output point)",
        "",
        "## Positioning",
        "",
        "_VESP-UQ is an uncertainty/risk-calibration layer, not a better residual surrogate. The "
        "posterior mean equals the ridge point estimate; the contribution is calibrated, "
        "altitude-aware error bars and the trajectory screen they enable._",
        "",
    ]
    return "\n".join(lines) + "\n"


def run(config: dict) -> dict:
    report = run_vespuq(config)
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs"))
    run_name = str(output_cfg.get("run_name", "vespuq"))
    layout = ensure_run_layout(output_dir / run_name)
    atomic_write_json(layout.run_dir / "vespuq_report.json", report)
    markdown = build_report_md(report)
    atomic_write_text(layout.run_dir / "vespuq_report.md", markdown)
    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"saved_vespuq_report: {layout.run_dir / 'vespuq_report.md'}")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ: calibration + trajectory risk screening.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
