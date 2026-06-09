"""Compare VESP-UQ trajectory force-risk scores against simple baseline selectors.

Target: trajectory-level true FORCE-MODEL error (never position error). Baselines:

    random | min_altitude | low_altitude_exposure | domain_support (if enabled)
    | uncertainty_only (mean sigma) | supervisor (supervisor_rel_p95)

    python scripts/compare_risk_baselines.py --config configs/vespuq/vespuq_smoke.yaml

Outputs (under --out-dir, default outputs/baselines):
    baseline_comparison.json, baseline_comparison.csv, baseline_comparison.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vesp.common.config import get_dtype, load_config
from vesp.uq.baselines import (
    domain_support_scores,
    low_altitude_exposure_scores,
    min_altitude_scores,
    random_scores,
    vespuq_scores,
)
from vesp.uq.benchmarking import METRIC_KEYS, _best_by, compare_baselines
from vesp.uq.data import split_uq_samples
from vesp.uq.ensemble import nearest_neighbor_error_magnitude
from vesp.uq.experiment import _build_trajectories, _load_samples
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.scoring import aggregate_trajectory_error

# Baseline scoring modes used for the two VESP-UQ entries.
_UNCERTAINTY_SCORING = "mean"  # mean predictive sigma -- uncertainty-only (no bias / altitude)
_SUPERVISOR_SCORING = "supervisor_rel_p95"  # full supervisor (expected error * altitude * domain)


def prepare(config: dict):
    """Fit VESP-UQ on the train split; return (plugin, samples, train, held, dtype, seed)."""

    dtype = get_dtype(config)
    samples = _load_samples(config, dtype)
    seed = int(config.get("seed", 0))
    train, held = split_uq_samples(
        samples, train_fraction=float(config.get("data", {}).get("train_fraction", 0.7)), seed=seed
    )
    plugin = VESPUQPlugin.from_config(config)
    plugin.fit(train.positions, train.surrogate, train.reference)
    return plugin, samples, train, held, dtype, seed


def _true_force_error(trajectories, *, residuals, held, aggregator, dtype):
    """Trajectory-level true FORCE error: direct residual pairs if present, else held-out NN oracle."""

    true_error = torch.empty(len(trajectories), dtype=torch.float64)
    if residuals is not None:
        for i, res in enumerate(residuals):
            mag = torch.linalg.norm(torch.as_tensor(res, dtype=torch.float64), dim=-1)
            true_error[i] = aggregate_trajectory_error(mag, aggregator)
        return true_error, "residual_csv"
    for i, traj in enumerate(trajectories):
        nn = nearest_neighbor_error_magnitude(traj.to(dtype), held.positions, held.error)
        true_error[i] = aggregate_trajectory_error(nn.to(torch.float64), aggregator)
    return true_error, "nn_oracle_heldout"


def baseline_scores_for(config: dict, plugin, trajectories, *, seed: int):
    """Assemble the baseline -> per-trajectory-score mapping for a fitted plugin + trajectories."""

    low_alt = float(config.get("uq", {}).get("risk", {}).get("low_altitude_radius", 1.15))
    n = len(trajectories)
    scores = {
        "random": random_scores(n, seed=seed),
        "min_altitude": min_altitude_scores(trajectories),
        "low_altitude_exposure": low_altitude_exposure_scores(trajectories, low_altitude_radius=low_alt),
        "uncertainty_only": vespuq_scores(plugin, trajectories, _UNCERTAINTY_SCORING),
        "supervisor": vespuq_scores(plugin, trajectories, _SUPERVISOR_SCORING),
    }
    if getattr(plugin, "domain_support", False):
        scores["domain_support"] = domain_support_scores(plugin, trajectories)
    return scores


def run_baseline_comparison(config: dict, *, rerun_fraction: float = 0.10, prepared=None) -> dict:
    """Run the full baseline comparison; return a payload dict (no file I/O)."""

    plugin, samples, train, held, dtype, seed = prepared or prepare(config)
    screen_cfg = config.get("uq", {}).get("screening", {})
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()

    traj_info = _build_trajectories(screen_cfg, seed=seed, dtype=dtype)
    trajectories = traj_info["trajectories"]
    true_error, te_source = _true_force_error(
        trajectories, residuals=traj_info["residuals"], held=held, aggregator=aggregator, dtype=dtype
    )

    scores = baseline_scores_for(config, plugin, trajectories, seed=seed)
    results = compare_baselines(scores, true_error, rerun_fraction=rerun_fraction)
    return {
        "config_dataset": str(config.get("data", {}).get("path") or samples.metadata.get("mode", "synthetic")),
        "n_trajectories": len(trajectories),
        "trajectory_source": traj_info["source"],
        "true_force_error_source": te_source,
        "true_force_error_aggregator": aggregator,
        "rerun_fraction": rerun_fraction,
        "uncertainty_scoring": _UNCERTAINTY_SCORING,
        "supervisor_scoring": _SUPERVISOR_SCORING,
        "baselines": results,
        "best_by_spearman": _best_by(results, "spearman"),
        "best_by_lift": _best_by(results, "lift_over_random"),
    }


def _fmt(x, spec=".4f"):
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _comparison_md(p: dict) -> str:
    results = p["baselines"]
    cols = ["spearman", "capture_rate", "precision", "lift_over_random",
            "mean_true_error_flagged", "mean_true_error_accepted", "force_error_ratio_flagged_to_accepted"]
    short = {"spearman": "spearman", "capture_rate": "capture", "precision": "precision",
             "lift_over_random": "lift", "mean_true_error_flagged": "err_flag",
             "mean_true_error_accepted": "err_acc", "force_error_ratio_flagged_to_accepted": "ratio"}
    lines = [
        "# VESP-UQ Baseline Comparison (trajectory force-risk screening)",
        "",
        "Target: trajectory-level true **force-model** error (NOT position error). Each selector",
        "flags the top trajectories; higher score = higher risk.",
        "",
        f"- dataset: `{p['config_dataset']}`  |  trajectories: {p['n_trajectories']} "
        f"({p['trajectory_source']})",
        f"- true force error: `{p['true_force_error_source']}` "
        f"(aggregator `{p['true_force_error_aggregator']}`)  |  rerun fraction: {p['rerun_fraction']:.0%}",
        f"- uncertainty_only scoring = `{p['uncertainty_scoring']}`  |  supervisor scoring = `{p['supervisor_scoring']}`",
        "",
        "| baseline | " + " | ".join(short[c] for c in cols) + " |",
        "| --- | " + " | ".join("---:" for _ in cols) + " |",
    ]
    for name, m in results.items():
        row = " | ".join(
            _fmt(m.get(c), ".3e" if c.startswith("mean_true") else (".2f" if c in ("lift_over_random", "force_error_ratio_flagged_to_accepted") else ".4f"))
            for c in cols
        )
        lines.append(f"| `{name}` | {row} |")
    lines += [
        "",
        f"- best by Spearman: **{p['best_by_spearman']}**",
        f"- best by lift over random: **{p['best_by_lift']}**",
        "",
        "Interpretation: a higher Spearman / lift means the selector concentrates the surrogate's",
        "true force-model error better. `min_altitude` and `low_altitude_exposure` are strong",
        "trivial baselines because force error usually grows toward low altitude; VESP-UQ adds value",
        "when its score (especially `supervisor`, which folds in expected bias and OOD risk) beats",
        "them, and adds none if it does not. This is a force-risk ranking comparison only -- it says",
        "nothing about long-horizon trajectory position error.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, out_dir: Path, *, config: dict | None = None) -> None:
    header = ["baseline", *METRIC_KEYS]
    rows = [header]
    for name, m in payload["baselines"].items():
        rows.append([name, *[m.get(k) for k in METRIC_KEYS]])
    csv_text = "\n".join(",".join("" if v is None else str(v) for v in row) for row in rows) + "\n"
    write_run_artifacts(
        out_dir,
        tool="compare_risk_baselines",
        config=config,
        json_files={"baseline_comparison.json": payload},
        text_files={"baseline_comparison.csv": csv_text, "baseline_comparison.md": _comparison_md(payload)},
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare VESP-UQ force-risk scores against simple baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/baselines")
    parser.add_argument("--rerun-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("_config_path", args.config)
    payload = run_baseline_comparison(config, rerun_fraction=args.rerun_fraction)
    out_dir = Path(args.out_dir)
    write_outputs(payload, out_dir, config=config)
    print(_comparison_md(payload))
    print(f"saved_baseline_comparison: {out_dir / 'baseline_comparison.md'}")


if __name__ == "__main__":
    main()
