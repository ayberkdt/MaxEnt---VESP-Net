"""Direct force-error ranking benchmark for VESP-UQ.

Evaluates the *core* VESP-UQ claim directly:

    Does the VESP-UQ force-risk score rank the surrogate's TRUE FORCE-MODEL error along a
    trajectory?

This is the right question for a force-risk / OOD layer -- unlike asking whether force-risk ranks
long-horizon *position* error (a separate diagnostic; position error is often not force-error
dominated). The true force error is read from held-out residual samples by nearest neighbour (no
leakage), or directly from surrogate/reference acceleration pairs if an external CSV supplies them.

    python scripts/run_force_error_benchmark.py --config configs/vespuq/vespuq_smoke.yaml

Outputs (under --out-dir, default outputs/iac):
    force_error_benchmark.md, force_error_benchmark.json, force_error_scores.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vesp.common.config import get_dtype, load_config
from vesp.uq.data import split_uq_samples
from vesp.uq.ensemble import generate_orbit_ensemble, nearest_neighbor_error_magnitude
from vesp.uq.experiment import _load_samples
from vesp.uq.io import load_trajectory_csv
from vesp.uq.plugin import VESPUQPlugin
from vesp.uq.scoring import aggregate_trajectory_error
from vesp.uq.selection import select_reruns


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


def force_error_benchmark(
    config: dict,
    *,
    scoring: str | None = None,
    rerun_fraction: float = 0.10,
    prepared=None,
) -> dict:
    """Run the force-error ranking benchmark; return a numbers dict (no file I/O)."""

    plugin, samples, train, held, dtype, seed = prepared or prepare(config)
    screen_cfg = config.get("uq", {}).get("screening", {})
    scoring = scoring or "supervisor_rel_p95"
    aggregator = str(screen_cfg.get("true_error_aggregator", "p95")).lower()

    # trajectory ensemble: external CSV if requested, else generated Keplerian orbits
    residuals = None
    if str(screen_cfg.get("trajectory_source", "generated")).lower() == "csv" and screen_cfg.get("trajectory_path"):
        ds = load_trajectory_csv(screen_cfg["trajectory_path"], dtype=dtype)
        trajectories = ds.trajectories
        residuals = ds.residual_accelerations
    else:
        ens = generate_orbit_ensemble(
            n_orbits=int(screen_cfg.get("n_orbits", 200)),
            n_points=int(screen_cfg.get("n_points", 48)),
            r_peri_range=tuple(screen_cfg.get("r_peri_range", (1.02, 1.30))),
            r_apo_range=tuple(screen_cfg.get("r_apo_range", (1.30, 1.60))),
            seed=seed,
            dtype=dtype,
        )
        trajectories = ens.trajectories

    scores = plugin.score_ensemble(trajectories, scoring=scoring)
    risk = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)

    # TRUE force error per trajectory (NOT position error): direct residual if available, else
    # nearest-neighbour read from held-out residual samples (no leakage).
    true_fe = torch.empty(len(trajectories), dtype=torch.float64)
    if residuals is not None:
        true_error_mode = "residual_csv"
        for i, res in enumerate(residuals):
            true_fe[i] = aggregate_trajectory_error(torch.linalg.norm(res.to(torch.float64), dim=-1), aggregator)
    else:
        true_error_mode = "nn_oracle_heldout"
        for i, traj in enumerate(trajectories):
            nn = nearest_neighbor_error_magnitude(traj.to(dtype), held.positions, held.error)
            true_fe[i] = aggregate_trajectory_error(nn.to(torch.float64), aggregator)

    rep = select_reruns(risk, rerun_fraction=rerun_fraction, true_error=true_fe)
    lift = (rep.capture_rate / rep.rerun_fraction) if rep.rerun_fraction else float("nan")
    numbers = {
        "benchmark": "force_error_ranking",
        "claim": "does VESP-UQ force-risk rank TRUE force-model error along trajectories?",
        "is_position_error_benchmark": False,
        "scoring": scoring,
        "true_force_error_aggregator": aggregator,
        "true_error_mode": true_error_mode,
        "n_trajectories": len(trajectories),
        "rerun_fraction": rerun_fraction,
        "spearman_force_risk_vs_true_force_error": rep.spearman_risk_vs_error,
        "capture_rate": rep.capture_rate,
        "precision": rep.precision,
        "lift_over_random": lift,
        "mean_true_force_error_flagged": rep.mean_error_flagged,
        "mean_true_force_error_accepted": rep.mean_error_accepted,
        "force_error_ratio_flagged_to_accepted": rep.error_ratio_flagged_to_accepted,
    }
    numbers["_scores"] = [
        {"trajectory_id": i, "force_risk": float(risk[i]), "true_force_error": float(true_fe[i]),
         "flagged": int(i in set(rep.flagged_indices))}
        for i in range(len(trajectories))
    ]
    return numbers


def _benchmark_md(n: dict) -> str:
    def f(x, s=".4f"):
        return "n/a" if x is None else format(float(x), s)

    return "\n".join([
        "# VESP-UQ Force-Error Ranking Benchmark",
        "",
        "**This is a FORCE-ERROR benchmark, not a position-error benchmark.** It asks whether the",
        "VESP-UQ force-risk score ranks the surrogate's true force-model error along a trajectory.",
        "",
        f"- scoring: `{n['scoring']}`  |  true force error: `{n['true_error_mode']}` "
        f"(aggregator `{n['true_force_error_aggregator']}`)",
        f"- trajectories: {n['n_trajectories']}  |  top fraction flagged: {n['rerun_fraction']:.0%}",
        "",
        f"- **Spearman(force-risk, true force error): {f(n['spearman_force_risk_vs_true_force_error'])}**",
        f"- capture rate: {f(n['capture_rate'])}  |  precision: {f(n['precision'])}  "
        f"|  lift over random: {f(n['lift_over_random'], '.2f')}x",
        f"- mean true force error flagged: {f(n['mean_true_force_error_flagged'], '.3e')}  vs  "
        f"accepted: {f(n['mean_true_force_error_accepted'], '.3e')}  "
        f"(ratio {f(n['force_error_ratio_flagged_to_accepted'], '.2f')}x)",
        "",
        "Interpretation: a positive Spearman / lift > 1 means the force-risk score concentrates the",
        "surrogate's true force-model error -- the core VESP-UQ value (force-risk / OOD detection).",
        "It does NOT imply prediction of long-horizon trajectory position error.",
        "",
    ]) + "\n"


def run_and_write(config: dict, *, out_dir: Path, scoring: str | None, rerun_fraction: float) -> dict:
    numbers = force_error_benchmark(config, scoring=scoring, rerun_fraction=rerun_fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = numbers.pop("_scores")
    (out_dir / "force_error_benchmark.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    (out_dir / "force_error_benchmark.md").write_text(_benchmark_md(numbers), encoding="utf-8")
    header = ["trajectory_id", "force_risk", "true_force_error", "flagged"]
    lines = [",".join(header)] + [",".join(str(r[h]) for h in header) for r in rows]
    (out_dir / "force_error_scores.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return numbers


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ direct force-error ranking benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/iac")
    parser.add_argument("--scoring", default=None, help="default supervisor_rel_p95")
    parser.add_argument("--rerun-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    numbers = run_and_write(
        load_config(args.config),
        out_dir=Path(args.out_dir),
        scoring=args.scoring,
        rerun_fraction=args.rerun_fraction,
    )
    print(_benchmark_md(numbers))
    print(f"saved_force_error_benchmark: {Path(args.out_dir) / 'force_error_benchmark.md'}")


if __name__ == "__main__":
    main()
