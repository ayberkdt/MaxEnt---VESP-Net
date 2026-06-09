#!/usr/bin/env python3
"""Driver: exploratory linearized (STM) force-error covariance propagation for VESP-UQ.

Parity with ``scripts/run_propagation.py`` (the Monte Carlo orbit-dispersion sampler), but
deterministic and sampling-free: it propagates the *variational* sensitivity of a nominal
trajectory to the equivalent-source strengths and maps the fitted force-error posterior
covariance into a ``6x6`` state covariance (see :mod:`vesp.uq.linear_propagation`).

    python scripts/run_linear_propagation.py --config configs/vespuq/vespuq_smoke.yaml

Outputs (under --out-dir, default outputs/propagation), written through the N1 artifact layer
(injected ``_provenance`` + ``run_manifest.json`` with config snapshot, seed, and SHA-256
checksums):

    linear_propagation.json   -- times, nominal states, 6x6 covariances, position/velocity sigma
    linear_propagation.md     -- human-readable summary + honest scope
    linear_propagation_states.csv -- per-step [time, r, v, position_sigma, velocity_sigma]

Propagation parameters come from the optional ``uq.propagation`` config block, overridable by CLI
flags (``--r-initial --mu --duration --dt --output-dt``).

SCOPE / HONESTY: this is an EXPLORATORY diagnostic, NOT validated operational orbit
determination. It maps the local FORCE-MODEL error posterior into a linearized state covariance;
it does not model measurement processing or realistic process noise, and the force-risk score is
NOT a position-error predictor. See ``docs/VESP_UQ_LIMITATIONS.md`` and
``benchmarks/covariance_propagation.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vesp.common.config import get_dtype, load_config
from vesp.uq.data import split_uq_samples
from vesp.uq.experiment import _load_samples
from vesp.uq.io.run_artifacts import write_run_artifacts
from vesp.uq.linear_propagation import LinearForceErrorCovariancePropagator
from vesp.uq.plugin import VESPUQPlugin

SCOPE_NOTE = (
    "Exploratory linearized (STM) covariance of the VESP-UQ FORCE-MODEL error posterior. It is NOT "
    "validated operational orbit determination, NOT a position-error predictor, and NOT a realistic "
    "operational state covariance (no measurement processing or process noise). Cross-check against "
    "the Monte Carlo sampler; large uncertainty or long horizons break the linearization."
)


def prepare(config: dict):
    """Fit VESP-UQ on the train split; return (plugin, dtype, seed)."""

    dtype = get_dtype(config)
    samples = _load_samples(config, dtype)
    seed = int(config.get("seed", 0))
    train, _held = split_uq_samples(
        samples, train_fraction=float(config.get("data", {}).get("train_fraction", 0.7)), seed=seed
    )
    plugin = VESPUQPlugin.from_config(config)
    plugin.fit(train.positions, train.surrogate, train.reference)
    return plugin, dtype, seed


def resolve_propagation_params(config: dict, args=None) -> dict:
    """Resolve propagation parameters from ``uq.propagation`` (config), overridden by CLI flags."""

    prop = dict(config.get("uq", {}).get("propagation", {}) or {})

    def pick(name, default):
        cli = getattr(args, name, None) if args is not None else None
        return cli if cli is not None else prop.get(name, default)

    return {
        "mu": float(pick("mu", 1.0)),
        "r_initial": float(pick("r_initial", 1.057)),
        "duration": float(pick("duration", 14.0)),
        "dt": float(pick("dt", 0.1)),
        "output_dt": float(pick("output_dt", 0.5)),
    }


def run_linear_propagation(config: dict, params: dict, *, prepared=None) -> dict:
    """Fit, build a low circular-orbit nominal state, and propagate the linearized ``6x6`` covariance."""

    plugin, _dtype, _seed = prepared or prepare(config)
    mu = params["mu"]
    r0 = params["r_initial"]
    v_circular = float(np.sqrt(mu / r0))  # circular-orbit speed for the point-mass base field
    y0 = np.array([r0, 0.0, 0.0, 0.0, v_circular, 0.0], dtype=np.float64)

    propagator = LinearForceErrorCovariancePropagator(
        plugin, dt_s=params["dt"], mu=mu, device=plugin.device, dtype=plugin.dtype
    )
    res = propagator.propagate(y0, duration_s=params["duration"], output_dt_s=params["output_dt"])

    return {
        "tool": "run_linear_propagation",
        "method": "linearized_stm_force_error_covariance",
        "error_basis": "force_model_error_posterior",
        "scope_note": SCOPE_NOTE,
        "config_path": config.get("_config_path"),
        "mu": mu,
        "initial_state": y0.tolist(),
        "duration": params["duration"],
        "dt": params["dt"],
        "output_dt": params["output_dt"],
        "n_sources": int(plugin.sources.n_sources),
        "fit": plugin.fit_info,
        "n_steps": int(res.times.shape[0]),
        "times": res.times.tolist(),
        "nominal_states": res.states.tolist(),
        "covariances_6x6": res.covariances.tolist(),
        "position_sigma": res.position_sigma.tolist(),
        "velocity_sigma": res.velocity_sigma.tolist(),
        "summary": {
            "final_position_sigma": float(res.position_sigma[-1]),
            "final_velocity_sigma": float(res.velocity_sigma[-1]),
            "max_position_sigma": float(np.max(res.position_sigma)),
            "max_velocity_sigma": float(np.max(res.velocity_sigma)),
        },
    }


def _propagation_md(result: dict) -> str:
    def f(x, fmt=".4e"):
        return "n/a" if x is None else format(float(x), fmt)

    s = result["summary"]
    init = [round(float(x), 6) for x in result["initial_state"]]
    return "\n".join([
        "# VESP-UQ Linearized (STM) Force-Error Covariance Propagation",
        "",
        "**EXPLORATORY diagnostic, not validated.** " + result["scope_note"],
        "",
        f"- config: `{result.get('config_path')}`",
        f"- method: `{result['method']}`  |  sources: {result['n_sources']}  |  mu: {f(result['mu'], '.3f')}",
        f"- initial state [r, v]: {init}",
        f"- duration: {f(result['duration'], '.3f')}  |  integration dt: {f(result['dt'], '.3f')}  |  "
        f"output dt: {f(result['output_dt'], '.3f')}  |  steps: {result['n_steps']}",
        "",
        "## 1-sigma state dispersion implied by the force-error posterior",
        "",
        f"- final position sigma: **{f(s['final_position_sigma'])}** body radii  "
        f"(max {f(s['max_position_sigma'])})",
        f"- final velocity sigma: **{f(s['final_velocity_sigma'])}** (body radii / time unit)",
        "",
        "Interpretation: the covariance starts at zero (`J(0) = 0`) and accumulates as the fitted "
        "force-error posterior is integrated along the nominal trajectory. This is the linearization "
        "of the Monte Carlo sampler's static force-error field model, so it matches the MC sample "
        "covariance in the small-perturbation regime (see `benchmarks/covariance_propagation.md`) "
        "without sampling noise. It is a FORCE-MODEL-error diagnostic only -- it does not predict "
        "long-horizon position error and is not an operational orbit covariance.",
        "",
    ]) + "\n"


def _states_csv(result: dict) -> str:
    header = ["time", "x", "y", "z", "vx", "vy", "vz", "position_sigma", "velocity_sigma"]
    rows = [
        ",".join(str(v) for v in [t, *state, pos_sigma, vel_sigma])
        for t, state, pos_sigma, vel_sigma in zip(
            result["times"],
            result["nominal_states"],
            result["position_sigma"],
            result["velocity_sigma"],
            strict=True,
        )
    ]
    return "\n".join([",".join(header), *rows]) + "\n"


def run_and_write(config: dict, params: dict, *, out_dir: Path, prepared=None) -> dict:
    result = run_linear_propagation(config, params, prepared=prepared)
    markdown = _propagation_md(result)
    csv_text = _states_csv(result)
    write_run_artifacts(
        out_dir,
        tool="run_linear_propagation",
        config=config,
        json_files={"linear_propagation.json": result},
        text_files={"linear_propagation.md": markdown, "linear_propagation_states.csv": csv_text},
    )
    result["_markdown"] = markdown
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="VESP-UQ exploratory linearized (STM) force-error covariance propagation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/propagation")
    parser.add_argument("--r-initial", type=float, default=None, dest="r_initial",
                        help="initial circular-orbit radius in body radii (default 1.057)")
    parser.add_argument("--mu", type=float, default=None, help="gravitational parameter (default 1.0)")
    parser.add_argument("--duration", type=float, default=None, help="total propagation time (time units)")
    parser.add_argument("--dt", type=float, default=None, help="RK4 integration step (time units)")
    parser.add_argument("--output-dt", type=float, default=None, dest="output_dt",
                        help="snapshot interval (time units)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("_config_path", args.config)
    params = resolve_propagation_params(config, args)
    result = run_and_write(config, params, out_dir=Path(args.out_dir))
    print(result["_markdown"].encode("ascii", "replace").decode("ascii"))
    print(f"saved_linear_propagation: {Path(args.out_dir) / 'linear_propagation.md'}")


if __name__ == "__main__":
    main()
