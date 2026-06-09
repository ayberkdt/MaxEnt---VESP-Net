#!/usr/bin/env python3
"""N6 benchmark: online force-model correction -- accuracy delta vs per-RHS cost.

Builds a synthetic "world" whose truth force-error field is an equivalent-source field, fits VESP-UQ
on samples of it, and integrates three trajectories from one initial orbit under:

    surrogate : a_base(x)                       -- the surrogate (here it omits the residual)
    corrected : a_base(x) + K(x) @ sigma_mean   -- surrogate + VESP-UQ posterior-mean correction
    reference : a_base(x) + truth_residual(x)   -- the truth

and reports the position-error reduction (corrected vs surrogate, both against the reference) AND the
per-RHS evaluation cost (the corrected RHS evaluates the full equivalent-source field every call).

    python scripts/run_force_correction_benchmark.py --config configs/vespuq/vespuq_smoke.yaml

Outputs (under --out-dir, default outputs/correction), through the N1 artifact layer:
    force_correction_benchmark.json, force_correction_benchmark.md, force_correction_errors.csv

SCOPE / HONESTY: the posterior mean is the ridge point estimate, so this is a FORCE-MODEL
correction with no guaranteed long-horizon position-accuracy claim. The synthetic truth lies in the
equivalent-source span, so this is a best-case illustration of the mechanism, not evidence it
transfers to real residuals. Report measured numbers only. See
``benchmarks/online_force_correction.md`` and ``docs/VESP_UQ_LIMITATIONS.md``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vesp.common.config import get_dtype, load_config
from vesp.core.operators import build_acceleration_operator
from vesp.core.sources import make_shell_sources
from vesp.uq.correction import CorrectedForceField, integrate_trajectory
from vesp.uq.plugin import VESPUQPlugin

SCOPE_NOTE = (
    "Online FORCE-MODEL correction a_corrected = a_surrogate + posterior-mean force error. The "
    "posterior mean is the ridge point estimate, so this corrects the force model with NO guaranteed "
    "long-horizon position-accuracy claim. The synthetic truth lies in the equivalent-source span, so "
    "this is a best-case illustration of the mechanism, not evidence it transfers to real residuals."
)


def build_synthetic_world(config: dict, dtype: torch.dtype):
    """Synthetic truth force-error field + samples of it.

    Mirrors :func:`vesp.uq.data.make_synthetic_uq_samples` (interior truth sources -> analytic
    acceleration + noise) but *returns the truth sources and strengths* so the benchmark can build a
    continuous reference acceleration field, not just samples. Returns
    ``(truth_sources, sigma_truth, positions, error)``.
    """

    data = config.get("data", {})
    n = int(data.get("n", 512))
    n_truth = int(data.get("n_truth_sources", 24))
    noise_std = float(data.get("noise_std", 1.0e-4))
    truth_shell = float(data.get("truth_shell", 0.7))
    qlo, qhi = data.get("query_r_range", (1.03, 1.6))
    seed = int(config.get("seed", 0))

    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, dtype=dtype)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    radii = float(qlo) + (float(qhi) - float(qlo)) * torch.rand(n, generator=g, dtype=dtype)
    positions = dirs * radii.unsqueeze(-1)

    truth = make_shell_sources([truth_shell], n_truth, dtype=dtype)
    sigma_truth = 0.02 * torch.randn(truth.n_sources, generator=g, dtype=dtype)
    op = build_acceleration_operator(positions, truth, eps=0.0, sign=1.0)
    error = (op @ sigma_truth).reshape(3, n).transpose(0, 1).contiguous()
    if noise_std > 0.0:
        error = error + noise_std * torch.randn(n, 3, generator=g, dtype=dtype)
    return truth, sigma_truth, positions, error


def make_fields(truth_sources, sigma_truth, *, mu: float):
    """Return ``(a_base, a_surrogate, a_reference)`` acceleration fields over positions ``(..., 3)``.

    Each accepts a single position ``(3,)`` (from the integrator) or a batch ``(N, 3)`` (cost timing).
    The surrogate omits the residual (``a_surrogate = a_base``); the reference adds the truth field.
    """

    def a_base(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(-1, keepdim=True)
        return -float(mu) * x / (r2 * torch.sqrt(r2))

    def truth_residual(x: torch.Tensor) -> torch.Tensor:
        single = x.ndim == 1
        xb = x.unsqueeze(0) if single else x
        op = build_acceleration_operator(xb, truth_sources, eps=0.0, sign=1.0)
        out = (op @ sigma_truth.to(dtype=xb.dtype, device=xb.device)).reshape(3, xb.shape[0]).transpose(0, 1)
        return out[0] if single else out

    def a_surrogate(x: torch.Tensor) -> torch.Tensor:
        return a_base(x)

    def a_reference(x: torch.Tensor) -> torch.Tensor:
        return a_base(x) + truth_residual(x)

    return a_base, a_surrogate, a_reference


def resolve_orbit_params(config: dict, args=None) -> dict:
    """Resolve the orbit/integration params from ``uq.propagation`` (config), overridden by CLI."""

    prop = dict(config.get("uq", {}).get("propagation", {}) or {})

    def pick(name, default):
        cli = getattr(args, name, None) if args is not None else None
        return cli if cli is not None else prop.get(name, default)

    return {
        "mu": float(pick("mu", 1.0)),
        "r_initial": float(pick("r_initial", 1.057)),
        "duration": float(pick("duration", 14.0)),
        "dt": float(pick("dt", 0.05)),
        "output_dt": float(pick("output_dt", 0.5)),
    }


def run_force_correction_benchmark(config: dict, params: dict, *, cost_reps: int = 200) -> dict:
    """Fit on a synthetic world, integrate surrogate/corrected/reference orbits, measure accuracy+cost."""

    dtype = get_dtype(config)
    device = torch.device(config.get("device", "cpu"))
    mu = params["mu"]

    truth_sources, sigma_truth, positions, error = build_synthetic_world(config, dtype)
    plugin = VESPUQPlugin.from_config(config)
    plugin.fit_error(positions.to(device), error.to(device))

    a_base, a_surrogate, a_reference = make_fields(truth_sources.to(device), sigma_truth.to(device), mu=mu)
    corrected = CorrectedForceField(plugin, surrogate_accel_fn=a_surrogate, device=device, dtype=dtype)

    v_circular = float(np.sqrt(mu / params["r_initial"]))
    y0 = np.array([params["r_initial"], 0.0, 0.0, 0.0, v_circular, 0.0], dtype=np.float64)
    kw = dict(dt=params["dt"], duration=params["duration"], output_dt=params["output_dt"], dtype=dtype, device=device)
    times, ref_states = integrate_trajectory(a_reference, y0, **kw)
    _, sur_states = integrate_trajectory(a_surrogate, y0, **kw)
    _, cor_states = integrate_trajectory(corrected, y0, **kw)

    sur_err = np.linalg.norm(sur_states[:, :3] - ref_states[:, :3], axis=1)
    cor_err = np.linalg.norm(cor_states[:, :3] - ref_states[:, :3], axis=1)

    # Per-RHS cost: time batched evals of the surrogate vs corrected field on the reference path.
    pts = torch.tensor(ref_states[:, :3], dtype=dtype, device=device)
    a_surrogate(pts), corrected(pts)  # warmup

    def per_call_seconds(fn) -> float:
        t0 = time.perf_counter()
        for _ in range(cost_reps):
            fn(pts)
        return (time.perf_counter() - t0) / cost_reps

    t_sur = per_call_seconds(a_surrogate)
    t_cor = per_call_seconds(corrected)

    final_sur, final_cor = float(sur_err[-1]), float(cor_err[-1])
    return {
        "tool": "run_force_correction_benchmark",
        "method": "online_force_model_correction",
        "error_basis": "force_model_error_posterior_mean",
        "scope_note": SCOPE_NOTE,
        "config_path": config.get("_config_path"),
        "mu": mu,
        "initial_state": y0.tolist(),
        "duration": params["duration"],
        "dt": params["dt"],
        "output_dt": params["output_dt"],
        "n_sources_fit": int(plugin.sources.n_sources),
        "n_truth_sources": int(truth_sources.n_sources),
        "fit": plugin.fit_info,
        "n_steps": int(times.shape[0]),
        "times": times.tolist(),
        "surrogate_position_error": sur_err.tolist(),
        "corrected_position_error": cor_err.tolist(),
        "summary": {
            "final_surrogate_position_error": final_sur,
            "final_corrected_position_error": final_cor,
            "max_surrogate_position_error": float(np.max(sur_err)),
            "max_corrected_position_error": float(np.max(cor_err)),
            # < 1 means the correction reduced the final position error; > 1 means it made it worse
            "final_error_reduction_ratio": (final_cor / final_sur) if final_sur > 0 else float("nan"),
            "final_improvement_factor": (final_sur / final_cor) if final_cor > 0 else float("inf"),
        },
        "cost": {
            "per_rhs_surrogate_us": t_sur * 1.0e6,
            "per_rhs_corrected_us": t_cor * 1.0e6,
            "per_rhs_cost_ratio": (t_cor / t_sur) if t_sur > 0 else float("inf"),
            "cost_eval_points": int(pts.shape[0]),
            "cost_reps": int(cost_reps),
        },
    }


def _benchmark_md(r: dict) -> str:
    def f(x, fmt=".4e"):
        return "n/a" if x is None else format(float(x), fmt)

    s, c = r["summary"], r["cost"]
    init = [round(float(x), 6) for x in r["initial_state"]]
    return "\n".join([
        "# VESP-UQ Online Force-Model Correction Benchmark",
        "",
        "**EXPLORATORY, force-model correction (not validated).** " + r["scope_note"],
        "",
        f"- config: `{r.get('config_path')}`  |  fit sources: {r['n_sources_fit']}  |  "
        f"truth sources: {r['n_truth_sources']}  |  mu: {f(r['mu'], '.3f')}",
        f"- initial state [r, v]: {init}  |  duration: {f(r['duration'], '.3f')}  |  "
        f"dt: {f(r['dt'], '.3f')}  |  steps: {r['n_steps']}",
        "",
        "## Accuracy (position error vs the reference trajectory)",
        "",
        f"- final surrogate error: **{f(s['final_surrogate_position_error'])}** body radii  "
        f"(max {f(s['max_surrogate_position_error'])})",
        f"- final corrected error: **{f(s['final_corrected_position_error'])}** body radii  "
        f"(max {f(s['max_corrected_position_error'])})",
        f"- final error reduction ratio (corrected/surrogate): **{f(s['final_error_reduction_ratio'], '.4f')}**  "
        f"(improvement factor {f(s['final_improvement_factor'], '.2f')}x)",
        "",
        "## Cost (per-RHS acceleration evaluation)",
        "",
        f"- surrogate RHS: {f(c['per_rhs_surrogate_us'], '.2f')} us/call  |  "
        f"corrected RHS: {f(c['per_rhs_corrected_us'], '.2f')} us/call  "
        f"(**{f(c['per_rhs_cost_ratio'], '.2f')}x** cost, {c['cost_eval_points']} pts x {c['cost_reps']} reps)",
        "",
        "Interpretation: the correction adds the VESP-UQ posterior-mean force-error field to the "
        "surrogate, reducing the integrated position error toward the reference -- but it evaluates "
        "the full equivalent-source field every RHS call, which costs more than the bare surrogate. "
        "The posterior mean is the ridge estimate, so this is a FORCE-MODEL correction: it improves "
        "accuracy only insofar as the force error is captured by the equivalent-source mean, with no "
        "guaranteed long-horizon position-accuracy claim. The synthetic truth lies in the "
        "equivalent-source span (best case); real residuals need not.",
        "",
    ]) + "\n"


def _errors_csv(r: dict) -> str:
    header = ["time", "surrogate_position_error", "corrected_position_error"]
    rows = [
        ",".join(str(v) for v in [t, se, ce])
        for t, se, ce in zip(
            r["times"], r["surrogate_position_error"], r["corrected_position_error"], strict=True
        )
    ]
    return "\n".join([",".join(header), *rows]) + "\n"


def run_and_write(config: dict, params: dict, *, out_dir: Path, cost_reps: int = 200) -> dict:
    from vesp.uq.io.run_artifacts import write_run_artifacts

    result = run_force_correction_benchmark(config, params, cost_reps=cost_reps)
    markdown = _benchmark_md(result)
    write_run_artifacts(
        out_dir,
        tool="run_force_correction_benchmark",
        config=config,
        json_files={"force_correction_benchmark.json": result},
        text_files={"force_correction_benchmark.md": markdown, "force_correction_errors.csv": _errors_csv(result)},
    )
    result["_markdown"] = markdown
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ online force-model correction benchmark (accuracy vs cost).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/correction")
    parser.add_argument("--r-initial", type=float, default=None, dest="r_initial")
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--output-dt", type=float, default=None, dest="output_dt")
    parser.add_argument("--cost-reps", type=int, default=200, help="repetitions for per-RHS cost timing")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("_config_path", args.config)
    params = resolve_orbit_params(config, args)
    result = run_and_write(config, params, out_dir=Path(args.out_dir), cost_reps=args.cost_reps)
    print(result["_markdown"].encode("ascii", "replace").decode("ascii"))
    print(f"saved_force_correction_benchmark: {Path(args.out_dir) / 'force_correction_benchmark.md'}")


if __name__ == "__main__":
    main()
