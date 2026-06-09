"""Compare trajectory-risk scores against trajectory-level true FORCE-MODEL error.

These utilities evaluate any per-trajectory risk score (a VESP-UQ supervisor score or a trivial
baseline from :mod:`vesp.uq.baselines`) against the same target: trajectory-level true
*force-model* error. The target is never position error.

Selection reuses :func:`vesp.uq.selection.select_reruns` (top-k fraction + capture/precision/
Spearman validation) so there is a single source of truth for the selection logic.
"""

from __future__ import annotations

import math

import torch

from vesp.uq.selection import select_reruns

# Per-baseline metric keys (stable column order for CSV / report tables).
METRIC_KEYS = (
    "n_trajectories",
    "rerun_fraction",
    "n_flagged",
    "spearman",
    "capture_rate",
    "precision",
    "lift_over_random",
    "mean_true_error_flagged",
    "mean_true_error_accepted",
    "force_error_ratio_flagged_to_accepted",
)


def evaluate_score_against_true_error(scores, true_error, rerun_fraction: float = 0.10) -> dict:
    """Evaluate one risk score against trajectory-level true force error.

    Flags the top ``rerun_fraction`` of trajectories by ``scores`` (exact top-k, tie-robust) and
    reports the Spearman correlation, capture rate / precision against the truly-high-error
    trajectories, lift over a random screen, and the flagged-vs-accepted true-force-error split.
    Safe for small arrays and ties (constant scores yield a ``nan`` Spearman, never a crash).
    """

    s = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    e = torch.as_tensor(true_error, dtype=torch.float64).reshape(-1)
    if s.numel() == 0:
        raise ValueError("scores is empty")
    if s.numel() != e.numel():
        raise ValueError(f"scores ({s.numel()}) and true_error ({e.numel()}) must have equal length")

    report = select_reruns(s, rerun_fraction=float(rerun_fraction), true_error=e)
    cap = report.capture_rate
    rf = report.rerun_fraction
    lift = (cap / rf) if (cap is not None and rf and not math.isnan(float(cap))) else float("nan")
    return {
        "n_trajectories": int(s.numel()),
        "rerun_fraction": rf,
        "n_flagged": report.n_flagged,
        "spearman": report.spearman_risk_vs_error,
        "capture_rate": report.capture_rate,
        "precision": report.precision,
        "lift_over_random": lift,
        "mean_true_error_flagged": report.mean_error_flagged,
        "mean_true_error_accepted": report.mean_error_accepted,
        "force_error_ratio_flagged_to_accepted": report.error_ratio_flagged_to_accepted,
    }


def compare_baselines(baseline_scores: dict, true_error, rerun_fraction: float = 0.10) -> dict:
    """Evaluate every named baseline score against the same true-force-error target.

    ``baseline_scores`` maps a baseline name to its per-trajectory score tensor. Returns a dict
    mapping each name to its metrics from :func:`evaluate_score_against_true_error`.
    """

    if not baseline_scores:
        raise ValueError("baseline_scores is empty")
    return {
        name: evaluate_score_against_true_error(scores, true_error, rerun_fraction=rerun_fraction)
        for name, scores in baseline_scores.items()
    }


def _best_by(results: dict, key: str) -> str | None:
    """Name of the baseline with the largest finite value of ``key`` (``None`` if all nan)."""

    best, best_val = None, -math.inf
    for name, m in results.items():
        v = m.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        if v > best_val:
            best, best_val = name, v
    return best
