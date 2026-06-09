"""Simple baseline trajectory-risk selectors for force-error screening comparison.

Each function returns ONE scalar score per trajectory, where a HIGHER score always means HIGHER
risk. The target these scores are compared against is always trajectory-level true *force-model*
error (never position error) -- see :mod:`vesp.uq.benchmarking` and
``scripts/compare_risk_baselines.py``.

The baselines are intentionally cheap and dependency-free (torch only) so a VESP-UQ supervisor
score can be judged against trivial heuristics on the same target.
"""

from __future__ import annotations

import torch

_H_FLOOR = 1.0e-3


def _as_traj_list(trajectories) -> list[torch.Tensor]:
    trajs = list(trajectories)
    if not trajs:
        raise ValueError("trajectories is empty")
    for t in trajs:
        tt = torch.as_tensor(t)
        if tt.ndim != 2 or tt.shape[-1] != 3:
            raise ValueError("each trajectory must have shape (T, 3)")
    return trajs


def random_scores(n: int, seed: int = 0) -> torch.Tensor:
    """Deterministic uniform random risk scores in ``[0, 1)`` -- the chance-level reference."""

    if int(n) <= 0:
        raise ValueError("n must be positive")
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(int(n), generator=g, dtype=torch.float64)


def min_altitude_scores(trajectories) -> torch.Tensor:
    """Minimum-altitude heuristic: lower periapsis -> higher risk.

    ``score = 1 / max(min_radius - 1, h_floor)`` (monotonically larger as the lowest point of the
    trajectory approaches the surface). A strong, trivial baseline because force-model error
    typically grows toward low altitude.
    """

    trajs = _as_traj_list(trajectories)
    out = torch.empty(len(trajs), dtype=torch.float64)
    for i, traj in enumerate(trajs):
        r = torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1)
        out[i] = 1.0 / max(float(r.min()) - 1.0, _H_FLOOR)
    return out


def low_altitude_exposure_scores(
    trajectories, low_altitude_radius: float = 1.15, weights=None
) -> torch.Tensor:
    """Low-altitude exposure heuristic: the (weighted) fraction of points below ``low_altitude_radius``.

    ``0`` means the trajectory never dips below the threshold, ``1`` means it is entirely below it.
    ``weights`` (optional) is an iterable of per-trajectory weight vectors (e.g. ~dt) aligned with
    ``trajectories``; ``None`` weights all points uniformly.
    """

    trajs = _as_traj_list(trajectories)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(trajs):
            raise ValueError("weights must be None or one weight vector per trajectory")
    thr = float(low_altitude_radius)
    out = torch.empty(len(trajs), dtype=torch.float64)
    for i, traj in enumerate(trajs):
        r = torch.linalg.norm(torch.as_tensor(traj, dtype=torch.float64), dim=-1)
        below = (r <= thr).to(torch.float64)
        if weights is None or weights[i] is None:
            out[i] = float(below.mean())
        else:
            w = torch.as_tensor(weights[i], dtype=torch.float64).reshape(-1)
            if w.numel() != below.numel():
                raise ValueError("weight vector length must match the trajectory length")
            total = float(w.sum())
            if total <= 0.0:
                raise ValueError("weights must sum to a positive value")
            out[i] = float((below * (w / total)).sum())
    return out


def domain_support_scores(plugin, trajectories) -> torch.Tensor:
    """Domain-support-only risk: mean per-point out-of-support (OOD) score along each trajectory.

    Requires a fitted ``plugin`` exposing ``domain_support_score(positions)`` (raises a clear error
    otherwise). Higher = more of the trajectory lies outside the calibration support.
    """

    if not hasattr(plugin, "domain_support_score"):
        raise ValueError("plugin does not expose domain_support_score(...)")
    trajs = _as_traj_list(trajectories)
    out = torch.empty(len(trajs), dtype=torch.float64)
    for i, traj in enumerate(trajs):
        ds = plugin.domain_support_score(traj)  # per-point; raises if the plugin is not fitted
        out[i] = float(torch.as_tensor(ds, dtype=torch.float64).mean())
    return out


def vespuq_scores(plugin, trajectories, scoring: str) -> torch.Tensor:
    """VESP-UQ trajectory risk scores for a given ``scoring`` mode (one scalar per trajectory).

    ``scoring`` must be a supported VESP-UQ mode (e.g. ``mean`` for an uncertainty-only baseline,
    ``supervisor_rel_p95`` for the full supervisor); unsupported modes raise a clear ``ValueError``.
    """

    from vesp.uq.scoring import SCORING_FUNCTIONS

    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"unsupported scoring {scoring!r}; must be one of {SCORING_FUNCTIONS}")
    if not hasattr(plugin, "score_ensemble"):
        raise ValueError("plugin does not expose score_ensemble(...)")
    scores = plugin.score_ensemble(list(trajectories), scoring=scoring)
    return torch.tensor([s.risk_score for s in scores], dtype=torch.float64)
