"""Trajectory-level risk scoring and selective high-fidelity rerun logic (Phases 3-4).

VESP-UQ scores a *whole trajectory* by aggregating the per-position predictive uncertainty
that :class:`~vesp.uq.plugin.VESPUQPlugin` produces along it, then flags the riskiest
trajectories for recomputation with a higher-fidelity force model. The point is operational:
run a cheap surrogate Monte Carlo, score every trajectory here, and rerun only the small
flagged subset -- preserving most of the speed advantage while removing blind trust in the
surrogate where it is least reliable (low altitude / ill-conditioned / out-of-support regimes).

Per-point risk comes in three families:

- the original ``sigma`` (predictive std) modes, kept verbatim for backward compatibility;
- *relative* expected-error / supervisor modes, normalized per trajectory -- good for RANKING
  one ensemble (which orbits to prioritize), not for cross-trajectory absolute thresholds;
- *absolute* expected-error / supervisor modes, normalized by a fixed altitude reference -- so a
  single physical risk budget means the same thing for every trajectory (zero-alarm screening).

``expected_error = sqrt(bias^2 + sigma^2)`` underpins both the relative and absolute supervisor
modes. Nothing here evaluates a gravity model; it consumes per-point arrays, so it is fully
surrogate-agnostic and cheap.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

# Scoring functions that turn a per-position profile into one trajectory number.
# Legacy sigma-only modes:
_SIGMA_MODES = ("max", "mean", "low_alt_integral", "time_above", "combined")
# Expected-error modes. `expected`/`expected_p95` are aliases of the absolute variants (no
# altitude weighting); `supervisor`/`supervisor_p95` are aliases of the RELATIVE variants
# (per-trajectory altitude normalization, for ranking).
_EXPECTED_ONLY_MODES = (
    "expected",
    "expected_abs",
    "expected_p95",
    "expected_abs_p95",
    "expected_low_alt",
)
_SUPERVISOR_MODES = (
    "supervisor",
    "supervisor_rel",
    "supervisor_p95",
    "supervisor_rel_p95",
    "supervisor_abs",
    "supervisor_abs_p95",
)
SCORING_FUNCTIONS = _SIGMA_MODES + _EXPECTED_ONLY_MODES + _SUPERVISOR_MODES

# Modes that need a per-point ``expected_error`` profile (and so cannot run on sigma alone).
_EXPECTED_MODES = frozenset(_EXPECTED_ONLY_MODES + _SUPERVISOR_MODES)

# Aggregators for collapsing a per-point true-error profile into one trajectory scalar.
TRUE_ERROR_AGGREGATORS = ("max", "mean", "p95")

# Fraction-mode selection policies for select_reruns.
FRACTION_POLICIES = ("topk", "quantile")

# ---- scoring-mode classification (relative ranking vs absolute physical-budget scale) ----
# RELATIVE modes normalize altitude per trajectory (good for ranking one ensemble, NOT for
# cross-trajectory absolute thresholds). ABSOLUTE / absolute-like modes are on a fixed
# expected-force-error scale, so a single physical budget means the same for every trajectory.
_RELATIVE_SCORINGS = frozenset(
    {"supervisor", "supervisor_rel", "supervisor_p95", "supervisor_rel_p95"}
)
_ABSOLUTE_SCORINGS = frozenset(
    {
        "expected",
        "expected_abs",
        "expected_p95",
        "expected_abs_p95",
        "expected_low_alt",
        "supervisor_abs",
        "supervisor_abs_p95",
    }
)
_EXPECTED_ONLY_SCORINGS = frozenset(_EXPECTED_ONLY_MODES)
# Canonical names for the backward-compatible aliases.
_CANONICAL_ALIASES = {
    "expected": "expected_abs",
    "expected_p95": "expected_abs_p95",
    "supervisor": "supervisor_rel",
    "supervisor_p95": "supervisor_rel_p95",
}


def _validate_scoring(scoring: str) -> str:
    """Return ``scoring`` if it is a known mode, else raise a clear ``ValueError``."""

    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"unknown scoring {scoring!r}; must be one of {SCORING_FUNCTIONS}")
    return scoring


def canonical_scoring_name(scoring: str) -> str:
    """Map a (possibly aliased) scoring name to its canonical name.

    ``expected -> expected_abs``, ``expected_p95 -> expected_abs_p95``,
    ``supervisor -> supervisor_rel``, ``supervisor_p95 -> supervisor_rel_p95``; all other known
    names map to themselves. Unknown names raise ``ValueError``.
    """

    return _CANONICAL_ALIASES.get(_validate_scoring(scoring), scoring)


def is_relative_scoring(scoring: str) -> bool:
    """True for per-trajectory-normalized supervisor modes (ranking only, not absolute budgets)."""

    return _validate_scoring(scoring) in _RELATIVE_SCORINGS


def is_absolute_scoring(scoring: str) -> bool:
    """True for absolute / absolute-like expected-force-error modes (safe for physical budgets)."""

    return _validate_scoring(scoring) in _ABSOLUTE_SCORINGS


def is_expected_only_scoring(scoring: str) -> bool:
    """True for the pure expected-error modes (no altitude weighting at all)."""

    return _validate_scoring(scoring) in _EXPECTED_ONLY_SCORINGS


@dataclass
class TrajectoryScore:
    """Aggregated risk summary for a single trajectory's output points.

    The ``*_sigma`` / ``*_altitude_risk`` fields are the legacy sigma-based aggregations. The
    ``*_expected_error`` / ``*_point_risk*`` / ``*_domain_risk`` fields are the supervisor
    metrics; they are ``nan`` when the relevant per-point profile was not supplied (e.g. calling
    :func:`score_sigma_profile` with sigma only, or with domain support disabled).

    ``mean_point_risk`` / ``p95_point_risk`` are the RELATIVE supervisor risk (per-trajectory
    altitude normalization -- for ranking). ``mean_point_risk_abs`` / ``p95_point_risk_abs`` are
    the ABSOLUTE supervisor risk (fixed altitude reference -- for cross-trajectory thresholds).
    """

    n_points: int
    max_sigma: float
    mean_sigma: float
    low_altitude_sigma_integral: float
    time_above_threshold: float
    combined_altitude_risk: float
    risk_score: float
    scoring: str
    min_radius: float
    mean_radius: float
    mean_epistemic_sigma: float
    # --- supervisor metrics (expected error = sqrt(bias^2 + sigma^2)) ---
    max_expected_error: float = float("nan")
    mean_expected_error: float = float("nan")
    p95_expected_error: float = float("nan")
    low_altitude_expected_error_integral: float = float("nan")
    max_mean_error_magnitude: float = float("nan")
    mean_mean_error_magnitude: float = float("nan")
    # relative supervisor point risk (per-trajectory altitude normalization)
    mean_point_risk: float = float("nan")
    p95_point_risk: float = float("nan")
    # absolute supervisor point risk (fixed altitude reference)
    mean_point_risk_abs: float = float("nan")
    p95_point_risk_abs: float = float("nan")
    # --- domain-support metrics (only when domain support is supplied) ---
    max_domain_risk: float = float("nan")
    time_outside_support: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def _as_1d(x, n: int | None = None, name: str = "array") -> torch.Tensor:
    t = torch.as_tensor(x).reshape(-1).to(torch.float64)
    if n is not None and t.numel() != n:
        raise ValueError(f"{name} must have length {n}, got {t.numel()}")
    return t


def _normalize_weights(weights, n: int) -> torch.Tensor | None:
    """Return weights normalized to sum 1, or ``None`` for the uniform/legacy path."""

    if weights is None:
        return None
    w = _as_1d(weights, n, "weights")
    if bool((w < 0).any()):
        raise ValueError("weights must be nonnegative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def _wmean(x: torch.Tensor, w: torch.Tensor | None) -> float:
    """Weighted mean (``w`` already normalized to sum 1); uniform when ``w is None``."""

    return float((x * w).sum()) if w is not None else float(x.mean())


def _weighted_quantile(x: torch.Tensor, q: float, w: torch.Tensor | None) -> float:
    """``q``-quantile of ``x``; weighted (``w`` sums to 1) or plain ``torch.quantile``.

    The weighted branch uses the standard cumulative-weight definition (lower value at the
    first point whose cumulative weight reaches ``q``), which is robust and dependency-free.
    """

    if x.numel() == 0:
        return float("nan")
    if w is None:
        return float(torch.quantile(x, q))
    order = torch.argsort(x)
    xs, ws = x[order], w[order]
    cum = torch.cumsum(ws, dim=0)
    cum = cum / cum[-1]
    idx = int(torch.searchsorted(cum, torch.tensor(float(q), dtype=x.dtype)))
    idx = min(idx, xs.numel() - 1)
    return float(xs[idx])


def aggregate_trajectory_error(values, mode: str = "p95") -> float:
    """Collapse a per-point error profile into one trajectory scalar (``max`` / ``mean`` / ``p95``).

    Shared by the nearest-neighbour oracle and the report so that risk and true error are
    aggregated consistently. ``p95`` (the default) is robust to a single nearest-neighbour
    spike while still rewarding a sustained high-error pass, unlike ``max`` (spike-dominated)
    or ``mean`` (washes the pass out).
    """

    if mode not in TRUE_ERROR_AGGREGATORS:
        raise ValueError(f"mode must be one of {TRUE_ERROR_AGGREGATORS}, got {mode!r}")
    v = _as_1d(values)
    if v.numel() == 0:
        return float("nan")
    if mode == "max":
        return float(v.max())
    if mode == "mean":
        return float(v.mean())
    return float(torch.quantile(v, 0.95))


def calibrate_risk_threshold(values, quantile: float = 0.95, multiplier: float = 1.0) -> float:
    """Absolute risk threshold from a held-out risk distribution: ``quantile(values) * multiplier``.

    ``values`` is a 1-D array of in-distribution risk samples -- either per-point
    ``expected_error`` or per-trajectory risk scores from a calibration set. The returned
    threshold is meant for the absolute ``select_reruns(threshold=...)`` path so a downstream
    test set with lower risk can legitimately raise zero alarms.
    """

    v = _as_1d(values)
    if v.numel() == 0:
        return float("nan")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return float(torch.quantile(v, float(quantile))) * float(multiplier)


def _altitude_weight(radius: torch.Tensor, *, h_floor: float) -> torch.Tensor:
    """Weight that grows toward the surface: ``1 / max(r - 1, h_floor)``.

    Concentrates risk where the residual-gravity surrogate is known to be least reliable, so
    a trajectory that is uncertain *and* low scores higher than one uncertain only up high.
    """

    return 1.0 / (radius - 1.0).clamp_min(h_floor)


def _relative_altitude_weight(radius: torch.Tensor, *, h_floor: float) -> torch.Tensor:
    """Altitude weight rescaled by its OWN median so a typical point on THIS trajectory weighs ~1.

    Good for ranking within an ensemble (keeps the supervisor point risk on the scale of
    ``expected_error``), but NOT comparable across trajectories -- a constant-altitude orbit
    always normalizes to 1 regardless of how low it is. Use the absolute weight for thresholds.
    """

    w = _altitude_weight(radius, h_floor=h_floor)
    med = torch.median(w)
    return w / med.clamp_min(torch.finfo(w.dtype).tiny)


def _absolute_altitude_weight(
    radius: torch.Tensor, *, h_floor: float, reference_h: float
) -> torch.Tensor:
    """Altitude weight rescaled by a FIXED reference altitude so it means the same everywhere.

    ``abs_weight = raw_weight / reference_weight`` with ``reference_weight = 1 / reference_h``;
    i.e. a point at altitude ``h = reference_h`` weighs 1, lower points weigh >1, higher <1 --
    the same mapping for every trajectory, so absolute thresholds are consistent.
    """

    raw = _altitude_weight(radius, h_floor=h_floor)
    reference_weight = 1.0 / max(float(reference_h), h_floor)
    return raw / reference_weight


def score_sigma_profile(
    sigma: torch.Tensor,
    radius: torch.Tensor,
    *,
    scoring: str = "max",
    sigma_threshold: float | None = None,
    low_altitude_radius: float = 1.15,
    h_floor: float = 1.0e-3,
    altitude_reference_h: float | None = None,
    epistemic_sigma: torch.Tensor | None = None,
    expected_error: torch.Tensor | None = None,
    mean_error_magnitude: torch.Tensor | None = None,
    domain_risk: torch.Tensor | None = None,
    domain_weight: float = 1.0,
    weights: torch.Tensor | None = None,
) -> TrajectoryScore:
    """Aggregate a per-output-point profile into a :class:`TrajectoryScore`.

    ``sigma`` and ``radius`` are 1-D tensors over the trajectory's output points. By default the
    points are assumed roughly uniform in time (a discrete sum approximates a time integral);
    pass ``weights`` (one per point, e.g. proportional to local dt) to correct for non-uniform
    sampling -- ``None`` preserves the legacy uniform behavior exactly.

    Legacy sigma modes:
      - ``max`` / ``mean``: extreme / average uncertainty along the trajectory.
      - ``low_alt_integral``: summed uncertainty over points below ``low_altitude_radius``.
      - ``time_above``: (weighted) fraction of points whose sigma exceeds ``sigma_threshold``.
      - ``combined``: mean of ``sigma`` times an altitude weight (uncertain-and-low).

    Expected-error modes (require ``expected_error``):
      - ``expected`` / ``expected_abs``: mean expected error.
      - ``expected_p95`` / ``expected_abs_p95``: 95th-percentile expected error.
      - ``expected_low_alt``: summed expected error below ``low_altitude_radius``.

    Supervisor modes (``point_risk = expected_error * altitude_weight * (1 + domain_weight *
    domain_risk)``):
      - ``supervisor`` / ``supervisor_rel`` (+ ``_p95``): RELATIVE altitude weight (per-trajectory
        median) -- for ranking an ensemble.
      - ``supervisor_abs`` (+ ``_p95``): ABSOLUTE altitude weight (fixed reference
        ``altitude_reference_h``, default ``low_altitude_radius - 1``) -- for absolute thresholds.

    ``risk_score`` is whichever of the above ``scoring`` selects.
    """

    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"scoring must be one of {SCORING_FUNCTIONS}, got {scoring!r}")
    sigma = _as_1d(sigma)
    radius = _as_1d(radius)
    if sigma.shape != radius.shape:
        raise ValueError("sigma and radius must have the same length")
    n = int(sigma.numel())
    if n == 0:
        raise ValueError("cannot score an empty trajectory")
    if scoring in _EXPECTED_MODES and expected_error is None:
        raise ValueError(
            f"scoring={scoring!r} requires an expected_error profile; score via "
            "VESPUQPlugin.score_trajectory or pass expected_error explicitly"
        )

    reference_h = (
        float(altitude_reference_h)
        if altitude_reference_h is not None
        else max(float(low_altitude_radius) - 1.0, h_floor)
    )

    w = _normalize_weights(weights, n)
    low_mask = radius <= float(low_altitude_radius)

    # ---- legacy sigma aggregations (unchanged when weights is None) ----
    max_sigma = float(sigma.max())
    mean_sigma = _wmean(sigma, w)
    if bool(low_mask.any()):
        low_alt_integral = (
            float(sigma[low_mask].sum()) if w is None else float((sigma * w)[low_mask].sum())
        )
    else:
        low_alt_integral = 0.0
    if sigma_threshold is not None:
        above = (sigma > float(sigma_threshold)).to(torch.float64)
        time_above = float(above.mean()) if w is None else float((above * w).sum())
    else:
        time_above = float("nan")
    alt_weight = _altitude_weight(radius, h_floor=h_floor)
    combined = _wmean(sigma * alt_weight, w)

    mean_epi = (
        _wmean(_as_1d(epistemic_sigma, n, "epistemic_sigma"), w)
        if epistemic_sigma is not None
        else float("nan")
    )

    # ---- expected-error + supervisor metrics ----
    max_ee = mean_ee = p95_ee = low_alt_ee = float("nan")
    mean_pr_rel = p95_pr_rel = float("nan")
    mean_pr_abs = p95_pr_abs = float("nan")
    if expected_error is not None:
        ee = _as_1d(expected_error, n, "expected_error")
        max_ee = float(ee.max())
        mean_ee = _wmean(ee, w)
        p95_ee = _weighted_quantile(ee, 0.95, w)
        if bool(low_mask.any()):
            low_alt_ee = float(ee[low_mask].sum()) if w is None else float((ee * w)[low_mask].sum())
        else:
            low_alt_ee = 0.0

        if domain_risk is not None:
            dr = _as_1d(domain_risk, n, "domain_risk")
            domain_factor = 1.0 + float(domain_weight) * dr
        else:
            domain_factor = torch.ones_like(ee)

        rel_alt = _relative_altitude_weight(radius, h_floor=h_floor)
        point_risk_rel = ee * rel_alt * domain_factor
        mean_pr_rel = _wmean(point_risk_rel, w)
        p95_pr_rel = _weighted_quantile(point_risk_rel, 0.95, w)

        abs_alt = _absolute_altitude_weight(radius, h_floor=h_floor, reference_h=reference_h)
        point_risk_abs = ee * abs_alt * domain_factor
        mean_pr_abs = _wmean(point_risk_abs, w)
        p95_pr_abs = _weighted_quantile(point_risk_abs, 0.95, w)

    max_mem = mean_mem = float("nan")
    if mean_error_magnitude is not None:
        mem = _as_1d(mean_error_magnitude, n, "mean_error_magnitude")
        max_mem = float(mem.max())
        mean_mem = _wmean(mem, w)

    max_domain_risk = time_outside_support = float("nan")
    if domain_risk is not None:
        dr = _as_1d(domain_risk, n, "domain_risk")
        max_domain_risk = float(dr.max())
        outside = (dr > 1.0).to(torch.float64)
        time_outside_support = float(outside.mean()) if w is None else float((outside * w).sum())

    table = {
        "max": max_sigma,
        "mean": mean_sigma,
        "low_alt_integral": low_alt_integral,
        "time_above": time_above,
        "combined": combined,
        # expected-error (absolute scale; `expected`/`expected_p95` are backward-compat aliases)
        "expected": mean_ee,
        "expected_abs": mean_ee,
        "expected_p95": p95_ee,
        "expected_abs_p95": p95_ee,
        "expected_low_alt": low_alt_ee,
        # relative supervisor (ranking) -- `supervisor`/`supervisor_p95` are aliases
        "supervisor": mean_pr_rel,
        "supervisor_rel": mean_pr_rel,
        "supervisor_p95": p95_pr_rel,
        "supervisor_rel_p95": p95_pr_rel,
        # absolute supervisor (thresholds)
        "supervisor_abs": mean_pr_abs,
        "supervisor_abs_p95": p95_pr_abs,
    }

    return TrajectoryScore(
        n_points=n,
        max_sigma=max_sigma,
        mean_sigma=mean_sigma,
        low_altitude_sigma_integral=low_alt_integral,
        time_above_threshold=time_above,
        combined_altitude_risk=combined,
        risk_score=table[scoring],
        scoring=scoring,
        min_radius=float(radius.min()),
        mean_radius=_wmean(radius, w),
        mean_epistemic_sigma=mean_epi,
        max_expected_error=max_ee,
        mean_expected_error=mean_ee,
        p95_expected_error=p95_ee,
        low_altitude_expected_error_integral=low_alt_ee,
        max_mean_error_magnitude=max_mem,
        mean_mean_error_magnitude=mean_mem,
        mean_point_risk=mean_pr_rel,
        p95_point_risk=p95_pr_rel,
        mean_point_risk_abs=mean_pr_abs,
        p95_point_risk_abs=p95_pr_abs,
        max_domain_risk=max_domain_risk,
        time_outside_support=time_outside_support,
    )


@dataclass
class RiskScreeningReport:
    """Outcome of selecting which trajectories to rerun at high fidelity."""

    n_trajectories: int
    threshold: float
    rerun_fraction: float
    n_flagged: int
    flagged_indices: list[int]
    mean_risk_flagged: float | None = None
    mean_risk_accepted: float | None = None
    # Selection-policy bookkeeping (so the report says *why* a given count was flagged):
    selection_mode: str = "fraction"  # fraction | threshold | threshold+max_fraction
    fraction_policy: str | None = None  # topk | quantile (fraction mode only)
    requested_rerun_fraction: float | None = None
    n_requested: int | None = None
    n_ties_at_cutoff: int | None = None
    max_rerun_fraction: float | None = None
    n_above_threshold: int | None = None
    threshold_source: str | None = None  # manual | calibration_quantile
    threshold_quantile: float | None = None
    # Validation against a ground-truth error metric (only when ``true_error`` is supplied):
    capture_rate: float | None = None  # share of truly-high-error trajectories that got flagged
    precision: float | None = None  # share of flagged trajectories that were truly high-error
    spearman_risk_vs_error: float | None = None
    mean_error_flagged: float | None = None
    mean_error_accepted: float | None = None
    error_ratio_flagged_to_accepted: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation (no scipy); ties broken by argsort order."""

    if a.numel() < 2:
        return float("nan")

    def _rank(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(x.numel(), dtype=torch.float64)
        return ranks

    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt((ra * ra).sum() * (rb * rb).sum())
    if float(denom) <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def select_reruns(
    risk_scores,
    *,
    rerun_fraction: float | None = None,
    threshold: float | None = None,
    max_rerun_fraction: float | None = None,
    fraction_policy: str = "topk",
    true_error=None,
    true_error_quantile: float = 0.90,
    threshold_source: str | None = None,
    threshold_quantile: float | None = None,
) -> RiskScreeningReport:
    """Flag the riskiest trajectories for high-fidelity rerun.

    Three selection policies:

    - ``rerun_fraction`` only: rerun the top fraction. ``fraction_policy="topk"`` (default) flags
      EXACTLY ``ceil(rerun_fraction * n)`` trajectories by stable top-k (robust to ties);
      ``"quantile"`` keeps the legacy quantile-threshold behavior (which can over-flag on ties).
    - ``threshold`` only: rerun every trajectory at or above an absolute risk threshold. If
      nothing exceeds it, *zero* trajectories are flagged -- a safe set may raise no alarms.
    - ``threshold`` + ``max_rerun_fraction``: take everything above the absolute threshold, but
      if that exceeds the budget ``ceil(max_rerun_fraction * n)``, keep only the top of them
      (at least 1 whenever anything is above the threshold).

    When ``true_error`` (one scalar per trajectory) is supplied, the report also validates the
    screen: ``capture_rate`` is the share of the truly-high-error trajectories (top
    ``1 - true_error_quantile``) that were flagged, and ``spearman_risk_vs_error`` measures
    monotonic agreement between risk and real error.
    """

    risk = _as_1d(risk_scores)
    n = int(risk.numel())
    if n == 0:
        raise ValueError("risk_scores is empty")
    if fraction_policy not in FRACTION_POLICIES:
        raise ValueError(f"fraction_policy must be one of {FRACTION_POLICIES}, got {fraction_policy!r}")

    has_frac = rerun_fraction is not None
    has_thr = threshold is not None
    has_max = max_rerun_fraction is not None

    if has_max:
        if not has_thr:
            raise ValueError("max_rerun_fraction requires an absolute threshold")
        if has_frac:
            raise ValueError("combine max_rerun_fraction with threshold, not rerun_fraction")
        selection_mode = "threshold+max_fraction"
    elif has_frac and has_thr:
        raise ValueError(
            "provide exactly one of rerun_fraction or threshold (or threshold + max_rerun_fraction)"
        )
    elif has_frac:
        selection_mode = "fraction"
    elif has_thr:
        selection_mode = "threshold"
    else:
        raise ValueError("provide rerun_fraction, threshold, or threshold + max_rerun_fraction")

    order = torch.argsort(risk, descending=True, stable=True)  # stable top-k by risk
    n_above_threshold: int | None = None
    n_requested: int | None = None
    n_ties_at_cutoff: int | None = None
    applied_fraction_policy: str | None = None

    if selection_mode == "fraction":
        f = float(rerun_fraction)
        if not 0.0 < f <= 1.0:
            raise ValueError("rerun_fraction must be in (0, 1]")
        applied_fraction_policy = fraction_policy
        k = n if f >= 1.0 else min(n, int(math.ceil(f * n)))
        n_requested = k
        if fraction_policy == "topk":
            flagged_mask = torch.zeros(n, dtype=torch.bool)
            flagged_mask[order[:k]] = True
            cutoff = float(risk[order[k - 1]]) if k > 0 else float("inf")
            thr = cutoff
            n_ties_at_cutoff = int((risk == risk[order[k - 1]]).sum()) if k > 0 else 0
        else:  # quantile (legacy)
            thr = float(torch.quantile(risk, 1.0 - f))
            flagged_mask = risk >= thr
            n_ties_at_cutoff = int((risk == thr).sum())
    else:
        thr = float(threshold)
        above_mask = risk >= thr
        n_above_threshold = int(above_mask.sum())
        if threshold_source is None:
            threshold_source = "manual"
        if selection_mode == "threshold":
            flagged_mask = above_mask
        else:  # threshold + max_fraction
            mf = float(max_rerun_fraction)
            if not 0.0 < mf <= 1.0:
                raise ValueError("max_rerun_fraction must be in (0, 1]")
            cap = int(math.ceil(mf * n))
            cap = max(1, cap) if n_above_threshold > 0 else 0
            n_requested = cap
            if n_above_threshold <= cap:
                flagged_mask = above_mask
            else:
                flagged_mask = torch.zeros(n, dtype=torch.bool)
                flagged_mask[order[:cap]] = True

    accepted_mask = ~flagged_mask
    flagged_indices = [int(i) for i in torch.nonzero(flagged_mask, as_tuple=False).reshape(-1).tolist()]
    n_flagged = len(flagged_indices)

    report = RiskScreeningReport(
        n_trajectories=n,
        threshold=thr,
        rerun_fraction=n_flagged / n,
        n_flagged=n_flagged,
        flagged_indices=flagged_indices,
        mean_risk_flagged=float(risk[flagged_mask].mean()) if n_flagged > 0 else float("nan"),
        mean_risk_accepted=float(risk[accepted_mask].mean()) if bool(accepted_mask.any()) else float("nan"),
        selection_mode=selection_mode,
        fraction_policy=applied_fraction_policy,
        requested_rerun_fraction=float(rerun_fraction) if has_frac else None,
        n_requested=n_requested,
        n_ties_at_cutoff=n_ties_at_cutoff,
        max_rerun_fraction=float(max_rerun_fraction) if has_max else None,
        n_above_threshold=n_above_threshold,
        threshold_source=threshold_source,
        threshold_quantile=threshold_quantile,
    )

    if true_error is not None:
        err = _as_1d(true_error)
        if err.numel() != n:
            raise ValueError("true_error must have one value per trajectory")
        high_thr = float(torch.quantile(err, float(true_error_quantile)))
        truly_high = err >= high_thr
        n_high = int(truly_high.sum())
        true_positive = int((flagged_mask & truly_high).sum())
        report.capture_rate = (true_positive / n_high) if n_high > 0 else float("nan")
        report.precision = (true_positive / n_flagged) if n_flagged > 0 else float("nan")
        report.spearman_risk_vs_error = _spearman(risk, err)
        report.mean_error_flagged = float(err[flagged_mask].mean()) if n_flagged > 0 else float("nan")
        report.mean_error_accepted = float(err[accepted_mask].mean()) if bool(accepted_mask.any()) else float("nan")
        if report.mean_error_accepted and report.mean_error_accepted > 0.0:
            report.error_ratio_flagged_to_accepted = report.mean_error_flagged / report.mean_error_accepted

    return report


def run_risk_screening(
    plugin,
    trajectories,
    *,
    true_error=None,
    rerun_fraction: float | None = 0.20,
    threshold: float | None = None,
    max_rerun_fraction: float | None = None,
    fraction_policy: str = "topk",
    scoring: str = "max",
    weights=None,
    threshold_source: str | None = None,
    threshold_quantile: float | None = None,
) -> dict:
    """Score a trajectory ensemble with ``plugin`` and select the high-fidelity rerun subset.

    ``plugin`` is any object exposing ``score_ensemble(trajectories, scoring=..., weights=...)``
    (the :class:`~vesp.uq.plugin.VESPUQPlugin`). Returns a dict with:
      - ``trajectory_scores``: list of :class:`TrajectoryScore` (one per trajectory),
      - ``selected_reruns``: indices flagged for high-fidelity rerun,
      - ``risk_screening_report``: the :class:`RiskScreeningReport` (validated when
        ``true_error`` -- one scalar per trajectory -- is supplied).

    Selection follows :func:`select_reruns`: ``threshold`` (optionally with
    ``max_rerun_fraction``) takes precedence over ``rerun_fraction`` when supplied.
    """

    scores = plugin.score_ensemble(trajectories, scoring=scoring, weights=weights)
    risk = torch.tensor([s.risk_score for s in scores], dtype=torch.float64)
    if threshold is not None:
        report = select_reruns(
            risk,
            threshold=threshold,
            max_rerun_fraction=max_rerun_fraction,
            true_error=true_error,
            threshold_source=threshold_source,
            threshold_quantile=threshold_quantile,
        )
    else:
        report = select_reruns(
            risk,
            rerun_fraction=rerun_fraction,
            fraction_policy=fraction_policy,
            true_error=true_error,
        )
    return {
        "trajectory_scores": scores,
        "selected_reruns": report.flagged_indices,
        "risk_screening_report": report,
    }
