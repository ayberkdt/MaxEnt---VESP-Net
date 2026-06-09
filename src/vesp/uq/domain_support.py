"""Domain-support (out-of-calibration-support / OOD) scoring helpers for VESP-UQ.

A query trajectory point can be *in distribution* in altitude yet far from the calibration
samples in direction; the domain-support score measures that geometric extrapolation so the
supervisor can raise risk where the equivalent-source fit was never constrained. The score is a
nonnegative, weighted sum of three robust, decomposable terms (``0`` = well supported, ``>1`` =
increasingly extrapolated):

- ``distance_score``: query ``k``-th nearest training-point Euclidean distance vs the median
  training ``k``-NN spacing (this already captures *angular* OOD at fixed radius);
- ``radius_penalty``: radial extrapolation below ``r_min`` / above ``r_max``;
- ``angular_score``: optional nearest-direction angular term (off by default).

The nearest-neighbour distance is computed by a pluggable backend: :class:`TorchDomainSupportBackend`
(default, dependency-free ``cdist``+``topk``) or :class:`KDTreeDomainSupportBackend` (scipy
``cKDTree``, used only when scipy is importable; otherwise falls back to torch).
"""

from __future__ import annotations

import warnings

import torch

DOMAIN_BACKENDS = ("torch", "kdtree")


def unit_directions(x: torch.Tensor) -> torch.Tensor:
    """Row-normalize positions to unit direction vectors (for the angular component)."""

    r = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    return x / r


class TorchDomainSupportBackend:
    """Default nearest-neighbour backend: chunked ``torch.cdist`` + ``topk`` (no extra deps)."""

    name = "torch"

    def kth_distances(self, query: torch.Tensor, train: torch.Tensor, k: int, *, chunk: int = 1024) -> torch.Tensor:
        out = torch.empty(query.shape[0], dtype=query.dtype, device=query.device)
        for start in range(0, query.shape[0], chunk):
            d = torch.cdist(query[start : start + chunk], train)
            out[start : start + chunk] = torch.topk(d, k, largest=False).values[:, -1]
        return out


class KDTreeDomainSupportBackend:
    """Optional scipy ``cKDTree`` backend (exact Euclidean kth distance). Requires scipy."""

    name = "kdtree"

    def __init__(self) -> None:
        from scipy.spatial import cKDTree  # raises ImportError when scipy is unavailable

        self._cKDTree = cKDTree

    def kth_distances(self, query: torch.Tensor, train: torch.Tensor, k: int, *, chunk: int = 1024) -> torch.Tensor:
        import numpy as np

        tree = self._cKDTree(train.detach().cpu().numpy())
        d, _ = tree.query(query.detach().cpu().numpy(), k=k)
        d = np.asarray(d)
        kth = d.reshape(-1) if k == 1 else d[:, -1]
        return torch.as_tensor(np.ascontiguousarray(kth), dtype=query.dtype, device=query.device)


def make_domain_backend(name: str = "torch", *, warn: bool = True):
    """Construct a domain-support backend; fall back to torch if ``kdtree`` is requested but scipy
    is unavailable. Raises ``ValueError`` for an unknown name."""

    name = str(name).lower()
    if name == "torch":
        return TorchDomainSupportBackend()
    if name == "kdtree":
        try:
            return KDTreeDomainSupportBackend()
        except ImportError:
            if warn:
                warnings.warn(
                    "scipy is not available; domain_backend='kdtree' falls back to 'torch'",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return TorchDomainSupportBackend()
    raise ValueError(f"domain_backend must be one of {DOMAIN_BACKENDS}, got {name!r}")


def angular_kth_distances(
    query_unit: torch.Tensor, train_unit: torch.Tensor, k: int, *, chunk: int = 1024
) -> torch.Tensor:
    """``k``-th nearest training-direction angle (radians) for each unit-vector query."""

    out = torch.empty(query_unit.shape[0], dtype=query_unit.dtype, device=query_unit.device)
    for start in range(0, query_unit.shape[0], chunk):
        cos = (query_unit[start : start + chunk] @ train_unit.transpose(0, 1)).clamp(-1.0, 1.0)
        kth_cos = torch.topk(cos, k, largest=True).values[:, -1]
        out[start : start + chunk] = torch.arccos(kth_cos)
    return out


def median_knn_scale(
    train: torch.Tensor,
    k: int,
    *,
    backend,
    seed: int,
    subset: int = 512,
    chunk: int = 1024,
) -> float:
    """Robust training length scale: median ``k``-th NN distance over a random training subset
    (the ``k+1``-th neighbour is taken so the self-distance is dropped)."""

    n = int(train.shape[0])
    if n < 2:
        return 1.0
    g = torch.Generator().manual_seed(int(seed))
    sub = train[torch.randperm(n, generator=g)[: min(subset, n)]]
    kk = min(k + 1, n)
    knn = backend.kth_distances(sub, train, kk, chunk=chunk)
    return max(float(torch.median(knn)), 1.0e-12)


def median_angular_scale(
    train: torch.Tensor, k: int, *, seed: int, subset: int = 512, chunk: int = 1024
) -> float:
    """Robust angular spacing: median ``k``-th nearest training-direction angle (radians)."""

    n = int(train.shape[0])
    if n < 2:
        return 1.0
    u = unit_directions(train)
    g = torch.Generator().manual_seed(int(seed))
    sub = u[torch.randperm(n, generator=g)[: min(subset, n)]]
    kk = min(k + 1, n)
    ang = angular_kth_distances(sub, u, kk, chunk=chunk)
    return max(float(torch.median(ang)), 1.0e-9)


def domain_support_components(
    query: torch.Tensor,
    train: torch.Tensor,
    train_radii: torch.Tensor,
    *,
    k: int,
    distance_scale: float,
    distance_weight: float,
    radial_weight: float,
    angular_weight: float,
    angular_scale: float | None,
    backend,
    chunk: int = 1024,
) -> dict[str, torch.Tensor]:
    """Pure decomposed domain-support score: weighted, nonnegative components that sum to total.

    ``distance_scale`` / ``angular_scale`` are the robust training spacings from
    :func:`median_knn_scale` / :func:`median_angular_scale`. Returns ``distance_score``,
    ``radius_penalty``, ``angular_score`` (each already weight-applied) and ``total_score``.
    """

    n_train = int(train.shape[0])
    k_eff = max(1, min(k, n_train))
    knn = backend.kth_distances(query, train, k_eff, chunk=chunk)
    distance_score = (knn / distance_scale - 1.0).clamp_min(0.0)

    radius = torch.linalg.norm(query, dim=-1)
    r_min = float(train_radii.min())
    r_max = float(train_radii.max())
    below = (r_min - radius).clamp_min(0.0)
    above = (radius - r_max).clamp_min(0.0)
    radius_penalty = (below + above) / distance_scale

    if angular_weight > 0.0 and angular_scale is not None:
        ang = angular_kth_distances(unit_directions(query), unit_directions(train), k_eff, chunk=chunk)
        angular_score = (ang / angular_scale - 1.0).clamp_min(0.0)
    else:
        angular_score = torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)

    d_c = (distance_weight * distance_score).clamp_min(0.0)
    r_c = (radial_weight * radius_penalty).clamp_min(0.0)
    a_c = (angular_weight * angular_score).clamp_min(0.0)
    return {
        "distance_score": d_c,
        "radius_penalty": r_c,
        "angular_score": a_c,
        "total_score": (d_c + r_c + a_c).clamp_min(0.0),
    }
