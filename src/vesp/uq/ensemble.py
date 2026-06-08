"""Synthetic trajectory ensemble + ground-truth error assignment for the risk-screening demo.

Experiment 3 needs (a) an ensemble of trajectories standing in for a fast-surrogate Monte
Carlo and (b) a ground-truth "how wrong was the surrogate here" signal to validate the screen.

We generate eccentric Keplerian orbits with random orientation so each trajectory sweeps a
*range* of altitudes (a low periapsis pass plus a high apoapsis), which is what makes
``max``-style risk scoring non-trivial. The ground-truth error magnitude at any orbit point is
read from real held-out residual samples by nearest neighbour -- no second gravity model and
no circularity with the plugin's own posterior, so "did the screen catch the high-error
trajectories?" is an honest question.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OrbitEnsemble:
    trajectories: list[torch.Tensor]  # each (T, 3) in normalized body radii
    periapsis: torch.Tensor  # (n_orbits,)
    apoapsis: torch.Tensor  # (n_orbits,)


def _random_rotations(n: int, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """``(n, 3, 3)`` Haar-ish random rotations via QR of Gaussian matrices."""

    mats = torch.randn(n, 3, 3, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(mats)
    # fix the sign ambiguity so columns are deterministic given the seed
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.unsqueeze(-2)


def generate_orbit_ensemble(
    *,
    n_orbits: int = 200,
    n_points: int = 48,
    r_peri_range: tuple[float, float] = (1.02, 1.30),
    r_apo_range: tuple[float, float] = (1.30, 1.60),
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> OrbitEnsemble:
    """Generate ``n_orbits`` eccentric, randomly-oriented orbits sampled at ``n_points`` each.

    Periapsis is drawn from ``r_peri_range`` and apoapsis from ``max(peri, r_apo_range)`` so
    every orbit is valid (apoapsis >= periapsis). Points are sampled uniformly in true anomaly.
    """

    generator = torch.Generator().manual_seed(int(seed))
    u_peri = torch.rand(n_orbits, generator=generator, dtype=dtype)
    u_apo = torch.rand(n_orbits, generator=generator, dtype=dtype)
    r_peri = r_peri_range[0] + u_peri * (r_peri_range[1] - r_peri_range[0])
    apo_lo = torch.clamp(r_peri, min=r_apo_range[0])
    r_apo = apo_lo + u_apo * torch.clamp(torch.tensor(r_apo_range[1], dtype=dtype) - apo_lo, min=0.0)

    a = 0.5 * (r_peri + r_apo)
    e = (r_apo - r_peri) / (r_apo + r_peri).clamp_min(torch.finfo(dtype).tiny)
    p = a * (1.0 - e * e)

    theta = torch.linspace(0.0, 2.0 * torch.pi, n_points + 1, dtype=dtype)[:-1]
    rotations = _random_rotations(n_orbits, generator, dtype)

    trajectories: list[torch.Tensor] = []
    for i in range(n_orbits):
        r = p[i] / (1.0 + e[i] * torch.cos(theta))
        plane = torch.stack([r * torch.cos(theta), r * torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        trajectories.append(plane @ rotations[i].transpose(0, 1))

    return OrbitEnsemble(trajectories=trajectories, periapsis=r_peri, apoapsis=r_apo)


def nearest_neighbor_error_magnitude(
    query_positions: torch.Tensor,
    ref_positions: torch.Tensor,
    ref_error: torch.Tensor,
    *,
    chunk: int = 4096,
) -> torch.Tensor:
    """``|error|`` at each query point, read from the nearest reference sample.

    ``ref_error`` is ``(M, 3)`` force-error vectors at ``ref_positions``; returns ``(Q,)``
    error magnitudes. Chunked over queries to bound memory.
    """

    ref_mag = torch.linalg.norm(ref_error, dim=-1)
    out = torch.empty(query_positions.shape[0], dtype=query_positions.dtype, device=query_positions.device)
    for start in range(0, query_positions.shape[0], chunk):
        q = query_positions[start : start + chunk]
        d = torch.cdist(q, ref_positions)
        out[start : start + chunk] = ref_mag[torch.argmin(d, dim=1)]
    return out
