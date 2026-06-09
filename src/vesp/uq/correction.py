"""Phase 5 (EXPLORATORY): online force-model correction ``a_corrected = a_surrogate + mean_error``.

The VESP-UQ posterior MEAN is the ridge estimate of the surrogate's force-model error,

    mean_error(x) = K(x) @ sigma_mean,

i.e. the equivalent-source acceleration operator applied to the posterior mean, honoring the fitted
softening ``eps``, ``acceleration_sign``, and source quadrature weights. Adding it to a surrogate's
acceleration inside an integrator RHS gives an **online force-model correction**:

    a_corrected(x) = a_surrogate(x) + mean_error(x).

This is exactly the posterior-mean force-error field used by :meth:`VESPUQPlugin.predict_uncertainty`
and as the nominal field of the MC / STM propagators -- so the trajectory it produces is the
**posterior-mean** trajectory, not a calibrated position-accuracy product.

**Scope / honesty caveat.** The posterior mean is a regularized point estimate. This corrects the
*force model*; it improves trajectory accuracy only insofar as the surrogate's force error is
captured by the equivalent-source posterior mean, and it carries **no guaranteed long-horizon
position-accuracy improvement** (the force-risk score does not predict position error). Evaluating
the full equivalent-source field every RHS call costs more than the bare surrogate, which can erode
the surrogate's speed advantage. Report measured accuracy **and** cost; see
``benchmarks/online_force_correction.md`` and ``docs/VESP_UQ_LIMITATIONS.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from vesp.core.operators import build_acceleration_operator
from vesp.uq.plugin import VESPUQPlugin


class CorrectedForceField:
    """Online ``a_corrected(x) = a_surrogate(x) + mean_error(x)`` RHS hook for a fitted plugin.

    ``surrogate_accel_fn`` maps positions ``(N, 3)`` -> surrogate total acceleration ``(N, 3)`` (the
    field the integrator would use without VESP-UQ). The correction reuses the plugin's operator
    convention, so :meth:`correction` equals ``plugin.predict_uncertainty(x).mean_error`` exactly.
    """

    def __init__(
        self,
        plugin: VESPUQPlugin,
        *,
        surrogate_accel_fn,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if plugin.posterior is None:
            raise RuntimeError("Plugin must be fitted before building a force correction.")
        self.dtype = dtype or plugin.dtype
        self.device = torch.device(device) if device is not None else plugin.device
        self.sources = plugin.sources
        self.eps = float(plugin.eps)
        self.sign = float(plugin.acceleration_sign)
        self.source_chunk_size = plugin.source_chunk_size
        self.mean_sigma = plugin.posterior.mean.to(dtype=self.dtype, device=self.device)
        self.surrogate_accel_fn = surrogate_accel_fn

    def _prep(self, r) -> tuple[torch.Tensor, bool]:
        x = torch.as_tensor(r, dtype=self.dtype, device=self.device)
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("positions must be (3,) or (N, 3)")
        return x, single

    def _correction_2d(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        op = build_acceleration_operator(
            x, self.sources, eps=self.eps, sign=self.sign, source_chunk_size=self.source_chunk_size
        )
        return (op @ self.mean_sigma).reshape(3, n).transpose(0, 1)

    def correction(self, r) -> torch.Tensor:
        """Posterior-mean force-error vector ``mean_error(x) = K(x) @ sigma_mean``."""

        x, single = self._prep(r)
        c = self._correction_2d(x)
        return c[0] if single else c

    def __call__(self, r) -> torch.Tensor:
        """Corrected acceleration ``a_surrogate(x) + mean_error(x)``."""

        x, single = self._prep(r)
        a = torch.as_tensor(self.surrogate_accel_fn(x), dtype=self.dtype, device=self.device)
        a = a.reshape(x.shape) + self._correction_2d(x)
        return a[0] if single else a


def integrate_trajectory(
    accel_fn,
    y0,
    *,
    dt: float,
    duration: float,
    output_dt: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """RK4-integrate ``[r, v]' = [v, accel_fn(r)]`` for one trajectory.

    ``accel_fn`` maps a single position ``(3,)`` -> acceleration ``(3,)``. Uses the same
    snap / sub-step structure as the MC and STM propagators. Returns ``(times (T,), states (T, 6))``.
    """

    y0 = np.asarray(y0, dtype=np.float64)
    if y0.shape != (6,):
        raise ValueError("y0 must be a 1D array of shape (6,)")
    device = torch.device(device)
    snap = float(output_dt)
    steps_per_snap = max(1, round(snap / float(dt)))
    dt_eff = snap / steps_per_snap
    n_snaps = max(1, round(float(duration) / snap))

    times = np.linspace(0.0, n_snaps * snap, n_snaps + 1, dtype=np.float64)
    states = np.empty((n_snaps + 1, 6), dtype=np.float64)
    state = torch.tensor(y0, dtype=dtype, device=device)
    states[0] = state.cpu().numpy()

    def deriv(s: torch.Tensor) -> torch.Tensor:
        a = torch.as_tensor(accel_fn(s[:3]), dtype=dtype, device=device).reshape(3)
        return torch.cat([s[3:], a])

    for i in range(n_snaps):
        for _ in range(steps_per_snap):
            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt_eff * k1)
            k3 = deriv(state + 0.5 * dt_eff * k2)
            k4 = deriv(state + dt_eff * k3)
            state = state + (dt_eff / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[i + 1] = state.cpu().numpy()
    return times, states
