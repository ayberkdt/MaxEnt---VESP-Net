"""VESP-UQ: a surrogate-agnostic equivalent-source uncertainty calibration layer (Phase 2).

VESP-UQ is *not* a residual-gravity surrogate. It is an uncertainty layer that wraps any
model with the interface ``x -> residual acceleration`` and answers a different question:
*where should that surrogate be trusted?* It samples the surrogate's error against a
higher-fidelity reference,

    e_a(x) = a_reference(x) - a_surrogate(x),

and fits a physics-consistent equivalent-source error model ``e_a(x) ~ A(x) sigma`` whose
sources live strictly inside the Moon. Because the model is linear in ``sigma``, the
Tikhonov/ridge solution has an exact linear-Gaussian posterior (see
:class:`~vesp.extensions.probabilistic.LinearGaussianPosterior`), turning the deterministic
error fit into calibrated, altitude-aware predictive uncertainty over the force-error field.

Pipeline (matching the VESP-UQ plan):

    fit(positions, surrogate_acc, reference_acc)   # Steps 1-2, 4-5
    predict_uncertainty(positions) -> mean error, std, per-point risk    # Step 6
    score_trajectory(positions_over_time) -> TrajectoryScore             # Steps 6-7

The posterior MEAN equals the ridge point estimate, so this never claims to improve
deterministic accuracy (the entropy/point-estimate story is kept only as an ablation). Its
value is the *error bars* and the trajectory risk screen they enable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from vesp.core.diagnostics import source_diagnostics
from vesp.core.operators import build_acceleration_operator
from vesp.core.regularization import lcurve_lambda
from vesp.core.sources import SourceSet, make_shell_sources
from vesp.extensions.probabilistic import (
    AltitudeNoiseModel,
    LinearGaussianPosterior,
    calibration_metrics,
)
from vesp.uq.trajectory import TrajectoryScore, score_sigma_profile


@dataclass
class UncertaintyPrediction:
    """Per-position output of :meth:`VESPUQPlugin.predict_uncertainty`.

    All tensors are indexed by query position. ``sigma`` is the scalar predictive std of the
    error-vector magnitude scale (``sqrt`` of the summed per-component variance) and is the
    default per-position ``risk_score``.
    """

    positions: torch.Tensor  # (N, 3)
    radius: torch.Tensor  # (N,)
    mean_error: torch.Tensor  # (N, 3) predicted mean residual-force error vector
    std_components: torch.Tensor  # (N, 3) per-component predictive std
    sigma: torch.Tensor  # (N,) total predictive std
    epistemic_sigma: torch.Tensor  # (N,) epistemic-only (source-posterior) std
    risk_score: torch.Tensor  # (N,)

    def to_numpy(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in asdict(self).items()}


def _flatten_acc(acc: torch.Tensor) -> torch.Tensor:
    """(N, 3) acceleration -> (3N,) in the [x-block, y-block, z-block] row order of the operator."""

    return torch.cat([acc[:, 0], acc[:, 1], acc[:, 2]])


class VESPUQPlugin:
    """Equivalent-source uncertainty calibration layer for residual-gravity surrogates."""

    def __init__(
        self,
        sources: SourceSet,
        *,
        eps: float = 0.0,
        acceleration_sign: float = 1.0,
        source_chunk_size: int | None = 1024,
        reg_method: str = "lcurve",
        lambda_l2: float = 30.0,
        noise_model: str = "heteroscedastic",
        val_fraction: float = 0.25,
        low_altitude_radius: float = 1.15,
        risk_scoring: str = "max",
        sigma_threshold: float | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if reg_method not in {"lcurve", "evidence", "fixed"}:
            raise ValueError("reg_method must be 'lcurve', 'evidence', or 'fixed'")
        if noise_model not in {"homoscedastic", "heteroscedastic"}:
            raise ValueError("noise_model must be 'homoscedastic' or 'heteroscedastic'")
        self.dtype = dtype
        self.device = torch.device(device)
        self.sources = sources.to(self.device)
        self.eps = float(eps)
        self.acceleration_sign = float(acceleration_sign)
        self.source_chunk_size = source_chunk_size
        self.reg_method = reg_method
        self.lambda_l2 = float(lambda_l2)
        self.noise_model = noise_model
        self.val_fraction = float(val_fraction)
        self.low_altitude_radius = float(low_altitude_radius)
        self.risk_scoring = risk_scoring
        self.sigma_threshold = sigma_threshold
        self.seed = int(seed)

        self.posterior: LinearGaussianPosterior | None = None
        self.altitude_noise: AltitudeNoiseModel | None = None
        self.fit_info: dict = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: dict) -> "VESPUQPlugin":
        """Build a plugin from a config dict (reuses the ``model``/``kernel`` conventions)."""

        dtype = torch.float64 if str(config.get("dtype", "float64")).lower() in {"float64", "double"} else torch.float32
        device = torch.device(config.get("device", "cpu"))
        model = config.get("model", {})
        if model.get("type") == "multishell":
            alphas = [float(a) for a in model["shell_alphas"]]
            counts = model["n_sources_per_shell"]
        else:
            alphas = [float(model.get("shell_alpha", 0.86))]
            counts = int(model.get("n_source", 512))
        sources = make_shell_sources(
            alphas,
            counts,
            weight_mode=str(model.get("weight_mode", "surface_area")),
            dtype=dtype,
            device=device,
        )
        kernel = config.get("kernel", {})
        uq = config.get("uq", config.get("uncertainty", {}))
        reg = uq.get("regularization", {})
        reg_method = str(reg.get("method", uq.get("reg_method", "lcurve"))).lower()
        # accept a numeric lambda either as the fixed value or as the seed for other methods
        lam_raw = reg.get("lambda_l2", config.get("solver", {}).get("lambda_l2", 30.0))
        try:
            lambda_l2 = float(lam_raw)
        except (TypeError, ValueError):
            lambda_l2 = 30.0
            if reg_method == "fixed":
                reg_method = "lcurve"
        risk = uq.get("risk", {})
        bands = config.get("evaluation", {}).get("altitude_bands", {}) or {}
        low_band = bands.get("low") or [1.03, 1.15]
        return cls(
            sources,
            eps=float(kernel.get("eps", kernel.get("softening", 0.0))),
            acceleration_sign=float(kernel.get("acceleration_sign", 1.0)),
            source_chunk_size=kernel.get("source_chunk_size", 1024),
            reg_method=reg_method,
            lambda_l2=lambda_l2,
            noise_model=str(uq.get("noise_model", "heteroscedastic")).lower(),
            val_fraction=float(uq.get("val_fraction", 0.25)),
            low_altitude_radius=float(risk.get("low_altitude_radius", low_band[1])),
            risk_scoring=str(risk.get("scoring", "max")).lower(),
            sigma_threshold=risk.get("sigma_threshold"),
            dtype=dtype,
            device=device,
            seed=int(config.get("seed", 0)),
        )

    # ------------------------------------------------------------------ internals
    def _prep_positions(self, positions) -> torch.Tensor:
        x = torch.as_tensor(positions, dtype=self.dtype, device=self.device)
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        return x

    def _operator(self, positions: torch.Tensor) -> torch.Tensor:
        return build_acceleration_operator(
            positions,
            self.sources,
            eps=self.eps,
            sign=self.acceleration_sign,
            source_chunk_size=self.source_chunk_size,
        )

    def _require_fitted(self) -> None:
        if self.posterior is None:
            raise RuntimeError("VESPUQPlugin is not fitted; call fit(...) first")

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        positions,
        surrogate_acceleration,
        reference_acceleration,
        *,
        val_positions=None,
        val_surrogate_acceleration=None,
        val_reference_acceleration=None,
    ) -> "VESPUQPlugin":
        """Fit the equivalent-source error posterior from surrogate/reference acceleration samples.

        Computes ``error = reference - surrogate`` and delegates to :meth:`fit_error`.
        """

        positions = self._prep_positions(positions)
        error = self._prep_positions(reference_acceleration) - self._prep_positions(surrogate_acceleration)
        val_error = None
        if val_positions is not None:
            val_positions = self._prep_positions(val_positions)
            val_error = self._prep_positions(val_reference_acceleration) - self._prep_positions(
                val_surrogate_acceleration
            )
        return self.fit_error(positions, error, val_positions=val_positions, val_error=val_error)

    def fit_error(self, positions, error, *, val_positions=None, val_error=None) -> "VESPUQPlugin":
        """Fit directly from sampled force-error vectors ``error = a_reference - a_surrogate``.

        The Tikhonov weight is selected automatically (L-curve corner by default). The posterior
        mean is the ridge solution; the global noise floor and the altitude-dependent excess
        noise are calibrated on a HELD-OUT validation split (an internal random split unless an
        explicit ``val_positions``/``val_error`` is supplied), because training residuals are
        optimistic and underestimate the altitude-dependent generalization error.
        """

        positions = self._prep_positions(positions)
        error = self._prep_positions(error)

        if val_positions is None:
            generator = torch.Generator().manual_seed(self.seed)
            n = positions.shape[0]
            perm = torch.randperm(n, generator=generator)
            n_val = max(1, int(round(self.val_fraction * n))) if n > 1 else 0
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            train_pos, train_err = positions[train_idx], error[train_idx]
            val_pos = positions[val_idx] if n_val > 0 else None
            val_err = error[val_idx] if n_val > 0 else None
        else:
            train_pos, train_err = positions, error
            val_pos = self._prep_positions(val_positions)
            val_err = self._prep_positions(val_error)

        operator = self._operator(train_pos)
        target = _flatten_acc(train_err)

        # --- Step 2: equivalent-source ridge fit with automatic regularization ---
        lcurve_points: list[dict] | None = None
        if self.reg_method == "evidence":
            posterior = LinearGaussianPosterior.fit_evidence(operator, target)
            lambda_used = posterior.lambda_l2
        else:
            if self.reg_method == "lcurve":
                lambda_used, lcurve_points = lcurve_lambda(operator, target)
            else:  # fixed
                lambda_used = self.lambda_l2
            # noise floor from HELD-OUT residuals (honest), falling back to the training fit
            noise_var = None
            if val_pos is not None:
                tmp = LinearGaussianPosterior.fit(operator, target, lambda_l2=lambda_used)
                val_resid = tmp.predict(self._operator(val_pos), include_noise=False)["mean"] - _flatten_acc(val_err)
                noise_var = float(torch.mean(val_resid * val_resid).detach().cpu())
            posterior = LinearGaussianPosterior.fit(
                operator, target, lambda_l2=lambda_used, noise_var=noise_var
            )
        self.posterior = posterior

        # --- Step 5: altitude-dependent heteroscedastic recalibration on held-out residuals ---
        self.altitude_noise = None
        if self.noise_model == "heteroscedastic" and val_pos is not None:
            val_op = self._operator(val_pos)
            val_pred = posterior.predict(val_op, include_noise=False)
            val_resid = val_pred["mean"] - _flatten_acc(val_err)
            val_row_radii = torch.linalg.norm(val_pos, dim=-1).repeat(3)
            self.altitude_noise = AltitudeNoiseModel.fit(
                val_row_radii, val_resid, val_pred["epistemic_variance"] + posterior.noise_var
            )

        self.fit_info = {
            "n_train": int(train_pos.shape[0]),
            "n_val": int(val_pos.shape[0]) if val_pos is not None else 0,
            "reg_method": self.reg_method,
            "lambda_l2": float(lambda_used) if lambda_used is not None else None,
            "noise_var": posterior.noise_var,
            "noise_std": float(posterior.noise_var ** 0.5),
            "noise_model": self.noise_model,
            "n_sources": int(self.sources.n_sources),
        }
        if self.altitude_noise is not None:
            self.fit_info["altitude_noise_a"] = self.altitude_noise.a
            self.fit_info["altitude_noise_b"] = self.altitude_noise.b
        if lcurve_points is not None:
            self.fit_info["lcurve"] = lcurve_points
        return self

    # ------------------------------------------------------------------ prediction
    def predict_uncertainty(self, positions) -> UncertaintyPrediction:
        """Predict the mean force-error and calibrated per-position predictive uncertainty."""

        self._require_fitted()
        positions = self._prep_positions(positions)
        n = positions.shape[0]
        op = self._operator(positions)
        radius = torch.linalg.norm(positions, dim=-1)
        if self.altitude_noise is not None:
            row_radii = radius.repeat(3)
            noise = self.posterior.noise_var + self.altitude_noise.variance(row_radii)
            pred = self.posterior.predict(op, noise_variance=noise)
        else:
            pred = self.posterior.predict(op, include_noise=True)

        # operator rows are [x-block, y-block, z-block]; reshape(3, N).T -> (N, 3)
        mean3 = pred["mean"].reshape(3, n).transpose(0, 1)
        var3 = pred["variance"].reshape(3, n).transpose(0, 1)
        epi3 = pred["epistemic_variance"].reshape(3, n).transpose(0, 1)
        std3 = torch.sqrt(var3.clamp_min(0.0))
        sigma = torch.sqrt(var3.sum(dim=1).clamp_min(0.0))
        epistemic_sigma = torch.sqrt(epi3.sum(dim=1).clamp_min(0.0))
        return UncertaintyPrediction(
            positions=positions,
            radius=radius,
            mean_error=mean3,
            std_components=std3,
            sigma=sigma,
            epistemic_sigma=epistemic_sigma,
            risk_score=sigma,
        )

    # ------------------------------------------------------------------ trajectory scoring
    def score_trajectory(self, positions_over_time, *, scoring: str | None = None) -> TrajectoryScore:
        """Score one trajectory (``(T, 3)`` output positions) into a :class:`TrajectoryScore`."""

        pred = self.predict_uncertainty(positions_over_time)
        return score_sigma_profile(
            pred.sigma,
            pred.radius,
            scoring=scoring or self.risk_scoring,
            sigma_threshold=self.sigma_threshold,
            low_altitude_radius=self.low_altitude_radius,
            epistemic_sigma=pred.epistemic_sigma,
        )

    def score_ensemble(self, trajectories, *, scoring: str | None = None) -> list[TrajectoryScore]:
        """Score an iterable of trajectories (each ``(T_i, 3)``)."""

        return [self.score_trajectory(traj, scoring=scoring) for traj in trajectories]

    # ------------------------------------------------------------------ calibration report
    def evaluate_calibration(self, positions, error, *, altitude_bands: dict | None = None) -> dict:
        """Per-band calibration metrics (PICP, z_std, NLL, CRPS) for held-out error samples.

        This is Experiment 1: does the layer's nominal interval cover the held-out residuals,
        and does its uncertainty grow toward low altitude where the surrogate is overconfident?
        """

        self._require_fitted()
        positions = self._prep_positions(positions)
        error = self._prep_positions(error)
        op = self._operator(positions)
        radius = torch.linalg.norm(positions, dim=-1)
        row_radii = radius.repeat(3)
        if self.altitude_noise is not None:
            noise = self.posterior.noise_var + self.altitude_noise.variance(row_radii)
            pred = self.posterior.predict(op, noise_variance=noise)
        else:
            pred = self.posterior.predict(op, include_noise=True)
        mean, std = pred["mean"], pred["std"]
        epistemic_std = torch.sqrt(pred["epistemic_variance"].clamp_min(0.0))
        target = _flatten_acc(error)

        bands = altitude_bands or {"low": [1.03, 1.15], "mid": [1.15, 1.35], "high": [1.35, 1.60]}

        def _band(mask: torch.Tensor) -> dict:
            m = calibration_metrics(mean[mask], std[mask], target[mask])
            m["mean_epistemic_std"] = float(torch.mean(epistemic_std[mask]).detach().cpu())
            m["mean_radius"] = float(torch.mean(row_radii[mask]).detach().cpu())
            return m

        report: dict = {"all": _band(torch.ones_like(row_radii, dtype=torch.bool))}
        for name, rng in bands.items():
            if rng is None:
                continue
            lo, hi = float(rng[0]), float(rng[1])
            mask = (row_radii >= lo) & (row_radii <= hi)
            if int(mask.sum()) >= 30:
                report[name] = _band(mask)
        low, high = report.get("low"), report.get("high")
        if low and high and high.get("mean_epistemic_std"):
            report["low_high_epistemic_std_ratio"] = low["mean_epistemic_std"] / max(
                high["mean_epistemic_std"], 1.0e-30
            )
        return report

    # ------------------------------------------------------------------ diagnostics
    def source_health(self) -> dict:
        """Step 3 source-health diagnostics on the fitted posterior mean (sigma)."""

        self._require_fitted()
        return source_diagnostics(
            source_positions=self.sources.positions,
            source_weights=self.sources.weights,
            shell_ids=self.sources.shell_ids,
            sigma=self.posterior.mean,
        )
