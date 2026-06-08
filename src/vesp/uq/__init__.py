"""VESP-UQ: equivalent-source uncertainty calibration layer for residual-gravity surrogates.

This package reframes the equivalent-source machinery as a surrogate-agnostic *uncertainty*
layer (not a better point-estimate surrogate). The headline object is :class:`VESPUQPlugin`,
which fits the calibrated linear-Gaussian error posterior and scores Monte Carlo trajectories
for selective high-fidelity rerun. See ``VESP_UQ_pipeline_and_usefulness_plan`` for the full
positioning.
"""

from vesp.uq.plugin import UncertaintyPrediction, VESPUQPlugin
from vesp.uq.trajectory import (
    RiskScreeningReport,
    TrajectoryScore,
    score_sigma_profile,
    select_reruns,
)

__all__ = [
    "VESPUQPlugin",
    "UncertaintyPrediction",
    "TrajectoryScore",
    "RiskScreeningReport",
    "score_sigma_profile",
    "select_reruns",
]
