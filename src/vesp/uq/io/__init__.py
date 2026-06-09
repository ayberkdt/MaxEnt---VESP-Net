"""External trajectory / surrogate-output I/O for VESP-UQ.

Loads externally generated surrogate trajectory ensembles (post-processing risk scoring) and,
optionally, the surrogate/reference acceleration pairs needed to fit the residual-force error.
See :func:`vesp.uq.io.trajectory_loader.load_trajectory_csv`.
"""

from vesp.uq.io.trajectory_schema import TrajectoryDataset
from vesp.uq.io.trajectory_loader import flatten_acceleration_pairs, load_trajectory_csv

__all__ = ["TrajectoryDataset", "load_trajectory_csv", "flatten_acceleration_pairs"]
