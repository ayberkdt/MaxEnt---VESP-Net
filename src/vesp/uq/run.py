"""CLI driver for VESP-UQ: calibration + trajectory force-risk screening.

    python -m vesp.uq.run --config configs/vespuq/vespuq_real_lunar.yaml
    python -m vesp.uq.run --config configs/vespuq/vespuq_smoke.yaml

The pipeline lives in :mod:`vesp.uq.experiment` (fit / calibrate / score / screen), threshold
resolution in :mod:`vesp.uq.thresholds`, and report/CSV construction in :mod:`vesp.uq.reporting`.
This module is the thin CLI + artifact writer. The historical symbols ``run_vespuq``,
``build_report_md``, ``_resolve_threshold`` and ``_resolve_time_weighting`` are re-exported here
for backward compatibility.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from vesp.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    ensure_run_layout,
    write_run_manifest,
)
from vesp.common.config import load_config
from vesp.uq.experiment import _resolve_time_weighting, _time_weights, run_vespuq
from vesp.uq.reporting import build_report_md, calibration_table, csv_text
from vesp.uq.thresholds import resolve_threshold as _resolve_threshold

__all__ = [
    "run",
    "run_vespuq",
    "build_report_md",
    "main",
    "_resolve_threshold",
    "_resolve_time_weighting",
    "_time_weights",
]


def run(config: dict) -> dict:
    report = run_vespuq(config)
    tables = report.pop("_tables")
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs"))
    run_name = str(output_cfg.get("run_name", "vespuq"))
    layout = ensure_run_layout(output_dir / run_name)
    run_dir = layout.run_dir

    atomic_write_json(run_dir / "vespuq_report.json", report)
    atomic_write_json(run_dir / "fit_summary.json", report["fit"])
    markdown = build_report_md(report)
    atomic_write_text(run_dir / "vespuq_report.md", markdown)

    cal_header, cal_rows = calibration_table(report["experiment_1_calibration"])
    atomic_write_text(run_dir / "calibration_by_band.csv", csv_text(cal_header, cal_rows))
    atomic_write_text(
        run_dir / "trajectory_scores.csv", csv_text(tables["trajectory_header"], tables["trajectory_rows"])
    )
    atomic_write_text(
        run_dir / "flagged_trajectories.csv", csv_text(tables["trajectory_header"], tables["flagged_rows"])
    )

    # Provenance manifest: config snapshot + SHA-256 checksums of every emitted artifact.
    write_run_manifest(
        run_dir,
        config=config,
        metrics=report.get("summary", {}),
        artifacts={
            "vespuq_report_json": run_dir / "vespuq_report.json",
            "vespuq_report_md": run_dir / "vespuq_report.md",
            "fit_summary_json": run_dir / "fit_summary.json",
            "calibration_by_band_csv": run_dir / "calibration_by_band.csv",
            "trajectory_scores_csv": run_dir / "trajectory_scores.csv",
            "flagged_trajectories_csv": run_dir / "flagged_trajectories.csv",
        },
    )

    print(markdown.encode("ascii", "replace").decode("ascii"))
    print(f"saved_vespuq_report: {run_dir / 'vespuq_report.md'}")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VESP-UQ: calibration + trajectory force-risk screening.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
