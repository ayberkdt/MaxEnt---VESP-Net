"""Render publication figures for the VESP-UQ IAC evidence pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.figures import FIGURE_STEMS, render_iac_figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render VESP-UQ IAC evidence figures.")
    parser.add_argument("--train-run", default="outputs/vespuq_smoke", help="Directory with calibration_by_band.csv")
    parser.add_argument("--iac-dir", default="outputs/iac", help="Directory with IAC benchmark outputs")
    parser.add_argument("--linear-dir", default="outputs/linear_propagation", help="Directory with linear propagation outputs")
    parser.add_argument("--benchmarks-dir", default="benchmarks", help="Directory with checked-in benchmark reports")
    parser.add_argument("--out-dir", default="outputs/iac_pack/figures", help="Output directory for PNG/PDF figures")
    args = parser.parse_args(argv)

    manifest = render_iac_figures(
        train_run=args.train_run,
        iac_dir=args.iac_dir,
        linear_dir=args.linear_dir,
        benchmarks_dir=args.benchmarks_dir,
        out_dir=args.out_dir,
    )
    statuses = {entry["name"]: entry.get("status", "ok") for entry in manifest["figures"]}
    print(f"Rendered {len(manifest['figures'])}/{len(FIGURE_STEMS)} figure groups into {Path(args.out_dir)}")
    for name in FIGURE_STEMS:
        print(f"  - {name}: {statuses.get(name, 'missing')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
