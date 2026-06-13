"""Tests for IAC publication figure rendering."""

from __future__ import annotations

import json

import pytest

from vesp.uq.figures import FIGURE_STEMS, render_iac_figures

pytest.importorskip("matplotlib")


def test_render_iac_figures_writes_png_pdf_and_manifest(tmp_path):
    train_run = tmp_path / "run"
    iac_dir = tmp_path / "iac"
    linear_dir = tmp_path / "linear"
    benchmarks_dir = tmp_path / "benchmarks"
    out_dir = tmp_path / "figures"
    for path in (train_run, iac_dir, linear_dir, benchmarks_dir):
        path.mkdir()

    (train_run / "calibration_by_band.csv").write_text(
        "\n".join(
            [
                "band,n,mean_radius,rmse,mean_pred_std,mean_epistemic_std,z_std,picp_50,picp_68,picp_90,picp_95",
                "all,30,1.30,1e-3,2e-3,8e-4,0.80,0.55,0.70,0.90,0.96",
                "low,10,1.08,2e-3,4e-3,2e-3,1.00,0.50,0.68,0.90,0.95",
                "mid,10,1.25,1e-3,2e-3,8e-4,0.70,0.60,0.75,0.94,0.98",
                "high,10,1.48,5e-4,1e-3,3e-4,0.40,0.80,0.90,1.00,1.00",
            ]
        ),
        encoding="utf-8",
    )
    (train_run / "trajectory_scores.csv").write_text(
        "\n".join(
            [
                "trajectory_id,risk_score,flagged_for_rerun,true_error",
                "0,0.10,1,0.08",
                "1,0.03,0,0.02",
                "2,0.07,1,0.05",
                "3,0.01,0,0.01",
            ]
        ),
        encoding="utf-8",
    )
    (benchmarks_dir / "covariance_propagation.md").write_text(
        "\n".join(
            [
                "| Propagator | position sigma | rel. error vs STM | wall time* |",
                "| --- | ---: | ---: | ---: |",
                "| STM (deterministic) | 1.38495 | -- | ~11 ms |",
                "| MC, N = 500 | 1.33440 | 3.65% | ~68 ms |",
                "| MC, N = 2000 | 1.36133 | 1.71% | ~209 ms |",
                "| MC, N = 8000 | 1.38484 | 0.008% | ~806 ms |",
            ]
        ),
        encoding="utf-8",
    )
    _write_report(benchmarks_dir / "vespuq_real_lunar_report.md", z_low=1.09, picp_low=0.87)
    _write_report(benchmarks_dir / "vespuq_real_lunar_L90_report.md", z_low=0.17, picp_low=1.00)

    manifest = render_iac_figures(
        train_run=train_run,
        iac_dir=iac_dir,
        linear_dir=linear_dir,
        benchmarks_dir=benchmarks_dir,
        out_dir=out_dir,
    )

    assert [entry["name"] for entry in manifest["figures"]] == list(FIGURE_STEMS)
    assert {entry["status"] for entry in manifest["figures"]} == {"ok"}
    for stem in FIGURE_STEMS:
        for ext in ("png", "pdf"):
            path = out_dir / f"{stem}.{ext}"
            assert path.exists()
            assert path.stat().st_size > 0

    manifest_file = json.loads((out_dir / "figures_manifest.json").read_text(encoding="utf-8"))
    assert manifest_file["figure_schema_version"] == 1
    assert len(manifest_file["figures"]) == len(FIGURE_STEMS)


def test_render_iac_figures_degrades_to_placeholders_when_inputs_missing(tmp_path):
    manifest = render_iac_figures(
        train_run=tmp_path / "missing_run",
        iac_dir=tmp_path / "missing_iac",
        linear_dir=tmp_path / "missing_linear",
        benchmarks_dir=tmp_path / "missing_benchmarks",
        out_dir=tmp_path / "figures",
    )

    assert [entry["name"] for entry in manifest["figures"]] == list(FIGURE_STEMS)
    assert {entry["status"] for entry in manifest["figures"]} == {"missing_data"}
    assert (tmp_path / "figures" / "figures_manifest.json").exists()
    for stem in FIGURE_STEMS:
        assert (tmp_path / "figures" / f"{stem}.png").exists()
        assert (tmp_path / "figures" / f"{stem}.pdf").exists()


def _write_report(path, *, z_low: float, picp_low: float):
    path.write_text(
        "\n".join(
            [
                "| band | mean_radius | rmse | mean_pred_std | mean_epi_std | z_std | picp_90 | ell_picp_90 | mean_d2 | nll |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| all | 1.30 | 1e-4 | 2e-4 | 3e-5 | 0.67 | 0.96 | 0.97 | 1.34 | -7.2 |",
                f"| low | 1.09 | 3e-4 | 3e-4 | 7e-5 | {z_low} | {picp_low} | 0.87 | 3.56 | -6.6 |",
                "| mid | 1.25 | 1e-4 | 2e-4 | 2e-5 | 0.64 | 0.97 | 0.99 | 1.24 | -7.3 |",
                "| high | 1.48 | 5e-5 | 2e-4 | 1e-5 | 0.27 | 1.00 | 1.00 | 0.22 | -7.5 |",
            ]
        ),
        encoding="utf-8",
    )
