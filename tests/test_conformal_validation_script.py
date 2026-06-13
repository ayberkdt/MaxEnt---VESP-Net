"""Tests for the L60/L90 conformal validation driver."""

from __future__ import annotations

import scripts.run_conformal_validation as cv


def test_conformal_config_enables_apply_without_mutating_base():
    base = {
        "uq": {"conformal": {"alpha": 0.2}, "screening": {"n_orbits": 10}},
        "output": {"output_dir": "old", "run_name": "base", "save_model": False},
    }

    cfg = cv.conformal_config(
        base,
        run_name="validated",
        output_root="out",
        n_orbits=5,
        n_points=7,
        save_model=True,
    )

    assert cfg["uq"]["conformal"]["apply"] is True
    assert cfg["uq"]["conformal"]["by_band"] is True
    assert cfg["uq"]["conformal"]["alpha"] == 0.2
    assert cfg["uq"]["screening"]["n_orbits"] == 5
    assert cfg["uq"]["screening"]["n_points"] == 7
    assert cfg["output"]["run_name"] == "validated"
    assert cfg["output"]["save_model"] is True
    assert base["uq"]["screening"]["n_orbits"] == 10
    assert "apply" not in base["uq"]["conformal"]


def test_band_acceptance_and_markdown_summary():
    calibration = {
        "low": {"z_std": 0.9, "picp_90": 0.9},
        "mid": {"z_std": 0.6, "picp_90": 0.9},
        "high": {"z_std": 1.0, "picp_90": 0.99},
    }
    acceptance = cv.band_acceptance(calibration)
    assert acceptance["low"]["z_std_in_range"] is True
    assert acceptance["low"]["picp90_in_range"] is True
    assert acceptance["mid"]["z_std_in_range"] is False
    assert acceptance["high"]["picp90_in_range"] is False
    assert acceptance["all_bands_pass"] is False

    md = cv.build_markdown(
        {
            "cases": [
                {
                    "label": "L90",
                    "run_dir": "outputs/l90_conformal",
                    "baseline_calibration": {"low": {"z_std": 0.17, "picp_90": 1.0}},
                    "calibration": {"low": {"z_std": 0.9, "picp_90": 0.9}},
                    "conformal": {
                        "global": {"scale": 0.5},
                        "bands": [{"name": "low", "scale": 0.8, "used": True}],
                    },
                    "acceptance": {"low": {"z_std_in_range": True, "picp90_in_range": True}},
                }
            ]
        }
    )
    assert "Operational Conformal Validation" in md
    assert "| L90 | low | 0.17 | 0.9 | 1 | 0.9 | 0.8 | yes |" in md
