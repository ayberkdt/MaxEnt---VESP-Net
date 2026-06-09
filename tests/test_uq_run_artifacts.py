"""Tests for the shared run-artifact writer (provenance manifest + checksums)."""

from __future__ import annotations

import json

import scripts.run_calibration_audit as ca
from vesp.common.artifacts import compute_file_sha256
from vesp.uq.io.run_artifacts import write_run_artifacts


def test_manifest_records_checksums_and_provenance(tmp_path):
    manifest = write_run_artifacts(
        tmp_path,
        tool="unit_test",
        config={"seed": 7, "_config_path": "x.yaml", "a": 1},
        json_files={"r.json": {"value": 1}},
        text_files={"r.csv": "a,b\n1,2\n"},
    )
    for name in ("r.json", "r.csv", "run_manifest.json"):
        assert (tmp_path / name).exists()

    assert manifest["tool"] == "unit_test"
    assert manifest["seed"] == 7
    assert manifest["config_path"] == "x.yaml"
    assert manifest["config"]["a"] == 1
    # the manifest does not list itself
    assert "run_manifest.json" not in manifest["artifacts"]
    # every recorded checksum + size matches the file on disk
    for name, info in manifest["artifacts"].items():
        assert info["sha256"] == compute_file_sha256(tmp_path / name)
        assert info["bytes"] == (tmp_path / name).stat().st_size


def test_provenance_injected_into_json(tmp_path):
    write_run_artifacts(
        tmp_path, tool="unit_test", config={"seed": 5}, json_files={"r.json": {"value": 2}}
    )
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["value"] == 2
    assert data["_provenance"]["tool"] == "unit_test"
    assert data["_provenance"]["seed"] == 5


def test_seed_inferred_from_config(tmp_path):
    manifest = write_run_artifacts(tmp_path, tool="t", config={"seed": 3}, json_files={"a.json": {}})
    assert manifest["seed"] == 3


def test_existing_provenance_not_overwritten(tmp_path):
    write_run_artifacts(
        tmp_path, tool="t", config={"seed": 1},
        json_files={"a.json": {"_provenance": {"tool": "keep"}}},
    )
    data = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert data["_provenance"] == {"tool": "keep"}


def _audit_config():
    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "data": {"type": "synthetic", "n": 240, "noise_std": 1.0e-4, "train_fraction": 0.7},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {
            "risk": {"scoring": "supervisor_rel", "low_altitude_radius": 1.15},
            "screening": {"n_orbits": 12, "n_points": 16, "rerun_fraction": 0.25},
        },
        "_config_path": "audit.yaml",
    }


def test_calibration_audit_script_writes_manifest(tmp_path):
    ca.run_and_write(_audit_config(), out_dir=tmp_path)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool"] == "run_calibration_audit"
    assert manifest["config_path"] == "audit.yaml"
    for name in ("calibration_audit.json", "calibration_audit.md", "sentinel_audit.csv"):
        assert name in manifest["artifacts"]
        assert manifest["artifacts"][name]["sha256"] == compute_file_sha256(tmp_path / name)
    # the emitted JSON carries provenance
    data = json.loads((tmp_path / "calibration_audit.json").read_text(encoding="utf-8"))
    assert data["_provenance"]["tool"] == "run_calibration_audit"
