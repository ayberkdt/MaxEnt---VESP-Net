"""Tests for the linearized (STM) covariance propagation driver (scripts/run_linear_propagation.py).

Locks the script's artifact contract: it writes the N1 manifest + provenance, a JSON holding the
nominal states / 6x6 covariances / position+velocity sigma, and a per-step CSV. (Full script-level
coverage of every driver is the separate N4 phase; this is the focused N3 check.)
"""

from __future__ import annotations

import json

import scripts.run_linear_propagation as rlp
from vesp.common.artifacts import compute_file_sha256


def _tiny_config():
    return {
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "data": {"type": "synthetic", "n": 240, "noise_std": 1.0e-4, "train_fraction": 0.7},
        "model": {"type": "multishell", "shell_alphas": [0.75, 0.9], "n_sources_per_shell": [24, 32]},
        "kernel": {"eps": 0.0},
        "uq": {
            "regularization": {"method": "lcurve"},
            "noise_model": "heteroscedastic",
            # config-driven propagation params (short, for a fast test)
            "propagation": {"r_initial": 1.1, "mu": 1.0, "duration": 3.0, "dt": 0.25, "output_dt": 0.5},
        },
        "_config_path": "lin_prop.yaml",
    }


def test_resolve_params_reads_config_and_cli_overrides():
    cfg = _tiny_config()
    params = rlp.resolve_propagation_params(cfg)
    assert params["r_initial"] == 1.1 and params["duration"] == 3.0 and params["output_dt"] == 0.5

    # a CLI flag (argparse Namespace) overrides the config value; unset (None) flags fall back
    args = rlp.argparse.Namespace(r_initial=1.2, mu=None, duration=None, dt=None, output_dt=None)
    overridden = rlp.resolve_propagation_params(cfg, args)
    assert overridden["r_initial"] == 1.2  # CLI wins
    assert overridden["duration"] == 3.0  # config still used where CLI is None


def test_script_writes_manifest_and_covariance_artifacts(tmp_path):
    cfg = _tiny_config()
    params = rlp.resolve_propagation_params(cfg)
    result = rlp.run_and_write(cfg, params, out_dir=tmp_path)

    # ---- manifest + provenance (N1 contract) ----
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool"] == "run_linear_propagation"
    assert manifest["config_path"] == "lin_prop.yaml"
    for name in ("linear_propagation.json", "linear_propagation.md", "linear_propagation_states.csv"):
        assert name in manifest["artifacts"]
        assert manifest["artifacts"][name]["sha256"] == compute_file_sha256(tmp_path / name)

    data = json.loads((tmp_path / "linear_propagation.json").read_text(encoding="utf-8"))
    assert data["_provenance"]["tool"] == "run_linear_propagation"

    # ---- covariance contract ----
    n = data["n_steps"]
    assert len(data["covariances_6x6"]) == n
    assert all(len(P) == 6 and all(len(row) == 6 for row in P) for P in data["covariances_6x6"])
    # covariance starts at zero (J(0) = 0) and the implied position dispersion grows from zero
    assert all(v == 0.0 for row in data["covariances_6x6"][0] for v in row)
    assert data["position_sigma"][0] == 0.0
    assert all(s >= 0.0 for s in data["position_sigma"])
    assert data["position_sigma"][-1] > 0.0
    assert data["summary"]["final_position_sigma"] == data["position_sigma"][-1]

    # ---- per-step CSV: 9-column header + one row per output step ----
    lines = (tmp_path / "linear_propagation_states.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == [
        "time", "x", "y", "z", "vx", "vy", "vz", "position_sigma", "velocity_sigma"
    ]
    assert len(lines) - 1 == n

    # ---- honest scope is stated in the human-readable report ----
    assert "not validated" in result["_markdown"].lower()


def test_main_runs_end_to_end(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg = _tiny_config()
    cfg.pop("_config_path")
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out"
    rlp.main(["--config", str(cfg_path), "--out-dir", str(out), "--duration", "2.0"])
    assert (out / "linear_propagation.json").exists()
    assert (out / "run_manifest.json").exists()
