import json
import math

import torch

from vesp.common.config import merge_defaults
from vesp.data.target_scaling import compute_target_scales
from vesp.training.train_discrete import make_data_splits, run


def _small_config(tmp_path, *, normalize_targets: bool) -> dict:
    return {
        "seed": 7,
        "device": "cpu",
        "dtype": "float64",
        "body": {"R_body": 1.0, "normalize_positions": True},
        "data": {
            "type": "synthetic",
            "path": None,
            "seed": 7,
            "synthetic_n_query": 32,
            "synthetic_n_truth_sources": 8,
            "synthetic_query_radius_min": 1.05,
            "synthetic_query_radius_max": 1.30,
            "synthetic_truth_shell_radius": 0.65,
        },
        "model": {"type": "discrete", "shell_alpha": 0.72, "n_source": 12},
        "kernel": {"eps": 0.0, "acceleration_sign": 1.0, "source_chunk_size": 64},
        "solver": {"type": "ridge", "ridge_method": "augmented_lstsq", "column_normalize": True},
        "loss": {
            "use_potential": True,
            "use_acceleration": True,
            "normalize_targets": normalize_targets,
            "potential_scale": "auto",
            "acceleration_scale": "auto",
            "lambda_potential": 0.2,
            "lambda_acceleration": 1.0,
            "lambda_l2": 1.0e-8,
            "lambda_moment": 0.0,
            "lambda_dipole": 1.0,
        },
        "split": {"type": "random", "train_fraction": 0.75},
        "output": {"output_dir": str(tmp_path), "run_name": f"scales_{normalize_targets}"},
    }


def test_target_scales_written_when_normalization_disabled(tmp_path):
    config = _small_config(tmp_path, normalize_targets=False)

    run(config)

    run_dir = tmp_path / "scales_False"
    payload = json.loads((run_dir / "target_scales.json").read_text(encoding="utf-8"))
    summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert payload["normalize_targets"] is False
    assert payload["potential_scale"] == 1.0
    assert payload["acceleration_scale"] == 1.0
    assert payload["potential_source"] == "disabled"
    assert payload["acceleration_source"] == "disabled"
    assert "normalize_targets: False" in summary
    assert "potential_scale_source: disabled" in summary
    assert "target_scales" in manifest["artifacts"]


def test_target_scales_written_from_train_split_when_enabled(tmp_path):
    config = _small_config(tmp_path, normalize_targets=True)
    merged = merge_defaults(config)
    splits = make_data_splits(merged, dtype=torch.float64)
    expected = compute_target_scales(splits.train, merged)

    run(config)

    run_dir = tmp_path / "scales_True"
    payload = json.loads((run_dir / "target_scales.json").read_text(encoding="utf-8"))
    summary = (run_dir / "summary.txt").read_text(encoding="utf-8")

    assert payload["normalize_targets"] is True
    assert math.isclose(payload["potential_scale"], expected.potential_scale)
    assert math.isclose(payload["acceleration_scale"], expected.acceleration_scale)
    assert payload["potential_source"] == "auto_rms"
    assert payload["acceleration_source"] == "auto_rms"
    assert "training_loss_units: target-normalized" in summary
