import json
from pathlib import Path

import pytest
import torch

from vesp.training import evaluate
from vesp.core.sources import make_shell_sources


def _write_csv(path: Path, *, with_metadata: bool = True) -> Path:
    path.write_text(
        "x,y,z,Delta U,Delta a_x,Delta a_y,Delta a_z\n"
        "20.0,0.0,0.0,1.0,0.1,0.0,0.0\n",
        encoding="utf-8",
    )
    if with_metadata:
        path.with_suffix(path.suffix + ".metadata.json").write_text(
            json.dumps(
                {
                    "position_units": "km",
                    "R_body": 10.0,
                    "R_body_units": "km",
                    "potential_units": "model",
                    "acceleration_units": "model",
                }
            ),
            encoding="utf-8",
        )
    return path


def _write_checkpoint(path: Path, *, include_config: bool = True) -> Path:
    sources = make_shell_sources([0.5], 4, dtype=torch.float64)
    payload = {
        "source_positions": sources.positions,
        "source_weights": sources.weights,
        "shell_ids": sources.shell_ids,
        "shell_radii": sources.shell_radii,
        "sigma": torch.ones(sources.n_sources, dtype=torch.float64),
        "metrics": {},
    }
    if include_config:
        payload["config"] = {
            "body": {
                "R_body": 1.0,
                "normalize_positions": True,
                "position_units": "normalized",
                "physical_R_body": 10.0,
                "physical_R_body_units": "km",
            },
            "kernel": {"eps": 0.0, "acceleration_sign": 1.0},
        }
    torch.save(payload, path)
    return path


def test_standalone_evaluate_uses_checkpoint_unit_config_for_physical_csv(tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "sigma.pt")
    csv_path = _write_csv(tmp_path / "residual.csv")
    seen = {}

    def fake_evaluate_model(model, data, **kwargs):
        seen["positions"] = data.positions.detach().clone()
        seen["dtype"] = data.positions.dtype
        return {"potential_rmse": 0.0}

    monkeypatch.setattr(evaluate, "evaluate_model", fake_evaluate_model)

    evaluate.main(["--checkpoint", str(checkpoint), "--data", str(csv_path), "--device", "cpu"])

    assert torch.allclose(seen["positions"], torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64))
    assert seen["dtype"] == torch.float64


def test_standalone_evaluate_requires_config_for_unit_safe_loading(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "sigma.pt", include_config=False)
    csv_path = _write_csv(tmp_path / "residual.csv")

    with pytest.raises(ValueError, match="checkpoint does not contain config"):
        evaluate.main(["--checkpoint", str(checkpoint), "--data", str(csv_path), "--device", "cpu"])


def test_standalone_evaluate_requires_metadata_sidecar(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "sigma.pt")
    csv_path = _write_csv(tmp_path / "residual.csv", with_metadata=False)

    with pytest.raises(ValueError, match="CSV metadata sidecar is required"):
        evaluate.main(["--checkpoint", str(checkpoint), "--data", str(csv_path), "--device", "cpu"])
