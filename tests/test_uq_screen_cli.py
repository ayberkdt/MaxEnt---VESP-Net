"""Tests for the serve-side screening CLI (``python -m vesp.uq.screen``).

The serve contract: load a persisted model, score WITHOUT refitting, apply the packaged (or
explicitly overridden) decision policy, and write provenance-checked artifacts whose scores
match the training driver exactly on the same ensemble.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
import torch

from vesp.common.config import load_config
from vesp.uq.run import run
from vesp.uq.screen import main as screen_main
from vesp.uq.screen import run_screen

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """One smoke training run with save_model: true, shared across the module's tests."""

    out = tmp_path_factory.mktemp("train")
    cfg = load_config(ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml")
    cfg["output"]["output_dir"] = str(out)
    cfg["output"]["run_name"] = "train"
    cfg["output"]["save_model"] = True
    report = run(cfg)
    run_dir = out / "train"
    return {"run_dir": run_dir, "model": run_dir / "vespuq_plugin.pt", "report": report, "config": cfg}


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_training_run_packages_policy_and_card(trained_model):
    from vesp.uq import VESPUQPlugin

    assert trained_model["model"].exists()
    card = trained_model["run_dir"] / "vespuq_plugin_card.md"
    assert card.exists(), "save_model must write a model card next to the artifact"
    text = card.read_text(encoding="utf-8")
    assert "Model card" in text
    assert "Decision policy" in text
    assert "VESP_UQ_LIMITATIONS" in text

    plugin = VESPUQPlugin.load(trained_model["model"])
    policy = plugin.user_metadata.get("decision_policy", {})
    assert policy.get("scoring") == "supervisor_rel"
    assert policy.get("rerun_fraction") == pytest.approx(0.25)
    assert plugin.user_metadata.get("provenance", {}).get("created_at_utc")

    manifest = json.loads((trained_model["run_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert "vespuq_plugin_card_md" in manifest["artifacts"]


def test_serve_scores_match_training_run_exactly(tmp_path, trained_model):
    # Same config -> same generated ensemble (same seed); the serve scores must equal the
    # training driver's trajectory_scores.csv row for row.
    report = run_screen(
        model_path=trained_model["model"],
        out_dir=tmp_path / "serve",
        config=trained_model["config"],
    )
    assert report["mode"] == "serve"
    train_rows = _read_csv(trained_model["run_dir"] / "trajectory_scores.csv")
    serve_rows = _read_csv(tmp_path / "serve" / "trajectory_scores.csv")
    assert len(train_rows) == len(serve_rows) > 0
    for tr, sr in zip(train_rows, serve_rows, strict=True):
        assert float(sr["risk_score"]) == pytest.approx(float(tr["risk_score"]), rel=1.0e-12)
    # identical policy -> identical flag set
    assert [r["flagged_for_rerun"] for r in serve_rows] == [r["flagged_for_rerun"] for r in train_rows]


def test_serve_artifacts_and_manifest_inputs(tmp_path, trained_model):
    out = tmp_path / "serve"
    run_screen(model_path=trained_model["model"], out_dir=out, config=trained_model["config"])
    for fname in (
        "screening_report.json",
        "screening_report.md",
        "trajectory_scores.csv",
        "flagged_trajectories.csv",
        "run_manifest.json",
    ):
        assert (out / fname).exists(), f"missing serve artifact {fname}"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    model_input = manifest["inputs"]["vespuq_plugin_pt"]
    assert model_input["sha256"], "serve manifest must checksum the consumed model"
    md = (out / "screening_report.md").read_text(encoding="utf-8")
    assert "serve mode" in md
    assert "no refit" in md


def test_serve_uses_packaged_fraction_and_reports_origin(tmp_path, trained_model):
    report = run_screen(
        model_path=trained_model["model"], out_dir=tmp_path / "serve", config=trained_model["config"]
    )
    sc = report["screening"]
    assert sc["selection_origin"] == "model"  # smoke config packages rerun_fraction=0.25
    assert sc["screen"]["selection_mode"] == "fraction"
    assert sc["screen"]["requested_rerun_fraction"] == pytest.approx(0.25)
    assert sc["true_error_mode"] == "none"  # generated orbits: no serve-time oracle


def test_cli_threshold_override_beats_packaged_policy(tmp_path, trained_model):
    report = run_screen(
        model_path=trained_model["model"],
        out_dir=tmp_path / "serve",
        config=trained_model["config"],
        threshold=1.0e9,  # absurdly high budget -> zero alarms
    )
    sc = report["screening"]
    assert sc["selection_origin"] == "cli"
    assert sc["screen"]["selection_mode"] != "fraction"
    assert sc["screen"]["n_flagged"] == 0


def test_packaged_threshold_refuses_mismatched_scoring(tmp_path, trained_model):
    from vesp.uq import VESPUQPlugin

    # Package a threshold calibrated for an absolute mode, then serve with a relative mode.
    plugin = VESPUQPlugin.load(trained_model["model"])
    plugin.save(
        tmp_path / "abs_model.pt",
        extra_metadata={
            "decision_policy": {"scoring": "expected_abs_p95", "threshold": 0.123, "rerun_fraction": 0.2}
        },
    )
    with pytest.raises(ValueError, match="do not transfer across score scales"):
        run_screen(
            model_path=tmp_path / "abs_model.pt",
            out_dir=tmp_path / "serve",
            config=trained_model["config"],
            scoring="supervisor_rel",
        )


def test_serve_with_external_csv_and_residual_oracle(tmp_path, trained_model):
    # Format B CSV (positions + surrogate/reference acceleration pairs) -> residual diagnostic.
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["trajectory_id", "t", "x", "y", "z", "ax_sur", "ay_sur", "az_sur", "ax_ref", "ay_ref", "az_ref"])
    g = torch.Generator().manual_seed(11)
    for tid in range(3):
        for k in range(12):
            d = torch.randn(3, generator=g, dtype=torch.float64)
            p = 1.2 * d / torch.linalg.norm(d)
            sur = 1.0e-4 * torch.randn(3, generator=g, dtype=torch.float64)
            ref = sur + 1.0e-5 * torch.randn(3, generator=g, dtype=torch.float64)
            writer.writerow([tid, k] + [float(v) for v in p] + [float(v) for v in sur] + [float(v) for v in ref])
    csv_path = tmp_path / "ensemble.csv"
    csv_path.write_text(buf.getvalue(), encoding="utf-8")

    report = run_screen(
        model_path=trained_model["model"],
        out_dir=tmp_path / "serve",
        trajectories_csv=str(csv_path),
        rerun_fraction=0.34,
    )
    sc = report["screening"]
    assert sc["trajectory_source"] == "csv"
    assert sc["n_trajectories"] == 3
    assert sc["true_error_mode"] == "residual_csv"
    rows = _read_csv(tmp_path / "serve" / "trajectory_scores.csv")
    assert all(r["true_error"] not in ("", "nan") for r in rows)
    manifest = json.loads((tmp_path / "serve" / "run_manifest.json").read_text(encoding="utf-8"))
    assert "trajectory_csv" in manifest["inputs"]


def test_serve_requires_a_trajectory_source(tmp_path, trained_model):
    with pytest.raises(ValueError, match="--trajectories|--config"):
        run_screen(model_path=trained_model["model"], out_dir=tmp_path / "serve")


def test_cli_entrypoint_smoke(tmp_path, trained_model):
    cfg_path = ROOT / "configs" / "vespuq" / "vespuq_smoke.yaml"
    screen_main(
        [
            "--model",
            str(trained_model["model"]),
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "serve_cli"),
            "--rerun-fraction",
            "0.10",
        ]
    )
    report = json.loads((tmp_path / "serve_cli" / "screening_report.json").read_text(encoding="utf-8"))
    assert report["screening"]["selection_origin"] == "cli"
    assert report["screening"]["screen"]["n_flagged"] >= 1
