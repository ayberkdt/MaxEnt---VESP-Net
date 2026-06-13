"""Tests for IAC evidence-pack assembly policy."""

from __future__ import annotations

import json

import pytest

import scripts.build_iac_pack as pack

pytest.importorskip("matplotlib")


def test_iac_pack_fails_on_placeholder_figures_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="placeholder figures"):
        pack.main([
            "--collect-only",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--train-run",
            str(tmp_path / "missing_run"),
            "--out-dir",
            str(tmp_path / "pack"),
        ])


def test_iac_pack_can_allow_placeholder_figures_and_records_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "pack"
    pack.main([
        "--collect-only",
        "--config",
        str(tmp_path / "missing.yaml"),
        "--train-run",
        str(tmp_path / "missing_run"),
        "--out-dir",
        str(out_dir),
        "--allow-placeholder-figures",
    ])

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    statuses = manifest["config"]["figure_statuses"]
    assert statuses
    assert set(statuses.values()) == {"missing_data"}
    figure_entries = {
        name: entry
        for name, entry in manifest["artifacts"].items()
        if name.startswith("figures/")
    }
    assert figure_entries
    assert {entry["status"] for entry in figure_entries.values()} == {"missing_data"}
    assert {entry["origin"] for entry in figure_entries.values()} == {"prewritten"}
    assert manifest["config"]["train_run"].endswith("missing_run")
