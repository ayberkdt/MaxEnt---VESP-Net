"""Fast subprocess smoke tests for lightweight script and CLI output contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vesp.common.version import package_version
from vesp.uq.figures import FIGURE_STEMS

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    roots = [str(ROOT / "src"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["MPLBACKEND"] = "Agg"
    return env


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_render_iac_figures_cli_output_contract(tmp_path):
    pytest.importorskip("matplotlib")
    out_dir = tmp_path / "figures"
    result = _run(
        str(ROOT / "scripts" / "render_iac_figures.py"),
        "--train-run",
        str(tmp_path / "missing_train"),
        "--iac-dir",
        str(tmp_path / "missing_iac"),
        "--linear-dir",
        str(tmp_path / "missing_linear"),
        "--benchmarks-dir",
        str(tmp_path / "missing_benchmarks"),
        "--out-dir",
        str(out_dir),
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert f"Rendered {len(FIGURE_STEMS)}/{len(FIGURE_STEMS)} figure groups" in result.stdout
    manifest = json.loads((out_dir / "figures_manifest.json").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in manifest["figures"]] == list(FIGURE_STEMS)
    assert {entry["status"] for entry in manifest["figures"]} == {"missing_data"}
    for stem in FIGURE_STEMS:
        assert (out_dir / f"{stem}.png").is_file()
        assert (out_dir / f"{stem}.pdf").is_file()


def test_build_iac_pack_collect_only_cli_output_contract(tmp_path):
    out_dir = tmp_path / "iac_pack"
    result = _run(
        str(ROOT / "scripts" / "build_iac_pack.py"),
        "--collect-only",
        "--skip-figures",
        "--out-dir",
        str(out_dir),
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "Running benchmarks" not in result.stdout
    assert "IAC Pack assembled at:" in result.stdout
    assert (out_dir / "EVIDENCE.md").is_file()
    assert (tmp_path / "iac_pack.zip").is_file()

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool"] == "build_iac_pack"
    assert manifest["config"]["collected_files"] == []
    assert manifest["config"]["figure_stems"] == []
    assert manifest["inputs"] == {}
    evidence = manifest["artifacts"]["EVIDENCE.md"]
    assert evidence["origin"] == "generated"
    assert evidence["sha256"]


@pytest.mark.parametrize("module", ["vesp.uq.run", "vesp.uq.screen"])
def test_uq_cli_version_exits_without_required_runtime_arguments(tmp_path, module):
    result = _run("-m", module, "--version", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.strip() == f"vesp-uq {package_version()}"
