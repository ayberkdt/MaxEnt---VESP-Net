"""Regression tests for the scoped mypy gate."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 local runs
    tomllib = None

import pytest


def test_pyproject_declares_scoped_mypy_gate():
    if tomllib is None:
        pytest.skip("tomllib is unavailable")
    data = tomllib.loads(_read("pyproject.toml"))

    assert "mypy" in data["project"]["optional-dependencies"]["dev"]
    mypy = data["tool"]["mypy"]
    assert mypy["files"] == ["src/vesp/uq", "src/vesp/common", "src/vesp/ui"]
    assert mypy["ignore_missing_imports"] is True
    assert mypy["check_untyped_defs"] is False
    assert "strict" not in mypy


def test_ci_has_blocking_typecheck_job():
    ci = _read(".github/workflows/ci.yml")
    assert "\n  typecheck:\n" in ci
    typecheck_job = ci.split("\n  typecheck:\n", 1)[1].split("\n  test:\n", 1)[0]
    assert 'python -m pip install -e ".[dev]"' in typecheck_job
    assert "--no-deps" not in typecheck_job
    assert "mypy src/vesp/uq src/vesp/common src/vesp/ui" in ci


def test_ci_runs_offscreen_ui_smoke_with_pyqt():
    ci = _read(".github/workflows/ci.yml")
    test_job = ci.split("\n  test:\n", 1)[1].split("\n  package:\n", 1)[0]
    assert "PyQt6" in test_job


def test_ci_has_source_parse_hygiene_gate():
    ci = _read(".github/workflows/ci.yml")
    assert "Source parse / UTF-8 hygiene" in ci
    assert "python scripts/check_source_parse.py src scripts tests ui" in ci


def _read(path: str) -> str:
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8")
