"""Offscreen GUI smoke tests for complete Mission Console workflows."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication

from vesp.core.sources import make_shell_sources
from vesp.uq.plugin import VESPUQPlugin


def _make_models(tmp_path: Path) -> tuple[Path, Path]:
    generator = torch.Generator().manual_seed(17)
    positions = torch.randn(36, 3, generator=generator, dtype=torch.float64)
    positions = positions / torch.linalg.norm(positions, dim=-1, keepdim=True)
    positions *= 1.05 + 0.45 * torch.rand(36, 1, generator=generator, dtype=torch.float64)
    errors = 1.0e-4 * torch.randn(36, 3, generator=generator, dtype=torch.float64)

    sources = make_shell_sources([0.86], 24, dtype=torch.float64)
    plugin = VESPUQPlugin(sources, reg_method="fixed", lambda_l2=1.0e-6, domain_support=True)
    plugin.fit_error(positions, errors)

    model_a = tmp_path / "model_a.pt"
    model_b = tmp_path / "model_b.pt"
    plugin.save(model_a)
    plugin.save(model_b)
    return model_a, model_b


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    data_csv = tmp_path / "held_out.csv"
    with data_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "y", "z", "ax_ref", "ay_ref", "az_ref", "ax_sur", "ay_sur", "az_sur"))
        for index in range(20):
            radius = 1.05 + 0.025 * index
            axis = index % 3
            position = [0.0, 0.0, 0.0]
            position[axis] = radius if index % 2 == 0 else -radius
            error = [1.0e-5 * (index + 1), -5.0e-6 * index, 2.5e-6 * (index % 5)]
            writer.writerow((*position, *error, 0.0, 0.0, 0.0))
    Path(str(data_csv) + ".metadata.json").write_text(
        json.dumps({"position_units": "normalized"}),
        encoding="utf-8",
    )

    trajectory_csv = tmp_path / "trajectories.csv"
    with trajectory_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("trajectory_id", "t", "x", "y", "z"))
        directions = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.70710678, 0.70710678, 0.0),
        )
        for trajectory_id, direction in enumerate(directions):
            for step in range(4):
                radius = 1.06 + 0.10 * trajectory_id + 0.02 * step
                writer.writerow(
                    (
                        trajectory_id,
                        step,
                        radius * direction[0],
                        radius * direction[1],
                        radius * direction[2],
                    )
                )
    return data_csv, trajectory_csv


def test_compare_identity_run_renders_perfect_agreement(tmp_path, monkeypatch):
    from vesp.ui.pages import compare as compare_page

    app = QApplication.instance() or QApplication([])
    model_a, model_b = _make_models(tmp_path)
    data_csv, trajectory_csv = _write_inputs(tmp_path)
    out_dir = tmp_path / "comparison"
    monkeypatch.setattr(compare_page, "list_models", lambda: [model_a, model_b])

    page = compare_page.ComparePage()
    page.show()
    app.processEvents()
    page.data_picker.set_path(data_csv)
    page.trajectory_picker.set_path(trajectory_csv)
    page.out_picker.set_path(out_dir)

    exit_codes: list[int] = []
    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    page.job.finished.connect(lambda code: (exit_codes.append(code), loop.quit()))

    page._run()
    timeout.start(120_000)
    loop.exec()
    timeout.stop()

    if not exit_codes:
        page.job.cancel()
    assert exit_codes == [0], page.console.toPlainText()
    assert page.status.text() == "completed"
    assert float(page.kpi_spearman.value.text()) == pytest.approx(1.0)
    assert float(page.kpi_iou.value.text()) == pytest.approx(1.0)
    assert page.cal_table.rowCount() > 0
    assert page.open_dir.isEnabled()
    assert page.open_report is not None and page.open_report.isEnabled()
    page.close()
