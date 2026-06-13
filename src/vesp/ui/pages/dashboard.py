"""Dashboard: fleet-level overview of models, runs, and the latest screening outcome."""

from __future__ import annotations

from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.paths import OUTPUTS_DIR, list_models, scan_runs
from vesp.ui.widgets import Card, KpiTile, PageHeader, make_button


class DashboardPage(QWidget):
    """Landing page: KPI tiles, quick actions into the workflow, and recent runs."""

    def __init__(self, navigate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._navigate = navigate

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        header = PageHeader(
            "Mission overview",
            "Surrogate-agnostic force-error uncertainty: calibrate once, screen every ensemble.",
        )
        header.add_action(make_button("Refresh", variant="ghost", on_click=self.refresh))
        root.addWidget(header)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.tile_models = KpiTile("Trained models")
        self.tile_runs = KpiTile("Recorded runs")
        self.tile_train = KpiTile("Last training")
        self.tile_screen = KpiTile("Last screening")
        for tile in (self.tile_models, self.tile_runs, self.tile_train, self.tile_screen):
            tiles.addWidget(tile)
        root.addLayout(tiles)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        for title, body, page in (
            ("Train a layer", "Fit the equivalent-source posterior from calibration data and package it with its decision policy + model card.", "train"),
            ("Screen an ensemble", "Load a persisted model and risk-screen new trajectories. No refitting.", "screen"),
            ("Inspect a model", "Provenance, packaged policy, calibration table and the uncertainty-vs-altitude profile.", "model"),
        ):
            card = Card(title)
            text = QLabel(body)
            text.setWordWrap(True)
            text.setObjectName("PageSubtitle")
            card.add(text)
            card.add_stretch()
            card.add(make_button("Open", variant="primary", on_click=partial(self._navigate, page)))
            actions.addWidget(card)
        root.addLayout(actions)

        recent = Card("Recent runs")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["created (UTC)", "kind", "run", "summary"])
        vertical_header = self.table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        horizontal_header = self.table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setStretchLastSection(True)
        self.table.doubleClicked.connect(lambda _i: self._navigate("runs"))
        recent.add(self.table)
        hint = QLabel(f"Scanning `{OUTPUTS_DIR}` -- double-click a row to open the runs browser.")
        hint.setObjectName("KpiHint")
        recent.add(hint)
        root.addWidget(recent, 1)

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        models = list_models()
        runs = scan_runs()
        self.tile_models.set(str(len(models)), models[0].parent.name if models else "save one from Train")
        self.tile_runs.set(str(len(runs)), "manifest-backed run directories")

        last_train = next((r for r in runs if r.kind == "train"), None)
        if last_train is not None:
            self.tile_train.set(last_train.name, last_train.created_at)
        else:
            self.tile_train.set("--", "no training runs yet")

        last_serve = next((r for r in runs if r.kind == "serve"), None)
        if last_serve is not None:
            flagged = last_serve.metrics.get("n_flagged")
            total = last_serve.metrics.get("n_trajectories")
            value = f"{flagged}/{total} flagged" if flagged is not None else last_serve.name
            self.tile_screen.set(value, last_serve.created_at)
        else:
            self.tile_screen.set("--", "no screening runs yet")

        self.table.setRowCount(0)
        for record in runs[:12]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            summary = ", ".join(
                f"{k}={v}" for k, v in list(record.metrics.items())[:3] if not isinstance(v, dict)
            )
            for col, text in enumerate((record.created_at, record.kind, record.name, summary)):
                item = QTableWidgetItem(str(text))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def showEvent(self, event) -> None:  # refresh whenever the page becomes visible
        super().showEvent(event)
        self.refresh()
