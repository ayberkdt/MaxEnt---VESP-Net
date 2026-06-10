"""Runs page: provenance browser over every manifest-bearing run directory in ``outputs/``."""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.jobs import open_in_file_manager
from vesp.ui.paths import OUTPUTS_DIR, RunRecord, scan_runs
from vesp.ui.widgets import Card, InfoGrid, PageHeader, StatusChip, make_button

COLUMNS = ("created (UTC)", "kind", "run", "artifacts", "inputs")


class RunsPage(QWidget):
    """Table of runs + manifest detail; every result traceable to exact input bytes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[RunRecord] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        header = PageHeader(
            "Runs & provenance",
            f"Every manifest-backed run under `{OUTPUTS_DIR}` -- outputs and consumed inputs are "
            "SHA-256 checksummed, so results trace to exact bytes.",
        )
        self.count_chip = StatusChip("0 runs")
        header.actions.addWidget(self.count_chip)
        header.add_action(make_button("Refresh", variant="ghost", on_click=self.refresh))
        root.addWidget(header)

        split = QSplitter()
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        table_card = Card("Run directories")
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.doubleClicked.connect(lambda _i: self._open_selected())
        table_card.add(self.table)
        split.addWidget(table_card)

        detail_card = Card("Manifest")
        self.detail_grid = InfoGrid()
        detail_card.add(self.detail_grid)
        button_row = QHBoxLayout()
        self.open_button = make_button("Open folder", variant="ghost", on_click=self._open_selected)
        self.open_button.setEnabled(False)
        button_row.addWidget(self.open_button)
        button_row.addStretch(1)
        detail_card.add_layout(button_row)
        self.manifest_view = QPlainTextEdit()
        self.manifest_view.setReadOnly(True)
        detail_card.add(self.manifest_view)
        split.addWidget(detail_card)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        self._records = scan_runs()
        self.count_chip.set_state(f"{len(self._records)} runs", "accent" if self._records else "neutral")
        self.table.setRowCount(0)
        for record in self._records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = (
                record.created_at,
                record.kind,
                record.name,
                str(len(record.artifacts)),
                str(len(record.inputs)),
            )
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.manifest_view.clear()
        self.detail_grid.clear()
        self.open_button.setEnabled(False)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()

    def _selected_record(self) -> RunRecord | None:
        row = self.table.currentRow()
        if 0 <= row < len(self._records):
            return self._records[row]
        return None

    def _on_select(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        self.open_button.setEnabled(True)
        metrics = ", ".join(f"{k}={v}" for k, v in record.metrics.items() if not isinstance(v, dict)) or "--"
        self.detail_grid.set_mapping(
            {
                "run dir": str(record.run_dir),
                "created": record.created_at or "--",
                "kind": record.kind,
                "metrics": metrics,
                "artifacts": ", ".join(sorted(record.artifacts)) or "--",
                "inputs": ", ".join(sorted(record.inputs)) or "--",
            }
        )
        try:
            manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
            self.manifest_view.setPlainText(json.dumps(manifest, indent=2, sort_keys=True))
        except (OSError, json.JSONDecodeError) as exc:
            self.manifest_view.setPlainText(f"failed to read manifest: {exc}")

    def _open_selected(self) -> None:
        record = self._selected_record()
        if record is not None:
            open_in_file_manager(record.run_dir)
