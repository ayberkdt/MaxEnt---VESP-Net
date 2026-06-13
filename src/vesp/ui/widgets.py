"""Reusable building blocks for the Mission Console pages (cards, tiles, chips, pickers)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.jobs import open_file, open_in_file_manager


class Card(QFrame):
    """Elevated rounded container; the visual unit every page is composed of."""

    def __init__(self, title: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", "true")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(10)
        if title:
            label = QLabel(title.upper())
            label.setObjectName("SectionTitle")
            self._layout.addWidget(label)

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)


class KpiTile(Card):
    """One headline number with a label and an optional hint line."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._layout.setSpacing(2)
        self.label = QLabel(label.upper())
        self.label.setObjectName("KpiLabel")
        self.value = QLabel("--")
        self.value.setObjectName("KpiValue")
        self.hint = QLabel("")
        self.hint.setObjectName("KpiHint")
        self.hint.setWordWrap(True)
        for w in (self.label, self.value, self.hint):
            self._layout.addWidget(w)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set(self, value: str, hint: str = "") -> None:
        self.value.setText(value)
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))


class StatusChip(QLabel):
    """Small semantic state pill: neutral / accent / ok / warn / danger."""

    def __init__(self, text: str = "idle", state: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.set_state(text, state)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_state(self, text: str, state: str = "neutral") -> None:
        self.setText(text)
        self.setProperty("chip", state)
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)


class PageHeader(QWidget):
    """Page title + subtitle row with an optional right-aligned action area."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        text = QVBoxLayout()
        text.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("PageTitle")
        s = QLabel(subtitle)
        s.setObjectName("PageSubtitle")
        s.setWordWrap(True)
        text.addWidget(t)
        text.addWidget(s)
        row.addLayout(text, 1)
        self.action_layout = QHBoxLayout()
        self.action_layout.setSpacing(8)
        row.addLayout(self.action_layout)

    def add_action(self, widget: QWidget) -> QWidget:
        self.action_layout.addWidget(widget)
        return widget


class PathPicker(QWidget):
    """Line edit + browse button for a file or directory path."""

    def __init__(
        self,
        placeholder: str = "",
        *,
        mode: str = "file",  # file | save | dir
        name_filter: str = "All files (*)",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._filter = name_filter
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse")
        browse.setProperty("variant", "ghost")
        browse.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(browse)

    def _browse(self) -> None:
        start = self.edit.text() or str(Path.home())
        if self._mode == "dir":
            chosen = QFileDialog.getExistingDirectory(self, "Select folder", start)
        elif self._mode == "save":
            chosen, _ = QFileDialog.getSaveFileName(self, "Select output file", start, self._filter)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "Select file", start, self._filter)
        if chosen:
            self.edit.setText(chosen)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None

    def set_path(self, path: str | Path) -> None:
        self.edit.setText(str(path))


class ModelArtifactPicker(QWidget):
    """Saved-model combo with a custom-path fallback."""

    def __init__(
        self,
        placeholder: str = "custom vespuq_plugin.pt",
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.combo = QComboBox()
        self.custom_picker = PathPicker(placeholder, name_filter="VESP-UQ model (*.pt)")
        self.combo.currentIndexChanged.connect(self._sync_custom_visibility)
        layout.addWidget(self.combo)
        layout.addWidget(self.custom_picker)

    def refresh(self, models: list[Path], *, default_index: int = 0) -> None:
        """Refresh discovered artifacts while preserving the current selection."""

        current = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        for path in models:
            self.combo.addItem(f"{path.parent.name}/{path.name}", str(path))
        self.combo.addItem("Browse...", "")
        if current:
            index = self.combo.findData(current)
            if index >= 0:
                self.combo.setCurrentIndex(index)
        elif 0 <= default_index < len(models):
            self.combo.setCurrentIndex(default_index)
        self.combo.blockSignals(False)
        self._sync_custom_visibility()

    def selected_path(self) -> Path | None:
        data = self.combo.currentData()
        return Path(data) if data else self.custom_picker.path()

    def _sync_custom_visibility(self, _index: int | None = None) -> None:
        self.custom_picker.setVisible(self.combo.currentData() == "")


class RunOutputActions(QWidget):
    """Open-folder/open-report actions with one shared output target."""

    def __init__(
        self,
        *,
        report_name: str | None = None,
        folder_label: str = "Open run folder",
        report_label: str = "Open report",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._directory: Path | None = None
        self._report_name = report_name
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.open_dir = make_button(folder_label, variant="ghost", on_click=self._open_dir)
        row.addWidget(self.open_dir)
        self.open_report: QPushButton | None = None
        if report_name is not None:
            self.open_report = make_button(report_label, variant="ghost", on_click=self._open_report)
            row.addWidget(self.open_report)
        row.addStretch(1)
        self.set_actions_enabled(False)

    def set_output(self, directory: Path | None, *, report_name: str | None = None) -> None:
        self._directory = directory
        if report_name is not None:
            self._report_name = report_name

    def set_actions_enabled(self, enabled: bool) -> None:
        self.open_dir.setEnabled(enabled)
        if self.open_report is not None:
            self.open_report.setEnabled(enabled)

    def _open_dir(self) -> None:
        if self._directory is not None and self._directory.exists():
            open_in_file_manager(self._directory)

    def _open_report(self) -> None:
        if self._directory is None or self._report_name is None:
            return
        report_path = self._directory / self._report_name
        if report_path.is_file():
            open_file(report_path)


class LogConsole(QPlainTextEdit):
    """Read-only streaming log with a bounded scrollback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.setPlaceholderText("Job output will stream here.")

    def append_line(self, line: str) -> None:
        self.appendPlainText(line)
        scrollbar = self.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())


class InfoGrid(QWidget):
    """Two-column key/value grid used for fit info, policy, provenance panels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(18)
        self._grid.setVerticalSpacing(5)
        self._row = 0

    def clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._row = 0

    def add_row(self, key: str, value: str) -> None:
        k = QLabel(key)
        k.setObjectName("KpiLabel")
        v = QLabel(value if value else "--")
        v.setWordWrap(True)
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._grid.addWidget(k, self._row, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self._grid.addWidget(v, self._row, 1)
        self._row += 1

    def set_mapping(self, mapping: dict[str, str]) -> None:
        self.clear()
        for key, value in mapping.items():
            self.add_row(key, value)


def make_button(
    text: str,
    *,
    variant: str | None = None,
    on_click: Callable[[], None] | None = None,
) -> QPushButton:
    button = QPushButton(text)
    if variant:
        button.setProperty("variant", variant)
    if on_click is not None:
        button.clicked.connect(on_click)
    return button


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #1c2532; background: #1c2532; max-height: 1px; border: none;")
    return line
