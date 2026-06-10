"""Reusable building blocks for the Mission Console pages (cards, tiles, chips, pickers)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addLayout(self.actions)

    def add_action(self, button: QPushButton) -> QPushButton:
        self.actions.addWidget(button)
        return button


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
