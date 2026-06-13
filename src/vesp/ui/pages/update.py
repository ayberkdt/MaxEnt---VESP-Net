"""Update page: exact sequential posterior update of a persisted layer with new error samples.

Wraps :meth:`vesp.uq.plugin.VESPUQPlugin.update_error` on a worker thread. The honesty rules
from ``docs/VESP_UQ_LIMITATIONS.md`` are surfaced in the page itself: the Tikhonov weight stays
fixed, and the noise/altitude law only recalibrates when fresh held-out data is supplied.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.helpers import fmt
from vesp.ui.jobs import FnWorker, open_in_file_manager
from vesp.ui.paths import list_models
from vesp.ui.widgets import (
    Card,
    InfoGrid,
    LogConsole,
    ModelArtifactPicker,
    PageHeader,
    PathPicker,
    StatusChip,
    make_button,
)


def _run_update(model_path: Path, update_csv: Path, val_csv: Path | None, out_path: Path) -> dict:
    """Worker-side: load -> update_error -> save; return a before/after summary."""

    from vesp.uq.data import load_uq_samples_from_csv
    from vesp.uq.plugin import VESPUQPlugin

    plugin = VESPUQPlugin.load(model_path)
    before = dict(plugin.fit_info)

    samples = load_uq_samples_from_csv(update_csv)
    kwargs: dict = {}
    if val_csv is not None:
        val = load_uq_samples_from_csv(val_csv)
        kwargs = {"val_positions": val.positions, "val_error": val.error}
    plugin.update_error(samples.positions, samples.error, **kwargs)

    plugin.save(
        out_path,
        extra_metadata={
            "last_update": {
                "update_csv": str(update_csv),
                "val_csv": str(val_csv) if val_csv is not None else None,
                "n_new_samples": int(samples.positions.shape[0]),
                "recalibrated": val_csv is not None,
            }
        },
    )
    return {"before": before, "after": dict(plugin.fit_info), "out_path": str(out_path)}


class UpdatePage(QWidget):
    """Load a model, feed it new reference samples, save the updated artifact."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: FnWorker | None = None
        self._out_path: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        header = PageHeader(
            "Sequential update",
            "Condition a persisted posterior on new error samples in closed form -- exactly "
            "equal to the batch refit on the concatenated data (same lambda and noise floor).",
        )
        self.status = StatusChip("idle")
        header.add_action(self.status)
        root.addWidget(header)

        warn = QLabel(
            "The L-curve is NOT re-run, and the noise floor / altitude law only recalibrate when "
            "a fresh validation CSV is supplied. After large updates without one, re-validate "
            "calibration before relying on per-band coverage (docs/VESP_UQ_LIMITATIONS.md)."
        )
        warn.setProperty("chip", "warn")
        warn.setWordWrap(True)
        root.addWidget(warn)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        form_card = Card("Inputs")
        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        self.model_selector = ModelArtifactPicker()
        self.model_combo = self.model_selector.combo
        self.model_picker = self.model_selector.custom_picker
        form.addRow("Model", self.model_selector)
        self.update_csv = PathPicker(
            "new samples: x,y,z + ax_err/.. or ax_ref/..+ax_sur/.. columns", name_filter="CSV (*.csv)"
        )
        form.addRow("Update CSV", self.update_csv)
        self.val_csv = PathPicker("optional fresh held-out CSV (recalibrates noise law)", name_filter="CSV (*.csv)")
        form.addRow("Validation CSV", self.val_csv)
        self.out_picker = PathPicker("output .pt (default: <model>_updated.pt)", mode="save", name_filter="Model (*.pt)")
        form.addRow("Save as", self.out_picker)
        form_card.add_layout(form)

        run_row = QHBoxLayout()
        self.run_button = make_button("Apply update", variant="primary", on_click=self._run)
        run_row.addWidget(self.run_button)
        self.open_button = make_button("Open output folder", variant="ghost", on_click=self._open_out)
        self.open_button.setEnabled(False)
        run_row.addWidget(self.open_button)
        run_row.addStretch(1)
        form_card.add_layout(run_row)
        body.addWidget(form_card, 1)

        result_card = Card("Before -> after")
        self.summary = InfoGrid()
        result_card.add(self.summary)
        self.console = LogConsole()
        self.console.setMaximumHeight(160)
        result_card.add(self.console)
        body.addWidget(result_card, 1)

    # ------------------------------------------------------------------ helpers
    def refresh_models(self) -> None:
        self.model_selector.refresh(list_models())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_models()

    def _selected_model(self) -> Path | None:
        return self.model_selector.selected_path()

    # ------------------------------------------------------------------ run
    def _run(self) -> None:
        model = self._selected_model()
        update_csv = self.update_csv.path()
        if model is None or not model.is_file():
            self.status.set_state("pick a model", "warn")
            return
        if update_csv is None or not update_csv.is_file():
            self.status.set_state("pick an update CSV", "warn")
            return
        val_csv = self.val_csv.path()
        if val_csv is not None and not val_csv.is_file():
            self.status.set_state("validation CSV missing", "danger")
            return
        out_path = self.out_picker.path() or model.with_name(model.stem + "_updated.pt")
        self._out_path = out_path

        if self._worker is not None and self._worker.isRunning():
            return
        self.status.set_state("updating", "accent")
        self.run_button.setEnabled(False)
        self.console.append_line(f"[ui] update {model.name} with {update_csv.name}"
                                 + (f" + val {val_csv.name}" if val_csv else " (no recalibration)"))
        self._worker = FnWorker(partial(_run_update, model, update_csv, val_csv, out_path), self)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, message: str) -> None:
        self.status.set_state("update failed", "danger")
        self.run_button.setEnabled(True)
        for line in message.splitlines()[-12:]:
            self.console.append_line(line)

    def _on_done(self, payload: object) -> None:
        data: dict = payload  # type: ignore[assignment]
        before, after = data.get("before", {}), data.get("after", {})
        self.status.set_state("updated", "ok")
        self.run_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.summary.set_mapping(
            {
                "train samples": f"{before.get('n_train', '--')} -> {after.get('n_train', '--')}",
                "sequential updates": f"{before.get('n_updates', 0)} -> {after.get('n_updates', '--')}",
                "noise std": f"{fmt(before.get('noise_std'))} -> {fmt(after.get('noise_std'))}",
                "altitude law b": (
                    f"{fmt(before.get('altitude_noise_b'))} -> {fmt(after.get('altitude_noise_b'))}"
                ),
                "saved to": str(data.get("out_path", "--")),
            }
        )
        self.console.append_line(f"[ui] saved updated model: {data.get('out_path')}")

    def _open_out(self) -> None:
        if self._out_path is not None and self._out_path.exists():
            open_in_file_manager(self._out_path)
