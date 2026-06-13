"""Compare page: run ``scripts.compare_models`` and inspect model drift."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.helpers import fmt, safe_read_json
from vesp.ui.jobs import ProcessJob
from vesp.ui.paths import OUTPUTS_DIR, list_models
from vesp.ui.widgets import (
    Card,
    InfoGrid,
    KpiTile,
    LogConsole,
    ModelArtifactPicker,
    PageHeader,
    PathPicker,
    RunOutputActions,
    StatusChip,
    make_button,
)

CAL_COLUMNS = ("band", "rmse A", "rmse B", "sigma A", "sigma B", "PICP90 A", "PICP90 B")


class ComparePage(QWidget):
    """Compare two persisted VESP-UQ layers on drift, calibration, and screening agreement."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.job = ProcessJob(self)
        self.job.line.connect(lambda line: self.console.append_line(line))
        self.job.finished.connect(self._on_finished)
        self._out_dir: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        header = PageHeader(
            "Compare models",
            "Side-by-side posterior drift, calibration deltas, and optional screening agreement for "
            "promotion or regression checks.",
        )
        self.status = StatusChip("idle")
        header.add_action(self.status)
        root.addWidget(header)

        split = QSplitter()
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        model_card = Card("Model artifacts")
        self.model_a_selector = ModelArtifactPicker("custom model A (*.pt)")
        self.model_b_selector = ModelArtifactPicker("custom model B (*.pt)")
        self.model_a_combo = self.model_a_selector.combo
        self.model_b_combo = self.model_b_selector.combo
        self.model_a_picker = self.model_a_selector.custom_picker
        self.model_b_picker = self.model_b_selector.custom_picker
        model_card.add(self.model_a_selector)
        model_card.add(self.model_b_selector)
        col.addWidget(model_card)

        input_card = Card("Evaluation inputs")
        self.data_picker = PathPicker("held-out calibration CSV", name_filter="CSV (*.csv)")
        self.trajectory_picker = PathPicker("optional trajectory CSV", name_filter="CSV (*.csv)")
        self.out_picker = PathPicker("output folder (optional)", mode="dir")
        input_card.add(self.data_picker)
        input_card.add(self.trajectory_picker)
        input_card.add(self.out_picker)
        col.addWidget(input_card)

        run_row = QHBoxLayout()
        self.run_button = make_button("Compare", variant="primary", on_click=self._run)
        self.cancel_button = make_button("Cancel", variant="danger", on_click=self.job.cancel)
        self.cancel_button.setEnabled(False)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_row.addStretch(1)
        col.addLayout(run_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        col.addWidget(self.progress)

        result_card = Card("Result")
        self.output_actions = RunOutputActions(report_name="model_comparison.md")
        self.open_dir = self.output_actions.open_dir
        self.open_report = self.output_actions.open_report
        result_card.add(self.output_actions)
        col.addWidget(result_card)
        col.addStretch(1)

        right = QWidget()
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(0, 0, 0, 0)
        rcol.setSpacing(12)

        drift_card = Card("Posterior and domain drift")
        self.drift_grid = InfoGrid()
        drift_card.add(self.drift_grid)
        rcol.addWidget(drift_card)

        agreement_card = Card("Screening agreement")
        agreement_tiles = QHBoxLayout()
        self.kpi_spearman = KpiTile("risk Spearman")
        self.kpi_iou = KpiTile("flag IoU")
        self.kpi_counts = KpiTile("flag counts")
        for tile in (self.kpi_spearman, self.kpi_iou, self.kpi_counts):
            agreement_tiles.addWidget(tile)
        agreement_card.add_layout(agreement_tiles)
        rcol.addWidget(agreement_card)

        cal_card = Card("Calibration comparison")
        self.cal_table = QTableWidget(0, len(CAL_COLUMNS))
        self.cal_table.setHorizontalHeaderLabels(CAL_COLUMNS)
        vertical_header = self.cal_table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.cal_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        horizontal_header = self.cal_table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setStretchLastSection(True)
        cal_card.add(self.cal_table)
        rcol.addWidget(cal_card, 1)

        log_card = Card("Live log")
        self.console = LogConsole()
        self.console.setMaximumHeight(150)
        log_card.add(self.console)
        rcol.addWidget(log_card)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

    def refresh_models(self) -> None:
        models = list_models()
        self.model_a_selector.refresh(models, default_index=0)
        self.model_b_selector.refresh(models, default_index=1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_models()

    def _run(self) -> None:
        model_a = self.model_a_selector.selected_path()
        model_b = self.model_b_selector.selected_path()
        if model_a is None or not model_a.is_file():
            self.status.set_state("pick model A", "warn")
            return
        if model_b is None or not model_b.is_file():
            self.status.set_state("pick model B", "warn")
            return
        data = self.data_picker.path()
        if data is None or not data.is_file():
            self.status.set_state("pick held-out CSV", "warn")
            return

        args = ["--model-a", str(model_a), "--model-b", str(model_b), "--data", str(data)]
        trajectory_csv = self.trajectory_picker.path()
        if trajectory_csv is not None:
            if not trajectory_csv.is_file():
                self.status.set_state("bad trajectory CSV", "warn")
                return
            args += ["--trajectories", str(trajectory_csv)]

        out_dir = self.out_picker.path()
        if out_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = OUTPUTS_DIR / f"ui_compare_{stamp}"
        self._out_dir = out_dir
        self.output_actions.set_output(out_dir)
        args += ["--out", str(out_dir)]

        self.console.clear()
        self.console.append_line("[ui] python -m scripts.compare_models " + " ".join(args))
        self.status.set_state("running", "accent")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.output_actions.set_actions_enabled(False)
        self.job.start_module("scripts.compare_models", args)

    def _on_finished(self, code: int) -> None:
        self.progress.setRange(0, 1)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if code != 0:
            self.status.set_state(f"failed (exit {code})", "danger")
            return
        self._load_results()

    def _load_results(self) -> None:
        if self._out_dir is None:
            return
        report_path = self._out_dir / "model_comparison.json"
        report, error = safe_read_json(report_path)
        if report is None:
            self.status.set_state("report missing", "danger")
            self.console.append_line(f"[ui] failed to read {report_path}: {error}")
            return

        self.status.set_state("completed", "ok")
        self.drift_grid.set_mapping(_drift_mapping(report))
        agreement = _agreement_summary(report.get("screening_agreement"))
        self.kpi_spearman.set(agreement["spearman"], agreement["spearman_hint"])
        self.kpi_iou.set(agreement["iou"], agreement["iou_hint"])
        self.kpi_counts.set(agreement["counts"], agreement["counts_hint"])
        self._fill_calibration(report.get("calibration", {}) or {})
        self.output_actions.set_actions_enabled(True)

    def _fill_calibration(self, calibration: dict) -> None:
        self.cal_table.setRowCount(0)
        for band in ("all", "low", "mid", "high"):
            metrics = calibration.get(band)
            if not isinstance(metrics, dict):
                continue
            row = self.cal_table.rowCount()
            self.cal_table.insertRow(row)
            values = (
                band,
                _ab_fmt(metrics, "rmse", "A"),
                _ab_fmt(metrics, "rmse", "B"),
                _ab_fmt(metrics, "mean_pred_std", "A"),
                _ab_fmt(metrics, "mean_pred_std", "B"),
                _ab_fmt(metrics, "picp_90", "A"),
                _ab_fmt(metrics, "picp_90", "B"),
            )
            for col, text in enumerate(values):
                self.cal_table.setItem(row, col, QTableWidgetItem(text))
        self.cal_table.resizeColumnsToContents()

def _drift_mapping(report: dict) -> dict[str, str]:
    post = report.get("posterior_distance", {}) or {}
    domain = report.get("domain_shift", {}) or {}
    return {
        "posterior mean L2": fmt(post.get("mean_l2_diff")),
        "posterior covariance Frobenius": fmt(post.get("cov_frob_diff")),
        "noise variance delta": fmt(post.get("noise_var_delta")),
        "domain score on A (mean)": fmt(domain.get("mean_score_on_A")),
        "domain score on A (max)": fmt(domain.get("max_score_on_A")),
    }


def _agreement_summary(agreement: dict | None) -> dict[str, str]:
    if not agreement:
        return {
            "spearman": "--",
            "spearman_hint": "provide a trajectory CSV",
            "iou": "--",
            "iou_hint": "screening comparison skipped",
            "counts": "--",
            "counts_hint": "no flags computed",
        }
    return {
        "spearman": fmt(agreement.get("risk_spearman")),
        "spearman_hint": "risk score rank agreement",
        "iou": fmt(agreement.get("flag_overlap")),
        "iou_hint": "intersection over union of flagged sets",
        "counts": f"{agreement.get('n_flagged_A', '--')} / {agreement.get('n_flagged_B', '--')}",
        "counts_hint": "model A / model B",
    }


def _ab_fmt(metrics: dict, metric_name: str, side: str) -> str:
    value = (metrics.get(metric_name) or {}).get(side)
    return fmt(value)
