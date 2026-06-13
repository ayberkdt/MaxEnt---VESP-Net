"""Model page: inspect a persisted layer -- provenance, packaged policy, card, and the
uncertainty-vs-altitude profile that is the layer's core value proposition.

Loading and probing run on a worker thread (lazy torch import); the matplotlib figure is drawn
on the GUI thread from precomputed arrays.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from vesp.ui.helpers import fmt
from vesp.ui.jobs import FnWorker, open_in_file_manager
from vesp.ui.paths import list_models
from vesp.ui.theme import TOKENS
from vesp.ui.widgets import Card, InfoGrid, ModelArtifactPicker, PageHeader, StatusChip, make_button


def _probe_model(path: Path) -> dict:
    """Worker-side: load the plugin and sample its uncertainty profile over altitude."""

    import torch

    from vesp.uq.plugin import VESPUQPlugin

    plugin = VESPUQPlugin.load(path)
    meta = plugin.user_metadata or {}

    radii = torch.linspace(1.03, 1.60, 24, dtype=plugin.dtype)
    generator = torch.Generator().manual_seed(0)
    dirs = torch.randn(48, 3, generator=generator, dtype=plugin.dtype)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    points = (radii.unsqueeze(1).unsqueeze(2) * dirs.unsqueeze(0)).reshape(-1, 3)
    pred = plugin.predict_uncertainty(points)
    sigma = pred.sigma.reshape(len(radii), -1)
    epistemic = pred.epistemic_sigma.reshape(len(radii), -1)

    card_path = path.parent / "vespuq_plugin_card.md"
    return {
        "path": str(path),
        "fit_info": dict(plugin.fit_info),
        "metadata": meta,
        "conformal_calibration": meta.get("conformal_prediction") or plugin.conformal_calibration,
        "card_markdown": card_path.read_text(encoding="utf-8") if card_path.is_file() else None,
        "profile": {
            "radius": radii.tolist(),
            "sigma_mean": sigma.mean(dim=1).tolist(),
            "sigma_p95": sigma.quantile(0.95, dim=1).tolist(),
            "epistemic_mean": epistemic.mean(dim=1).tolist(),
        },
    }


class ModelPage(QWidget):
    """Pick a saved artifact and surface everything packaged inside it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: FnWorker | None = None
        self._current: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        header = PageHeader(
            "Model inspector",
            "Everything a model artifact carries: fit provenance, the packaged decision policy, "
            "the model card, and the calibrated uncertainty profile.",
        )
        self.status = StatusChip("idle")
        header.add_action(self.status)
        root.addWidget(header)

        picker_card = Card("Artifact")
        row = QHBoxLayout()
        self.model_selector = ModelArtifactPicker()
        self.model_combo = self.model_selector.combo
        self.model_picker = self.model_selector.custom_picker
        row.addWidget(self.model_selector, 1)
        row.addWidget(make_button("Inspect", variant="primary", on_click=self._inspect))
        self.open_folder = make_button("Open folder", variant="ghost", on_click=self._open_folder)
        self.open_folder.setEnabled(False)
        row.addWidget(self.open_folder)
        picker_card.add_layout(row)
        root.addWidget(picker_card)

        split = QSplitter()
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # left: info panels
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)
        fit_card = Card("Fit")
        self.fit_grid = InfoGrid()
        fit_card.add(self.fit_grid)
        policy_card = Card("Packaged decision policy")
        self.policy_grid = InfoGrid()
        policy_card.add(self.policy_grid)
        prov_card = Card("Provenance")
        self.prov_grid = InfoGrid()
        prov_card.add(self.prov_grid)
        col.addWidget(fit_card)
        col.addWidget(policy_card)
        col.addWidget(prov_card)
        col.addStretch(1)

        # right: tabs (profile plot, model card)
        self.tabs = QTabWidget()
        self.plot_host = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_host)
        self.plot_layout.setContentsMargins(8, 8, 8, 8)
        self._canvas: QWidget | None = None
        self.tabs.addTab(self.plot_host, "Uncertainty profile")
        self.card_view = QTextBrowser()
        self.card_view.setOpenExternalLinks(True)
        self.tabs.addTab(self.card_view, "Model card")

        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

    # ------------------------------------------------------------------ data
    def refresh_models(self) -> None:
        self.model_selector.refresh(list_models())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_models()

    def _selected_model(self) -> Path | None:
        return self.model_selector.selected_path()

    def _inspect(self) -> None:
        path = self._selected_model()
        if path is None or not path.is_file():
            self.status.set_state("pick a model", "warn")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._current = path
        self.status.set_state("loading", "accent")
        self._worker = FnWorker(partial(_probe_model, path), self)
        self._worker.done.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, message: str) -> None:
        self.status.set_state("load failed", "danger")
        self.card_view.setPlainText(message)
        self.tabs.setCurrentWidget(self.card_view)

    def _on_loaded(self, payload: object) -> None:
        data: dict = payload  # type: ignore[assignment]
        self.status.set_state("loaded", "ok")
        self.open_folder.setEnabled(True)

        fit = data.get("fit_info", {})
        self.fit_grid.set_mapping(
            {
                "sources": str(fit.get("n_sources", "--")),
                "train / val": f"{fit.get('n_train', '--')} / {fit.get('n_val', '--')}",
                "regularization": f"{fit.get('reg_method', '--')} (lambda={fmt(fit.get('lambda_l2'))})",
                "noise model": str(fit.get("noise_model", "--")),
                "noise std": fmt(fit.get("noise_std")),
                "altitude law": (
                    f"a={fmt(fit.get('altitude_noise_a'))}, b={fmt(fit.get('altitude_noise_b'))}"
                    if fit.get("altitude_noise_b") is not None
                    else "--"
                ),
                "domain support": "on" if fit.get("domain_support_enabled") else "off",
                "sequential updates": str(fit.get("n_updates", 0)),
            }
        )
        metadata = data.get("metadata", {}) or {}
        conformal = data.get("conformal_calibration") or {}
        policy = metadata.get("decision_policy", {}) or {}
        self.policy_grid.set_mapping(
            {
                "scoring": f"{policy.get('scoring', '--')} ({policy.get('scoring_scale', '--')})",
                "threshold": (
                    f"{fmt(policy.get('threshold'))} (source {policy.get('threshold_source')})"
                    if policy.get("threshold") is not None
                    else "none (fraction mode)"
                ),
                "rerun fraction": fmt(policy.get("rerun_fraction")),
                "time weighting": str(policy.get("time_weighting", "--")),
                "conformal prediction": _conformal_summary(conformal),
            }
            if policy
            else {
                "policy": "none packaged (saved via plugin.save without run metadata)",
                "conformal prediction": _conformal_summary(conformal),
            }
        )
        provenance = metadata.get("provenance", {}) or {}
        self.prov_grid.set_mapping(
            {
                "created (UTC)": str(provenance.get("created_at_utc", "--")),
                "dataset": str(provenance.get("dataset", "--")),
                "dataset sha256": str(provenance.get("dataset_sha256") or "--")[:24],
                "state version": str(metadata.get("state_version", "--")),
                "file": data.get("path", "--"),
            }
        )

        card_md = data.get("card_markdown")
        if card_md:
            self.card_view.setMarkdown(card_md)
        else:
            self.card_view.setPlainText("No vespuq_plugin_card.md found next to the artifact.")
        self._draw_profile(data.get("profile", {}))

    # ------------------------------------------------------------------ plot
    def _draw_profile(self, profile: dict) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception:  # matplotlib missing/broken -> degrade gracefully
            return
        if self._canvas is not None:
            self.plot_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        if not profile:
            return

        fig = Figure(figsize=(5.2, 3.6), facecolor=TOKENS["card"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(TOKENS["surface"])
        radius = profile["radius"]
        ax.plot(radius, profile["sigma_mean"], color=TOKENS["accent"], lw=2.0, label="predictive sigma (mean)")
        ax.plot(radius, profile["sigma_p95"], color=TOKENS["accent"], lw=1.0, ls="--", alpha=0.7, label="sigma (p95 over directions)")
        ax.plot(radius, profile["epistemic_mean"], color=TOKENS["ok"], lw=1.6, label="epistemic component")
        ax.set_yscale("log")
        ax.set_xlabel("radius  [body radii]", color=TOKENS["text_muted"])
        ax.set_ylabel("force-error std  [model units]", color=TOKENS["text_muted"])
        ax.tick_params(colors=TOKENS["text_muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(TOKENS["border"])
        ax.grid(True, color=TOKENS["border_soft"], lw=0.6, alpha=0.6)
        legend = ax.legend(loc="upper right", fontsize=8, facecolor=TOKENS["card"], edgecolor=TOKENS["border"])
        for text in legend.get_texts():
            text.set_color(TOKENS["text"])
        fig.tight_layout()
        self._canvas = FigureCanvasQTAgg(fig)
        self.plot_layout.addWidget(self._canvas)

    def _open_folder(self) -> None:
        if self._current is not None:
            open_in_file_manager(self._current)


def _conformal_summary(conformal: dict | None) -> str:
    """Compact display string for the persisted operational conformal layer."""

    if not conformal or not conformal.get("enabled"):
        return "off"
    global_cal = conformal.get("global") or {}
    scale = fmt(global_cal.get("scale"), digits=3)
    mode = conformal.get("mode", "--")
    scope = conformal.get("scope", "--")
    return f"{scale} ({mode}, {scope})"
