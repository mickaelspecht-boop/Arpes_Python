"""Dialog d'analyse multi-fichier kF, m*, Gamma0."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from arpes.analysis.aggregation import MultiFileSeries, aggregate_session_entries
from arpes.core.session import Session
from arpes.ui.widgets.canvas import MplCanvas


class MultiFileAnalysisDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Analyse multi-fichier")
        self.resize(920, 680)
        self._build()
        self._populate_files()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Direction"))
        self._txt_direction = QLineEdit()
        self._txt_direction.setPlaceholderText("ex: Γ-M")
        top.addWidget(self._txt_direction, stretch=1)
        top.addWidget(QLabel("X"))
        self._cmb_x = QComboBox()
        self._cmb_x.addItems(["T (K)", "hν", "polarisation"])
        top.addWidget(self._cmb_x)
        self._btn_plot = QPushButton("Tracer")
        self._btn_plot.clicked.connect(self._plot)
        top.addWidget(self._btn_plot)
        root.addLayout(top)

        mid = QHBoxLayout()
        self._list = QListWidget()
        mid.addWidget(self._list, stretch=1)
        self._canvas = MplCanvas(figsize=(7, 6), toolbar=True, nrows=3)
        mid.addWidget(self._canvas, stretch=3)
        root.addLayout(mid, stretch=1)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color:#9fc;font-size:10px;")
        root.addWidget(self._lbl_status)

    def _populate_files(self) -> None:
        self._list.clear()
        for name, entry in self._session.files.items():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if entry.fit_result else Qt.CheckState.Unchecked)
            if not entry.fit_result:
                item.setToolTip("Ignoré: pas de fit_result.")
            else:
                item.setToolTip(f"T={entry.meta.temperature:g} K, hν={entry.meta.hv:g}, dir={entry.meta.direction}")
            self._list.addItem(item)

    def _selected_names(self) -> list[str]:
        out = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _plot(self) -> None:
        series = aggregate_session_entries(
            self._session,
            self._selected_names(),
            x_axis=self._cmb_x.currentText(),
            direction_filter=self._txt_direction.text().strip(),
        )
        self._draw_series(series)
        self._lbl_status.setText(
            f"{len(series.points)} point(s), {series.skipped} ignoré(s). {series.warning}".strip()
        )

    def _draw_series(self, series: MultiFileSeries) -> None:
        axes = self._canvas.axes
        for ax in axes:
            ax.cla()
            ax.set_facecolor("#1a1a1a")
        x = np.asarray([p.x_value for p in series.points], dtype=float)
        labels = [p.x_label for p in series.points]
        panels = [
            ("kF (π/a)", [p.kF for p in series.points], [p.kF_sigma for p in series.points]),
            ("m*/me", [p.m_star for p in series.points], [p.m_star_sigma for p in series.points]),
            ("Γ0 (π/a)", [p.gamma_zero for p in series.points], [p.gamma_zero_sigma for p in series.points]),
        ]
        for ax, (ylabel, values, errors) in zip(axes, panels):
            y = np.asarray(values, dtype=float)
            err = np.asarray(errors, dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.any():
                yerr = np.where(np.isfinite(err) & (err > 0), err, np.nan)
                ax.errorbar(x[valid], y[valid], yerr=yerr[valid],
                            fmt="o-", color="#38bdf8", ecolor="#93c5fd",
                            lw=1.0, ms=4, capsize=2)
            ax.set_ylabel(ylabel, color="w")
            ax.tick_params(colors="w", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
        axes[-1].set_xlabel(self._cmb_x.currentText(), color="w")
        if self._cmb_x.currentText() == "polarisation" and labels:
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(labels, rotation=20, ha="right")
        self._canvas.fig.tight_layout(pad=0.6)
        self._canvas.redraw()
