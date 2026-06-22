"""Multi-file analysis dialog for kF, m*, and Gamma0."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from arpes.analysis.aggregation import MultiFileSeries, aggregate_session_entries
from arpes.core.session import Session
from arpes.ui.widgets.canvas import MplCanvas


class MultiFileAnalysisDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Multi-file Analysis")
        self.resize(920, 680)
        self._build()
        self._populate_files()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        filters = QGridLayout()
        filters.addWidget(QLabel("Direction"), 0, 0)
        self._cmb_direction = QComboBox()
        filters.addWidget(self._cmb_direction, 0, 1)
        filters.addWidget(QLabel("Polarisation"), 0, 2)
        self._cmb_pol = QComboBox()
        filters.addWidget(self._cmb_pol, 0, 3)
        filters.addWidget(QLabel("hν"), 1, 0)
        self._cmb_hv = QComboBox()
        filters.addWidget(self._cmb_hv, 1, 1)
        filters.addWidget(QLabel("T"), 1, 2)
        self._cmb_temp = QComboBox()
        filters.addWidget(self._cmb_temp, 1, 3)
        filters.addWidget(QLabel("Axe X"), 2, 0)
        self._cmb_x = QComboBox()
        self._cmb_x.addItems(["T (K)", "hν", "polarisation"])
        filters.addWidget(self._cmb_x, 2, 1)
        self._btn_plot = QPushButton("Plot")
        self._btn_plot.clicked.connect(self._plot)
        filters.addWidget(self._btn_plot, 2, 3)
        root.addLayout(filters)
        for cmb in (self._cmb_direction, self._cmb_pol, self._cmb_hv, self._cmb_temp):
            cmb.currentTextChanged.connect(self._apply_filters)

        mid = QHBoxLayout()
        self._list = QListWidget()
        mid.addWidget(self._list, stretch=1)
        self._canvas = MplCanvas(figsize=(8, 6), toolbar=True, nrows=4)
        self._canvas.fig.clear()
        self._canvas.axes = list(self._canvas.fig.subplots(2, 2).ravel())
        self._canvas.ax = self._canvas.axes[0]
        mid.addWidget(self._canvas, stretch=3)
        root.addLayout(mid, stretch=1)

        anim_row = QHBoxLayout()
        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.toggled.connect(self._on_play_toggled)
        anim_row.addWidget(self._btn_play)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        anim_row.addWidget(self._slider, stretch=1)
        anim_row.addWidget(QLabel("speed (ms):"))
        self._cmb_speed = QComboBox()
        self._cmb_speed.addItems(["300", "600", "1000", "2000"])
        self._cmb_speed.setCurrentText("1000")
        self._cmb_speed.currentTextChanged.connect(self._on_speed_changed)
        anim_row.addWidget(self._cmb_speed)
        root.addLayout(anim_row)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._step_animation)
        self._highlight_artists: list = []
        self._series = None

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color:#333;font-size:10px;")
        root.addWidget(self._lbl_status)

    def _populate_files(self) -> None:
        self._list.clear()
        self._populate_filter_combos()
        for name, entry in self._session.files.items():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if entry.fit_result else Qt.CheckState.Unchecked)
            if not entry.fit_result:
                item.setToolTip("Ignoré: aucun fit_result.")
            else:
                item.setToolTip(
                    f"T={entry.meta.temperature:g} K, hν={entry.meta.hv:g}, "
                    f"dir={entry.meta.direction}, pol={entry.meta.polarization}"
                )
            self._list.addItem(item)
        self._apply_filters()

    def _populate_filter_combos(self) -> None:
        directions = sorted({str(e.meta.direction or "").strip() for e in self._session.files.values()
                             if e.fit_result and str(e.meta.direction or "").strip()})
        pols = sorted({str(e.meta.polarization or "").strip() for e in self._session.files.values()
                       if e.fit_result and str(e.meta.polarization or "").strip()})
        hvs = sorted({float(e.meta.hv) for e in self._session.files.values()
                      if e.fit_result and np.isfinite(float(e.meta.hv or np.nan))})
        temps = sorted({float(e.meta.temperature) for e in self._session.files.values()
                        if e.fit_result and np.isfinite(float(e.meta.temperature or np.nan))})
        specs = [
            (self._cmb_direction, directions, lambda v: v),
            (self._cmb_pol, pols, lambda v: v),
            (self._cmb_hv, hvs, lambda v: f"{v:g}"),
            (self._cmb_temp, temps, lambda v: f"{v:g}"),
        ]
        for cmb, values, fmt in specs:
            cmb.blockSignals(True)
            current = cmb.currentText()
            cmb.clear()
            cmb.addItem("Tous")
            for value in values:
                cmb.addItem(fmt(value))
            if current:
                idx = cmb.findText(current)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)
            cmb.blockSignals(False)

    def _entry_matches_filters(self, name: str) -> bool:
        entry = self._session.files.get(name)
        if entry is None or not entry.fit_result:
            return False
        meta = entry.meta
        checks = [
            (self._cmb_direction.currentText(), str(meta.direction or "").strip()),
            (self._cmb_pol.currentText(), str(meta.polarization or "").strip()),
            (self._cmb_hv.currentText(), f"{float(meta.hv or np.nan):g}"),
            (self._cmb_temp.currentText(), f"{float(meta.temperature or np.nan):g}"),
        ]
        return all(sel == "Tous" or sel == value for sel, value in checks)

    def _apply_filters(self) -> None:
        shown = 0
        for i in range(self._list.count()):
            item = self._list.item(i)
            match = self._entry_matches_filters(item.text())
            item.setHidden(not match)
            if match:
                shown += 1
        self._lbl_status.setText(f"{shown} fichier(s) affichés par les filtres.")

    def _selected_names(self) -> list[str]:
        out = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden() and item.checkState() == Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _plot(self) -> None:
        series = aggregate_session_entries(
            self._session,
            self._selected_names(),
            x_axis=self._x_axis_key(),
            direction_filter="" if self._cmb_direction.currentText() == "Tous" else self._cmb_direction.currentText(),
        )
        self._series = series
        self._highlight_artists = []
        self._draw_series(series)
        n = len(series.points)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.setEnabled(n > 1)
        self._slider.blockSignals(False)
        self._btn_play.setEnabled(n > 1)
        if self._btn_play.isChecked():
            self._btn_play.setChecked(False)
        self._lbl_status.setText(
            f"{n} point(s), {series.skipped} ignoré(s). {series.warning}".strip()
        )

    def _on_play_toggled(self, checked: bool) -> None:
        if checked and self._series and len(self._series.points) > 1:
            self._timer.start()
            self._btn_play.setText("■ Stop")
        else:
            self._timer.stop()
            self._btn_play.setText("▶ Play")

    def _on_speed_changed(self, text: str) -> None:
        try:
            self._timer.setInterval(int(text))
        except (TypeError, ValueError):
            pass

    def _step_animation(self) -> None:
        if self._series is None:
            return
        n = len(self._series.points)
        if n == 0:
            return
        next_val = (self._slider.value() + 1) % n
        self._slider.setValue(next_val)

    def _on_slider_changed(self, idx: int) -> None:
        if self._series is None:
            return
        for art in self._highlight_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._highlight_artists = []
        if not (0 <= idx < len(self._series.points)):
            return
        point = self._series.points[idx]
        x = float(point.x_value)
        ys = (point.kF, point.vF, point.m_star, point.gamma_zero)
        for ax, y in zip(self._canvas.axes, ys):
            try:
                if y == y:  # not NaN
                    art = ax.scatter([x], [y], s=120, facecolor="none",
                                     edgecolor="#fcd34d", lw=2.0, zorder=10)
                    self._highlight_artists.append(art)
            except Exception:
                pass
        self._lbl_status.setText(
            f"Point {idx + 1}/{len(self._series.points)} "
            f"({self._cmb_x.currentText()} = {x:g})"
        )
        self._canvas.redraw()

    def _draw_series(self, series: MultiFileSeries) -> None:
        axes = self._canvas.axes
        for ax in axes:
            ax.cla()
            ax.set_facecolor("white")
        self._canvas.fig.set_facecolor("white")
        x = np.asarray([p.x_value for p in series.points], dtype=float)
        labels = [p.x_label for p in series.points]
        panels = [
            ("kF (π/a)", [p.kF for p in series.points], [p.kF_sigma for p in series.points]),
            ("vF (eV·π/a)", [p.vF for p in series.points], [p.vF_sigma for p in series.points]),
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
                            fmt="o-", color="#1f77b4", ecolor="#6aaed6",
                            lw=1.2, ms=4, capsize=2)
            ax.set_ylabel(ylabel, color="black")
            ax.grid(True, color="#d0d0d0", lw=0.6, alpha=0.85)
            ax.tick_params(colors="black", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("black")
            ax.set_title(ylabel, fontsize=9, color="black")
        for ax in axes[-2:]:
            ax.set_xlabel(self._cmb_x.currentText(), color="black")
        if self._x_axis_key() == "polarisation" and labels:
            for ax in axes[-2:]:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=20, ha="right")
        self._canvas.fig.tight_layout(pad=0.6)
        self._canvas.redraw()

    def _x_axis_key(self) -> str:
        txt = self._cmb_x.currentText()
        return "polarisation" if txt == "polarisation" else txt
