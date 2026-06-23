"""Multi-file analysis dialog — one coloured series per compound.

Article-style comparison of the physical results (kF, vF, m*, Γ₀) across the
fitted entries. Points are **grouped by compound** (sample folder) so each
compound is its own coloured line, instead of a single blurred series. The
x-axis is selectable (temperature, hν, polarisation, dopant), which covers both
"several compounds at one T" (x = dopant) and "one compound vs T" (x = T).
"""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from arpes.analysis.aggregation import aggregate_session_entries
from arpes.core.session import Session
from arpes.ui.widgets.canvas import MplCanvas

_PANELS = [
    (r"$k_F$ (π/a)", "kF", "kF_sigma"),
    (r"$v_F$ (eV·π/a)", "vF", "vF_sigma"),
    (r"$m^*/m_e$", "m_star", "m_star_sigma"),
    (r"$\Gamma_0$ (π/a)", "gamma_zero", "gamma_zero_sigma"),
]
_X_ITEMS = ["T (K)", "hν", "polarisation", "dopant"]


def _frame_y_on_values(ax, values) -> None:
    """Set y-limits from the data values only (error bars clipped at the edge)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return
    lo, hi = float(v.min()), float(v.max())
    if hi > lo:
        pad = 0.18 * (hi - lo)
    else:
        pad = max(abs(hi) * 0.1, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)


class MultiFileAnalysisDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Multi-file Analysis")
        self.resize(980, 720)
        self._build()
        self._populate_files()

    # ── UI ────────────────────────────────────────────────────────────────
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
        filters.addWidget(QLabel("X axis"), 2, 0)
        self._cmb_x = QComboBox()
        self._cmb_x.addItems(_X_ITEMS)
        self._cmb_x.setToolTip(
            "Compare several compounds at one T → set filters then X = dopant.\n"
            "Follow one compound vs T → X = T. Each compound is its own line.")
        filters.addWidget(self._cmb_x, 2, 1)
        self._cmb_x.currentTextChanged.connect(self._plot)
        self._btn_plot = QPushButton("Plot")
        self._btn_plot.clicked.connect(self._plot)
        filters.addWidget(self._btn_plot, 2, 3)
        root.addLayout(filters)
        for cmb in (self._cmb_direction, self._cmb_pol, self._cmb_hv, self._cmb_temp):
            cmb.currentTextChanged.connect(self._apply_filters)

        mid = QHBoxLayout()
        self._list = QListWidget()
        self._list.setMaximumWidth(220)
        self._list.itemChanged.connect(lambda *_: self._plot())
        mid.addWidget(self._list, stretch=0)
        self._canvas = MplCanvas(figsize=(8, 6), toolbar=True, nrows=4)
        self._canvas.fig.clear()
        self._canvas.axes = list(self._canvas.fig.subplots(2, 2).ravel())
        self._canvas.ax = self._canvas.axes[0]
        mid.addWidget(self._canvas, stretch=3)
        root.addLayout(mid, stretch=1)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color:#333;font-size:10px;")
        root.addWidget(self._lbl_status)

    # ── population / filters ───────────────────────────────────────────────
    def _populate_files(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        self._populate_filter_combos()
        for name, entry in self._session.files.items():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if entry.fit_result else Qt.CheckState.Unchecked)
            if not entry.fit_result:
                item.setToolTip("Ignored: no fit_result.")
            else:
                item.setToolTip(
                    f"T={entry.meta.temperature:g} K, hν={entry.meta.hv:g}, "
                    f"dir={entry.meta.direction}, pol={entry.meta.polarization}")
            self._list.addItem(item)
        self._list.blockSignals(False)
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
            cmb.addItem("All")
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
        return all(sel in ("All", "") or sel == value for sel, value in checks)

    def _apply_filters(self) -> None:
        shown = 0
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            match = self._entry_matches_filters(item.text())
            item.setHidden(not match)
            if match:
                shown += 1
        self._list.blockSignals(False)
        self._lbl_status.setText(f"{shown} file(s) shown by the filters.")
        self._plot()

    def _selected_names(self) -> list[str]:
        out = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden() and item.checkState() == Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _x_axis_key(self) -> str:
        return self._cmb_x.currentText()

    # ── plotting ────────────────────────────────────────────────────────────
    def _plot(self) -> None:
        names = self._selected_names()
        series = aggregate_session_entries(
            self._session, names, x_axis=self._x_axis_key(),
            direction_filter="" if self._cmb_direction.currentText() in ("All", "")
            else self._cmb_direction.currentText(),
        )
        self._draw_series(series)
        n = len(series.points)
        n_comp = len({p.compound for p in series.points})
        self._lbl_status.setText(
            f"{n} point(s) · {n_comp} compound(s) · {series.skipped} skipped. "
            f"{series.warning}".strip())

    def _draw_series(self, series) -> None:
        axes = self._canvas.axes
        for ax in axes:
            ax.cla()
            ax.set_facecolor("white")
        self._canvas.fig.set_facecolor("white")

        groups: "OrderedDict[str, list]" = OrderedDict()
        for p in series.points:
            groups.setdefault(p.compound or p.filename, []).append(p)
        cmap = plt.get_cmap("tab10")
        colours = {c: cmap(i % 10) for i, c in enumerate(groups)}

        x_key = self._x_axis_key()
        categorical = x_key in ("polarisation", "dopant")
        tick_pos: dict[float, str] = {}

        for ax, (ylabel, attr, sattr) in zip(axes, _PANELS):
            panel_y: list[float] = []
            for comp, pts in groups.items():
                pts_s = sorted(pts, key=lambda p: p.x_value)
                x = np.asarray([p.x_value for p in pts_s], dtype=float)
                y = np.asarray([getattr(p, attr) for p in pts_s], dtype=float)
                e = np.asarray([getattr(p, sattr) for p in pts_s], dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if not valid.any():
                    continue
                yerr = np.where(np.isfinite(e) & (e > 0), e, np.nan)
                ax.errorbar(x[valid], y[valid], yerr=yerr[valid], fmt="o-",
                            color=colours[comp], ecolor=colours[comp], lw=1.6,
                            ms=5, capsize=2, elinewidth=1.0, label=comp)
                panel_y.extend(y[valid].tolist())
                for p in pts_s:
                    tick_pos[float(p.x_value)] = p.x_label
            # Frame the panel on the VALUES, not the error bars: an ill-conditioned
            # σ (e.g. C05 m* ±46 %) then no longer blows the scale — its bar is
            # simply clipped at the axis edge while every point stays visible.
            _frame_y_on_values(ax, panel_y)
            ax.set_ylabel(ylabel, color="black", fontsize=11)
            ax.set_title(ylabel, fontsize=10, color="black")
            ax.grid(True, color="#d8d8d8", lw=0.6, alpha=0.9)
            ax.tick_params(colors="black", labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")

        xlabel = {"T (K)": "Temperature (K)", "hν": "hν (eV)",
                  "polarisation": "Polarisation", "dopant": "Compound"}.get(x_key, x_key)
        for ax in axes[-2:]:
            ax.set_xlabel(xlabel, color="black", fontsize=11)
        if categorical and tick_pos:
            xs = sorted(tick_pos)
            for ax in axes:
                ax.set_xticks(xs)
                ax.set_xticklabels([tick_pos[v] for v in xs], rotation=25,
                                   ha="right", fontsize=8)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            ncol = min(len(labels), 5)
            self._canvas.fig.legend(handles, labels, loc="upper center",
                                    ncol=ncol, fontsize=9, frameon=True,
                                    facecolor="white", edgecolor="#bbb")
            self._canvas.fig.tight_layout(rect=(0, 0, 1, 0.93))
        else:
            self._canvas.fig.tight_layout()
        self._canvas.redraw()
