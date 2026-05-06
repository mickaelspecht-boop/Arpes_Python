"""Panneau résultats — table fits + carte couleur."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from arpes.core.session import Session
from arpes.io.export import result_rows, write_results_csv
from arpes.ui.widgets.canvas import MplCanvas


class ResultsPanel(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)

        # canvas dispersion
        self._canvas = MplCanvas(figsize=(6, 5))
        lay.addWidget(self._canvas, stretch=2)

        # droite : table + boutons
        right = QVBoxLayout()
        right.addWidget(QLabel("Résultats fittés"))

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["Fichier", "hν", "T (K)", "Dir.", "kF+ (π/a)", "xg (π/a)", "Γ brut", "Γ corr."])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setStyleSheet(
            "QTableWidget{background:#222;color:#ddd;font-size:10px;}"
            "QHeaderView::section{background:#333;color:#ddd;}")
        right.addWidget(self._table, stretch=1)

        btn_ref = QPushButton("🔄  Actualiser")
        btn_ref.clicked.connect(self.refresh)
        btn_csv = QPushButton("💾  Export CSV")
        btn_csv.clicked.connect(self._export_csv)
        btn_pdf = QPushButton("🖼  Export figure")
        btn_pdf.clicked.connect(self._export_fig)
        for b in (btn_ref, btn_csv, btn_pdf):
            right.addWidget(b)

        rw = QWidget(); rw.setLayout(right)
        rw.setMaximumWidth(350)
        lay.addWidget(rw, stretch=1)

    def refresh(self):
        self._table.setRowCount(0)
        ax = self._canvas.ax
        ax.cla(); ax.set_facecolor("#1a1a1a")
        self._canvas.fig.set_facecolor("#2b2b2b")

        colors = plt.cm.plasma(np.linspace(0.1, 0.9,
                                           max(1, len(self._session.files))))
        row = 0
        for ci, (name, entry) in enumerate(self._session.files.items()):
            if entry.fit_result is None:
                continue
            fr   = entry.fit_result
            meta = entry.meta
            ev_f = np.asarray(fr["e_fitted"])
            n    = entry.fit_params.n_pairs
            label = f"{name}  T={meta.temperature:.0f}K  {meta.direction}"
            c = colors[ci]

            for i in range(n):
                km = np.asarray(fr["kF_minus"][i]) if i < len(fr["kF_minus"]) else []
                kp = np.asarray(fr["kF_plus"][i])  if i < len(fr["kF_plus"])  else []
                ax.scatter(km, ev_f, s=8, color=c, marker="o", alpha=0.8,
                           label=label if i == 0 else "_")
                ax.scatter(kp, ev_f, s=8, color=c, marker="^", alpha=0.8)

            # Table row
            kf_ef = np.nan
            if len(fr["kF_plus"]) > 0:
                idx_ef = np.argmin(np.abs(ev_f))
                kf_arr = np.asarray(fr["kF_plus"][0])
                if len(kf_arr) > idx_ef:
                    kf_ef = kf_arr[idx_ef]
            xg_m = float(np.nanmean(fr.get("xg", [np.nan])))
            gamma_b = np.nan
            gamma_c = np.nan
            if fr.get("gamma_brut"):
                gamma_b = float(np.nanmedian(np.asarray(fr["gamma_brut"][0], dtype=float)))
            if fr.get("gamma_corrige"):
                gamma_c = float(np.nanmedian(np.asarray(fr["gamma_corrige"][0], dtype=float)))

            self._table.insertRow(row)
            for col, val in enumerate([
                name, f"{meta.hv:.0f}", f"{meta.temperature:.0f}",
                meta.direction, f"{kf_ef:.4f}", f"{xg_m:.4f}",
                f"{gamma_b:.4f}", f"{gamma_c:.4f}",
            ]):
                self._table.setItem(row, col, QTableWidgetItem(val))
            row += 1

        ax.axhline(0, color="cyan", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="w",    lw=0.5, ls="--", alpha=0.3)
        ax.set_xlabel("k// (π/a)", fontsize=10, color="w")
        ax.set_ylabel("E − EF (eV)", fontsize=10, color="w")
        ax.set_title("Dispersions kF — tous fichiers fittés", fontsize=10, color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        if row > 0:
            ax.legend(fontsize=8, facecolor="#333", labelcolor="w",
                      loc="upper right", markerscale=2)
        self._canvas.redraw()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(self._session.folder or Path.home()),
            "CSV (*.csv)")
        if not path:
            return
        rows = result_rows(self._session)
        if not rows:
            return
        write_results_csv(path, rows)

    def _export_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", str(self._session.folder or Path.home()),
            "PDF (*.pdf);;PNG (*.png)")
        if path:
            self._canvas.fig.savefig(path, dpi=200, bbox_inches="tight",
                                     facecolor=self._canvas.fig.get_facecolor())



