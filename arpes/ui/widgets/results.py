"""Panneau résultats — table fits + carte couleur."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
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

from arpes.analysis.results import compute_results
from arpes.core.session import Session
from arpes.io.export import physics_rows, result_rows, write_physics_csv, write_results_csv
from arpes.ui.widgets.canvas import MplCanvas

DEFAULT_CRYSTAL_A_ANGSTROM = 4.143  # Fallback BaNi₂As₂ si meta.crystal_a_angstrom = 0.


class ResultsPanel(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)

        # canvas gauche : dispersion (top) + Γ(E) (bottom) empilés
        canvases = QVBoxLayout()
        self._canvas = MplCanvas(figsize=(6, 4))
        self._canvas_gamma = MplCanvas(figsize=(6, 3))
        canvases.addWidget(self._canvas, stretch=3)
        canvases.addWidget(self._canvas_gamma, stretch=2)
        cw = QWidget(); cw.setLayout(canvases)
        lay.addWidget(cw, stretch=2)

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

        self._chk_bootstrap = QCheckBox("Bootstrap σ (N=500, robuste outliers)")
        self._chk_bootstrap.setToolTip(
            "Remplace σ statistique propagée par σ bootstrap (rééchantillonnage\n"
            "des points fittés près de E_F). Plus robuste si points aberrants\n"
            "résiduels. ~1 s pour 4 branches × 500 itérations."
        )
        self._chk_bootstrap.toggled.connect(self.refresh)
        right.addWidget(self._chk_bootstrap)
        right.addWidget(QLabel("Résultats physiques ± σ (fit MDC stat.)"))
        self._table_phys = QTableWidget(0, 6)
        self._table_phys.setHorizontalHeaderLabels([
            "Fichier", "Paire/Branche",
            "kF (π/a) ± σ", "vF (eV·π/a) ± σ",
            "m*/me ± σ", "Γ₀ (π/a) ± σ",
        ])
        self._table_phys.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table_phys.setStyleSheet(
            "QTableWidget{background:#222;color:#ddd;font-size:10px;}"
            "QHeaderView::section{background:#333;color:#ddd;}")
        right.addWidget(self._table_phys, stretch=1)

        btn_ref = QPushButton("Actualiser")
        btn_ref.clicked.connect(self.refresh)
        btn_csv = QPushButton("Export CSV (par slice)")
        btn_csv.clicked.connect(self._export_csv)
        btn_csv_phys = QPushButton("Export CSV physique (± σ)")
        btn_csv_phys.clicked.connect(self._export_physics_csv)
        btn_pdf = QPushButton("Export figure")
        btn_pdf.clicked.connect(self._export_fig)
        for b in (btn_ref, btn_csv, btn_csv_phys, btn_pdf):
            right.addWidget(b)

        rw = QWidget(); rw.setLayout(right)
        rw.setMaximumWidth(350)
        lay.addWidget(rw, stretch=1)

    def refresh(self):
        self._table.setRowCount(0)
        self._table_phys.setRowCount(0)
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

            self._populate_physics_rows(name, fr, n, entry.meta)

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
        self._draw_gamma_panel(colors)

    def _populate_physics_rows(self, filename: str, fr: dict, n_pairs: int, meta=None) -> None:
        a_val = float(getattr(meta, "crystal_a_angstrom", 0.0) or 0.0)
        if a_val <= 0:
            a_val = DEFAULT_CRYSTAL_A_ANGSTROM
        bundle = compute_results(
            fr, e_window_kF=0.10, e_window_gamma=0.30,
            crystal_a_angstrom=a_val,
        )
        if self._chk_bootstrap.isChecked():
            from arpes.analysis.bootstrap import bootstrap_branch_result
            bs_branches = []
            for br in bundle.branches:
                bs_branches.append(bootstrap_branch_result(
                    fr, branch=br.branch, pair_index=br.pair_index,
                    e_window=0.10, crystal_a_angstrom=a_val, n_iter=500,
                ))
            branches = bs_branches
        else:
            branches = bundle.branches
        gamma_by_pair = {g.pair_index: g for g in bundle.gamma_fl}
        for br in branches:
            row = self._table_phys.rowCount()
            self._table_phys.insertRow(row)
            label = f"P{br.pair_index + 1} {br.branch.replace('kF_', '')}"
            kf = self._fmt(br.kF_at_EF, br.kF_at_EF_sigma, dec=4)
            vf = self._fmt(br.vF_eV_pi_a, br.vF_sigma, dec=2)
            mstar = self._fmt(br.m_star_over_me, br.m_star_sigma, dec=2)
            g_fl = gamma_by_pair.get(br.pair_index)
            g0 = self._fmt(g_fl.gamma_zero, g_fl.gamma_zero_sigma, dec=4) if g_fl else "—"
            for col, val in enumerate([filename, label, kf, vf, mstar, g0]):
                self._table_phys.setItem(row, col, QTableWidgetItem(val))

    def _draw_gamma_panel(self, colors) -> None:
        from arpes.analysis.results import fit_gamma_fermi_liquid
        ax = self._canvas_gamma.ax
        ax.cla(); ax.set_facecolor("#1a1a1a")
        self._canvas_gamma.fig.set_facecolor("#2b2b2b")
        plotted = 0
        for ci, (name, entry) in enumerate(self._session.files.items()):
            if entry.fit_result is None:
                continue
            fr = entry.fit_result
            ev = np.asarray(fr.get("e_fitted", []), dtype=float)
            g_arrays = fr.get("gamma_corrige") or fr.get("gamma") or []
            sg_arrays = fr.get("sigma_gamma") or []
            color = colors[ci]
            for i, g_raw in enumerate(g_arrays):
                g = np.asarray(g_raw, dtype=float)
                n = min(len(ev), len(g))
                if n == 0:
                    continue
                e_n, g_n = ev[:n], g[:n]
                valid = np.isfinite(e_n) & np.isfinite(g_n)
                if int(valid.sum()) < 3:
                    continue
                ax.plot(e_n[valid], g_n[valid], "o-", ms=3, lw=0.8, color=color,
                        alpha=0.85, label=f"{name} P{i+1}" if plotted < 6 else "_")
                if i < len(sg_arrays):
                    sg = np.asarray(sg_arrays[i], dtype=float)[:n]
                    band_valid = valid & np.isfinite(sg) & (sg > 0)
                    if band_valid.any():
                        ax.fill_between(e_n[band_valid],
                                        g_n[band_valid] - sg[band_valid],
                                        g_n[band_valid] + sg[band_valid],
                                        color=color, alpha=0.18, lw=0)
                fl = fit_gamma_fermi_liquid(fr, pair_index=i, e_window=0.30)
                if np.isfinite(fl.gamma_zero) and np.isfinite(fl.coef_E2):
                    e_grid = np.linspace(float(np.nanmin(e_n[valid])),
                                         float(np.nanmax(e_n[valid])), 80)
                    ax.plot(e_grid, fl.gamma_zero + fl.coef_E2 * e_grid ** 2,
                            "--", color=color, lw=1.0, alpha=0.7)
                plotted += 1
        ax.set_xlabel("E − EF (eV)", fontsize=10, color="w")
        ax.set_ylabel("Γ (π/a)", fontsize=10, color="w")
        ax.set_title("Γ(E) — bandes ±σ et fit Fermi liquide (Γ₀ + a·E²)",
                     fontsize=10, color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        if plotted > 0:
            ax.legend(fontsize=7, facecolor="#333", labelcolor="w",
                      loc="upper right", ncol=1)
        self._canvas_gamma.fig.tight_layout(pad=0.6)
        self._canvas_gamma.redraw()

    @staticmethod
    def _fmt(value: float, sigma: float, *, dec: int = 4) -> str:
        if not (np.isfinite(value) and np.isfinite(sigma)):
            return "—"
        return f"{value:.{dec}f} ± {sigma:.{dec}f}"

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

    def _export_physics_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV physique (± σ)",
            str(self._session.folder or Path.home()), "CSV (*.csv)",
        )
        if not path:
            return
        rows = physics_rows(self._session)
        if not rows:
            return
        write_physics_csv(path, rows)

    def _export_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", str(self._session.folder or Path.home()),
            "PDF (*.pdf);;PNG (*.png)")
        if path:
            self._canvas.fig.savefig(path, dpi=200, bbox_inches="tight",
                                     facecolor=self._canvas.fig.get_facecolor())

