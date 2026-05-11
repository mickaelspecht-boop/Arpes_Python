"""Panneau résultats — table fits + carte couleur."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from arpes.analysis.results import compute_results
from arpes.core.session import Session
from arpes.io.export import (
    physics_rows,
    physics_to_latex,
    result_rows,
    write_physics_csv,
    write_physics_txt,
    write_results_csv,
    write_results_txt,
)
from arpes.io.export_styles import PRESETS, savefig_with_preset
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
        right.addWidget(QLabel("Afficher fichiers fittés"))
        filter_btn_row = QHBoxLayout()
        btn_filter_all = QPushButton("Tout")
        btn_filter_all.setMaximumWidth(60)
        btn_filter_all.clicked.connect(lambda: self._set_all_filter(True))
        btn_filter_none = QPushButton("Aucun")
        btn_filter_none.setMaximumWidth(60)
        btn_filter_none.clicked.connect(lambda: self._set_all_filter(False))
        filter_btn_row.addWidget(btn_filter_all)
        filter_btn_row.addWidget(btn_filter_none)
        filter_btn_row.addStretch(1)
        right.addLayout(filter_btn_row)
        self._file_filter = QListWidget()
        self._file_filter.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._file_filter.setMaximumHeight(110)
        self._file_filter.setStyleSheet(
            "QListWidget{background:#222;color:#ddd;font-size:10px;}"
        )
        self._file_filter.itemChanged.connect(self._on_file_filter_changed)
        self._file_filter_unchecked: set[str] = set()
        right.addWidget(self._file_filter)

        right.addWidget(QLabel("Résultats fittés"))
        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(
            ["Fichier", "hν", "T (K)", "Dir.", "kF+ (π/a)", "xg (π/a)",
             "Γ brut", "Γ corr.", "chi2_red méd."])
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

        btn_ref = QPushButton("Actualiser tout")
        btn_ref.setToolTip("Redessine la dispersion + Γ(E) et recalcule les tables.")
        btn_ref.clicked.connect(self.refresh)
        btn_recalc = QPushButton("Recalculer résultats physiques")
        btn_recalc.setToolTip(
            "Recalcule uniquement la table 'Résultats physiques ± σ' "
            "et le panneau Γ(E). Utile après suppression de points."
        )
        btn_recalc.clicked.connect(self.refresh_physics_only)
        btn_multi = QPushButton("Analyse multi-fichier...")
        btn_multi.setToolTip("Trace kF, m* et Γ0 pour les entrées fittées sélectionnées.")
        btn_multi.clicked.connect(self._open_multi_file_analysis)
        btn_export = QPushButton("Exporter résultats…")
        btn_export.setToolTip(
            "Choisir le contenu (par slice ou physique ± σ) et le format\n"
            "(CSV, TXT aligné, LaTeX booktabs)."
        )
        btn_export.clicked.connect(self._export_results)
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style export"))
        self._cmb_export_style = QComboBox()
        self._cmb_export_style.addItems(list(PRESETS.keys()))
        self._cmb_export_style.setCurrentText("default")
        self._cmb_export_style.setToolTip(
            "Style matplotlib utilise pour l'export figure.\n"
            "PRB utilise LaTeX si disponible, sinon fallback sans LaTeX."
        )
        style_row.addWidget(self._cmb_export_style, stretch=1)
        right.addLayout(style_row)
        btn_pdf = QPushButton("Export figure")
        btn_pdf.clicked.connect(self._export_fig)
        for b in (btn_ref, btn_recalc, btn_multi, btn_export, btn_pdf):
            right.addWidget(b)

        rw = QWidget(); rw.setLayout(right)
        rw.setMaximumWidth(350)
        lay.addWidget(rw, stretch=1)

    def refresh(self):
        self._sync_file_filter()
        visible = self._visible_files()
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
            if name not in visible:
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
                self._plot_branch_segments(ax, km, ev_f, color=c, alpha=0.60)
                self._plot_branch_segments(ax, kp, ev_f, color=c, alpha=0.60)

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
            chi2_med = np.nan
            chi2 = np.asarray(fr.get("chi2_red", []), dtype=float)
            if chi2.size and np.isfinite(chi2).any():
                chi2_med = float(np.nanmedian(chi2))

            self._table.insertRow(row)
            for col, val in enumerate([
                name, f"{meta.hv:.0f}", f"{meta.temperature:.0f}",
                meta.direction, f"{kf_ef:.4f}", f"{xg_m:.4f}",
                f"{gamma_b:.4f}", f"{gamma_c:.4f}", f"{chi2_med:.3f}",
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
            leg = ax.legend(
                fontsize=8, facecolor="#333", labelcolor="w",
                loc="upper left", bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0, markerscale=2, frameon=True,
            )
            leg.set_draggable(True)
            self._canvas.fig.subplots_adjust(right=0.74)
        else:
            self._canvas.fig.subplots_adjust(right=0.97)
        self._canvas.redraw()
        self._draw_gamma_panel(colors)

    @staticmethod
    def _plot_branch_segments(ax, k_values, e_values, *, color, alpha=0.6) -> None:
        """Relie une branche kF(E) sans franchir NaN ni saut manifeste."""
        k = np.asarray(k_values, dtype=float)
        e = np.asarray(e_values, dtype=float)
        n = min(k.size, e.size)
        if n < 2:
            return
        k = k[:n]
        e = e[:n]
        finite = np.isfinite(k) & np.isfinite(e)
        if int(finite.sum()) < 2:
            return

        start = None
        prev = None
        for idx, ok in enumerate(finite):
            if not ok:
                if start is not None and prev is not None and prev - start + 1 >= 2:
                    ax.plot(k[start:prev + 1], e[start:prev + 1], "-", lw=0.9,
                            color=color, alpha=alpha, zorder=2)
                start = None
                prev = None
                continue
            if start is None:
                start = idx
            elif prev is not None:
                # Garde les ruptures visibles: typiquement changement de branche,
                # mauvais accrochage ou trou dans la fenêtre fit.
                if abs(k[idx] - k[prev]) > 0.10 or abs(e[idx] - e[prev]) > 0.08:
                    if prev - start + 1 >= 2:
                        ax.plot(k[start:prev + 1], e[start:prev + 1], "-", lw=0.9,
                                color=color, alpha=alpha, zorder=2)
                    start = idx
            prev = idx

        if start is not None and prev is not None and prev - start + 1 >= 2:
            ax.plot(k[start:prev + 1], e[start:prev + 1], "-", lw=0.9,
                    color=color, alpha=alpha, zorder=2)

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

    def refresh_physics_only(self) -> None:
        """Re-popule table physique + redessine Γ(E) sans toucher à la dispersion."""
        import matplotlib.pyplot as _plt
        self._sync_file_filter()
        visible = self._visible_files()
        self._table_phys.setRowCount(0)
        for name, entry in self._session.files.items():
            if entry.fit_result is None:
                continue
            if name not in visible:
                continue
            n = entry.fit_params.n_pairs
            self._populate_physics_rows(name, entry.fit_result, n, entry.meta)
        colors = _plt.cm.plasma(np.linspace(0.1, 0.9, max(1, len(self._session.files))))
        self._draw_gamma_panel(colors)

    def _draw_gamma_panel(self, colors) -> None:
        from arpes.analysis.results import fit_gamma_fermi_liquid
        visible = self._visible_files()
        ax = self._canvas_gamma.ax
        ax.cla(); ax.set_facecolor("#1a1a1a")
        self._canvas_gamma.fig.set_facecolor("#2b2b2b")
        plotted = 0
        for ci, (name, entry) in enumerate(self._session.files.items()):
            if entry.fit_result is None:
                continue
            if name not in visible:
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
            leg = ax.legend(
                fontsize=7, facecolor="#333", labelcolor="w",
                loc="upper left", bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0, ncol=1, frameon=True,
            )
            leg.set_draggable(True)
            self._canvas_gamma.fig.subplots_adjust(right=0.74)
        else:
            self._canvas_gamma.fig.subplots_adjust(right=0.97)
        self._canvas_gamma.redraw()

    @staticmethod
    def _fmt(value: float, sigma: float, *, dec: int = 4) -> str:
        if not (np.isfinite(value) and np.isfinite(sigma)):
            return "—"
        return f"{value:.{dec}f} ± {sigma:.{dec}f}"

    # ── filtre fichiers ──────────────────────────────────────────────────────
    def _sync_file_filter(self) -> None:
        """Synchronise QListWidget avec session.files (que les fittés)."""
        self._file_filter.blockSignals(True)
        # Mémorise l'état courant avant rebuild
        current_unchecked = set(self._file_filter_unchecked)
        for i in range(self._file_filter.count()):
            it = self._file_filter.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                current_unchecked.discard(it.text())
            else:
                current_unchecked.add(it.text())
        self._file_filter.clear()
        for name, entry in self._session.files.items():
            if entry.fit_result is None:
                continue
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(
                Qt.CheckState.Unchecked if name in current_unchecked
                else Qt.CheckState.Checked
            )
            self._file_filter.addItem(it)
        self._file_filter_unchecked = current_unchecked
        self._file_filter.blockSignals(False)

    def _visible_files(self) -> set[str]:
        out: set[str] = set()
        for i in range(self._file_filter.count()):
            it = self._file_filter.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.add(it.text())
        return out

    def _set_all_filter(self, checked: bool) -> None:
        if self._file_filter.count() == 0:
            return
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._file_filter.blockSignals(True)
        for i in range(self._file_filter.count()):
            self._file_filter.item(i).setCheckState(state)
        self._file_filter.blockSignals(False)
        self.refresh()

    def _on_file_filter_changed(self, _item) -> None:
        self.refresh()

    def _export_results(self):
        from arpes.ui.widgets.dialogs import ExportDialog
        dlg = ExportDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        rows = (physics_rows(self._session) if dlg.content_key == "physics"
                else result_rows(self._session))
        if not rows:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export", "Aucun résultat à exporter.")
            return
        suggested = str(self._session.folder or Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter résultats", suggested, dlg.file_filter(),
        )
        if not path:
            return
        if not path.lower().endswith(dlg.extension()):
            path = path + dlg.extension()
        try:
            self._dispatch_export(path, rows, dlg.content_key, dlg.format_key)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export", f"Échec écriture : {exc}")

    @staticmethod
    def _dispatch_export(path: str, rows: list[dict], content: str, fmt: str) -> None:
        if fmt == "csv":
            if content == "physics":
                write_physics_csv(path, rows)
            else:
                write_results_csv(path, rows)
        elif fmt == "txt":
            if content == "physics":
                write_physics_txt(path, rows)
            else:
                write_results_txt(path, rows)
        elif fmt == "latex":
            text = physics_to_latex(rows)
            Path(path).write_text(text, encoding="utf-8")

    def _open_multi_file_analysis(self):
        from arpes.ui.widgets.dialogs import MultiFileAnalysisDialog
        dialog = MultiFileAnalysisDialog(self._session, self)
        dialog.exec()

    def _export_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", str(self._session.folder or Path.home()),
            "PDF (*.pdf);;PNG (*.png)")
        if path:
            savefig_with_preset(
                self._canvas.fig,
                path,
                self._cmb_export_style.currentText(),
                bbox_inches="tight",
                facecolor=self._canvas.fig.get_facecolor(),
            )
            self._write_figure_metadata_sidecar(path)

    def _write_figure_metadata_sidecar(self, fig_path: str) -> None:
        import json
        meta_path = Path(fig_path).with_suffix(".meta.json")
        visible = sorted(self._visible_files())
        files_meta = []
        for name in visible:
            entry = self._session.files.get(name)
            if entry is None:
                continue
            m = entry.meta
            files_meta.append({
                "file": name,
                "hv": float(getattr(m, "hv", 0.0) or 0.0),
                "T_K": float(getattr(m, "temperature", 0.0) or 0.0),
                "direction": str(getattr(m, "direction", "") or ""),
                "polarization": str(getattr(m, "polarization", "") or ""),
                "formula": str(getattr(m, "formula", "") or ""),
                "mp_id": str(getattr(m, "mp_id", "") or ""),
                "crystal_a_angstrom": float(getattr(m, "crystal_a_angstrom", 0.0) or 0.0),
                "ef_offset": float(getattr(entry, "ef_offset", 0.0) or 0.0),
                "fitted": bool(entry.fit_result),
            })
        payload = {
            "figure": Path(fig_path).name,
            "export_style": self._cmb_export_style.currentText(),
            "session_folder": str(self._session.folder) if self._session.folder else "",
            "n_files_visible": len(files_meta),
            "files": files_meta,
            "session_notes": str(getattr(self._session, "session_notes", "") or "")[:500],
        }
        try:
            meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception:
            pass

