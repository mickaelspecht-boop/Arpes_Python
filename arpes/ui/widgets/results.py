"""Results panel: fit tables and color map."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from arpes.analysis.results import compute_results
from arpes.core.sample import require_lattice_a, sample_for_entry
from arpes.core.session import Session
from arpes.io.export import (
    export_provenance,
    physics_rows,
    physics_to_latex,
    result_rows,
    write_physics_csv,
    write_physics_txt,
    write_provenance_sidecar,
    write_results_csv,
    write_results_txt,
)
from arpes.ui.widgets.canvas import MplCanvas

class ResultsPanel(QWidget):
    def __init__(self, session: Session, host=None):
        super().__init__()
        self._session = session
        self._host = host
        self._result_point_refs: list[dict] = []
        self._linked_selection: dict | None = None
        self._linked_result_artist = None
        self._dispersion_offsets: dict[str, tuple[float, float]] = {}
        self._loading_alignment_controls = False
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)

        # Left: one plot at a time, full height — switch with the sub-tabs
        # (stacking both crushed each to half height).
        self._canvas = MplCanvas(figsize=(6, 4), toolbar=True)
        from arpes.ui.widgets import results_link
        self._canvas.canvas.mpl_connect(
            "button_press_event", lambda ev: results_link.on_results_click(self, ev)
        )
        self._canvas_gamma = MplCanvas(figsize=(6, 3), toolbar=True)
        cw = QTabWidget()
        cw.setStyleSheet(
            "QTabBar::tab{background:#303030;color:#bbb;padding:4px 10px;}"
            "QTabBar::tab:selected{background:#444;color:white;}"
        )
        cw.addTab(self._canvas, "kF dispersion")
        # Γ(E) tab = a small trend-model selector above the canvas.
        gamma_tab = QWidget()
        gtl = QVBoxLayout(gamma_tab)
        gtl.setContentsMargins(0, 0, 0, 0); gtl.setSpacing(2)
        grow = QHBoxLayout()
        grow.addWidget(QLabel("Trend fit:"))
        self._cmb_gamma_model = QComboBox()
        self._cmb_gamma_model.addItems(["Quadratic  Γ₀ + a·E²", "Linear  a + b·E"])
        self._cmb_gamma_model.setToolTip(
            "Overlay model for Γ(E):\n"
            "• Quadratic Γ₀ + a·E² — Fermi-liquid scattering (default).\n"
            "• Linear a + b·E — marginal-Fermi-liquid-like / quick trend.")
        self._cmb_gamma_model.currentIndexChanged.connect(
            lambda *_: self._draw_gamma_panel(getattr(self, "_gamma_colors", None)))
        grow.addWidget(self._cmb_gamma_model)
        from arpes.ui.widgets._qt_helpers import dspin
        grow.addSpacing(12)
        grow.addWidget(QLabel("Fit range E:"))
        self._sp_gamma_emin = dspin(-0.15, -5.0, 0.0, 0.01, dec=3)
        self._sp_gamma_emax = dspin(0.0, -5.0, 0.5, 0.01, dec=3)
        for _sp in (self._sp_gamma_emin, self._sp_gamma_emax):
            _sp.setToolTip(
                "Energy window (E−E_F, eV) used for the Γ(E) trend fit and the "
                "reported Γ₀. Combined with the reliability mask: only resolved, "
                "in-range slices are fitted. Set e.g. [−0.12, −0.05] to avoid the "
                "Fermi-edge blow-up.")
            _sp.valueChanged.connect(
                lambda *_: self._draw_gamma_panel(getattr(self, "_gamma_colors", None)))
        grow.addWidget(self._sp_gamma_emin)
        grow.addWidget(QLabel("…"))
        grow.addWidget(self._sp_gamma_emax)
        grow.addStretch(1)
        gtl.addLayout(grow)
        gtl.addWidget(self._canvas_gamma)
        cw.addTab(gamma_tab, "Γ(E) — lifetime")

        # MDC waterfall tab: stacked fitted MDCs (data+model) of one file, with
        # kF marked → the dispersion line drawn on top of the raw MDCs.
        wf_tab = QWidget()
        wtl = QVBoxLayout(wf_tab)
        wtl.setContentsMargins(0, 0, 0, 0); wtl.setSpacing(2)
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("File:"))
        self._cmb_wf_file = QComboBox()
        self._cmb_wf_file.setToolTip(
            "Fitted file shown as an MDC waterfall. Number of stacked MDCs = the "
            "fitted slices (set by the fit 'Energy step ΔE'); decimated above 40.")
        self._cmb_wf_file.currentTextChanged.connect(lambda *_: self._draw_mdc_waterfall())
        wrow.addWidget(self._cmb_wf_file, 1)
        self._chk_wf_model = QCheckBox("Model")
        self._chk_wf_model.setChecked(True)
        self._chk_wf_model.setToolTip("Overlay the fitted model on each MDC.")
        self._chk_wf_model.toggled.connect(lambda *_: self._draw_mdc_waterfall())
        wrow.addWidget(self._chk_wf_model)
        wrow.addStretch(1)
        wtl.addLayout(wrow)
        self._canvas_wf = MplCanvas(figsize=(6, 4), toolbar=True)
        wtl.addWidget(self._canvas_wf)
        cw.addTab(wf_tab, "MDC waterfall")

        cw.setTabToolTip(0, "kF(E) points of every fitted file (both branches).")
        cw.setTabToolTip(1, "MDC linewidth Γ(E) ± σ with a selectable linear or "
                            "Fermi-liquid (Γ₀ + a·E²) trend.")
        cw.setTabToolTip(2, "Stacked fitted MDCs of one file (data + model), "
                            "coloured by energy, kF dispersion drawn on top.")

        # droite : table + boutons
        right = QVBoxLayout()
        right.addWidget(QLabel("Show fitted files"))
        from arpes.ui.widgets import results_groups
        results_groups.build_group_filter(self, right)
        right.addWidget(QLabel("Bandes visibles / noms"))
        from arpes.ui.widgets import results_bands
        results_bands.build_band_registry(self, right)

        right.addWidget(QLabel("Alignement dispersion (affichage/export)"))
        self._chk_align_gamma = QCheckBox("Centrer chaque fichier sur Γ")
        self._chk_align_gamma.setToolTip(
            "Affichage non destructif: soustrait le centre Γ estimé par fichier "
            "(xg du fit, sinon milieu kF−/kF+). Les données fit_result ne sont pas modifiées."
        )
        self._chk_align_gamma.toggled.connect(self.refresh)
        right.addWidget(self._chk_align_gamma)
        align_row = QHBoxLayout()
        self._cmb_align_file = QComboBox()
        self._cmb_align_file.currentTextChanged.connect(self._load_alignment_controls)
        align_row.addWidget(self._cmb_align_file, stretch=1)
        self._sp_align_dk = QDoubleSpinBox()
        self._sp_align_dk.setRange(-5.0, 5.0)
        self._sp_align_dk.setDecimals(4)
        self._sp_align_dk.setSingleStep(0.005)
        self._sp_align_dk.setPrefix("Δk ")
        self._sp_align_dk.setSuffix(" π/a")
        self._sp_align_dk.setToolTip("Offset manuel k ajouté après centrage Γ (affichage/export seulement).")
        self._sp_align_dk.valueChanged.connect(self._on_alignment_offset_changed)
        align_row.addWidget(self._sp_align_dk)
        self._sp_align_de = QDoubleSpinBox()
        self._sp_align_de.setRange(-5.0, 5.0)
        self._sp_align_de.setDecimals(4)
        self._sp_align_de.setSingleStep(0.005)
        self._sp_align_de.setPrefix("ΔE ")
        self._sp_align_de.setSuffix(" eV")
        self._sp_align_de.setToolTip("Offset manuel énergie ajouté aux courbes (affichage/export seulement).")
        self._sp_align_de.valueChanged.connect(self._on_alignment_offset_changed)
        align_row.addWidget(self._sp_align_de)
        right.addLayout(align_row)
        align_btns = QHBoxLayout()
        btn_align_reset = QPushButton("Reset offset")
        btn_align_reset.setToolTip("Remet Δk=0 et ΔE=0 pour le fichier sélectionné.")
        btn_align_reset.clicked.connect(self._reset_selected_alignment_offset)
        align_btns.addWidget(btn_align_reset)
        self._lbl_align_state = QLabel("")
        self._lbl_align_state.setStyleSheet("color:#bbb;font-size:10px;")
        align_btns.addWidget(self._lbl_align_state, stretch=1)
        right.addLayout(align_btns)

        # Result tables were removed from the panel to declutter the UI: the
        # numbers now live behind "Export results…" which shows a live preview
        # (per-slice or physical ± σ) before writing. The plots stay here.
        right.addStretch(1)
        lbl_hint = QLabel(
            "Résultats chiffrés → bouton « Export results… » (aperçu avant écriture).")
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color:#9aa;font-size:11px;")
        right.addWidget(lbl_hint)

        btn_ref = QPushButton("Refresh all")
        btn_ref.setToolTip("Redraws dispersion + Γ(E) and recomputes the tables.")
        btn_ref.clicked.connect(self.refresh)
        btn_recalc = QPushButton("Recompute physical results")
        btn_recalc.setToolTip(
            "Only recomputes the 'Physical results ± σ' table "
            "and the Γ(E) panel. Useful after deleting points."
        )
        btn_recalc.clicked.connect(self.refresh_physics_only)
        btn_multi = QPushButton("Multi-file analysis...")
        btn_multi.setToolTip("Plots kF, m*, and Γ0 for the selected fitted entries.")
        btn_multi.clicked.connect(self._open_multi_file_analysis)
        btn_export = QPushButton("Export results...")
        btn_export.setToolTip(
            "Choose the content (per slice or physical ± σ) and the format\n"
            "(CSV, aligned TXT, LaTeX booktabs)."
        )
        btn_export.clicked.connect(self._export_results)
        btn_pdf = QPushButton("Export figure")
        btn_pdf.setToolTip(
            "Export scientifique unique: dispersion + Γ(E), fond blanc, axes, grille, légende."
        )
        btn_pdf.clicked.connect(self._export_fig)
        row1 = QHBoxLayout(); row1.addWidget(btn_ref); row1.addWidget(btn_recalc)
        row2 = QHBoxLayout(); row2.addWidget(btn_multi); row2.addWidget(btn_export)
        row2.addWidget(btn_pdf)
        right.addLayout(row1); right.addLayout(row2)

        rw = QWidget(); rw.setLayout(right)
        rw.setMinimumWidth(320)
        # User-draggable split instead of a hard 350 px cap: the plots take
        # the whole remaining screen and the divider can be moved at will.
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(cw)
        split.addWidget(rw)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1100, 380])
        self._split = split
        lay.addWidget(split)

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        # Apply the 65/35 split once the widget has its REAL width: showEvent
        # fires before the final layout pass (width ~650), which froze tiny
        # plot panes. resizeEvent with a realistic width is the reliable hook.
        if not getattr(self, "_split_initialized", False) and self.width() > 900:
            self._split_initialized = True
            w = self.width()
            self._split.setSizes([int(w * 0.65), int(w * 0.35)])

    def refresh(self):
        self._sync_file_filter()
        from arpes.ui.widgets import results_bands
        results_bands.sync_band_registry(self)
        visible = self._visible_files()
        ax = self._canvas.ax
        self._result_point_refs = []
        self._disp_uncertainty_labeled = False
        ax.cla(); ax.set_facecolor("#1a1a1a")
        self._canvas.fig.set_facecolor("#2b2b2b")

        colors = self._palette(len(self._session.files))
        row = 0
        for ci, (name, entry) in enumerate(self._session.files.items()):
            if entry.fit_result is None:
                continue
            if name not in visible:
                continue
            fr   = entry.fit_result
            ev_f = np.asarray(fr["e_fitted"])
            n = int(fr.get("n_pairs") or entry.fit_params.n_pairs or 1)
            c = self._color_for_file(name, ci, colors)

            for i in range(n):
                if not results_bands.band_visible(self, name, i):
                    continue
                km = np.asarray(fr["kF_minus"][i]) if i < len(fr["kF_minus"]) else []
                kp = np.asarray(fr["kF_plus"][i])  if i < len(fr["kF_plus"])  else []
                km_p, ev_p = self._aligned_dispersion_values(name, entry, km, ev_f)
                kp_p, _ = self._aligned_dispersion_values(name, entry, kp, ev_f)
                style = results_bands.band_style(c, i)
                band_lbl = results_bands.band_label(name, entry, i)
                ax.scatter(km_p, ev_p, s=11, color=style["color"],
                           marker=style["marker_minus"], alpha=0.85, label=band_lbl)
                ax.scatter(kp_p, ev_p, s=11, color=style["color"],
                           marker=style["marker_plus"], alpha=0.85)
                self._plot_branch_segments(
                    ax, km_p, ev_p, color=style["color"], alpha=0.68,
                    linestyle=style["linestyle"],
                )
                self._plot_branch_segments(
                    ax, kp_p, ev_p, color=style["color"], alpha=0.68,
                    linestyle=style["linestyle"],
                )
                sigma_m = (fr.get("sigma_kF_minus") or [])
                sigma_p = (fr.get("sigma_kF_plus") or [])
                sm = np.asarray(sigma_m[i], dtype=float) if i < len(sigma_m) else np.array([])
                sp = np.asarray(sigma_p[i], dtype=float) if i < len(sigma_p) else np.array([])
                for values, sigma in ((km_p, sm), (kp_p, sp)):
                    uncertain = results_bands.high_uncertainty_mask(values, sigma)
                    n_u = min(len(values), len(ev_p), len(uncertain))
                    uncertain = uncertain[:n_u]
                    if uncertain.any():
                        ax.scatter(
                            np.asarray(values)[:n_u][uncertain],
                            np.asarray(ev_p)[:n_u][uncertain],
                            s=24, marker="x", color="#d0d0d0", linewidths=0.9,
                            alpha=0.85, zorder=5,
                            label="incertitude kF élevée" if not getattr(
                                self, "_disp_uncertainty_labeled", False) else "_",
                        )
                        self._disp_uncertainty_labeled = True
                from arpes.ui.widgets.results_link import append_branch_refs
                append_branch_refs(self, name, "kF_minus", i, km_p, ev_p)
                append_branch_refs(self, name, "kF_plus", i, kp_p, ev_p)

            # Count plotted files to drive the legend below (tables removed).
            row += 1

        ax.axhline(0, color="cyan", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="w",    lw=0.5, ls="--", alpha=0.3)
        ax.set_xlabel(r"$k_\parallel$ (π/a)", fontsize=10, color="w")
        ax.set_ylabel(r"$E - E_F$ (eV)", fontsize=10, color="w")
        title = "kF dispersions — aligned on Γ" if self._chk_align_gamma.isChecked() else "kF dispersions — all fitted files"
        ax.set_title(title, fontsize=10, color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        if row > 0:
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) <= 8:  # a bigger legend would cover the data
                leg = ax.legend(
                    fontsize=7, facecolor="#333", labelcolor="w",
                    loc="best", markerscale=2, frameon=True, framealpha=0.75,
                )
                leg.set_draggable(True)
            self._canvas.fig.subplots_adjust(right=0.74)
        else:
            self._canvas.fig.subplots_adjust(right=0.97)
        from arpes.ui.widgets.results_link import highlight_results_selection
        highlight_results_selection(self)
        self._draw_gamma_panel(colors)
        from arpes.ui.widgets.results_waterfall import populate_waterfall_files
        populate_waterfall_files(self)
        self._draw_mdc_waterfall()

    def _auto_gamma_center(self, entry) -> float:
        fr = getattr(entry, "fit_result", None) or {}
        xg_raw = fr.get("xg")
        xg = np.asarray([] if xg_raw is None else xg_raw, dtype=float)
        if xg.size and np.isfinite(xg).any():
            return float(np.nanmedian(xg))
        centers = []
        for km_raw, kp_raw in zip(fr.get("kF_minus") or [], fr.get("kF_plus") or []):
            km = np.asarray(km_raw, dtype=float)
            kp = np.asarray(kp_raw, dtype=float)
            n = min(km.size, kp.size)
            if n:
                mid = 0.5 * (km[:n] + kp[:n])
                centers.extend(mid[np.isfinite(mid)].tolist())
        return float(np.nanmedian(centers)) if centers else 0.0

    def _alignment_for_file(self, name: str, entry) -> tuple[float, float, float]:
        dk, de = self._dispersion_offsets.get(name, (0.0, 0.0))
        center = self._auto_gamma_center(entry) if self._chk_align_gamma.isChecked() else 0.0
        return center, float(dk), float(de)

    def _aligned_dispersion_values(self, name: str, entry, k_values, e_values):
        k = np.asarray(k_values, dtype=float)
        e = np.asarray(e_values, dtype=float)
        center, dk, de = self._alignment_for_file(name, entry)
        return k - center + dk, e + de

    def sync_linked_fit_selection(self, filename: str, selection) -> None:
        """Highlight a BM-selected fit point on the Results kF plot when possible."""
        from arpes.ui.widgets.results_link import sync_from_bm_selection
        sync_from_bm_selection(self, filename, selection)

    @staticmethod
    def _plot_branch_segments(
        ax, k_values, e_values, *, color, alpha=0.6, linestyle="-"
    ) -> None:
        from arpes.ui.widgets.results_bands import plot_branch_segments
        plot_branch_segments(
            ax, k_values, e_values, color=color, alpha=alpha, linestyle=linestyle,
        )

    def refresh_physics_only(self) -> None:
        """Redraw Γ(E) and refresh the file filter without touching the dispersion plot."""
        self._sync_file_filter()
        from arpes.ui.widgets import results_bands
        results_bands.sync_band_registry(self)
        colors = self._palette(len(self._session.files))
        self._draw_gamma_panel(colors)

    @staticmethod
    def _palette(n: int):
        """n maximally-distinct qualitative colours (tab10/tab20 cycled).

        Replaces the old sequential plasma map whose adjacent samples were all
        near-identical yellows — files must be visually separable.
        """
        import matplotlib.pyplot as plt
        base = list(plt.cm.tab10.colors)
        if n > len(base):
            base = base + list(plt.cm.tab20.colors)
        return [base[i % len(base)] for i in range(max(1, n))]

    def _fallback_colors(self):
        return self._palette(len(self._session.files))

    def _gamma_e_range(self) -> tuple[float, float]:
        """User-chosen Γ(E) trend fit window (E−E_F, eV), low→high."""
        try:
            lo = float(self._sp_gamma_emin.value())
            hi = float(self._sp_gamma_emax.value())
            return (lo, hi) if lo <= hi else (hi, lo)
        except Exception:
            return (-0.15, 0.0)

    def _draw_mdc_waterfall(self) -> None:
        from arpes.ui.widgets.results_waterfall import draw_mdc_waterfall
        draw_mdc_waterfall(self)

    def _draw_gamma_panel(self, colors) -> None:
        from arpes.ui.widgets.results_gamma import draw_gamma_panel
        if colors is None or len(colors) == 0:
            colors = self._fallback_colors()
        self._gamma_colors = colors
        draw_gamma_panel(self, colors)

    @staticmethod
    def _fmt(value: float, sigma: float, *, dec: int = 4) -> str:
        if not (np.isfinite(value) and np.isfinite(sigma)):
            return "—"
        return f"{value:.{dec}f} ± {sigma:.{dec}f}"

    # -- file filter / grouping (delegated to results_groups) -----------------
    def _sync_file_filter(self) -> None:
        from arpes.ui.widgets import results_groups
        results_groups.sync_group_tree(self)
        align_names = [n for n, e in self._session.files.items()
                       if e.fit_result is not None]
        self._sync_alignment_combo(align_names)

    def _visible_files(self) -> set[str]:
        from arpes.ui.widgets import results_groups
        return results_groups.visible_files(self)

    def _set_all_filter(self, checked: bool) -> None:
        from arpes.ui.widgets import results_groups
        results_groups.set_all(self, checked)

    def _color_for_file(self, name: str, ci: int, default_colors):
        """Per-file plot colour, overridden by the group colour when
        'Colour by group' is on (delegated to results_groups)."""
        from arpes.ui.widgets import results_groups
        return results_groups.color_for_file(self, name, ci, default_colors)

    def _sync_alignment_combo(self, names: list[str]) -> None:
        current = self._cmb_align_file.currentText() if hasattr(self, "_cmb_align_file") else ""
        self._cmb_align_file.blockSignals(True)
        self._cmb_align_file.clear()
        self._cmb_align_file.addItems(names)
        if current in names:
            self._cmb_align_file.setCurrentText(current)
        self._cmb_align_file.blockSignals(False)
        self._load_alignment_controls(self._cmb_align_file.currentText())

    def _load_alignment_controls(self, name: str) -> None:
        if not hasattr(self, "_sp_align_dk"):
            return
        dk, de = self._dispersion_offsets.get(name, (0.0, 0.0))
        entry = self._session.files.get(name)
        center = self._auto_gamma_center(entry) if entry is not None and entry.fit_result else 0.0
        self._loading_alignment_controls = True
        self._sp_align_dk.setValue(float(dk))
        self._sp_align_de.setValue(float(de))
        self._loading_alignment_controls = False
        if name:
            self._lbl_align_state.setText(f"Γ auto={center:+.4f} π/a")
        else:
            self._lbl_align_state.setText("")

    def _on_alignment_offset_changed(self, *_args) -> None:
        if self._loading_alignment_controls:
            return
        name = self._cmb_align_file.currentText()
        if not name:
            return
        self._dispersion_offsets[name] = (
            float(self._sp_align_dk.value()),
            float(self._sp_align_de.value()),
        )
        self.refresh()

    def _reset_selected_alignment_offset(self) -> None:
        name = self._cmb_align_file.currentText()
        if not name:
            return
        self._dispersion_offsets.pop(name, None)
        self._load_alignment_controls(name)
        self.refresh()

    def _export_results(self):
        from arpes.ui.widgets.dialogs import ExportDialog
        dlg = ExportDialog(self._session, self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        try:
            rows = (physics_rows(self._session) if dlg.content_key == "physics"
                    else result_rows(self._session))
        except ValueError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export", str(exc))
            return
        if not rows:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export", "No results to export.")
            return
        suggested = str(self._session.folder or Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", suggested, dlg.file_filter(),
        )
        if not path:
            return
        if not path.lower().endswith(dlg.extension()):
            path = path + dlg.extension()
        try:
            self._dispatch_export(path, rows, dlg.content_key, dlg.format_key)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export", f"Write failed: {exc}")

    def _dispatch_export(self, path: str, rows: list[dict], content: str, fmt: str) -> None:
        if fmt == "csv":
            provenance = export_provenance(self._session, content=content)
            if content == "physics":
                write_physics_csv(path, rows, provenance=provenance)
            else:
                write_results_csv(path, rows, provenance=provenance)
            write_provenance_sidecar(path, provenance)
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
        from arpes.ui.widgets import results_export
        results_export.export_fig(self)
