"""Band-analysis panel — 3 sub-tabs (TB / Kink / Gap) for the MDC area.

Inserted as a sub-tab inside the existing `_mdc_fit_tabs` widget so that the
user can chain: fit MDC → TB / kink / gap analysis without changing area.

Each sub-tab has:
- 2-line header explaining what the tab computes + prerequisites
- Prerequisite badge (turns red if no MDC fit) + Run button auto-disabled
- Compact param block with rich tooltips + "Auto" button filling defaults
- "?" help button opening a short explanation dialog
- Mpl canvas + summary line + warnings (notes)
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets.canvas import MplCanvas


def _dspin(lo: float, hi: float, val: float, step: float = 0.01,
           dec: int = 4) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(dec)
    sb.setSingleStep(step)
    sb.setValue(val)
    sb.setKeyboardTracking(False)
    return sb


HELP_TB = (
    "<b>TB fit (Tight-Binding)</b><br><br>"
    "Ajuste la dispersion E(k) extraite du fit MDC sur le modèle "
    "<i>E(k) = ε₀ − 2t·cos(ka) − 4t'·cos(ka)cos(kb)</i>.<br><br>"
    "<b>Sortie</b> : ε₀, t, t' (eV), m*/m, W (bandwidth, eV).<br>"
    "<b>Usage</b> : comparer hopping mesuré vs DFT. Si t_ARPES &lt; t_DFT → "
    "renormalisation par corrélations électroniques.<br>"
    "<b>Choix lattice</b> : chain = 1D, square = a=b, hex = nid-d'abeille, "
    "rect = a≠b."
)

HELP_KINK = (
    "<b>Kink Σ(E) — couplage électron-boson</b><br><br>"
    "Compare dispersion expérimentale et bare band (parabolique ajustée "
    "sur fenêtre profonde où corrélations faibles).<br>"
    "<i>Re Σ = E_exp − E_bare</i> ; <i>Im Σ ≈ (v_bare/2)·Γ_MDC</i>.<br>"
    "<i>λ = −∂ReΣ/∂ω|_{ω=0}</i> par fit linéaire dans une petite fenêtre."
    "<br><br><b>Sortie</b> : λ (typique 0.3–1.5), v_bare (eV·Å), Re/Im Σ(E)."
    "<br><b>Usage</b> : quantifier couplage phonon ('kink' à ~50–70 meV) "
    "ou mode bosonique (oxydes supras)."
)

HELP_GAP = (
    "<b>Gap Δ (Dynes)</b><br><br>"
    "Symétrise EDC à kF puis ajuste densité d'états Dynes "
    "<i>I(ω) ∝ Re[(ω−iΓ)/√((ω−iΓ)² − Δ²)]</i>.<br><br>"
    "<b>Sortie</b> : Δ (gap, meV), Γ (broadening, meV), k_F (Å⁻¹).<br>"
    "<b>2 gaps</b> : pour supras s± (Fe-pnictides).<br>"
    "<b>Résolution</b> : convolution gaussienne (mettre dE instrumental en meV)."
)


class BandAnalysisPanel(QWidget):
    """3 sub-tabs (TB / Kink / Gap) — emits Run signals consumed by controller."""

    tb_fit_requested = pyqtSignal()
    kink_run_requested = pyqtSignal()
    gap_fit_requested = pyqtSignal()
    autofill_requested = pyqtSignal(str)  # tab name: "tb" | "kink" | "gap"
    preset_requested = pyqtSignal(str)    # material name
    csv_export_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build()
        self._n_pairs = 1
        self._has_fit = False
        self._last_ba: dict = {}
        self._last_n_points: int = 0

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addWidget(self._build_status_row())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tb_tab(), "TB fit")
        self.tabs.addTab(self._build_kink_tab(), "Kink Σ(E)")
        self.tabs.addTab(self._build_gap_tab(), "Gap Δ")
        self.tabs.addTab(self._build_summary_tab(), "Résumé")
        lay.addWidget(self.tabs)

    # ------------------------------------------------------------------
    # Top status row (multi-stage progress + presets)
    # ------------------------------------------------------------------

    PRESETS = {
        "Custom": {},
        "BaNi2P2": {"a": 4.143, "lattice": "square", "omega_max_meV": 25.0},
        "Bi2212":  {"a": 5.40,  "lattice": "square", "omega_max_meV": 80.0},
        "FeSe":    {"a": 3.77,  "lattice": "square", "omega_max_meV": 15.0},
        "Cu(111)": {"a": 2.56,  "lattice": "hex",    "omega_max_meV": 5.0},
    }

    def _build_status_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(4)
        self.stage_mdc = QLabel("○ MDC")
        self.stage_tb = QLabel("○ TB")
        self.stage_kink = QLabel("○ Kink")
        self.stage_gap = QLabel("○ Gap")
        for lbl in (self.stage_mdc, self.stage_tb, self.stage_kink, self.stage_gap):
            lbl.setStyleSheet(
                "color:#aaa; background:#222; padding:2px 6px;"
                " border-radius:3px; font-size:10px;"
            )
            lbl.setMinimumWidth(60)
            row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        self.preset_combo.setToolTip(
            "Pré-remplit tous les paramètres pour un matériau standard "
            "(crystal a, lattice, ω_max gap)."
        )
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        row.addWidget(self.preset_combo)
        return w

    def _on_preset_changed(self, name: str) -> None:
        if name == "Custom":
            return
        preset = self.PRESETS.get(name, {})
        # MED-7: block child-widget signals so we don't trigger 3 cascading
        # redraws / param-changed emissions while applying the preset.
        widgets = [self.tb_a, self.tb_lattice, self.gap_omega_max]
        prev = [w.blockSignals(True) for w in widgets]
        try:
            if "a" in preset:
                self.tb_a.setValue(float(preset["a"]))
            if "lattice" in preset:
                idx = self.tb_lattice.findText(preset["lattice"])
                if idx >= 0:
                    self.tb_lattice.setCurrentIndex(idx)
                    self.tb_b.setEnabled(preset["lattice"] == "rect")
            if "omega_max_meV" in preset:
                self.gap_omega_max.setValue(float(preset["omega_max_meV"]))
        finally:
            for w, state in zip(widgets, prev):
                w.blockSignals(state)
        self.preset_requested.emit(name)

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _make_canvas(self, nrows: int = 1) -> MplCanvas:
        c = MplCanvas(figsize=(4, 3), toolbar=True, nrows=nrows)
        c.setMinimumSize(180, 180)
        c.setSizePolicy(QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Expanding)
        return c

    def _branch_combo(self) -> QComboBox:
        c = QComboBox()
        c.addItems(["kF_minus", "kF_plus"])
        c.setToolTip(
            "Branche du fit MDC à analyser :\n"
            "  kF_minus = côté k < 0 (vers Γ négatif)\n"
            "  kF_plus  = côté k > 0\n"
            "Choisir celle qui contient la bande d'intérêt."
        )
        return c

    def _header(self, text: str, help_text: str) -> QWidget:
        row = QHBoxLayout()
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#aaa; font-size:10px; padding:2px 0px;")
        btn = QToolButton()
        btn.setText("?")
        btn.setToolTip("Méthode + interprétation détaillée.")
        btn.setStyleSheet("padding:0px 6px;")
        btn.clicked.connect(lambda: QMessageBox.information(self, "Aide", help_text))
        row.addWidget(lbl, 1)
        row.addWidget(btn, 0)
        w = QWidget(); w.setLayout(row)
        return w

    def _badge_row(self) -> tuple[QWidget, QLabel]:
        """Build prerequisite badge row. Returns (widget, label_to_update)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        badge = QLabel("● MDC : —")
        badge.setStyleSheet(
            "color:#aaa; background:#222; padding:2px 6px;"
            " border-radius:3px; font-size:10px;"
        )
        row.addWidget(badge)
        row.addStretch(1)
        w = QWidget(); w.setLayout(row)
        return w, badge

    # ------------------------------------------------------------------
    # TB tab
    # ------------------------------------------------------------------

    def _build_tb_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(2, 2, 2, 2)

        outer.addWidget(self._header(
            "Ajuste E(k) sur modèle TB → ε₀, t, t', m*/m. "
            "Compare à DFT/théorie. Requiert fit MDC.",
            HELP_TB,
        ))
        bw, self.tb_badge = self._badge_row()
        outer.addWidget(bw)

        form = QFormLayout()
        self.tb_lattice = QComboBox()
        self.tb_lattice.addItems(["chain", "square", "hex", "rect"])
        self.tb_lattice.setToolTip(
            "Modèle réseau :\n"
            "  chain   = 1D (1 paramètre t, suffisant pour MDC cut Γ-X)\n"
            "  square  = 2D carré a=b (t, t')\n"
            "  hex     = 2D nid-d'abeille (graphene-like)\n"
            "  rect    = 2D rectangle a≠b"
        )
        self.tb_a = _dspin(1.0, 20.0, 3.9, 0.01, 4)
        self.tb_a.setToolTip(
            "Paramètre de maille a en Å (distance inter-atomes).\n"
            "Auto-rempli depuis méta cristal si disponible."
        )
        self.tb_b = _dspin(1.0, 20.0, 3.9, 0.01, 4)
        self.tb_b.setToolTip("Paramètre b (rect uniquement).")
        self.tb_b.setEnabled(False)
        self.tb_lattice.currentTextChanged.connect(
            lambda s: self.tb_b.setEnabled(s == "rect")
        )
        self.tb_branch = self._branch_combo()
        self.tb_pair_label = QLabel("Paire #")
        self.tb_pair = QSpinBox(); self.tb_pair.setRange(0, 7)
        self.tb_pair.setToolTip(
            "Index de la paire de bandes ([0..n_pairs-1] du fit MDC).\n"
            "Caché si une seule paire fittée."
        )
        form.addRow("Lattice", self.tb_lattice)
        form.addRow("a (Å)", self.tb_a)
        form.addRow("b (Å, rect)", self.tb_b)
        form.addRow("Branche", self.tb_branch)
        self._tb_pair_form_row = (self.tb_pair_label, self.tb_pair)
        form.addRow(self.tb_pair_label, self.tb_pair)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.tb_auto_btn = QPushButton("Auto")
        self.tb_auto_btn.setToolTip(
            "Pré-remplit lattice a depuis méta cristal + sélectionne 1ère "
            "branche valide."
        )
        self.tb_auto_btn.clicked.connect(lambda: self.autofill_requested.emit("tb"))
        self.tb_run_btn = QPushButton("Run TB fit")
        self.tb_run_btn.setToolTip(
            "Ajuste E(k) sur le modèle choisi. Désactivé tant qu'aucun fit MDC."
        )
        self.tb_run_btn.clicked.connect(self.tb_fit_requested.emit)
        btn_row.addWidget(self.tb_auto_btn)
        btn_row.addWidget(self.tb_run_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.tb_summary = QLabel("Aucun fit TB.")
        self.tb_summary.setWordWrap(True)
        outer.addWidget(self.tb_summary)

        self.tb_canvas = self._make_canvas()
        outer.addWidget(self.tb_canvas, 1)

        self.tb_notes = QTextBrowser()
        self.tb_notes.setMaximumHeight(80)
        outer.addWidget(self.tb_notes)
        return w

    def tb_options(self) -> dict:
        return {
            "lattice_type": self.tb_lattice.currentText(),
            "a": self.tb_a.value(),
            "b": self.tb_b.value() if self.tb_lattice.currentText() == "rect" else None,
            "branch": self.tb_branch.currentText(),
            "pair": self.tb_pair.value(),
        }

    def show_tb_result(self, tb, *, k=None, E=None, E_fit=None):
        from arpes.ui.widgets.band_analysis_renders import show_tb_result
        return show_tb_result(self, tb, k=k, E=E, E_fit=E_fit)

    # ------------------------------------------------------------------
    # Kink tab
    # ------------------------------------------------------------------

    def _build_kink_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(2, 2, 2, 2)

        outer.addWidget(self._header(
            "Kink Σ(E) → couplage électron-boson λ. "
            "Re Σ = E_exp − E_bare, λ = −∂ReΣ/∂ω|_{ω=0}. Requiert fit MDC.",
            HELP_KINK,
        ))
        bw, self.kink_badge = self._badge_row()
        outer.addWidget(bw)

        form = QFormLayout()
        self.kink_branch = self._branch_combo()
        self.kink_pair_label = QLabel("Paire #")
        self.kink_pair = QSpinBox(); self.kink_pair.setRange(0, 7)
        self.kink_pair.setToolTip("Index paire ([0..n_pairs-1]).")
        self.kink_bare = QComboBox()
        self.kink_bare.addItems(["parabolic"])
        self.kink_bare.setToolTip(
            "Modèle bare band ajusté sur fenêtre profonde :\n"
            "  parabolic = E = v_F·(k-k0) + α·(k-k0)²"
        )
        self.kink_win_lo = _dspin(-2.0, 0.0, -0.30, 0.01, 3)
        self.kink_win_lo.setToolTip(
            "Borne basse de la fenêtre bare-band (eV, négatif sous E_F).\n"
            "Choisir profond pour éviter zone renormalisée."
        )
        self.kink_win_hi = _dspin(-2.0, 0.0, -0.08, 0.01, 3)
        self.kink_win_hi.setToolTip(
            "Borne haute fenêtre bare (eV). Doit éviter le 'kink' lui-même."
        )
        self.kink_lambda_win = _dspin(0.005, 0.20, 0.05, 0.005, 3)
        self.kink_lambda_win.setToolTip(
            "Demi-fenêtre |ω| pour fit linéaire de λ (eV).\n"
            "Typique : 0.03–0.08 eV. Tester 2 valeurs pour sensibilité."
        )
        self.kink_EF = _dspin(-1.0, 1.0, 0.0, 0.001, 3)
        self.kink_EF.setToolTip(
            "Décalage E_F si dispersion pas référencée à 0.\n"
            "Auto-rempli depuis sp_ef (EF offset principal)."
        )
        form.addRow("Branche", self.kink_branch)
        self._kink_pair_form_row = (self.kink_pair_label, self.kink_pair)
        form.addRow(self.kink_pair_label, self.kink_pair)
        form.addRow("Bare model", self.kink_bare)
        form.addRow("Window E_lo (eV)", self.kink_win_lo)
        form.addRow("Window E_hi (eV)", self.kink_win_hi)
        form.addRow("λ window |ω| (eV)", self.kink_lambda_win)
        form.addRow("E_F offset (eV)", self.kink_EF)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.kink_auto_btn = QPushButton("Auto")
        self.kink_auto_btn.setToolTip(
            "Auto-remplit E_F (depuis sp_ef), fenêtres depuis plage MDC fittée."
        )
        self.kink_auto_btn.clicked.connect(lambda: self.autofill_requested.emit("kink"))
        self.kink_run_btn = QPushButton("Run kink analysis")
        self.kink_run_btn.setToolTip("Calcule Re/Im Σ et λ. Désactivé sans fit MDC.")
        self.kink_run_btn.clicked.connect(self.kink_run_requested.emit)
        btn_row.addWidget(self.kink_auto_btn)
        btn_row.addWidget(self.kink_run_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.kink_summary = QLabel("Aucune analyse de kink.")
        self.kink_summary.setWordWrap(True)
        outer.addWidget(self.kink_summary)

        self.kink_canvas = self._make_canvas(nrows=2)
        outer.addWidget(self.kink_canvas, 1)

        self.kink_notes = QTextBrowser()
        self.kink_notes.setMaximumHeight(80)
        outer.addWidget(self.kink_notes)
        return w

    def kink_options(self) -> dict:
        return {
            "branch": self.kink_branch.currentText(),
            "pair": self.kink_pair.value(),
            "bare": self.kink_bare.currentText(),
            "window_lo": self.kink_win_lo.value(),
            "window_hi": self.kink_win_hi.value(),
            "lambda_window": self.kink_lambda_win.value(),
            "E_F": self.kink_EF.value(),
        }

    def show_kink_result(self, kink):
        from arpes.ui.widgets.band_analysis_renders import show_kink_result
        return show_kink_result(self, kink)

    # ------------------------------------------------------------------
    # Gap tab
    # ------------------------------------------------------------------

    def _build_gap_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(2, 2, 2, 2)

        outer.addWidget(self._header(
            "Gap Δ (Dynes) → ajuste EDC symétrisée à k_F. "
            "Sortie : Δ, Γ, k_F. Requiert fit MDC pour localiser k_F.",
            HELP_GAP,
        ))
        bw, self.gap_badge = self._badge_row()
        outer.addWidget(bw)

        form = QFormLayout()
        self.gap_branch = self._branch_combo()
        self.gap_pair_label = QLabel("Paire #")
        self.gap_pair = QSpinBox(); self.gap_pair.setRange(0, 7)
        self.gap_pair.setToolTip("Index paire ([0..n_pairs-1]).")
        self.gap_n_gaps = QSpinBox(); self.gap_n_gaps.setRange(1, 2)
        self.gap_n_gaps.setToolTip(
            "1 = supra simple gap.\n2 = supras s± (Fe-pnictides multi-poches)."
        )
        self.gap_resolution = _dspin(0.0, 30.0, 5.0, 0.5, 2)
        self.gap_resolution.setToolTip(
            "Résolution énergétique instrumentale (meV).\n"
            "Convoluée au modèle Dynes (gaussienne)."
        )
        self.gap_omega_max = _dspin(5.0, 200.0, 30.0, 1.0, 1)
        self.gap_omega_max.setToolTip(
            "Fenêtre de symétrisation autour de E_F (meV).\n"
            "Typique 2–5× Δ_attendu."
        )
        self.gap_EF = _dspin(-1.0, 1.0, 0.0, 0.001, 3)
        self.gap_EF.setToolTip("Décalage E_F (eV). Auto depuis sp_ef.")
        form.addRow("Branche", self.gap_branch)
        self._gap_pair_form_row = (self.gap_pair_label, self.gap_pair)
        form.addRow(self.gap_pair_label, self.gap_pair)
        form.addRow("# gaps", self.gap_n_gaps)
        form.addRow("Résolution (meV)", self.gap_resolution)
        form.addRow("ω_max (meV)", self.gap_omega_max)
        form.addRow("E_F offset (eV)", self.gap_EF)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.gap_auto_btn = QPushButton("Auto")
        self.gap_auto_btn.setToolTip("Auto-remplit E_F + ω_max heuristique.")
        self.gap_auto_btn.clicked.connect(lambda: self.autofill_requested.emit("gap"))
        self.gap_run_btn = QPushButton("Run gap fit")
        self.gap_run_btn.setToolTip("Symétrise EDC à k_F + fit Dynes. Désactivé sans fit MDC.")
        self.gap_run_btn.clicked.connect(self.gap_fit_requested.emit)
        btn_row.addWidget(self.gap_auto_btn)
        btn_row.addWidget(self.gap_run_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.gap_summary = QLabel("Aucun fit de gap.")
        self.gap_summary.setWordWrap(True)
        outer.addWidget(self.gap_summary)

        self.gap_canvas = self._make_canvas()
        outer.addWidget(self.gap_canvas, 1)

        self.gap_notes = QTextBrowser()
        self.gap_notes.setMaximumHeight(80)
        outer.addWidget(self.gap_notes)
        return w

    def gap_options(self) -> dict:
        return {
            "branch": self.gap_branch.currentText(),
            "pair": self.gap_pair.value(),
            "n_gaps": self.gap_n_gaps.value(),
            "resolution_meV": self.gap_resolution.value(),
            "omega_max_meV": self.gap_omega_max.value(),
            "E_F": self.gap_EF.value(),
        }

    def show_gap_result(self, gap):
        from arpes.ui.widgets.band_analysis_renders import show_gap_result
        return show_gap_result(self, gap)

    # ------------------------------------------------------------------
    # Summary tab
    # ------------------------------------------------------------------

    def _build_summary_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.addWidget(self._header(
            "Résumé consolidé : MDC + TB + Kink + Gap. Cohérence m*↔(1+λ). "
            "Export CSV.",
            "<b>Résumé</b><br>Toutes les métriques en une vue.<br><br>"
            "<b>Cohérence m*/m vs (1+λ)</b> : Migdal-Eliashberg prédit "
            "<i>m*/m_bare ≈ 1 + λ</i>. Écart &gt;30 %% signale soit un "
            "couplage non phononique, soit une bare-band mal choisie.<br><br>"
            "<b>CSV</b> : 1 ligne par métrique avec valeur, erreur, unité, source."
        ))
        self.summary_text = QTextBrowser()
        self.summary_text.setStyleSheet(
            "background:#1a1a1a; color:#ddd; font-family:monospace; font-size:11px;"
        )
        outer.addWidget(self.summary_text, 1)
        btn_row = QHBoxLayout()
        self.summary_csv_btn = QPushButton("Export CSV")
        self.summary_csv_btn.setToolTip(
            "Sauve un fichier CSV avec toutes les métriques mesurées."
        )
        self.summary_csv_btn.clicked.connect(self.csv_export_requested.emit)
        btn_row.addWidget(self.summary_csv_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)
        return w

    def update_summary(self, ba: dict, *, has_fit: bool, n_points: int,
                       n_pairs: int) -> None:
        from arpes.ui.widgets.band_analysis_summary import render_summary_html
        html = render_summary_html(
            ba, has_fit=has_fit, n_points=n_points, n_pairs=n_pairs,
        )
        self.summary_text.setHtml(html)

    # ------------------------------------------------------------------
    # Status row update (multi-stage badges)
    # ------------------------------------------------------------------

    def _stage_style(self, *, done: bool) -> str:
        if done:
            return (
                "color:#86efac; background:#14532d; padding:2px 6px;"
                " border-radius:3px; font-size:10px;"
            )
        return (
            "color:#aaa; background:#222; padding:2px 6px;"
            " border-radius:3px; font-size:10px;"
        )

    def update_stage_row(self, ba: dict, *, has_fit: bool, n_points: int,
                         n_pairs: int) -> None:
        if has_fit:
            self.stage_mdc.setText(f"✓ MDC {n_points} pts × {n_pairs}")
            self.stage_mdc.setStyleSheet(self._stage_style(done=True))
        else:
            self.stage_mdc.setText("○ MDC")
            self.stage_mdc.setStyleSheet(self._stage_style(done=False))
        tb = ba.get("tb") or {}
        if tb and tb.get("params"):
            t_val = tb["params"].get("t")
            txt = f"✓ TB t={t_val:+.3f}" if t_val is not None else "✓ TB"
            self.stage_tb.setText(txt)
            self.stage_tb.setStyleSheet(self._stage_style(done=True))
        else:
            self.stage_tb.setText("○ TB")
            self.stage_tb.setStyleSheet(self._stage_style(done=False))
        kink = ba.get("kink") or {}
        lam = kink.get("lambda")
        if lam is not None:
            self.stage_kink.setText(f"✓ Kink λ={lam:.2f}")
            self.stage_kink.setStyleSheet(self._stage_style(done=True))
        else:
            self.stage_kink.setText("○ Kink")
            self.stage_kink.setStyleSheet(self._stage_style(done=False))
        gap = ba.get("gap") or {}
        Ds = gap.get("deltas_meV") or []
        if Ds:
            self.stage_gap.setText(f"✓ Gap Δ={Ds[0]:.1f}")
            self.stage_gap.setStyleSheet(self._stage_style(done=True))
        else:
            self.stage_gap.setText("○ Gap")
            self.stage_gap.setStyleSheet(self._stage_style(done=False))

    # ------------------------------------------------------------------
    # Prerequisite + n_pairs UI sync
    # ------------------------------------------------------------------

    def update_prerequisites(
        self, *, has_fit: bool, n_pairs: int, n_points: int = 0,
    ) -> None:
        """Refresh badges + enable/disable Run, hide pair spinbox if 1 paire."""
        self._has_fit = bool(has_fit)
        self._n_pairs = max(1, int(n_pairs))
        # Badge + button state
        if has_fit:
            badge_txt = f"● MDC ✓ {n_points} pts, {self._n_pairs} paire(s)"
            badge_css = (
                "color:#86efac; background:#14532d; padding:2px 6px;"
                " border-radius:3px; font-size:10px;"
            )
            run_enabled = True
        else:
            badge_txt = "⚠ MDC non fitté — onglet désactivé"
            badge_css = (
                "color:#fca5a5; background:#7f1d1d; padding:2px 6px;"
                " border-radius:3px; font-size:10px;"
            )
            run_enabled = False
        for badge in (self.tb_badge, self.kink_badge, self.gap_badge):
            badge.setText(badge_txt)
            badge.setStyleSheet(badge_css)
        for btn in (self.tb_run_btn, self.kink_run_btn, self.gap_run_btn):
            btn.setEnabled(run_enabled)
        for spin in (self.tb_pair, self.kink_pair, self.gap_pair):
            spin.setMaximum(max(0, self._n_pairs - 1))
        # Hide "Paire #" row when only 1 pair
        show_pair = self._n_pairs > 1
        for lbl, spin in (
            self._tb_pair_form_row,
            self._kink_pair_form_row,
            self._gap_pair_form_row,
        ):
            lbl.setVisible(show_pair)
            spin.setVisible(show_pair)

    def apply_autofill(self, target: str, defaults: dict) -> None:
        """Apply auto-filled defaults to a specific tab's spinboxes."""
        if target == "tb":
            if "a" in defaults:
                self.tb_a.setValue(float(defaults["a"]))
            if "branch" in defaults:
                idx = self.tb_branch.findText(str(defaults["branch"]))
                if idx >= 0:
                    self.tb_branch.setCurrentIndex(idx)
        elif target == "kink":
            if "E_F" in defaults:
                self.kink_EF.setValue(float(defaults["E_F"]))
            if "window_lo" in defaults:
                self.kink_win_lo.setValue(float(defaults["window_lo"]))
            if "window_hi" in defaults:
                self.kink_win_hi.setValue(float(defaults["window_hi"]))
            if "branch" in defaults:
                idx = self.kink_branch.findText(str(defaults["branch"]))
                if idx >= 0:
                    self.kink_branch.setCurrentIndex(idx)
        elif target == "gap":
            if "E_F" in defaults:
                self.gap_EF.setValue(float(defaults["E_F"]))
            if "omega_max_meV" in defaults:
                self.gap_omega_max.setValue(float(defaults["omega_max_meV"]))
            if "branch" in defaults:
                idx = self.gap_branch.findText(str(defaults["branch"]))
                if idx >= 0:
                    self.gap_branch.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Restore from entry
    # ------------------------------------------------------------------

    def restore(self, ba: dict):
        from arpes.ui.widgets.band_analysis_renders import restore_all
        return restore_all(self, ba)
