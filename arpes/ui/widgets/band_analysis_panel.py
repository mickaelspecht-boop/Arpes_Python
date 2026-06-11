"""Band-analysis panel: TB, Kink and Gap sub-tabs for the MDC area."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QLocale, Qt, pyqtSignal
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
    sb.setLocale(QLocale(QLocale.Language.C))  # dot decimal regardless of system locale
    sb.setRange(lo, hi)
    sb.setDecimals(dec)
    sb.setSingleStep(step)
    sb.setValue(val)
    sb.setKeyboardTracking(False)
    return sb


HELP_TB = (
    "<b>TB fit (Tight-Binding)</b><br><br>"
    "Fits the E(k) dispersion extracted from the MDC fit on the model "
    "<i>E(k) = ε₀ − 2t·cos(ka) − 4t'·cos(ka)cos(kb)</i>.<br><br>"
    "<b>Output</b>: ε₀, t, t' (eV), m*/m, W (bandwidth, eV).<br>"
    "<b>Use</b>: compare measured hopping vs DFT. If t_ARPES &lt; t_DFT → "
    "renormalization by electronic correlations.<br>"
    "<b>Lattice choice</b>: chain = 1D, square = a=b, hex = honeycomb, "
    "rect = a≠b."
)

HELP_KINK = (
    "<b>Kink Σ(E) — electron-boson coupling</b><br><br>"
    "Compares experimental dispersion and bare band (parabolic fit "
    "over a deep window where correlations are weak).<br>"
    "<i>Re Σ = E_exp − E_bare</i> ; <i>Im Σ ≈ (v_bare/2)·Γ_MDC</i>.<br>"
    "<i>λ = −∂ReΣ/∂ω|_{ω=0}</i> from a linear fit in a small window."
    "<br><br><b>Output</b>: λ (typical 0.3–1.5), v_bare (eV·Å), Re/Im Σ(E)."
    "<br><b>Use</b>: quantify phonon coupling ('kink' near ~50–70 meV) "
    "or a bosonic mode (superconducting oxides)."
)

HELP_GAP = (
    "<b>Gap Δ (Dynes)</b><br><br>"
    "Symmetrizes the EDC at kF, then fits the Dynes density of states "
    "<i>I(ω) ∝ Re[(ω−iΓ)/√((ω−iΓ)² − Δ²)]</i>.<br><br>"
    "<b>Output</b>: Δ (gap, meV), Γ (broadening, meV), k_F (Å⁻¹).<br>"
    "<b>2 gaps</b>: for s± superconductors (Fe-pnictides).<br>"
    "<b>Resolution</b>: Gaussian convolution (set instrumental dE in meV)."
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
        # Summary first: it is the page you READ; the three analysis pages
        # (workflow order TB → Kink → Gap) are where you act.
        self.tabs.addTab(self._build_summary_tab(), "Summary")
        self.tabs.addTab(self._build_tb_tab(), "TB fit")
        self.tabs.addTab(self._build_kink_tab(), "Kink Σ(E)")
        self.tabs.addTab(self._build_gap_tab(), "Gap Δ")
        lay.addWidget(self.tabs)

    # ------------------------------------------------------------------
    # Top status row (multi-stage progress + presets)
    # ------------------------------------------------------------------

    from arpes.ui.widgets.band_analysis_presets import PRESETS  # noqa: E402

    def _build_status_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(4)
        self.stage_mdc = QLabel("○ MDC")
        self.stage_tb = QLabel("○ TB")
        self.stage_kink = QLabel("○ Kink")
        self.stage_gap = QLabel("○ Gap")
        _CHIP_TIPS = {
            "MDC": "Step 0 — run an MDC fit in the MDC Fit tab first; every "
                   "analysis below uses its kF(E) points.",
            "TB": "Step 1 — tight-binding fit of the dispersion (click to open).",
            "Kink": "Step 2 — self-energy / electron-boson coupling λ from the "
                    "deviation to a bare band (click to open).",
            "Gap": "Step 3 — superconducting/CDW gap Δ from the symmetrized "
                   "EDC (click to open).",
        }
        # Chips MDC→Gap mirror the workflow order; clicking one (except MDC,
        # which lives in the MDC Fit tab) jumps to the matching sub-tab.
        for chip_idx, lbl in enumerate(
                (self.stage_mdc, self.stage_tb, self.stage_kink, self.stage_gap)):
            lbl.setStyleSheet(
                "color:#aaa; background:#222; padding:2px 6px;"
                " border-radius:3px; font-size:10px;"
            )
            lbl.setMinimumWidth(60)
            name = ("MDC", "TB", "Kink", "Gap")[chip_idx]
            lbl.setToolTip(_CHIP_TIPS[name])
            if chip_idx > 0:  # TB/Kink/Gap → sub-tabs 1/2/3 (Summary is 0)
                lbl.setCursor(Qt.CursorShape.PointingHandCursor)
                lbl.mousePressEvent = (
                    lambda _e, i=chip_idx: self.tabs.setCurrentIndex(i))
            row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        self.preset_combo.setToolTip(
            "Pre-fills all parameters for a standard material "
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
            "MDC fit branch to analyze:\n"
            "  kF_minus = k < 0 side (toward negative Γ)\n"
            "  kF_plus  = k > 0 side\n"
            "Choose the one containing the band of interest."
        )
        return c

    def _header(self, text: str, help_text: str) -> QWidget:
        row = QHBoxLayout()
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#aaa; font-size:10px; padding:2px 0px;")
        btn = QToolButton()
        btn.setText("?")
        btn.setToolTip("Method + detailed interpretation.")
        btn.setStyleSheet("padding:0px 6px;")
        btn.clicked.connect(lambda: QMessageBox.information(self, "Help", help_text))
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
            "Fits E(k) on a TB model → ε₀, t, t', m*/m. "
            "Compares to DFT/theory. Requires MDC fit.",
            HELP_TB,
        ))
        bw, self.tb_badge = self._badge_row()
        outer.addWidget(bw)

        form = QFormLayout()
        self.tb_lattice = QComboBox()
        self.tb_lattice.addItems(["chain", "square", "hex", "rect"])
        self.tb_lattice.setToolTip(
            "Lattice model:\n"
            "  chain   = 1D (one t parameter, enough for MDC Γ-X cut)\n"
            "  square  = 2D square a=b (t, t')\n"
            "  hex     = 2D honeycomb (graphene-like)\n"
            "  rect    = 2D rectangle a≠b"
        )
        self.tb_a = _dspin(1.0, 20.0, 3.9, 0.01, 4)
        self.tb_a.setToolTip(
            "Lattice parameter a in Å (interatomic distance).\n"
            "Auto-filled from crystal metadata if available."
        )
        self.tb_b = _dspin(1.0, 20.0, 3.9, 0.01, 4)
        self.tb_b.setToolTip("Parameter b (rect only).")
        self.tb_b.setEnabled(False)
        self.tb_lattice.currentTextChanged.connect(
            lambda s: self.tb_b.setEnabled(s == "rect")
        )
        self.tb_branch = self._branch_combo()
        self.tb_pair_label = QLabel("Pair #")
        self.tb_pair = QSpinBox(); self.tb_pair.setRange(0, 7)
        self.tb_pair.setToolTip(
            "Band-pair index ([0..n_pairs-1] from MDC fit).\n"
            "Hidden if only one pair was fitted."
        )
        form.addRow("Lattice", self.tb_lattice)
        form.addRow("a (Å)", self.tb_a)
        form.addRow("b (Å, rect)", self.tb_b)
        form.addRow("Branch", self.tb_branch)
        self._tb_pair_form_row = (self.tb_pair_label, self.tb_pair)
        form.addRow(self.tb_pair_label, self.tb_pair)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.tb_auto_btn = QPushButton("Auto")
        self.tb_auto_btn.setToolTip(
            "Pre-fills lattice a from crystal metadata and selects the first "
            "valid branch."
        )
        self.tb_auto_btn.clicked.connect(lambda: self.autofill_requested.emit("tb"))
        self.tb_run_btn = QPushButton("Run TB fit")
        self.tb_run_btn.setToolTip(
            "Fits E(k) on the selected model. Disabled until an MDC fit exists."
        )
        self.tb_run_btn.clicked.connect(self.tb_fit_requested.emit)
        btn_row.addWidget(self.tb_auto_btn)
        btn_row.addWidget(self.tb_run_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.tb_summary = QLabel("No TB fit.")
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
            "Kink Σ(E) → electron-boson coupling λ. "
            "Re Σ = E_exp − E_bare, λ = −∂ReΣ/∂ω|_{ω=0}. Requires MDC fit.",
            HELP_KINK,
        ))
        bw, self.kink_badge = self._badge_row()
        outer.addWidget(bw)

        form = QFormLayout()
        self.kink_branch = self._branch_combo()
        self.kink_pair_label = QLabel("Pair #")
        self.kink_pair = QSpinBox(); self.kink_pair.setRange(0, 7)
        self.kink_pair.setToolTip("Pair index ([0..n_pairs-1]).")
        self.kink_bare = QComboBox()
        self.kink_bare.addItems(["parabolic"])
        self.kink_bare.setToolTip(
            "Bare-band model fit over a deep window:\n"
            "  parabolic = E = v_F·(k-k0) + α·(k-k0)²"
        )
        self.kink_win_lo = _dspin(-2.0, 0.0, -0.30, 0.01, 3)
        self.kink_win_lo.setToolTip(
            "Lower bound of the bare-band window (eV, negative below E_F).\n"
            "Choose a deep range to avoid the renormalized zone."
        )
        self.kink_win_hi = _dspin(-2.0, 0.0, -0.08, 0.01, 3)
        self.kink_win_hi.setToolTip(
            "Upper bound of the bare window (eV). Must avoid the kink itself."
        )
        self.kink_lambda_win = _dspin(0.005, 0.20, 0.05, 0.005, 3)
        self.kink_lambda_win.setToolTip(
            "Half-window |ω| for the linear λ fit (eV).\n"
            "Typical: 0.03–0.08 eV. Test two values for sensitivity."
        )
        self.kink_EF = _dspin(-1.0, 1.0, 0.0, 0.001, 3)
        self.kink_EF.setToolTip(
            "E_F offset if the dispersion is not referenced to 0.\n"
            "Auto-filled from sp_ef (main EF offset)."
        )
        form.addRow("Branch", self.kink_branch)
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
            "Auto-fills E_F (from sp_ef), windows from fitted MDC range."
        )
        self.kink_auto_btn.clicked.connect(lambda: self.autofill_requested.emit("kink"))
        self.kink_run_btn = QPushButton("Run kink analysis")
        self.kink_run_btn.setToolTip("Computes Re/Im Σ and λ. Disabled without MDC fit.")
        self.kink_run_btn.clicked.connect(self.kink_run_requested.emit)
        btn_row.addWidget(self.kink_auto_btn)
        btn_row.addWidget(self.kink_run_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.kink_summary = QLabel("No kink analysis.")
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
            "Gap Δ (Dynes) → fits the symmetrized EDC at k_F. "
            "Output: Δ, Γ, k_F. Requires MDC fit to locate k_F.",
            HELP_GAP,
        ))
        bw, self.gap_badge = self._badge_row()
        outer.addWidget(bw)

        form = QFormLayout()
        self.gap_branch = self._branch_combo()
        self.gap_pair_label = QLabel("Pair #")
        self.gap_pair = QSpinBox(); self.gap_pair.setRange(0, 7)
        self.gap_pair.setToolTip("Pair index ([0..n_pairs-1]).")
        self.gap_n_gaps = QSpinBox(); self.gap_n_gaps.setRange(1, 2)
        self.gap_n_gaps.setToolTip(
            "1 = simple-gap superconductor.\n2 = s± superconductors (multi-pocket Fe-pnictides)."
        )
        self.gap_resolution = _dspin(0.0, 30.0, 5.0, 0.5, 2)
        self.gap_resolution.setToolTip(
            "Instrumental energy resolution (meV).\n"
            "Convolved into the Dynes model (Gaussian)."
        )
        self.gap_omega_max = _dspin(5.0, 200.0, 30.0, 1.0, 1)
        self.gap_omega_max.setToolTip(
            "Symmetrization window around E_F (meV).\n"
            "Typical 2–5× expected Δ."
        )
        self.gap_EF = _dspin(-1.0, 1.0, 0.0, 0.001, 3)
        self.gap_EF.setToolTip("E_F offset (eV). Auto from sp_ef.")
        form.addRow("Branch", self.gap_branch)
        self._gap_pair_form_row = (self.gap_pair_label, self.gap_pair)
        form.addRow(self.gap_pair_label, self.gap_pair)
        form.addRow("# gaps", self.gap_n_gaps)
        form.addRow("Resolution (meV)", self.gap_resolution)
        form.addRow("ω_max (meV)", self.gap_omega_max)
        form.addRow("E_F offset (eV)", self.gap_EF)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.gap_auto_btn = QPushButton("Auto")
        self.gap_auto_btn.setToolTip("Auto-fills E_F + heuristic ω_max.")
        self.gap_auto_btn.clicked.connect(lambda: self.autofill_requested.emit("gap"))
        self.gap_run_btn = QPushButton("Run gap fit")
        self.gap_run_btn.setToolTip("Symmetrizes EDC at k_F + Dynes fit. Disabled without MDC fit.")
        self.gap_run_btn.clicked.connect(self.gap_fit_requested.emit)
        btn_row.addWidget(self.gap_auto_btn)
        btn_row.addWidget(self.gap_run_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.gap_summary = QLabel("No gap fit.")
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
            "Consolidated summary: MDC + TB + Kink + Gap. m*↔(1+λ) consistency. "
            "Export CSV.",
            "<b>Summary</b><br>All metrics in one view.<br><br>"
            "<b>m*/m vs (1+λ) consistency</b>: Migdal-Eliashberg predicts "
            "<i>m*/m_bare ≈ 1 + λ</i>. A discrepancy &gt;30%% flags either "
            "non-phononic coupling or a poorly chosen bare band.<br><br>"
            "<b>CSV</b>: one row per metric with value, error, unit, source."
        ))
        self.summary_text = QTextBrowser()
        self.summary_text.setStyleSheet(
            "background:#1a1a1a; color:#ddd; font-family:monospace; font-size:11px;"
        )
        outer.addWidget(self.summary_text, 1)
        btn_row = QHBoxLayout()
        self.summary_csv_btn = QPushButton("Export CSV")
        self.summary_csv_btn.setToolTip(
            "Saves a CSV file with all measured metrics."
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

    @classmethod
    def _stage_style(cls, *, done: bool) -> str:
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
        from arpes.ui.widgets.band_analysis_prereq import update_prerequisites
        return update_prerequisites(
            self, has_fit=has_fit, n_pairs=n_pairs, n_points=n_points,
        )

    def apply_autofill(self, target: str, defaults: dict) -> None:
        from arpes.ui.widgets.band_analysis_prereq import apply_autofill
        return apply_autofill(self, target, defaults)

    # ------------------------------------------------------------------
    # Restore from entry
    # ------------------------------------------------------------------

    def restore(self, ba: dict):
        from arpes.ui.widgets.band_analysis_renders import restore_all
        return restore_all(self, ba)
