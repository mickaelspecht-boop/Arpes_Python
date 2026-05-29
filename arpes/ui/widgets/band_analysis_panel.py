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

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build()
        self._n_pairs = 1
        self._has_fit = False

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tb_tab(), "TB fit")
        self.tabs.addTab(self._build_kink_tab(), "Kink Σ(E)")
        self.tabs.addTab(self._build_gap_tab(), "Gap Δ")
        lay.addWidget(self.tabs)

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

    def show_tb_result(self, tb: dict, *, k: np.ndarray | None = None,
                       E: np.ndarray | None = None,
                       E_fit: np.ndarray | None = None):
        p = tb.get("params", {})
        per = tb.get("perr", {})
        parts = [f"<b>Model:</b> {tb.get('model','')}"]
        for name, v in p.items():
            err = per.get(name, 0.0)
            parts.append(f"<b>{name}</b>={v:.4f}±{err:.4f} eV")
        if tb.get("m_eff_over_me") is not None:
            parts.append(f"<b>m*/m</b>={tb['m_eff_over_me']:.3f}")
        if tb.get("bandwidth_eV") is not None:
            parts.append(f"<b>W</b>={tb['bandwidth_eV']:.3f} eV")
        parts.append(f"χ²_red={tb.get('chi2_red',0.0):.2e} (N={tb.get('n_points',0)})")
        self.tb_summary.setText(" — ".join(parts))
        ax = self.tb_canvas.ax
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        if k is not None and E is not None:
            ax.plot(k, E, "o", ms=3, color="#fbbf24", label="MDC peaks")
        if k is not None and E_fit is not None:
            order = np.argsort(k)
            ax.plot(k[order], E_fit[order], "-", lw=1.5,
                    color="#60a5fa", label="TB fit")
        ax.set_xlabel("k (Å⁻¹)", color="#ddd")
        ax.set_ylabel("E − E_F (eV)", color="#ddd")
        ax.tick_params(colors="#ddd")
        ax.legend(facecolor="#2b2b2b", edgecolor="#444", labelcolor="#ddd",
                  fontsize=8)
        self.tb_canvas.redraw()
        notes = tb.get("notes") or []
        self.tb_notes.setHtml("<br>".join(f"⚠ {n}" for n in notes) if notes else "")

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

    def show_kink_result(self, kink: dict):
        lam = kink.get("lambda")
        lam_err = kink.get("lambda_err")
        vb = kink.get("v_bare")
        parts = []
        if lam is not None:
            parts.append(f"<b>λ</b>={lam:.3f}" + (f"±{lam_err:.3f}" if lam_err else ""))
        if vb is not None:
            parts.append(f"v_bare={vb:.3f} eV·Å")
        self.kink_summary.setText(" — ".join(parts) or "λ non extractible.")
        E = np.asarray(kink.get("E_exp") or [])
        re = np.asarray(kink.get("re_sigma") or [])
        im = kink.get("im_sigma")
        ax_re, ax_im = self.kink_canvas.axes
        for ax in (ax_re, ax_im):
            ax.clear(); ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#ddd")
        ax_re.plot(E, re, "-o", ms=3, color="#fbbf24")
        ax_re.set_ylabel("Re Σ (eV)", color="#ddd")
        ax_re.axhline(0, color="#666", lw=0.5)
        if im is not None:
            ax_im.plot(E, np.asarray(im), "-o", ms=3, color="#60a5fa")
            ax_im.set_ylabel("Im Σ (eV)", color="#ddd")
        else:
            ax_im.text(0.5, 0.5, "Γ_MDC absent → Im Σ N/A",
                       ha="center", va="center", color="#aaa",
                       transform=ax_im.transAxes)
        ax_im.set_xlabel("E − E_F (eV)", color="#ddd")
        self.kink_canvas.redraw()
        notes = kink.get("notes") or []
        self.kink_notes.setHtml("<br>".join(f"⚠ {n}" for n in notes) if notes else "")

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

    def show_gap_result(self, gap: dict):
        Ds = gap.get("deltas_meV") or []
        errs = gap.get("delta_err_meV") or []
        Gs = gap.get("gammas_meV") or []
        parts = []
        for i, D in enumerate(Ds):
            e = errs[i] if i < len(errs) else 0.0
            parts.append(f"Δ<sub>{i+1}</sub>={D:.2f}±{e:.2f} meV")
        for i, G in enumerate(Gs):
            parts.append(f"Γ<sub>{i+1}</sub>={G:.2f} meV")
        parts.append(f"k_F={gap.get('k_F_inv_A', 0.0):.3f} Å⁻¹")
        parts.append(f"χ²_red={gap.get('chi2_red', 0.0):.2e}")
        self.gap_summary.setText(" — ".join(parts))
        omega = np.asarray(gap.get("omega_meV") or [])
        I_sym = np.asarray(gap.get("I_sym") or [])
        I_fit = np.asarray(gap.get("I_fit") or [])
        ax = self.gap_canvas.ax
        ax.clear(); ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#ddd")
        ax.plot(omega, I_sym, "o", ms=3, color="#fbbf24", label="symmetrized")
        ax.plot(omega, I_fit, "-", lw=1.5, color="#60a5fa", label="Dynes fit")
        for D in Ds:
            ax.axvline(D, color="#a78bfa", ls="--", lw=0.6)
            ax.axvline(-D, color="#a78bfa", ls="--", lw=0.6)
        ax.set_xlabel("ω = E − E_F (meV)", color="#ddd")
        ax.set_ylabel("I_sym", color="#ddd")
        ax.legend(facecolor="#2b2b2b", edgecolor="#444", labelcolor="#ddd",
                  fontsize=8)
        self.gap_canvas.redraw()
        notes = gap.get("notes") or []
        self.gap_notes.setHtml("<br>".join(f"⚠ {n}" for n in notes) if notes else "")

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
        if "tb" in ba:
            tb = ba["tb"]
            self.show_tb_result(tb)
        else:
            self.tb_summary.setText("Aucun fit TB.")
            self.tb_canvas.ax.clear(); self.tb_canvas.redraw()
            self.tb_notes.clear()
        if "kink" in ba:
            self.show_kink_result(ba["kink"])
        else:
            self.kink_summary.setText("Aucune analyse de kink.")
            for ax in self.kink_canvas.axes: ax.clear()
            self.kink_canvas.redraw()
            self.kink_notes.clear()
        if "gap" in ba:
            self.show_gap_result(ba["gap"])
        else:
            self.gap_summary.setText("Aucun fit de gap.")
            self.gap_canvas.ax.clear(); self.gap_canvas.redraw()
            self.gap_notes.clear()
