"""Band-analysis panel — 3 sub-tabs (TB / Kink / Gap) for the MDC area.

Inserted as a sub-tab inside the existing `_mdc_fit_tabs` widget so that the
user can chain: fit MDC → TB / kink / gap analysis without changing area.

Each sub-tab has:
- Compact param block (QFormLayout)
- "Run" button (manual; no live recompute — fits can take >100ms)
- Small Mpl canvas (figsize(4,3) max, Expanding policy, min 150)
- QLabel summary line(s) for fitted values
- QTextBrowser for physicist warnings (notes)

Canvas size strictly bounded to avoid the _right_stack squash regression.
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
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
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


class BandAnalysisPanel(QWidget):
    """3 sub-tabs (TB / Kink / Gap) — emits Run signals consumed by controller."""

    tb_fit_requested = pyqtSignal()
    kink_run_requested = pyqtSignal()
    gap_fit_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build()

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
        return c

    # ------------------------------------------------------------------
    # TB tab
    # ------------------------------------------------------------------

    def _build_tb_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(2, 2, 2, 2)

        form = QFormLayout()
        self.tb_lattice = QComboBox()
        self.tb_lattice.addItems(["chain", "square", "hex", "rect"])
        self.tb_a = _dspin(1.0, 20.0, 3.9, 0.01, 4)
        self.tb_b = _dspin(1.0, 20.0, 3.9, 0.01, 4)
        self.tb_b.setEnabled(False)
        self.tb_lattice.currentTextChanged.connect(
            lambda s: self.tb_b.setEnabled(s == "rect")
        )
        self.tb_branch = self._branch_combo()
        self.tb_pair = QSpinBox(); self.tb_pair.setRange(0, 7)
        form.addRow("Lattice", self.tb_lattice)
        form.addRow("a (Å)", self.tb_a)
        form.addRow("b (Å, rect)", self.tb_b)
        form.addRow("Branche", self.tb_branch)
        form.addRow("Paire #", self.tb_pair)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.tb_run_btn = QPushButton("Run TB fit")
        self.tb_run_btn.clicked.connect(self.tb_fit_requested.emit)
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

        form = QFormLayout()
        self.kink_branch = self._branch_combo()
        self.kink_pair = QSpinBox(); self.kink_pair.setRange(0, 7)
        self.kink_bare = QComboBox()
        self.kink_bare.addItems(["parabolic"])  # custom requires controller hook
        self.kink_win_lo = _dspin(-2.0, 0.0, -0.30, 0.01, 3)
        self.kink_win_hi = _dspin(-2.0, 0.0, -0.08, 0.01, 3)
        self.kink_lambda_win = _dspin(0.005, 0.20, 0.05, 0.005, 3)
        self.kink_EF = _dspin(-1.0, 1.0, 0.0, 0.001, 3)
        form.addRow("Branche", self.kink_branch)
        form.addRow("Paire #", self.kink_pair)
        form.addRow("Bare model", self.kink_bare)
        form.addRow("Window E_lo (eV)", self.kink_win_lo)
        form.addRow("Window E_hi (eV)", self.kink_win_hi)
        form.addRow("λ window |ω| (eV)", self.kink_lambda_win)
        form.addRow("E_F offset (eV)", self.kink_EF)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.kink_run_btn = QPushButton("Run kink analysis")
        self.kink_run_btn.clicked.connect(self.kink_run_requested.emit)
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

        form = QFormLayout()
        self.gap_branch = self._branch_combo()
        self.gap_pair = QSpinBox(); self.gap_pair.setRange(0, 7)
        self.gap_n_gaps = QSpinBox(); self.gap_n_gaps.setRange(1, 2)
        self.gap_resolution = _dspin(0.0, 30.0, 5.0, 0.5, 2)
        self.gap_omega_max = _dspin(5.0, 200.0, 30.0, 1.0, 1)
        self.gap_EF = _dspin(-1.0, 1.0, 0.0, 0.001, 3)
        form.addRow("Branche", self.gap_branch)
        form.addRow("Paire #", self.gap_pair)
        form.addRow("# gaps", self.gap_n_gaps)
        form.addRow("Résolution (meV)", self.gap_resolution)
        form.addRow("ω_max (meV)", self.gap_omega_max)
        form.addRow("E_F offset (eV)", self.gap_EF)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        self.gap_run_btn = QPushButton("Run gap fit")
        self.gap_run_btn.clicked.connect(self.gap_fit_requested.emit)
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
