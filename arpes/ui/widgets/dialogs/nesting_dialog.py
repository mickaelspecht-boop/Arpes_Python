"""C(q) Fermi-surface autocorrelation (nesting) viewer.

Shows the geometric self-overlap C(q) = Σ_k A(k,E_F) A(k+q,E_F) of the current
FS map, marks the strongest off-Γ peaks (candidate nesting / folding vectors),
and lists them. A "remove background" toggle subtracts the mean first so weak
peaks stand out. The map is a geometric measure, not the susceptibility χ(q).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from arpes.physics.nesting import (
    autocorrelation_peaks,
    autocorrelation_q_axes,
    fs_autocorrelation,
)
from arpes.ui.widgets.canvas import MplCanvas


class NestingDialog(QDialog):
    def __init__(self, parent, kx, ky, fs, *, title: str = ""):
        super().__init__(parent)
        self.setWindowTitle("C(q) — Fermi-surface autocorrelation (nesting)")
        self.resize(640, 680)
        self._kx = np.asarray(kx, dtype=float)
        self._ky = np.asarray(ky, dtype=float)
        self._fs = np.asarray(fs, dtype=float)
        self._title = title

        lay = QVBoxLayout(self)
        self._canvas = MplCanvas(figsize=(6, 5), toolbar=True)
        lay.addWidget(self._canvas)
        opts = QHBoxLayout()
        self.chk_bg = QCheckBox("Remove background (subtract mean)")
        self.chk_bg.setChecked(True)
        self.chk_bg.setToolTip(
            "Subtract the mean FS intensity before C(q) so the trivial q=0 peak "
            "does not dominate and weak nesting/folding peaks become visible."
        )
        self.chk_bg.stateChanged.connect(self._recompute)
        opts.addWidget(self.chk_bg)
        opts.addStretch(1)
        lay.addLayout(opts)
        self.lbl_peaks = QLabel("")
        self.lbl_peaks.setWordWrap(True)
        self.lbl_peaks.setStyleSheet("color:#cbd5e1;font-size:11px;")
        lay.addWidget(self.lbl_peaks)
        self._recompute()

    def _recompute(self) -> None:
        sub = self.chk_bg.isChecked()
        ac = fs_autocorrelation(self._fs, subtract_mean=sub, normalize=True)
        qx, qy = autocorrelation_q_axes(self._kx, self._ky)
        peaks = autocorrelation_peaks(ac, qx, qy, n_peaks=3)
        ax = self._canvas.ax
        ax.clear()
        self._canvas._dark() if hasattr(self._canvas, "_dark") else None
        extent = [qx[0], qx[-1], qy[0], qy[-1]]
        im = ax.imshow(ac, origin="lower", extent=extent, aspect="equal",
                       cmap="inferno")
        ax.set_xlabel(r"$q_x$ (π/a)", color="w")
        ax.set_ylabel(r"$q_y$ (π/a)", color="w")
        ax.set_title("C(q)" + (f" — {self._title}" if self._title else ""),
                     color="w", fontsize=10)
        ax.tick_params(colors="w")
        for i, pk in enumerate(peaks):
            ax.plot(pk["qx"], pk["qy"], "o", mfc="none", mec="#34d399",
                    mew=1.6, ms=12, zorder=5)
            ax.annotate(f"Q{i+1}", (pk["qx"], pk["qy"]), color="#34d399",
                        fontsize=9, xytext=(5, 5), textcoords="offset points")
        try:
            self._canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            pass
        self._canvas.redraw()
        if peaks:
            txt = "Candidate vectors (geometric, not χ(q)):  " + "   ".join(
                f"Q{i+1}=({pk['qx']:+.3f}, {pk['qy']:+.3f}) |Q|={pk['q']:.3f} "
                f"(C={pk['value']:.2f})" for i, pk in enumerate(peaks)
            )
        else:
            txt = "No off-Γ peak found."
        self.lbl_peaks.setText(txt)
