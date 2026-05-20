"""Dialog Qt pour Im Sigma(E) calculé depuis Γ(E) du fit MDC."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout

from arpes.ui.widgets.canvas import MplCanvas


class ImagSelfEnergyDialog(QDialog):
    """Affiche Im Σ(E) = (vF/2)·Γ_k(E) en meV."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Self-energy Im Σ(E)")
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        e = np.asarray(result.get("energy") or [], dtype=float)
        im = np.asarray(result.get("im_sigma") or [], dtype=float)
        self._im_std = np.asarray(result.get("im_sigma_std") or [], dtype=float)
        vF = float(result.get("vF_eV_A") or float("nan"))
        pi = int(result.get("pair_index") or 0)
        n = int(e.size)
        info = (f"Paire P{pi + 1}  |  {n} points  |  "
                f"vF = {vF:.2f} eV·Å")
        if n:
            info += (f"  |  Im Σ med = {float(np.nanmedian(im)) * 1000:.1f} meV")
        lay.addWidget(QLabel(info))
        self._canvas = MplCanvas(figsize=(7, 4), toolbar=True)
        lay.addWidget(self._canvas, stretch=1)
        self._plot(e, im)

    def _plot(self, e, im) -> None:
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        if e.size:
            order = np.argsort(e)
            std = getattr(self, "_im_std", np.array([]))
            if std.size == im.size and np.any(np.isfinite(std)):
                yerr_ord = std[order] * 1000.0
                yerr_ord[~np.isfinite(yerr_ord)] = 0.0
                ax.errorbar(
                    e[order], im[order] * 1000.0, yerr=yerr_ord,
                    fmt="o-", color="#f97316", ecolor="#fdba74",
                    elinewidth=0.7, capsize=1.5, lw=1.2, ms=4, alpha=0.95,
                )
            else:
                ax.plot(e[order], im[order] * 1000.0, "o-",
                        color="#f97316", lw=1.2, ms=4)
        ax.axhline(0.0, color="#888", lw=0.8, ls="--")
        ax.set_xlabel("E - EF (eV)", color="w")
        ax.set_ylabel("Im Σ (meV)", color="w")
        ax.set_title("Im Σ(E) = (vF/2)·Γ_k(E)", color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        self._canvas.fig.tight_layout(pad=0.6)
        self._canvas.redraw()
