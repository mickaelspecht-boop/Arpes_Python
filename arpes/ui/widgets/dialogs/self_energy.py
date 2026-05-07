"""Dialog Qt pour affichage Re Sigma(E)."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout

from arpes.analysis.self_energy import RealSelfEnergyResult
from arpes.ui.widgets.canvas import MplCanvas


class SelfEnergyDialog(QDialog):
    def __init__(self, result: RealSelfEnergyResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Self-energy Re Sigma")
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        info = (
            f"{result.branch} P{result.pair_index + 1} / bande DFT {result.band_index}  "
            f"RMS={result.rms_e * 1000:.1f} meV"
        )
        if np.isfinite(result.kink_energy):
            info += f"  |  kink≈{result.kink_energy * 1000:.0f} meV"
        if np.isfinite(result.lambda_eff):
            info += f"  |  lambda_eff≈{result.lambda_eff:.2f}"
        lay.addWidget(QLabel(info))
        self._canvas = MplCanvas(figsize=(7, 4), toolbar=True)
        lay.addWidget(self._canvas, stretch=1)
        self._plot(result)

    def _plot(self, result: RealSelfEnergyResult) -> None:
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        order = np.argsort(result.energy)
        e = result.energy[order]
        sigma = result.re_sigma[order]
        ax.axhline(0.0, color="#888", lw=0.8, ls="--")
        ax.plot(e, sigma * 1000.0, "o-", color="#38bdf8", lw=1.2, ms=4)
        if np.isfinite(result.kink_energy):
            ax.axvline(result.kink_energy, color="#f97316", lw=1.0, ls=":")
        ax.set_xlabel("E - EF (eV)", color="w")
        ax.set_ylabel("Re Sigma (meV)", color="w")
        ax.set_title("Re Sigma(E) = E exp - E DFT(k exp)", color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        self._canvas.fig.tight_layout(pad=0.6)
        self._canvas.redraw()
