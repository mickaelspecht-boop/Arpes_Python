"""Dialog Qt pour Im Sigma(E) calculé depuis Γ(E) du fit MDC."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout

from arpes.ui.widgets.canvas import MplCanvas


_SIDE_COLORS = {"mean": "#f97316", "left": "#38bdf8", "right": "#a78bfa"}


class ImagSelfEnergyDialog(QDialog):
    """Affiche Im Σ(E) = (vF/2)·Γ_k(E) en meV.

    ``payload`` peut être :
    - un seul résultat dict (compat) → trace une courbe.
    - un mapping {label: result} → trace une courbe par entrée (utile
      pour comparer mean / left / right en mode width_mode=independent).
    """

    def __init__(self, payload, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Self-energy Im Σ(E)")
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        # Normalise vers mapping {label: result}
        if isinstance(payload, dict) and "energy" in payload:
            results = {"Moyenne": payload}
        elif isinstance(payload, dict):
            results = payload
        else:
            results = {"?": {}}
        self._results = results
        first = next(iter(results.values()), {})
        vF = float(first.get("vF_eV_A") or float("nan"))
        pi = int(first.get("pair_index") or 0)
        info = f"Paire P{pi + 1}  |  vF = {vF:.2f} eV·Å"
        if len(results) > 1:
            info += f"  |  {len(results)} courbes (mean/left/right)"
        lay.addWidget(QLabel(info))
        self._canvas = MplCanvas(figsize=(7, 4), toolbar=True)
        lay.addWidget(self._canvas, stretch=1)
        self._plot_multi()

    def _plot_multi(self) -> None:
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        for label, result in self._results.items():
            e = np.asarray(result.get("energy") or [], dtype=float)
            im = np.asarray(result.get("im_sigma") or [], dtype=float)
            std = np.asarray(result.get("im_sigma_std") or [], dtype=float)
            if e.size == 0:
                continue
            order = np.argsort(e)
            side = str(result.get("side", "")).lower()
            color = _SIDE_COLORS.get(side, "#f97316")
            if std.size == im.size and np.any(np.isfinite(std)):
                yerr = std[order] * 1000.0
                yerr[~np.isfinite(yerr)] = 0.0
                ax.errorbar(
                    e[order], im[order] * 1000.0, yerr=yerr,
                    fmt="o-", color=color, ecolor=color, alpha=0.85,
                    elinewidth=0.7, capsize=1.5, lw=1.2, ms=4,
                    label=label,
                )
            else:
                ax.plot(e[order], im[order] * 1000.0, "o-",
                        color=color, lw=1.2, ms=4, label=label, alpha=0.9)
        ax.axhline(0.0, color="#888", lw=0.8, ls="--")
        ax.set_xlabel("E - EF (eV)", color="w")
        ax.set_ylabel("Im Σ (meV)", color="w")
        ax.set_title("Im Σ(E) = (vF/2)·Γ_k(E)", color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        if len(self._results) > 1:
            leg = ax.legend(loc="best", fontsize=8, framealpha=0.75,
                            facecolor="#111827", edgecolor="#64748b",
                            labelcolor="w")
            if leg:
                leg.set_zorder(8)
        self._canvas.fig.tight_layout(pad=0.6)
        self._canvas.redraw()
