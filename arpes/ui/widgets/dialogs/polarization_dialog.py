"""P(k,E) linear-dichroism viewer: π map, σ map, and contrast side by side.

Takes two band maps (orthogonal polarizations, e.g. LH/LV) on possibly different
(k,E) grids, resamples the second onto the first, and shows
P = (I_π − I_σ)/(I_π + I_σ). LH is treated as π, LV as σ; if the labels are
ambiguous the first map is π. Internal convention: maps are stored ``[k, e]``
(app convention) and transposed to ``[e, k]`` for resampling/plotting.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout

from arpes.physics.polarization import pkE_contrast, resample_map
from arpes.ui.widgets.canvas import MplCanvas


def _is_pi(pol: str) -> bool:
    """LH / p / horizontal → π (even); LV / s / vertical → σ (odd)."""
    p = str(pol or "").strip().upper()
    return p.startswith("LH") or p in ("P", "H") or "HORIZ" in p


class PolarizationDialog(QDialog):
    def __init__(self, parent, kpar_a, ev_a, data_a, pol_a,
                 kpar_b, ev_b, data_b, pol_b):
        super().__init__(parent)
        self.setWindowTitle("P(k,E) — polarization contrast (linear dichroism)")
        self.resize(1000, 460)
        # App stores band maps as [k, e]; work in [e, k] for imshow/resample.
        self._ka = np.asarray(kpar_a, dtype=float)
        self._ea = np.asarray(ev_a, dtype=float)
        self._a = np.asarray(data_a, dtype=float).T  # [e, k]
        kb = np.asarray(kpar_b, dtype=float)
        eb = np.asarray(ev_b, dtype=float)
        b_keb = np.asarray(data_b, dtype=float).T    # [e, k]
        # Resample B onto A's grid.
        self._b = resample_map(b_keb, kb, eb, self._ka, self._ea)
        # Assign π / σ by polarization label.
        if _is_pi(pol_a) or not _is_pi(pol_b):
            self._pi, self._sigma = self._a, self._b
            self._pi_lbl, self._sigma_lbl = str(pol_a or "A"), str(pol_b or "B")
        else:
            self._pi, self._sigma = self._b, self._a
            self._pi_lbl, self._sigma_lbl = str(pol_b or "B"), str(pol_a or "A")

        lay = QVBoxLayout(self)
        self._canvas = MplCanvas(figsize=(10, 4), toolbar=True)
        lay.addWidget(self._canvas)
        opts = QHBoxLayout()
        self.chk_smooth = QCheckBox("Smooth (σ=1 px)")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.stateChanged.connect(self._draw)
        opts.addWidget(self.chk_smooth)
        opts.addStretch(1)
        lay.addLayout(opts)
        self.lbl = QLabel(
            f"π = {self._pi_lbl},  σ = {self._sigma_lbl}.  "
            "P>0: brighter in π (even orbital);  P<0: brighter in σ (odd).  "
            "A band suppressed in one polarization is not necessarily absent "
            "(matrix-element effect)."
        )
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#cbd5e1;font-size:11px;")
        lay.addWidget(self.lbl)
        self._draw()

    def _draw(self) -> None:
        sigma_px = 1.0 if self.chk_smooth.isChecked() else 0.0
        p = pkE_contrast(self._pi, self._sigma, smooth_sigma=sigma_px,
                         denom_floor_frac=0.03, clip=1.0)
        ext = [float(self._ka[0]), float(self._ka[-1]),
               float(self._ea[0]), float(self._ea[-1])]
        fig = self._canvas.fig
        fig.clf()
        axs = fig.subplots(1, 3)
        fig.set_facecolor("#2b2b2b")
        for ax, m, ttl, cmap, kw in (
            (axs[0], self._pi, f"I_π ({self._pi_lbl})", "inferno", {}),
            (axs[1], self._sigma, f"I_σ ({self._sigma_lbl})", "inferno", {}),
            (axs[2], p, "P(k,E)", "RdBu_r", {"vmin": -1, "vmax": 1}),
        ):
            im = ax.imshow(m, origin="lower", extent=ext, aspect="auto",
                           cmap=cmap, **kw)
            ax.set_title(ttl, color="w", fontsize=10)
            ax.set_xlabel(r"$k_\parallel$ (π/a)", color="w")
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="w")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axs[0].set_ylabel(r"$E-E_F$ (eV)", color="w")
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._canvas.redraw()
