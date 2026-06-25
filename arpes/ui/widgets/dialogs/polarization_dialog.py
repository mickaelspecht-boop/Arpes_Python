"""P(k,E) linear-dichroism viewer: π map, σ map, and contrast side by side.

Takes two band maps (orthogonal polarizations, e.g. LH/LV) on possibly different
(k,E) grids, resamples the second onto the first, and shows
P = (I_π − I_σ)/(I_π + I_σ). LH is treated as π, LV as σ; if the labels are
ambiguous the first map is π. Internal convention: maps are stored ``[k, e]``
(app convention) and transposed to ``[e, k]`` for resampling/plotting.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

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
        self.setWindowTitle("Polarization compare (LH/LV) — maps, contrast & MDC")
        self.resize(1000, 720)
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

        # MDC overlay: same cut, both polarizations at a chosen E — this is the
        # α(LH) vs β(LV) orbital-weight comparison along k.
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("MDC at  E−E_F (eV):"))
        emin, emax = float(self._ea.min()), float(self._ea.max())
        self.sp_e = QDoubleSpinBox()
        self.sp_e.setRange(emin, emax)
        self.sp_e.setDecimals(3)
        self.sp_e.setSingleStep(0.005)
        self.sp_e.setValue(float(np.clip(0.0, emin, emax)))
        self.sp_e.setToolTip("Binding energy of the MDC (E−E_F). 0 = Fermi level.")
        self.sp_e.valueChanged.connect(self._draw_mdc)
        mrow.addWidget(self.sp_e)
        mrow.addWidget(QLabel("± window (eV):"))
        self.sp_w = QDoubleSpinBox()
        self.sp_w.setRange(0.0, 0.2)
        self.sp_w.setDecimals(3)
        self.sp_w.setSingleStep(0.005)
        self.sp_w.setValue(0.010)
        self.sp_w.setToolTip("Half-width of the energy window integrated into the MDC.")
        self.sp_w.valueChanged.connect(self._draw_mdc)
        mrow.addWidget(self.sp_w)
        self.chk_mdc_norm = QCheckBox("Normalize")
        self.chk_mdc_norm.setChecked(True)
        self.chk_mdc_norm.setToolTip("Scale each MDC to its own max → compare peak shapes/positions.")
        self.chk_mdc_norm.stateChanged.connect(self._draw_mdc)
        mrow.addWidget(self.chk_mdc_norm)
        mrow.addStretch(1)
        lay.addLayout(mrow)
        self._mdc_canvas = MplCanvas(figsize=(10, 2.6), toolbar=True)
        lay.addWidget(self._mdc_canvas)

        self.lbl = QLabel(
            f"π = {self._pi_lbl},  σ = {self._sigma_lbl}.  "
            "Maps: P>0 brighter in π (even orbital), P<0 brighter in σ (odd).  "
            "MDC: π (LH) weights α, σ (LV) weights β — same cut, watch the weight "
            "transfer between peaks.  Suppression in one polarization is a "
            "matrix-element effect, not an absent band."
        )
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#cbd5e1;font-size:11px;")
        lay.addWidget(self.lbl)
        self._draw()
        self._draw_mdc()

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

    def _draw_mdc(self, *_args) -> None:
        e0 = float(self.sp_e.value())
        w = float(self.sp_w.value())
        mask = np.abs(self._ea - e0) <= w
        if not mask.any():
            mask = np.zeros_like(self._ea, dtype=bool)
            mask[int(np.argmin(np.abs(self._ea - e0)))] = True
        pi_mdc = np.nanmean(self._pi[mask, :], axis=0).astype(float)
        si_mdc = np.nanmean(self._sigma[mask, :], axis=0).astype(float)
        ylbl = "intensity (a.u.)"
        if self.chk_mdc_norm.isChecked():
            pi_mdc = pi_mdc / (np.nanmax(pi_mdc) or 1.0)
            si_mdc = si_mdc / (np.nanmax(si_mdc) or 1.0)
            ylbl = "intensity (normalized)"
        ax = self._mdc_canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        self._mdc_canvas.fig.set_facecolor("#2b2b2b")
        ax.plot(self._ka, pi_mdc, color="#f0a050", lw=1.4, label=f"π ({self._pi_lbl})")
        ax.plot(self._ka, si_mdc, color="#5ab0f0", lw=1.4, label=f"σ ({self._sigma_lbl})")
        ax.axvline(0, color="w", lw=0.5, ls="--", alpha=0.3)
        ax.set_xlabel(r"$k_\parallel$ (π/a)", color="w", fontsize=9)
        ax.set_ylabel(ylbl, color="w", fontsize=9)
        ax.set_title(f"MDC at E−E_F = {e0:+.3f} ± {w:.3f} eV", color="w", fontsize=9)
        ax.tick_params(colors="w", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        leg = ax.legend(fontsize=8, facecolor="#333", labelcolor="w", framealpha=0.7)
        if leg is not None:
            leg.set_draggable(True)
        try:
            self._mdc_canvas.fig.tight_layout()
        except Exception:
            pass
        self._mdc_canvas.redraw()
