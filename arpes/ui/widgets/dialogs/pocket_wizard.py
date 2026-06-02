"""3-step wizard guiding the user through pocket characterization.

Page 1 : seed + σ smoothing + ΔE (display local SNR)
Page 2 : algo choice (iso vs MDC radial) + level/level params
Page 3 : HS orientation confirm (Γ-X, Γ-M, tolerance)

Returns a settings dict via ``result_settings()`` after accept.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)


def _dspin(val, lo, hi, step, dec=3) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec); sp.setValue(val)
    sp.setKeyboardTracking(False)
    return sp


def _ispin(val, lo, hi, step=1) -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi); sp.setSingleStep(step); sp.setValue(val)
    sp.setKeyboardTracking(False)
    return sp


class _SeedSmoothingPage(QWizardPage):
    def __init__(self, seed_plot, defaults: dict, snr_provider):
        super().__init__()
        self.setTitle("1/3 — Seed + smoothing + ΔE")
        self.setSubTitle(
            f"Seed (Γ-relatif) : ({seed_plot[0]:+.3f}, {seed_plot[1]:+.3f}) π/a. "
            "Choisis le smoothing et la fenêtre EF cohérents avec la résolution instrument."
        )
        self._snr_provider = snr_provider
        lay = QFormLayout(self)
        self.sp_sigma_y = _dspin(defaults["smooth_sigma_y"], 0.0, 6.0, 0.25, dec=2)
        self.sp_sigma_x = _dspin(defaults["smooth_sigma_x"], 0.0, 12.0, 0.25, dec=2)
        self.sp_ef_window = _dspin(defaults["ef_window"], 0.001, 0.5, 0.005, dec=3)
        lay.addRow("σ smoothing ky (pixels) :", self.sp_sigma_y)
        lay.addRow("σ smoothing kx (pixels) :", self.sp_sigma_x)
        lay.addRow("Fenêtre EF ±eV :", self.sp_ef_window)
        self.lbl_snr = QLabel("SNR local autour seed : —")
        self.lbl_snr.setStyleSheet("color:#9cf; font-weight:bold;")
        lay.addRow(self.lbl_snr)
        warn = QLabel(
            "Rappel physique : σ·dk ne doit pas dépasser ~0.5·kF, sinon "
            "la mesure de kF est biaisée. Vérifier après caractérisation."
        )
        warn.setStyleSheet("color:#aaa; font-size:10px;")
        warn.setWordWrap(True)
        lay.addRow(warn)
        for sp in (self.sp_sigma_y, self.sp_sigma_x, self.sp_ef_window):
            sp.valueChanged.connect(self._refresh_snr)
        self._refresh_snr()

    def _refresh_snr(self) -> None:
        try:
            snr = float(self._snr_provider(
                self.sp_sigma_y.value(), self.sp_sigma_x.value(), self.sp_ef_window.value()
            ))
        except Exception:
            snr = float("nan")
        if not np.isfinite(snr):
            self.lbl_snr.setText("SNR local autour seed : non calculable")
            return
        color = "#9cf" if snr >= 3.0 else "#fa6"
        self.lbl_snr.setText(f"SNR local autour seed : {snr:.2f}")
        self.lbl_snr.setStyleSheet(f"color:{color}; font-weight:bold;")

    def values(self) -> dict[str, float]:
        return {
            "smooth_sigma_y": float(self.sp_sigma_y.value()),
            "smooth_sigma_x": float(self.sp_sigma_x.value()),
            "ef_window": float(self.sp_ef_window.value()),
        }


class _AlgoPage(QWizardPage):
    def __init__(self, defaults: dict):
        super().__init__()
        self.setTitle("2/3 — Algorithme")
        self.setSubTitle("MDC-radial = publication. Iso-contour = quicklook.")
        lay = QVBoxLayout(self)
        self.rb_mdc = QRadioButton(
            "MDC radial Lorentzian fit (kF = max de A(k,EF), rigoureux)"
        )
        self.rb_iso = QRadioButton(
            "Iso-contour à level fixe (heuristique, plus rapide)"
        )
        self.rb_mdc.setChecked(True)
        group = QButtonGroup(self); group.addButton(self.rb_mdc); group.addButton(self.rb_iso)
        lay.addWidget(self.rb_mdc); lay.addWidget(self.rb_iso)

        form = QFormLayout()
        self.sp_mdc_n = _ispin(int(defaults.get("mdc_n_directions", 36)), 8, 180, 4)
        self.sp_mdc_r2 = _dspin(float(defaults.get("mdc_r2_min", 0.5)), 0.0, 1.0, 0.05, dec=2)
        self.sp_iso_level = _dspin(float(defaults.get("level", 0.5) or 0.5), 0.0, 1.0, 0.01, dec=3)
        form.addRow("MDC : directions :", self.sp_mdc_n)
        form.addRow("MDC : R² min :", self.sp_mdc_r2)
        form.addRow("Iso : level :", self.sp_iso_level)
        lay.addLayout(form)
        note = QLabel(
            "MDC : fit Lorentzien sur chaque rayon, incertitude par direction "
            "(Damascelli RMP 2003). Iso : level utilisé tel quel — pas de fit."
        )
        note.setWordWrap(True); note.setStyleSheet("color:#aaa; font-size:10px;")
        lay.addWidget(note)

    def values(self) -> dict[str, Any]:
        return {
            "algo": "mdc" if self.rb_mdc.isChecked() else "iso",
            "mdc_n_directions": int(self.sp_mdc_n.value()),
            "mdc_r2_min": float(self.sp_mdc_r2.value()),
            "level": float(self.sp_iso_level.value()),
        }


class _HsOrientationPage(QWizardPage):
    def __init__(self, defaults: dict):
        super().__init__()
        self.setTitle("3/3 — Orientation HS Γ-X / Γ-M")
        self.setSubTitle(
            "À confirmer à chaque sample : mount casse l'alignement. "
            "Lis l'orientation sur l'overlay BZ théorique."
        )
        lay = QFormLayout(self)
        self.sp_x = _dspin(float(defaults.get("hs_dir_x_deg", 0.0)), -180.0, 180.0, 1.0, dec=1)
        self.sp_m = _dspin(float(defaults.get("hs_dir_m_deg", 45.0)), -180.0, 180.0, 1.0, dec=1)
        self.sp_tol = _dspin(float(defaults.get("hs_dir_tol_deg", 10.0)), 1.0, 45.0, 1.0, dec=1)
        lay.addRow("Direction Γ-X (deg) :", self.sp_x)
        lay.addRow("Direction Γ-M (deg) :", self.sp_m)
        lay.addRow("Tolérance secteur (±deg) :", self.sp_tol)
        note = QLabel(
            "kF(Γ-X) et kF(Γ-M) seront mesurés dans ces secteurs angulaires. "
            "Une erreur d'orientation biaise l'anisotropie."
        )
        note.setWordWrap(True); note.setStyleSheet("color:#aaa; font-size:10px;")
        lay.addRow(note)

    def values(self) -> dict[str, float]:
        return {
            "hs_dir_x_deg": float(self.sp_x.value()),
            "hs_dir_m_deg": float(self.sp_m.value()),
            "hs_dir_tol_deg": float(self.sp_tol.value()),
        }


class PocketWizardDialog(QWizard):
    def __init__(self, parent, *, seed_plot: tuple[float, float],
                 defaults: dict, snr_provider):
        super().__init__(parent)
        self.setWindowTitle("Caractérisation poche FS — guidée")
        self.resize(520, 460)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self._page_seed = _SeedSmoothingPage(seed_plot, defaults, snr_provider)
        self._page_algo = _AlgoPage(defaults)
        self._page_hs = _HsOrientationPage(defaults)
        self.addPage(self._page_seed)
        self.addPage(self._page_algo)
        self.addPage(self._page_hs)

    def result_settings(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out.update(self._page_seed.values())
        out.update(self._page_algo.values())
        out.update(self._page_hs.values())
        return out
