"""3-step wizard guiding the user through pocket characterization.

Page 1: seed + σ smoothing + ΔE (display local SNR)
Page 2: algorithm choice (iso vs MDC radial) + level/level parameters
Page 3: HS orientation confirmation (Γ-X, Γ-M, tolerance)

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
            f"Seed (Γ-relative): ({seed_plot[0]:+.3f}, {seed_plot[1]:+.3f}) π/a. "
            "Choose smoothing and the EF window consistently with the instrumental resolution."
        )
        self._snr_provider = snr_provider
        # Smoothing σ are no longer user-facing here (panel defaults are used);
        # the wizard only asks for the physically meaningful ΔE window.
        self._sigma_y = float(defaults["smooth_sigma_y"])
        self._sigma_x = float(defaults["smooth_sigma_x"])
        lay = QFormLayout(self)
        self.sp_ef_window = _dspin(defaults["ef_window"], 0.001, 0.5, 0.005, dec=3)
        lay.addRow("EF window ±eV:", self.sp_ef_window)
        self.lbl_snr = QLabel("Local SNR around seed: —")
        self.lbl_snr.setStyleSheet("color:#9cf; font-weight:bold;")
        lay.addRow(self.lbl_snr)
        self.sp_ef_window.valueChanged.connect(self._refresh_snr)
        self._refresh_snr()

    def _refresh_snr(self) -> None:
        try:
            snr = float(self._snr_provider(
                self._sigma_y, self._sigma_x, self.sp_ef_window.value()
            ))
        except Exception:
            snr = float("nan")
        if not np.isfinite(snr):
            self.lbl_snr.setText("Local SNR around seed: not computable")
            return
        color = "#9cf" if snr >= 3.0 else "#fa6"
        self.lbl_snr.setText(f"Local SNR around seed: {snr:.2f}")
        self.lbl_snr.setStyleSheet(f"color:{color}; font-weight:bold;")

    def values(self) -> dict[str, float]:
        return {"ef_window": float(self.sp_ef_window.value())}


class _AlgoPage(QWizardPage):
    def __init__(self, defaults: dict):
        super().__init__()
        self.setTitle("2/3 — Algorithm")
        self.setSubTitle("MDC-radial = publication. Iso-contour = quicklook.")
        lay = QVBoxLayout(self)
        self.rb_mdc = QRadioButton(
            "MDC radial Lorentzian fit (kF = max of A(k,EF), rigorous)"
        )
        self.rb_iso = QRadioButton(
            "Fixed-level iso-contour (heuristic, faster)"
        )
        self.rb_mdc.setChecked(True)
        group = QButtonGroup(self); group.addButton(self.rb_mdc); group.addButton(self.rb_iso)
        lay.addWidget(self.rb_mdc); lay.addWidget(self.rb_iso)

        form = QFormLayout()
        # MDC direction count / R² threshold are internal quality settings
        # (panel defaults apply); only the iso level and the mode remain.
        self.sp_iso_level = _dspin(float(defaults.get("level", 0.5) or 0.5), 0.0, 1.0, 0.01, dec=3)
        form.addRow("Iso : level :", self.sp_iso_level)
        from PyQt6.QtWidgets import QComboBox
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Auto (closed if possible, arc otherwise)",
            "Forced arc (pocket cut by scan edge)",
        ])
        self.cmb_mode.setToolTip(
            "Pocket mode. Forced arc: does not try to close; reports kF by "
            "direction + arc_coverage_deg. No area or Luttinger count."
        )
        form.addRow("Mode :", self.cmb_mode)
        lay.addLayout(form)
        note = QLabel(
            "MDC: Lorentzian fit on each ray, uncertainty by direction "
            "(Damascelli RMP 2003). Iso: level used as-is, no fit."
        )
        note.setWordWrap(True); note.setStyleSheet("color:#aaa; font-size:10px;")
        lay.addWidget(note)

    def values(self) -> dict[str, Any]:
        return {
            "algo": "mdc" if self.rb_mdc.isChecked() else "iso",
            "level": float(self.sp_iso_level.value()),
            "force_arc": bool(self.cmb_mode.currentIndex() == 1),
        }


class _HsOrientationPage(QWizardPage):
    def __init__(self, defaults: dict):
        super().__init__()
        self.setTitle("3/3 — Orientation HS Γ-X / Γ-M")
        self.setSubTitle(
            "Confirm for each sample: mounting breaks alignment. "
            "Read the orientation from the theoretical BZ overlay."
        )
        lay = QFormLayout(self)
        self.sp_x = _dspin(float(defaults.get("hs_dir_x_deg", 0.0)), -180.0, 180.0, 1.0, dec=1)
        self.sp_m = _dspin(float(defaults.get("hs_dir_m_deg", 45.0)), -180.0, 180.0, 1.0, dec=1)
        self.sp_tol = _dspin(float(defaults.get("hs_dir_tol_deg", 10.0)), 1.0, 45.0, 1.0, dec=1)
        lay.addRow("Direction Γ-X (deg) :", self.sp_x)
        lay.addRow("Direction Γ-M (deg) :", self.sp_m)
        lay.addRow("Sector tolerance (±deg):", self.sp_tol)
        note = QLabel(
            "kF(Γ-X) and kF(Γ-M) will be measured in these angular sectors. "
            "An orientation error biases the anisotropy."
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
        self.setWindowTitle("Guided FS Pocket Characterization")
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
