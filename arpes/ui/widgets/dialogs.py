"""Boîte de dialogue calibration EF — preview + fit (scalar / per-column)."""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from arpes.ui.widgets.canvas import MplCanvas
from arpes.ui.widgets.plots import (
    auto_ef_window,
    fit_fermi_edge,
    fit_fermi_edge_per_column,
)


class EFCalibrationDialog(QDialog):
    """Calibration EF interactive : scalaire ou par colonne (poly).

    Inputs : data (n_k, n_E), kpar, ev_arr, T_init, half_width_init, source_name.
    Outputs (via .result_payload après accept) :
        {"mode": "scalar"|"poly", "ef_offset": float | None,
         "poly_coefs": [...] | None, "T": float, "fwhm_res": float,
         "rms": float, "n_valid": int, "k_min": float, "k_max": float,
         "save_as_reference": bool}
    """

    def __init__(self, parent, data, kpar, ev_arr, T_init=28.0,
                 half_width_init=0.15, source_name="", current_offset=0.0,
                 metadata: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Calibration EF")
        self.resize(900, 620)
        self._data  = np.asarray(data, dtype=float)
        self._kpar  = np.asarray(kpar, dtype=float)
        self._ev    = np.asarray(ev_arr, dtype=float)
        self._fit   = None
        self.result_payload = None
        self._current_offset = float(current_offset)
        self._metadata = metadata or {}
        self._ef_search = self._default_ef_search_range()

        # ── widgets ────────────────────────────────────────────────────────────
        lay = QHBoxLayout(self)

        # Panneau gauche : contrôles
        left = QWidget(); fl = QFormLayout(left); left.setMaximumWidth(320)
        info = QLabel(f"Source : {source_name or '—'}\nDimensions : {self._data.shape[0]} k × {self._data.shape[1]} E")
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        fl.addRow(info)

        self.rb_scalar = QRadioButton("Scalaire (un EF moyen)")
        self.rb_poly   = QRadioButton("Par colonne (polynôme)")
        self.rb_scalar.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_scalar); grp.addButton(self.rb_poly)
        fl.addRow(self.rb_scalar)
        fl.addRow(self.rb_poly)

        self.sp_T = QDoubleSpinBox(); self.sp_T.setRange(1.0, 400.0); self.sp_T.setDecimals(1)
        self.sp_T.setValue(float(T_init)); self.sp_T.setSuffix(" K")
        self.sp_T.setToolTip("Température utilisée pour fixer kBT dans la FD.")
        fl.addRow("Température :", self.sp_T)

        self.sp_hw = QDoubleSpinBox(); self.sp_hw.setRange(0.03, 0.50); self.sp_hw.setDecimals(3)
        self.sp_hw.setSingleStep(0.01); self.sp_hw.setValue(float(half_width_init)); self.sp_hw.setSuffix(" eV")
        self.sp_hw.setToolTip("Demi-largeur de la fenêtre de fit autour de EF estimé.")
        fl.addRow("Demi-fenêtre :", self.sp_hw)

        self.chk_auto = QCheckBox("Auto-fenêtre (gradient max)")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "Centre la fenêtre sur le gradient max de l'EDC moyenne.\n"
            f"Recherche actuelle : {self._ef_search[0]:+.2f} à {self._ef_search[1]:+.2f} eV."
        )
        fl.addRow(self.chk_auto)

        self.sp_deg = QSpinBox(); self.sp_deg.setRange(0, 4); self.sp_deg.setValue(2)
        self.sp_deg.setToolTip("Degré du polynôme EF(k). 0=constant, 2=parabole (défaut).")
        fl.addRow("Degré poly :", self.sp_deg)

        self.sp_sigma = QDoubleSpinBox(); self.sp_sigma.setRange(0.005, 0.10); self.sp_sigma.setDecimals(3)
        self.sp_sigma.setSingleStep(0.005); self.sp_sigma.setValue(0.025); self.sp_sigma.setSuffix(" eV")
        self.sp_sigma.setToolTip("Sigma initiale de résolution gaussienne pour le fit (FWHM=2.355σ).")
        fl.addRow("σ init :", self.sp_sigma)

        self.btn_fit = QPushButton("▶ Fitter")
        self.btn_fit.clicked.connect(self._do_fit)
        fl.addRow(self.btn_fit)

        self.lbl_result = QLabel("—")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setStyleSheet("background:#222; padding:6px; border-radius:4px;")
        fl.addRow(self.lbl_result)

        self.chk_save_ref = QCheckBox("Sauvegarder comme référence dossier (Au)")
        self.chk_save_ref.setToolTip(
            "La correction sera proposée comme référence appliquable aux autres\n"
            "fichiers du dossier via 'Appliquer Au de référence'."
        )
        fl.addRow(self.chk_save_ref)

        self.btn_apply  = QPushButton("✓ Appliquer à ce fichier")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.clicked.connect(self.reject)
        fl.addRow(self.btn_apply)
        fl.addRow(self.btn_cancel)

        lay.addWidget(left)

        # Panneau droit : preview matplotlib
        self._fig = Figure(figsize=(6, 5))
        self._canvas = FigureCanvas(self._fig)
        self._ax_edc  = self._fig.add_subplot(2, 1, 1)
        self._ax_poly = self._fig.add_subplot(2, 1, 2)
        self._fig.tight_layout()
        right = QWidget(); rl = QVBoxLayout(right); rl.addWidget(self._canvas)
        lay.addWidget(right, 1)

        self._draw_initial_preview()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _default_ef_search_range(self) -> tuple[float, float]:
        fmt = str(self._metadata.get("fs_source") or self._metadata.get("source_format") or "").lower()
        lab = str(self._metadata.get("lab") or "").lower()
        ref = str(self._metadata.get("energy_reference") or "").lower()
        # BESSY Center Energy mode: l'expérimentateur peut avoir centré l'analyseur
        # à n'importe quel offset d'EF. On vise donc d'abord le bord détecté dans
        # l'EDC (max drop d'intensité), puis on élargit autour. Si la détection
        # échoue, fallback large couvrant tout l'axe.
        if "bessy" in fmt or "bessy" in lab or ref == "ses_center_energy":
            try:
                edc = np.nanmean(self._data, axis=0)
                e = np.asarray(self._ev, dtype=float)
                grad = np.gradient(edc, e)
                drop_idx = int(np.nanargmin(grad))
                ef_hint = float(e[drop_idx])
                e_min, e_max = float(e.min()), float(e.max())
                lo = max(e_min, ef_hint - 0.30)
                hi = min(e_max, ef_hint + 0.20)
                if hi - lo < 0.15:
                    return (e_min, e_max)
                return (lo, hi)
            except Exception:
                return (-0.5, 0.5)
        return (-0.5, 0.2)

    def _draw_initial_preview(self):
        edc = np.nanmean(self._data, axis=0)
        self._ax_edc.clear()
        self._ax_edc.plot(self._ev, edc, "k-", lw=1.2, label="EDC moyenne")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_edc.axvspan(self._ef_search[0], self._ef_search[1], color="orange", alpha=0.08,
                             label="recherche EF")
        self._ax_edc.set_xlabel("E − EF (eV)"); self._ax_edc.set_ylabel("Intensité")
        self._ax_edc.set_title("EDC moyennée sur k")
        self._ax_edc.legend(fontsize=8)
        self._ax_poly.clear()
        self._ax_poly.text(0.5, 0.5, "Cliquer 'Fitter' pour lancer la calibration",
                           ha="center", va="center", transform=self._ax_poly.transAxes,
                           fontsize=10, color="gray")
        self._ax_poly.set_axis_off()
        self._canvas.draw_idle()

    def _do_fit(self):
        T  = self.sp_T.value()
        hw = self.sp_hw.value()
        sig = self.sp_sigma.value()
        auto = self.chk_auto.isChecked()
        edc = np.nanmean(self._data, axis=0)

        if self.rb_scalar.isChecked():
            win = auto_ef_window(self._ev, edc, half_width=hw, search=self._ef_search) if auto else (-hw, hw)
            try:
                _ax = Figure().add_subplot(111)
                r = fit_fermi_edge(
                    self._ev, edc,
                    temperature_K=T, fit_range=win,
                    sigma_resolution_init=sig, fix_kBT=True,
                    units="binding", ax=_ax, verbose=False,
                )
            except Exception as e:
                self.lbl_result.setText(f"⚠ fit échoué : {e}")
                return
            ef     = float(r["EF"])
            efe    = float(r.get("EF_err", np.nan))
            fwhm   = float(r["fwhm_res"])
            resid  = float(r["residual"])
            self._fit = {
                "mode": "scalar",
                "ef_shift": ef,
                "ef_err":   efe,
                "fwhm_res": fwhm,
                "rms":      resid,
                "n_valid":  int(self._data.shape[0]),
                "k_min":    float(self._kpar.min()),
                "k_max":    float(self._kpar.max()),
                "T":        T,
                "window":   win,
            }
            self._draw_scalar_preview(r, win)
            new_offset = self._current_offset - ef
            self.lbl_result.setText(
                f"<b>Mode scalaire</b><br>"
                f"EF fit : {ef*1000:+.1f} meV (±{efe*1000:.1f} meV)<br>"
                f"FWHM résolution : {fwhm*1000:.0f} meV<br>"
                f"Résidu rms : {resid:.4f}<br>"
                f"Fenêtre : [{win[0]*1000:+.0f}, {win[1]*1000:+.0f}] meV<br>"
                f"→ nouvel offset proposé : {new_offset:.4f} eV"
            )
        else:
            try:
                r = fit_fermi_edge_per_column(
                    self._data, self._kpar, self._ev,
                    temperature_K=T, half_width=hw,
                    sigma_resolution_init=sig,
                    poly_deg=self.sp_deg.value(),
                    auto_window=auto,
                    ef_search=self._ef_search,
                    verbose=False,
                )
            except Exception as e:
                self.lbl_result.setText(f"⚠ fit par colonne échoué : {e}")
                return
            self._fit = {
                "mode": "poly",
                "poly_coefs": r["poly_coefs"].tolist(),
                "ef_per_col": r["ef_per_col"],
                "ef_smooth":  r["ef_smooth"],
                "fwhm_res":   r["mean_fwhm"],
                "rms":        r["rms"],
                "n_valid":    r["n_valid"],
                "k_min":      float(self._kpar.min()),
                "k_max":      float(self._kpar.max()),
                "T":          T,
                "window":     r["window"],
                "mean_ef":    r["mean_ef"],
            }
            self._draw_poly_preview(r)
            self.lbl_result.setText(
                f"<b>Mode par colonne (poly deg {self.sp_deg.value()})</b><br>"
                f"Colonnes valides : {r['n_valid']}/{self._data.shape[0]}<br>"
                f"&lt;EF&gt; : {r['mean_ef']*1000:+.1f} meV<br>"
                f"FWHM médian : {r['mean_fwhm']*1000:.0f} meV<br>"
                f"RMS résidu poly : {r['rms']*1000:.1f} meV<br>"
                f"Fenêtre : [{r['window'][0]*1000:+.0f}, {r['window'][1]*1000:+.0f}] meV"
            )
        self.btn_apply.setEnabled(True)

    def _draw_scalar_preview(self, fit_result, win):
        self._ax_edc.clear()
        edc = np.nanmean(self._data, axis=0)
        self._ax_edc.plot(self._ev, edc / max(np.nanmax(edc), 1e-9), "k-", lw=1.0,
                          label="EDC normée")
        self._ax_edc.plot(fit_result["model_ev"], fit_result["model_I"], "r-", lw=2.0,
                          label=f"FD fit  EF={fit_result['EF']*1000:+.0f} meV")
        self._ax_edc.axvline(fit_result["EF"], color="red", lw=1.0)
        self._ax_edc.axvspan(win[0], win[1], color="orange", alpha=0.10, label="fenêtre")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_edc.set_xlim(min(win[0]-0.05, -0.4), max(win[1]+0.05, 0.1))
        self._ax_edc.set_xlabel("E − EF (eV)"); self._ax_edc.set_ylabel("I norm.")
        self._ax_edc.legend(fontsize=8)
        self._ax_poly.clear()
        self._ax_poly.text(0.5, 0.5,
                           "Mode scalaire : pas de courbe EF(k).\n"
                           "Passer en 'Par colonne' pour voir la dispersion.",
                           ha="center", va="center", transform=self._ax_poly.transAxes,
                           fontsize=9, color="gray")
        self._ax_poly.set_axis_off()
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _draw_poly_preview(self, r):
        self._ax_edc.clear()
        edc = np.nanmean(self._data, axis=0)
        self._ax_edc.plot(self._ev, edc / max(np.nanmax(edc), 1e-9), "k-", lw=1.0,
                          label="EDC moyenne")
        win = r["window"]
        self._ax_edc.axvspan(win[0], win[1], color="orange", alpha=0.10, label="fenêtre")
        self._ax_edc.axvline(r["mean_ef"], color="red", lw=1.0,
                             label=f"<EF>={r['mean_ef']*1000:+.0f} meV")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_edc.set_xlim(min(win[0]-0.05, -0.4), max(win[1]+0.05, 0.1))
        self._ax_edc.set_xlabel("E − EF (eV)"); self._ax_edc.set_ylabel("I norm.")
        self._ax_edc.legend(fontsize=8)

        self._ax_poly.clear()
        kp = self._kpar
        ef_raw = r["ef_per_col"]
        ef_sm  = r["ef_smooth"]
        valid = np.isfinite(ef_raw)
        self._ax_poly.plot(kp[valid], ef_raw[valid] * 1000, ".", color="#888", ms=3,
                           label="fits par colonne")
        self._ax_poly.plot(kp, ef_sm * 1000, "r-", lw=2.0,
                           label=f"poly deg {self.sp_deg.value()}")
        self._ax_poly.axhline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_poly.set_xlabel("k (π/a)"); self._ax_poly.set_ylabel("EF (meV)")
        self._ax_poly.set_title("EF(k) — courbure du détecteur")
        self._ax_poly.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _on_apply(self):
        if not self._fit:
            return
        save_ref = bool(self.chk_save_ref.isChecked())
        if self._fit["mode"] == "scalar":
            self.result_payload = {
                "mode": "scalar",
                "ef_shift": self._fit["ef_shift"],
                "T": self._fit["T"],
                "fwhm_res": self._fit["fwhm_res"],
                "rms": self._fit["rms"],
                "k_min": self._fit["k_min"],
                "k_max": self._fit["k_max"],
                "save_as_reference": save_ref,
            }
        else:
            self.result_payload = {
                "mode": "poly",
                "poly_coefs": list(self._fit["poly_coefs"]),
                "T": self._fit["T"],
                "fwhm_res": self._fit["fwhm_res"],
                "rms": self._fit["rms"],
                "n_valid": self._fit["n_valid"],
                "k_min": self._fit["k_min"],
                "k_max": self._fit["k_max"],
                "save_as_reference": save_ref,
            }
        self.accept()
