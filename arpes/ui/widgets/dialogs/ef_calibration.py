"""EF calibration dialog — preview + fit (scalar / per-column)."""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets.canvas import MplCanvas
from arpes.ui.widgets.plots import (
    auto_ef_window,
    fit_fermi_edge,
    fit_fermi_edge_per_column,
)


class EFCalibrationDialog(QDialog):
    """Interactive EF calibration: scalar or per-column (polynomial).

    Inputs: data (n_k, n_E), kpar, ev_arr, T_init, half_width_init, source_name.
    Outputs (via .result_payload after accept):
        {"mode": "scalar"|"poly", "ef_offset": float | None,
         "poly_coefs": [...] | None, "T": float, "fwhm_res": float,
         "rms": float, "n_valid": int, "k_min": float, "k_max": float,
         "save_as_reference": bool}
    """

    def __init__(self, parent, data, kpar, ev_arr, T_init=28.0,
                 half_width_init=0.15, source_name="", current_offset=0.0,
                 metadata: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("EF Calibration")
        self.resize(940, 660)
        self._data  = np.asarray(data, dtype=float)
        self._kpar  = np.asarray(kpar, dtype=float)
        self._ev    = np.asarray(ev_arr, dtype=float)
        self._fit   = None
        self.result_payload = None
        self._current_offset = float(current_offset)
        self._metadata = metadata or {}
        self._ef_search = self._default_ef_search_range()
        self._manual_window: tuple[float, float] | None = None
        self._window_span = None
        self._window_drag: dict | None = None
        # Manual EDC x-range from scroll-zoom (None = auto-frame around the
        # edge). Large EF offsets push the edge far from 0, so the view is
        # framed on the fit window by default instead of the full axis.
        self._view_override: tuple[float, float] | None = None

        # ── widgets ────────────────────────────────────────────────────────────
        lay = QHBoxLayout(self)

        # Left panel: controls grouped into numbered steps (mode → params →
        # fit → result → apply), so the workflow reads top-to-bottom.
        left = QWidget(); left.setMaximumWidth(360)
        col = QVBoxLayout(left); col.setSpacing(8)

        info = QLabel(
            f"Source: {source_name or '—'}   "
            f"({self._data.shape[0]} k × {self._data.shape[1]} E)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa0a6; font-size:11px;")
        col.addWidget(info)

        # 1 · Fit mode ----------------------------------------------------------
        grp_mode = QGroupBox("1 · Fit mode")
        ml = QVBoxLayout(grp_mode); ml.setSpacing(2)
        self.rb_scalar = QRadioButton("Scalar — single average EF")
        self.rb_poly   = QRadioButton("Per column — EF(k) polynomial")
        self.rb_scalar.setChecked(True)
        self.rb_scalar.setToolTip(
            "One rigid EF shift for the whole map (fast, robust — the usual choice).")
        self.rb_poly.setToolTip(
            "Fit EF independently per k column and model the detector curvature EF(k).")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_scalar); grp.addButton(self.rb_poly)
        ml.addWidget(self.rb_scalar); ml.addWidget(self.rb_poly)
        self.rb_poly.toggled.connect(self._on_mode_changed)
        col.addWidget(grp_mode)

        # 2 · Fit parameters ----------------------------------------------------
        grp_params = QGroupBox("2 · Fit parameters")
        pf = QFormLayout(grp_params); pf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.sp_T = QDoubleSpinBox(); self.sp_T.setRange(1.0, 400.0); self.sp_T.setDecimals(1)
        self.sp_T.setValue(float(T_init)); self.sp_T.setSuffix(" K")
        self.sp_T.setToolTip("Temperature used to fix kBT in the Fermi-Dirac distribution.")
        pf.addRow("Temperature", self.sp_T)

        self.sp_hw = QDoubleSpinBox(); self.sp_hw.setRange(0.03, 0.50); self.sp_hw.setDecimals(3)
        self.sp_hw.setSingleStep(0.01); self.sp_hw.setValue(float(half_width_init)); self.sp_hw.setSuffix(" eV")
        self.sp_hw.setToolTip("Half-width of the fit window around the estimated EF.")
        pf.addRow("Half-window", self.sp_hw)

        self.chk_auto = QCheckBox("Auto-place window (max gradient)")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "Centers the window on the maximum gradient of the average EDC.\n"
            f"Current search range: {self._ef_search[0]:+.2f} to {self._ef_search[1]:+.2f} eV.\n"
            "Drag the yellow band on the plot to place it by hand instead."
        )
        self.chk_auto.stateChanged.connect(self._on_auto_window_toggled)
        pf.addRow(self.chk_auto)

        self.sp_sigma = QDoubleSpinBox(); self.sp_sigma.setRange(0.005, 0.10); self.sp_sigma.setDecimals(3)
        self.sp_sigma.setSingleStep(0.005); self.sp_sigma.setValue(0.025); self.sp_sigma.setSuffix(" eV")
        self.sp_sigma.setToolTip("Initial Gaussian resolution sigma for the fit (FWHM = 2.355·σ).")
        pf.addRow("Resolution σ", self.sp_sigma)
        col.addWidget(grp_params)

        # Per-column options — only meaningful in poly mode, greyed otherwise. --
        self.grp_poly = QGroupBox("Per-column options")
        qf = QFormLayout(self.grp_poly); qf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_deg = QSpinBox(); self.sp_deg.setRange(0, 4); self.sp_deg.setValue(2)
        self.sp_deg.setToolTip("Degree of the EF(k) polynomial. 0 = constant, 2 = parabola (default).")
        qf.addRow("Poly degree", self.sp_deg)
        col.addWidget(self.grp_poly)

        # 3 · Primary action ----------------------------------------------------
        self.btn_fit = QPushButton("▶  Fit")
        self.btn_fit.setStyleSheet(
            "background:#2a6099;color:white;font-weight:bold;padding:7px;")
        self.btn_fit.setToolTip("Run the Fermi-edge fit on the selected window (Enter).")
        self.btn_fit.setDefault(True)
        self.btn_fit.clicked.connect(self._do_fit)
        col.addWidget(self.btn_fit)

        # 4 · Result ------------------------------------------------------------
        grp_res = QGroupBox("Result")
        rl = QVBoxLayout(grp_res)
        self.lbl_result = QLabel("Run a fit to see the proposed EF offset.")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_result.setStyleSheet("color:#bbb; padding:2px;")
        rl.addWidget(self.lbl_result)
        col.addWidget(grp_res, 1)  # stretch: result box absorbs extra height

        # 5 · Apply -------------------------------------------------------------
        self.chk_save_ref = QCheckBox("Save as folder reference (Au)")
        self.chk_save_ref.setToolTip(
            "The correction will be offered as a reference applicable to other\n"
            "files in the folder via 'Apply reference Au'."
        )
        col.addWidget(self.chk_save_ref)

        btns = QDialogButtonBox()
        self.btn_apply = btns.addButton(
            "Apply to this file", QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_apply.setEnabled(False)
        # Accent style is applied only once a fit succeeds (see _do_fit): a
        # custom background overrides Fusion's disabled dimming, so a greened
        # button would look clickable while still disabled.
        self.btn_cancel = btns.addButton(QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_apply)
        btns.rejected.connect(self.reject)
        col.addWidget(btns)

        lay.addWidget(left)
        self._on_mode_changed()

        # Panneau droit : preview matplotlib
        self._fig = Figure(figsize=(6, 5))
        self._canvas = FigureCanvas(self._fig)
        self._ax_edc  = self._fig.add_subplot(2, 1, 1)
        self._ax_poly = self._fig.add_subplot(2, 1, 2)
        self._fig.tight_layout()
        right = QWidget(); rl = QVBoxLayout(right); rl.addWidget(self._canvas)
        view_bar = QHBoxLayout()
        self.chk_zoom_edge = QCheckBox("Zoom to edge")
        self.chk_zoom_edge.setChecked(True)
        self.chk_zoom_edge.setToolTip(
            "Frame the EDC on the fit window instead of the full energy axis —\n"
            "keeps the Fermi edge large even with a big EF offset.\n"
            "Scroll to zoom, double-click the EDC to reset the view."
        )
        self.chk_zoom_edge.stateChanged.connect(self._on_zoom_toggled)
        view_bar.addWidget(self.chk_zoom_edge)
        hint = QLabel("scroll = zoom · double-click = reset")
        hint.setStyleSheet("color:#777; font-size:10px;")
        view_bar.addWidget(hint)
        view_bar.addStretch(1)
        rl.addLayout(view_bar)
        lay.addWidget(right, 1)

        self._canvas.mpl_connect("button_press_event", self._on_window_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_window_motion)
        self._canvas.mpl_connect("button_release_event", self._on_window_release)
        self._canvas.mpl_connect("scroll_event", self._on_edc_scroll)
        self._draw_initial_preview()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _default_ef_search_range(self) -> tuple[float, float]:
        fmt = str(self._metadata.get("fs_source") or self._metadata.get("source_format") or "").lower()
        lab = str(self._metadata.get("lab") or "").lower()
        ref = str(self._metadata.get("energy_reference") or "").lower()
        # BESSY Center Energy mode: the experimenter may have centered the
        # analyzer at any EF offset. First target the detected edge in the EDC
        # (max intensity drop), then widen around it. If detection fails, use a
        # broad fallback covering the full axis.
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

    def _edc_mean(self) -> np.ndarray:
        return np.nanmean(self._data, axis=0)

    def _fit_window(self) -> tuple[float, float]:
        hw = float(self.sp_hw.value())
        if self.chk_auto.isChecked():
            return auto_ef_window(self._ev, self._edc_mean(), half_width=hw, search=self._ef_search)
        if self._manual_window is not None:
            return self._manual_window
        return (-hw, hw)

    def _clamp_window(self, lo: float, hi: float) -> tuple[float, float]:
        ev_min = float(np.nanmin(self._ev))
        ev_max = float(np.nanmax(self._ev))
        lo, hi = sorted((float(lo), float(hi)))
        min_width = max(0.01, float(np.nanmedian(np.abs(np.diff(np.sort(self._ev))))) * 4.0)
        width = max(hi - lo, min_width)
        if lo < ev_min:
            lo = ev_min
            hi = lo + width
        if hi > ev_max:
            hi = ev_max
            lo = hi - width
        lo = max(ev_min, lo)
        hi = min(ev_max, hi)
        if hi - lo < min_width:
            mid = 0.5 * (lo + hi)
            lo = max(ev_min, mid - min_width / 2.0)
            hi = min(ev_max, mid + min_width / 2.0)
        return (float(lo), float(hi))

    def _set_manual_window(self, lo: float, hi: float, *, redraw: bool = True) -> None:
        win = self._clamp_window(lo, hi)
        self._manual_window = win
        self.chk_auto.blockSignals(True)
        self.chk_auto.setChecked(False)
        self.chk_auto.blockSignals(False)
        half = max((win[1] - win[0]) / 2.0, self.sp_hw.minimum())
        self.sp_hw.blockSignals(True)
        self.sp_hw.setValue(min(half, self.sp_hw.maximum()))
        self.sp_hw.blockSignals(False)
        if redraw:
            self._draw_window_span(win, label="manual window")
            self._canvas.draw_idle()

    def _draw_window_span(self, win: tuple[float, float] | None = None, *, label: str = "window"):
        if self._window_span is not None:
            try:
                self._window_span.remove()
            except Exception:
                pass
            self._window_span = None
        win = win or self._fit_window()
        self._window_span = self._ax_edc.axvspan(
            win[0], win[1], color="gold", alpha=0.18, label=label, zorder=0
        )
        return self._window_span

    def _on_auto_window_toggled(self):
        if self.chk_auto.isChecked():
            self._manual_window = None
        if self._ax_edc.has_data():
            self._draw_initial_preview()

    def _on_mode_changed(self):
        """Show the per-column options only in polynomial mode (irrelevant in
        scalar mode, where a single EF is fitted)."""
        self.grp_poly.setVisible(self.rb_poly.isChecked())

    def _on_window_press(self, event):
        if event.inaxes is not self._ax_edc or event.xdata is None or event.button != 1:
            return
        if getattr(event, "dblclick", False):
            # Double-click resets the scroll-zoom back to the auto frame.
            self._view_override = None
            self._apply_edc_xlim(self._fit["window"] if self._fit else None)
            self._canvas.draw_idle()
            return
        lo, hi = self._fit_window()
        x = float(event.xdata)
        width = max(hi - lo, 1e-12)
        edge_tol = max(width * 0.18, 0.015)
        if lo - edge_tol <= x <= hi + edge_tol:
            if abs(x - lo) <= edge_tol:
                mode = "left"
            elif abs(x - hi) <= edge_tol:
                mode = "right"
            else:
                mode = "move"
        else:
            mode = "move"
            lo, hi = x - width / 2.0, x + width / 2.0
            lo, hi = self._clamp_window(lo, hi)
            self._set_manual_window(lo, hi, redraw=True)
        self._window_drag = {"mode": mode, "x0": x, "lo": float(lo), "hi": float(hi)}

    def _on_window_motion(self, event):
        if not self._window_drag or event.inaxes is not self._ax_edc or event.xdata is None:
            return
        x = float(event.xdata)
        d = self._window_drag
        lo = float(d["lo"])
        hi = float(d["hi"])
        if d["mode"] == "left":
            lo = min(x, hi - 0.01)
        elif d["mode"] == "right":
            hi = max(x, lo + 0.01)
        else:
            dx = x - float(d["x0"])
            lo, hi = lo + dx, hi + dx
        self._set_manual_window(lo, hi, redraw=True)

    def _on_window_release(self, _event):
        self._window_drag = None

    # ── EDC view framing (zoom) ──────────────────────────────────────────────
    def _view_xlim(self, focus: tuple[float, float] | None = None) -> tuple[float, float]:
        """x-range for the EDC axis.

        Full energy axis when zoom is off; an explicit scroll-zoom override when
        set; otherwise auto-framed on the fit window (with margin) so a large EF
        offset does not squash the edge into a few pixels.
        """
        emn, emx = float(np.nanmin(self._ev)), float(np.nanmax(self._ev))
        if not self.chk_zoom_edge.isChecked():
            return (emn, emx)
        if self._view_override is not None:
            lo, hi = self._view_override
            return (max(emn, float(lo)), min(emx, float(hi)))
        win = focus or self._fit_window()
        lo, hi = float(win[0]), float(win[1])
        pad = max(0.05, (hi - lo) * 1.5)
        return (max(emn, lo - pad), min(emx, hi + pad))

    def _apply_edc_xlim(self, focus: tuple[float, float] | None = None) -> None:
        lo, hi = self._view_xlim(focus)
        if hi > lo:
            self._ax_edc.set_xlim(lo, hi)

    def _on_zoom_toggled(self):
        self._view_override = None
        self._apply_edc_xlim(self._fit["window"] if self._fit else None)
        self._canvas.draw_idle()

    def _on_edc_scroll(self, event):
        if event.inaxes is not self._ax_edc or event.xdata is None:
            return
        lo, hi = self._ax_edc.get_xlim()
        x = float(event.xdata)
        factor = 0.8 if event.button == "up" else 1.25  # wheel up = zoom in
        nlo = x - (x - lo) * factor
        nhi = x + (hi - x) * factor
        emn, emx = float(np.nanmin(self._ev)), float(np.nanmax(self._ev))
        nlo, nhi = max(emn, nlo), min(emx, nhi)
        min_w = max(0.02, float(np.nanmedian(np.abs(np.diff(np.sort(self._ev))))) * 6.0)
        if nhi - nlo < min_w:
            return
        self._view_override = (nlo, nhi)
        if not self.chk_zoom_edge.isChecked():
            self.chk_zoom_edge.blockSignals(True)
            self.chk_zoom_edge.setChecked(True)
            self.chk_zoom_edge.blockSignals(False)
        self._ax_edc.set_xlim(nlo, nhi)
        self._canvas.draw_idle()

    def _draw_initial_preview(self):
        edc = self._edc_mean()
        self._ax_edc.clear()
        self._ax_edc.plot(self._ev, edc, "k-", lw=1.2, label="Average EDC")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._draw_window_span(label="fit window")
        self._ax_edc.set_xlabel(r"$E - E_F$ (eV)"); self._ax_edc.set_ylabel(r"$I$ (counts)")
        self._ax_edc.set_title("EDC averaged over k — drag the yellow region to select the fit range")
        self._ax_edc.legend(fontsize=8)
        self._apply_edc_xlim()
        self._ax_poly.clear()
        self._ax_poly.text(0.5, 0.5, "Click 'Fit' to start the calibration",
                           ha="center", va="center", transform=self._ax_poly.transAxes,
                           fontsize=10, color="gray")
        self._ax_poly.set_axis_off()
        self._canvas.draw_idle()

    def _do_fit(self):
        T  = self.sp_T.value()
        sig = self.sp_sigma.value()
        edc = self._edc_mean()
        self._view_override = None  # reframe the result on its fit window

        if self.rb_scalar.isChecked():
            win = self._fit_window()
            try:
                _ax = Figure().add_subplot(111)
                r = fit_fermi_edge(
                    self._ev, edc,
                    temperature_K=T, fit_range=win,
                    sigma_resolution_init=sig, fix_kBT=True,
                    units="binding", ax=_ax, verbose=False,
                )
            except Exception as e:
                self.lbl_result.setText(f"Warning: fit failed: {e}")
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
                f"<b>Scalar mode</b><br>"
                f"EF fit: {ef*1000:+.1f} meV (±{efe*1000:.1f} meV)<br>"
                f"Resolution FWHM: {fwhm*1000:.0f} meV<br>"
                f"RMS residual: {resid:.4f}<br>"
                f"Window: [{win[0]*1000:+.0f}, {win[1]*1000:+.0f}] meV"
                f"<div style='margin-top:6px;color:#5fb85f;font-size:13px;'>"
                f"<b>→ new offset: {new_offset:.4f} eV</b></div>"
            )
        else:
            try:
                r = fit_fermi_edge_per_column(
                    self._data, self._kpar, self._ev,
                    temperature_K=T, half_width=self.sp_hw.value(),
                    sigma_resolution_init=sig,
                    poly_deg=self.sp_deg.value(),
                    auto_window=self.chk_auto.isChecked(),
                    ef_search=self._ef_search,
                    fit_range=None if self.chk_auto.isChecked() else self._fit_window(),
                    verbose=False,
                )
            except Exception as e:
                self.lbl_result.setText(f"Warning: per-column fit failed: {e}")
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
                f"<b>Per-column mode (poly deg {self.sp_deg.value()})</b><br>"
                f"Valid columns: {r['n_valid']}/{self._data.shape[0]}<br>"
                f"&lt;EF&gt;: {r['mean_ef']*1000:+.1f} meV<br>"
                f"Median FWHM: {r['mean_fwhm']*1000:.0f} meV<br>"
                f"Poly RMS residual: {r['rms']*1000:.1f} meV<br>"
                f"Window: [{r['window'][0]*1000:+.0f}, {r['window'][1]*1000:+.0f}] meV"
            )
        self.btn_apply.setEnabled(True)
        self.btn_apply.setStyleSheet(
            "background:#1a6b3a;color:white;font-weight:bold;padding:6px;")
        # Fit succeeded → the commit action becomes the keyboard default so
        # Enter applies (it was the Fit button until now).
        self.btn_fit.setDefault(False)
        self.btn_apply.setDefault(True)

    def _draw_scalar_preview(self, fit_result, win):
        self._ax_edc.clear()
        edc = self._edc_mean()
        win_mask = (self._ev >= win[0]) & (self._ev <= win[1])
        norm = max(float(np.nanmax(edc[win_mask])) if np.any(win_mask) else float(np.nanmax(edc)), 1e-9)
        self._ax_edc.plot(self._ev, edc / norm, "k-", lw=1.0,
                          label="Normalised EDC")
        self._ax_edc.plot(fit_result["model_ev"], fit_result["model_I"], "r-", lw=2.0,
                          label=f"FD fit  EF={fit_result['EF']*1000:+.0f} meV")
        self._ax_edc.axvline(fit_result["EF"], color="red", lw=1.0)
        self._draw_window_span(win, label="window")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._apply_edc_xlim(win)
        self._ax_edc.set_xlabel(r"$E - E_F$ (eV)"); self._ax_edc.set_ylabel(r"$I/I_{\max}$")
        self._ax_edc.legend(fontsize=8)
        self._ax_poly.clear()
        self._ax_poly.text(0.5, 0.5,
                           "Scalar mode: no EF(k) curve.\n"
                           "Switch to 'Per column' to see the dispersion.",
                           ha="center", va="center", transform=self._ax_poly.transAxes,
                           fontsize=9, color="gray")
        self._ax_poly.set_axis_off()
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _draw_poly_preview(self, r):
        self._ax_edc.clear()
        edc = self._edc_mean()
        win = r["window"]
        win_mask = (self._ev >= win[0]) & (self._ev <= win[1])
        norm = max(float(np.nanmax(edc[win_mask])) if np.any(win_mask) else float(np.nanmax(edc)), 1e-9)
        self._ax_edc.plot(self._ev, edc / norm, "k-", lw=1.0,
                          label="Average EDC")
        self._draw_window_span(win, label="window")
        self._ax_edc.axvline(r["mean_ef"], color="red", lw=1.0,
                             label=f"<EF>={r['mean_ef']*1000:+.0f} meV")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._apply_edc_xlim(win)
        self._ax_edc.set_xlabel(r"$E - E_F$ (eV)"); self._ax_edc.set_ylabel(r"$I/I_{\max}$")
        self._ax_edc.legend(fontsize=8)

        self._ax_poly.clear()
        kp = self._kpar
        ef_raw = r["ef_per_col"]
        ef_sm  = r["ef_smooth"]
        valid = np.isfinite(ef_raw)
        self._ax_poly.plot(kp[valid], ef_raw[valid] * 1000, ".", color="#888", ms=3,
                           label="per-column fits")
        self._ax_poly.plot(kp, ef_sm * 1000, "r-", lw=2.0,
                           label=f"poly deg {self.sp_deg.value()}")
        self._ax_poly.axhline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_poly.set_xlabel(r"$k$ (π/a)"); self._ax_poly.set_ylabel(r"$E_F$ (meV)")
        self._ax_poly.set_title("EF(k) — detector curvature")
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
