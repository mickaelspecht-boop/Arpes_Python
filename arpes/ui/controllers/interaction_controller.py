"""Controller UI pour interactions souris/spinbox + scheduling redraws.

Sort de ArpesExplorer toute la logique d'interaction temps-réel :
- ROI fit (cliquer-glisser un rectangle sur la carte BM/MDC)
- click sur carte → MDC/EDC à (k, E)
- spinbox énergie → resync sélection
- view changed → redessine
- debouncers (`_redraw_timer`, `_fit_redraw_timer`) → redraws séquentiels
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from matplotlib.patches import Rectangle


class InteractionController:
    def __init__(self, parent):
        self._parent = parent

    @property
    def _params(self):
        return self._parent._params

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    # ---------------------------------------------------------------- callbacks
    def _on_view_changed(self):
        p = self._parent
        if p._current_path:
            entry = p._session.get_or_create(p._session.key_for_path(p._current_path))
            entry.view_mode = p._cmb_view.currentText()
        p._update_display_data()
        p._draw_bm()

    def _on_ev_spinbox_changed(self, val: float):
        p = self._parent
        if p._raw_data is None:
            return
        ev_arr = p._raw_data["ev_arr"]
        p._sel_ev = float(np.clip(val, ev_arr.min(), ev_arr.max()))
        p._draw_bm()
        p._draw_mdc_edc()
        if hasattr(p, "_mdc_fit_tabs") and p._tabs.currentIndex() == 1 and p._mdc_fit_tabs.currentIndex() == 1:
            p._draw_mdc_waterfall()

    def _schedule_model_redraw(self, _=None):
        self._parent._redraw_timer.start(120)

    def _schedule_fit_only_redraw(self, _=None):
        self._parent._fit_redraw_timer.start(120)

    def _on_model_changed(self, _=None):
        p = self._parent
        p._update_display_data()
        p._draw_bm()
        if p._tabs.currentIndex() == 1:
            p._draw_mdc_edc()
            if hasattr(p, "_mdc_fit_tabs") and p._mdc_fit_tabs.currentIndex() == 1:
                p._draw_mdc_waterfall()

    def _on_fit_only_changed(self, _=None):
        p = self._parent
        if p._tabs.currentIndex() != 1:
            return
        p._draw_mdc_edc()
        if hasattr(p, "_mdc_fit_tabs") and p._mdc_fit_tabs.currentIndex() == 1:
            p._draw_mdc_waterfall()

    # ---------------------------------------------------------------- ROI fit
    def _set_fit_roi_pick_mode(self, active: bool):
        p = self._parent
        active = bool(active)
        if not active and p._fit_roi_rect is not None:
            try:
                canvas = p._fit_roi_rect.figure.canvas
                p._fit_roi_rect.remove()
                canvas.draw_idle()
            except Exception:
                pass
        p._fit_roi_active = active
        p._fit_roi_start = None
        p._fit_roi_ax = None
        p._fit_roi_rect = None
        self._params.set_fit_roi_active(active)
        for canv in (getattr(p, "_bm_canvas", None), getattr(p, "_mdc_map_canvas", None)):
            if canv is None or not hasattr(canv, "canvas"):
                continue
            if active:
                canv.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else:
                canv.canvas.unsetCursor()
        if active:
            if p._tabs.currentIndex() not in (0, 1):
                p._tabs.setCurrentIndex(1)
            self._status("Sélection zone fit : cliquer-glisser un rectangle sur la carte.")

    def _on_fit_roi_press(self, event):
        p = self._parent
        if not p._fit_roi_active:
            return
        if event.inaxes not in (p._bm_canvas.ax, p._mdc_map_canvas.ax):
            return
        button = getattr(event.button, "value", event.button)
        if button != 1 or event.xdata is None or event.ydata is None:
            return
        p._fit_roi_start = (float(event.xdata), float(event.ydata))
        p._fit_roi_ax = event.inaxes
        if p._fit_roi_rect is not None:
            try:
                p._fit_roi_rect.remove()
            except Exception:
                pass
        p._fit_roi_rect = Rectangle(
            p._fit_roi_start, 0.0, 0.0,
            fill=False, edgecolor="#38bdf8", linewidth=1.4,
            linestyle="-", alpha=0.95, zorder=20,
        )
        event.inaxes.add_patch(p._fit_roi_rect)
        event.canvas.draw_idle()

    def _on_fit_roi_motion(self, event):
        p = self._parent
        if not p._fit_roi_active or p._fit_roi_start is None or p._fit_roi_rect is None:
            return
        if event.inaxes is not p._fit_roi_ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = p._fit_roi_start
        x1, y1 = float(event.xdata), float(event.ydata)
        p._fit_roi_rect.set_x(min(x0, x1))
        p._fit_roi_rect.set_y(min(y0, y1))
        p._fit_roi_rect.set_width(abs(x1 - x0))
        p._fit_roi_rect.set_height(abs(y1 - y0))
        event.canvas.draw_idle()

    def _on_fit_roi_release(self, event):
        p = self._parent
        if not p._fit_roi_active or p._fit_roi_start is None:
            return
        if event.inaxes is not p._fit_roi_ax or event.xdata is None or event.ydata is None:
            self._set_fit_roi_pick_mode(False)
            return
        x0, y0 = p._fit_roi_start
        x1, y1 = float(event.xdata), float(event.ydata)
        if abs(x1 - x0) < 1e-4 or abs(y1 - y0) < 1e-4:
            self._set_fit_roi_pick_mode(False)
            return
        self._apply_fit_roi_from_bounds(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        self._set_fit_roi_pick_mode(False)

    def _apply_fit_roi_from_bounds(self, k0: float, k1: float, e0: float, e1: float):
        p = self._parent
        if p._raw_data is None:
            return
        d = p._raw_data
        k0 = float(np.clip(k0, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        k1 = float(np.clip(k1, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        e0 = float(np.clip(e0, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        e1 = float(np.clip(e1, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        if k1 <= k0 or e1 <= e0:
            return
        for sp, val in (
            (self._params.sp_kmin, k0), (self._params.sp_kmax, k1),
            (self._params.sp_evs, e0), (self._params.sp_eve, e1),
        ):
            sp.blockSignals(True)
            sp.setValue(float(val))
            sp.blockSignals(False)
        p._sel_k = float((k0 + k1) * 0.5)
        p._sel_ev = float((e0 + e1) * 0.5)
        self._sync_ev_spinbox()
        self._params.params_changed.emit()
        p._draw_bm()
        p._draw_mdc_edc()
        self._status(
            f"Zone fit : k={k0:+.3f}→{k1:+.3f} π/a, "
            f"E={e0:+.3f}→{e1:+.3f} eV"
        )

    def _reset_fit_roi_range(self):
        p = self._parent
        if p._raw_data is None:
            return
        d = p._raw_data
        self._apply_fit_roi_from_bounds(
            float(np.nanmin(d["kpar"])), float(np.nanmax(d["kpar"])),
            float(np.nanmin(d["ev_arr"])), float(np.nanmax(d["ev_arr"])),
        )

    def _on_map_click(self, event):
        p = self._parent
        if p._fit_roi_active:
            return
        if event.inaxes not in (p._bm_canvas.ax, p._mdc_map_canvas.ax):
            return
        if event.xdata is None or event.ydata is None:
            return
        d = p._raw_data
        p._sel_ev = float(np.clip(event.ydata, d["ev_arr"].min(), d["ev_arr"].max()))
        p._sel_k = float(np.clip(event.xdata, d["kpar"].min(), d["kpar"].max()))
        self._sync_ev_spinbox()
        p._draw_bm()
        p._draw_mdc_edc()

    def _sync_ev_spinbox(self):
        self._params.sp_ev.blockSignals(True)
        self._params.sp_ev.setValue(self._parent._sel_ev)
        self._params.sp_ev.blockSignals(False)
