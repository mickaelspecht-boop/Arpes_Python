"""Drag the Γ center marker over the BM signal (marker-only model).

Lets the user grab the cyan Γ guide line on the band-map canvas and slide it onto
the high-symmetry point of the band. This moves **only the center marker**
(``sp_cx`` / ``fit_params.center_init``, the pair-symmetry center used by the MDC
fit) — the signal itself does NOT move and the k// axis is NOT recentered.

This is deliberately distinct from the automatic Γ detectors (``Auto Γ BM`` /
``Γ FS → BM``) which recenter the whole axis through
``GammaController.apply_resolved_gamma``. Here the user just says "the symmetry
center sits here on the data", with no axis shift, so the line follows the hand
and nothing jumps.

Why a manual path at all: the FS-derived Γ relies on matching hv (kz), azimuth
and the k-calibration between FS and BM. When those are uncertain, a hand-placed
center is the honest fallback.

Live drag only moves the existing guide-line artist (no data recompute → no lag);
the value is committed to ``sp_cx`` once on release. Free functions take
``window`` (ArpesExplorer) as first argument; wiring lives in ``panels.py`` via
closures so no PROXY_MAP entry is consumed. The handlers run on the BM canvas
only and guard the fit-ROI / fit-select press handlers via
``window._gamma_drag_active`` (set on press, cleared one event loop after release
so sibling release handlers still see the gesture).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer

_GRAB_PX = 8.0
_MOVE_THRESHOLD_PX = 3.0


def wire_gamma_drag(canvas_widget, window) -> None:
    """Connect the Γ-drag handlers on the BM canvas.

    Must be called before the generic map-click handlers so the press handler
    sets ``_gamma_drag_active`` before they run and they can bail on it.
    """
    canvas = canvas_widget.canvas
    canvas.mpl_connect("button_press_event", lambda ev: on_press(window, ev))
    canvas.mpl_connect("motion_notify_event", lambda ev: on_motion(window, ev))
    canvas.mpl_connect("button_release_event", lambda ev: on_release(window, ev))


def _bm_ax(window):
    canvas = getattr(window, "_bm_canvas", None)
    if canvas is None or not getattr(canvas, "fig", None) or not canvas.fig.axes:
        return None
    return canvas.fig.axes[0]


def _gamma_line_pixel_x(window, ax):
    """X pixel of the current Γ guide line (sp_cx value) on ``ax``, or None."""
    sp_cx = getattr(window._params, "sp_cx", None)
    if sp_cx is None:
        return None
    try:
        cx = float(sp_cx.value())
    except (TypeError, ValueError):
        return None
    ymid = float(np.mean(ax.get_ylim()))
    try:
        return float(ax.transData.transform((cx, ymid))[0])
    except Exception:
        return None


def _near_gamma_line(window, event, ax) -> bool:
    px = _gamma_line_pixel_x(window, ax)
    if px is None or event.x is None:
        return False
    return abs(float(event.x) - px) <= _GRAB_PX


def _draggable(window, event) -> bool:
    """Common preconditions for grabbing / hovering the Γ line."""
    if getattr(window, "_fit_roi_active", False):
        return False
    # A click here would otherwise both place an HS anchor and start a Γ drag.
    if getattr(window, "_theory_anchor_pick_active", False):
        return False
    raw = getattr(window, "_raw_data", None)
    if raw is None:
        return False
    ax = _bm_ax(window)
    if ax is None or event.inaxes is not ax:
        return False
    return True


def on_press(window, event) -> None:
    button = getattr(event.button, "value", event.button)
    if button != 1 or event.x is None:
        return
    if not _draggable(window, event):
        return
    ax = _bm_ax(window)
    if not _near_gamma_line(window, event, ax):
        return
    window._gamma_drag_active = True
    window._gamma_drag_press_px = float(event.x)
    window._gamma_drag_moved = False
    try:
        event.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
    except Exception:
        pass


def on_motion(window, event) -> None:
    if getattr(window, "_gamma_drag_active", False):
        if event.xdata is None:
            return
        press_px = getattr(window, "_gamma_drag_press_px", None)
        if press_px is not None and event.x is not None:
            if abs(float(event.x) - press_px) >= _MOVE_THRESHOLD_PX:
                window._gamma_drag_moved = True
        # Marker-only: the signal stays put, the guide line follows the cursor.
        try:
            window._update_gamma_preview(float(event.xdata))
            event.canvas.draw_idle()
        except Exception:
            pass
        window._status(f"Γ center → {float(event.xdata):+.3f} π/a (release to set)")
        return
    # Hover affordance: horizontal-resize cursor when over the grab zone.
    if button_idle_over_line(window, event):
        try:
            event.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
        except Exception:
            pass
    else:
        _restore_cursor(window, event)


def button_idle_over_line(window, event) -> bool:
    if not _draggable(window, event):
        return False
    ax = _bm_ax(window)
    return _near_gamma_line(window, event, ax)


def _restore_cursor(window, event) -> None:
    # Only reset if we are the ones who set it (avoid stomping ROI cross cursor).
    if getattr(window, "_fit_roi_active", False):
        return
    try:
        event.canvas.unsetCursor()
    except Exception:
        pass


def on_release(window, event) -> None:
    if not getattr(window, "_gamma_drag_active", False):
        return
    ax = _bm_ax(window)
    try:
        event.canvas.unsetCursor()
    except Exception:
        pass
    moved = bool(getattr(window, "_gamma_drag_moved", False))
    if moved and event.xdata is not None and event.inaxes is ax:
        _commit(window, float(event.xdata))
    elif not moved:
        # Treated as a click (no drag): snap the guide line back to the current
        # center so a stray press near it does not nudge anything.
        try:
            window._update_gamma_preview(float(window._params.sp_cx.value()))
            event.canvas.draw_idle()
        except Exception:
            pass
    # Clear one event loop later so sibling release handlers still see the
    # gesture and skip their own selection logic for this click.
    QTimer.singleShot(0, lambda: setattr(window, "_gamma_drag_active", False))


def _commit(window, k_center: float) -> None:
    """Set the pair-symmetry center to ``k_center`` (π/a). No axis shift."""
    raw = getattr(window, "_raw_data", None)
    if raw is None:
        return
    kpar = np.asarray(raw.get("kpar"), dtype=float)
    if kpar.size:
        k_center = float(np.clip(k_center, np.nanmin(kpar), np.nanmax(kpar)))
    sp_cx = getattr(window._params, "sp_cx", None)
    old_center = float(sp_cx.value()) if sp_cx is not None else 0.0
    if sp_cx is not None:
        # setValue fires valueChanged → gamma_center_preview (redraws the guide
        # line) + fit_only_changed (refreshes the fit overlay), same as a manual
        # spinbox edit. This is the single place the drag writes the center.
        sp_cx.setValue(float(k_center))
    # Persist on the entry so it survives reload (a plain spinbox edit only
    # persists on the next fit/save; an explicit gesture should stick now).
    path = getattr(window, "_current_path", None)
    if path:
        try:
            entry = window._session.get_or_create(window._session.key_for_path(path))
            entry.fit_params.center_init = float(k_center)
            window._session.save()
        except Exception:
            pass
    # The manual center is the master Γ reference: keep the DFT overlay glued to
    # it so its bands and the Γ mirror axis follow the drag.
    try:
        from arpes.ui.controllers.theory_anchor_ctrl import track_gamma_center_delta
        track_gamma_center_delta(window, float(k_center) - old_center)
    except Exception:
        pass
    window._status(f"Γ center set to {k_center:+.3f} π/a (signal unchanged)")
