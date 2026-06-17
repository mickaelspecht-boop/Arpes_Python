"""Place high-symmetry points on the BM and align the DFT onto them.

User flow: pick a high-symmetry label (Γ, X, M…) in the theory panel, enable
"Pick on BM", click where that point sits on the band map. Each click stores an
anchor ``{label, k}`` on the overlay. "Fit & align" turns the anchors into
``k_scale`` + ``k_shift`` (via ``arpes.theory.anchor_calib``) so the DFT overlay
maps onto the chosen points.

Free functions take ``window`` (ArpesExplorer); wiring lives in ``panels.py`` via
closures so no PROXY_MAP entry is consumed. Anchors persist on the overlay dict
(``overlay["anchors"]``) so they survive reload. Markers are drawn from
``draw_anchor_markers`` inside the theory overlay draw, on every view.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from arpes.theory.anchor_calib import fit_scale_shift, local_k_for_label

_MARKER_COLOR = "#f0abfc"
_PLACED_COLOR = "#34d399"  # green tint for already-placed labels in the combo


def _norm_label(value) -> str:
    return str(value or "").strip().upper().replace("GAMMA", "Γ")


def wire(window) -> None:
    """Connect the panel signals + the BM click handler (closures, 0 PROXY)."""
    p = getattr(window, "_params", None)
    if p is None:
        return
    p.theory_anchor_pick_toggled.connect(lambda on: set_pick_active(window, on))
    p.theory_anchor_apply_requested.connect(lambda: apply_calibration(window))
    p.theory_anchor_clear_requested.connect(lambda: clear_anchors(window))
    bm = getattr(window, "_bm_canvas", None)
    if bm is not None:
        bm.canvas.mpl_connect("button_press_event", lambda ev: on_bm_click(window, ev))


def _theory_ctrl(window):
    return getattr(window, "_theory_overlay_ctrl", None)


def _overlay(window) -> dict:
    ctrl = _theory_ctrl(window)
    if ctrl is None:
        return {}
    return dict(ctrl._current_overlay() or {})


def populate_labels(window) -> None:
    """Fill the HS-point combo from the imported DFT labels (keep selection)."""
    combo = getattr(window._params, "cmb_theory_anchor_label", None)
    if combo is None:
        return
    overlay = _overlay(window)
    labels = ((overlay.get("data") or {}).get("labels")) or []
    placed = {_norm_label(a.get("label")) for a in (overlay.get("anchors") or [])}
    names: list[str] = []
    for item in labels:
        name = str(item.get("label") or "").strip()
        if name and name not in names:
            names.append(name)
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(names)
    # Tint already-placed labels green so the user sees what's done at a glance.
    for i, name in enumerate(names):
        if _norm_label(name) in placed:
            combo.setItemData(i, QColor(_PLACED_COLOR), Qt.ItemDataRole.ForegroundRole)
    if current in names:
        combo.setCurrentText(current)
    combo.blockSignals(False)


def set_pick_active(window, on: bool) -> None:
    window._theory_anchor_pick_active = bool(on)
    if on:
        populate_labels(window)
        if not (_overlay(window).get("data") or {}).get("labels"):
            window._status("Import a DFT first: no high-symmetry labels to place.")
        else:
            window._status("Click on the BM map to place the selected Γ/X/M point.")
    bm = getattr(window, "_bm_canvas", None)
    if bm is not None:
        try:
            if on:
                bm.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else:
                bm.canvas.unsetCursor()
        except Exception:
            pass


def on_bm_click(window, event) -> None:
    if not getattr(window, "_theory_anchor_pick_active", False):
        return
    # Don't fire while another map gesture owns the click (ROI / Γ drag).
    if getattr(window, "_fit_roi_active", False) or getattr(window, "_gamma_drag_active", False):
        return
    bm = getattr(window, "_bm_canvas", None)
    ax = bm.fig.axes[0] if (bm is not None and bm.fig.axes) else None
    if ax is None or event.inaxes is not ax or event.xdata is None:
        return
    if getattr(event.button, "value", event.button) != 1:
        return
    combo = getattr(window._params, "cmb_theory_anchor_label", None)
    label = combo.currentText().strip() if combo is not None else ""
    if not label:
        window._status("Choose a high-symmetry point before placing it.")
        return
    overlay = _overlay(window)
    anchors = [a for a in (overlay.get("anchors") or [])
               if str(a.get("label")) != label]
    anchors.append({"label": label, "k": float(event.xdata)})
    overlay["anchors"] = anchors
    ctrl = _theory_ctrl(window)
    if ctrl is not None:
        ctrl._save_overlay(overlay)
    populate_labels(window)  # refresh the green "placed" tint
    _redraw(window)
    window._status(f"Placed {label} at k={float(event.xdata):+.3f} π/a "
                   f"({len(anchors)} point(s)). 'Fit & align' to apply.")


def apply_calibration(window) -> None:
    overlay = _overlay(window)
    anchors = overlay.get("anchors") or []
    if not anchors:
        window._status("No placed points. Enable 'Pick on BM' and click the map.")
        return
    data = overlay.get("data") or {}
    config = window._params.theory_overlay_config()
    pairs = []
    skipped = []
    for a in anchors:
        u = local_k_for_label(data, config, a.get("label"))
        if u is None:
            skipped.append(str(a.get("label")))
            continue
        pairs.append((u, float(a.get("k"))))
    if skipped:
        window._status(
            "Ignored points outside the current segment / unknown: "
            + ", ".join(skipped) + ". Pick the matching segment first."
        )
    res = fit_scale_shift(pairs, current_scale=float(window._params.sp_theory_kscale.value()))
    if res is None:
        window._status("Cannot align: need ≥1 usable point (≥2 for the scale).")
        return
    scale, shift = res
    for sp, val in ((window._params.sp_theory_kscale, scale),
                    (window._params.sp_theory_dk, shift)):
        sp.blockSignals(True)
        sp.setValue(float(val))
        sp.blockSignals(False)
    window._on_theory_overlay_changed()
    # Lock the manual Γ center to the placed Γ so the cyan center marker, the DFT
    # Γ and the mirror axis all coincide (one coherent reference).
    centered = _sync_center_to_gamma_anchor(window, anchors, scale, shift, data, config)
    msg = f"Aligned DFT on {len(pairs)} point(s): scale={scale:.3f}, Δk={shift:+.3f}."
    if centered is not None:
        msg += f" Γ center set to {centered:+.3f} π/a."
    window._status(msg)


def _sync_center_to_gamma_anchor(window, anchors, scale, shift, data, config) -> float | None:
    """Set sp_cx to the displayed Γ when a Γ anchor was placed; else no-op."""
    has_gamma = any(_norm_label(a.get("label")) == "Γ" for a in anchors)
    if not has_gamma:
        return None
    u = local_k_for_label(data, config, "Γ")
    if u is None:
        return None
    g = float(u) * float(scale) + float(shift)
    sp_cx = getattr(window._params, "sp_cx", None)
    if sp_cx is None:
        return None
    sp_cx.setValue(g)  # fires gamma_center_preview → redraws the marker live
    return g


def clear_anchors(window) -> None:
    overlay = _overlay(window)
    overlay["anchors"] = []
    ctrl = _theory_ctrl(window)
    if ctrl is not None:
        ctrl._save_overlay(overlay)
    populate_labels(window)  # clear the green "placed" tint
    _redraw(window)
    window._status("Cleared placed high-symmetry points.")


def track_gamma_center_delta(window, delta: float) -> None:
    """Shift the DFT overlay by ``delta`` so its Γ tracks the manual center.

    Called when the user drags the Γ center on the BM: the manual center is the
    master reference, so the DFT (and thus the Γ mirror axis) follows it. No-op
    when no overlay is enabled or the move is negligible.
    """
    if abs(float(delta)) < 1e-9:
        return
    p = getattr(window, "_params", None)
    sp_dk = getattr(p, "sp_theory_dk", None) if p is not None else None
    if sp_dk is None:
        return
    try:
        if not p.theory_overlay_config().get("enabled"):
            return
    except Exception:
        return
    sp_dk.blockSignals(True)
    sp_dk.setValue(float(sp_dk.value()) + float(delta))
    sp_dk.blockSignals(False)
    try:
        window._on_theory_overlay_changed()
    except Exception:
        pass


def _redraw(window) -> None:
    try:
        window._draw_current_view(include_curves=False, overlays_only=True)
    except Exception:
        pass


def draw_anchor_markers(ctrl, ax) -> None:
    """Draw the placed high-symmetry markers (called from the theory draw)."""
    anchors = (dict(ctrl._current_overlay() or {}).get("anchors")) or []
    if not anchors:
        return
    y_top = max(ax.get_ylim())
    for a in anchors:
        try:
            k = float(a.get("k"))
        except (TypeError, ValueError):
            continue
        ax.axvline(k, color=_MARKER_COLOR, lw=1.0, ls=":", alpha=0.85, zorder=8)
        ax.text(k, y_top, f" {a.get('label')}", color=_MARKER_COLOR,
                fontsize=7, va="top", ha="left", alpha=0.95, zorder=9)
