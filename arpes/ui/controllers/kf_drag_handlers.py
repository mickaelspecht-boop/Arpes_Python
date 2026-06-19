"""kF drag-handler helpers for the MDC/EDC view.

Free functions take the PlotController as ``ctrl`` and update the shared
MDC/EDC canvas state.
"""
from __future__ import annotations

import numpy as np


def install_kf_drag_handlers(ctrl) -> None:
    if getattr(ctrl._parent, "_kf_drag_cids", None):
        return
    canvas = ctrl._mdc_edc.canvas
    cids = [
        canvas.mpl_connect("pick_event", ctrl._on_kf_pick),
        canvas.mpl_connect("motion_notify_event", ctrl._on_kf_motion),
        canvas.mpl_connect("button_release_event", ctrl._on_kf_release),
    ]
    ctrl._parent._kf_drag_cids = cids
    ctrl._parent._kf_drag_active = None


def on_kf_pick(ctrl, event) -> None:
    art = event.artist
    meta = getattr(art, "_kf_meta", None)
    if meta is None:
        return
    ctrl._parent._kf_drag_active = (meta[0], meta[1], art)
    # Cache the MDC peak positions once at grab time so live snapping during the
    # drag never recomputes the MDC on every mouse-move.
    ctrl._parent._kf_snap_cache = _build_snap_cache(ctrl)


def on_kf_motion(ctrl, event) -> None:
    active = getattr(ctrl._parent, "_kf_drag_active", None)
    if not active or event.inaxes is None or event.xdata is None:
        return
    pi, _sign, line = active
    x = float(event.xdata)
    if pi != "center":
        # Snap live so the segment sticks to the MDC peak WHILE dragging: the
        # displayed position is exactly where it will land on release (no more
        # surprise jump). The Γ center line is never snapped.
        x = _maybe_snap(ctrl, event, x)
    line.set_xdata([x, x])
    guide = getattr(line, "_kf_guide", None)
    if guide is not None:
        guide.set_xdata([x, x])
    ctrl._mdc_edc.canvas.draw_idle()


def on_kf_release(ctrl, event) -> None:
    active = getattr(ctrl._parent, "_kf_drag_active", None)
    if not active:
        return
    pi, sign, line = active
    ctrl._parent._kf_drag_active = None
    cache = getattr(ctrl._parent, "_kf_snap_cache", None)
    ctrl._parent._kf_snap_cache = None
    if event.inaxes is None or event.xdata is None:
        return
    if pi == "center":
        _apply_center_drag(ctrl, float(event.xdata))
        return
    cx = float(ctrl._params.sp_cx.value())
    x_new = _maybe_snap(ctrl, event, float(event.xdata), cache=cache)
    line.set_xdata([x_new, x_new])
    kf_new = max(0.0, sign * (x_new - cx))
    ctrl._params.kf_init_drag_changed.emit(int(pi), int(sign), float(kf_new))


def _snap_enabled(ctrl) -> bool:
    chk = getattr(ctrl._params, "chk_snap_kf", None)
    return chk is None or bool(chk.isChecked())


def _shift_held(event) -> bool:
    key = getattr(event, "key", None)
    return bool(key) and "shift" in str(key).lower()


def _maybe_snap(ctrl, event, x: float, *, cache=None) -> float:
    """Snap x to the nearest cached MDC peak, unless disabled (checkbox) or
    temporarily bypassed (Shift held). Returns x unchanged when no peak is in
    range — so free placement away from peaks still works."""
    if _shift_held(event) or not _snap_enabled(ctrl):
        return x
    if cache is None:
        cache = getattr(ctrl._parent, "_kf_snap_cache", None)
    if not cache:
        return x
    try:
        gi = float(ctrl._params.sp_gi.value())
    except Exception:
        gi = 0.0
    xs = _snap_cached(cache, x, max(gi * 2.0, 0.05))
    return xs if xs is not None else x


def _build_snap_cache(ctrl) -> dict | None:
    res = ctrl._get_mdc()
    if res is None:
        return None
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    kpar, mdc = res
    kpar = np.asarray(kpar, dtype=float)
    try:
        s = max(1, int(ctrl._params.sp_sfd.value()))
    except Exception:
        s = 1
    m_sm = gaussian_filter1d(np.nan_to_num(mdc), sigma=s)
    rng = float(np.nanmax(m_sm) - np.nanmin(m_sm))
    if rng <= 0:
        return {"peaks": np.array([])}
    mn = (m_sm - np.nanmin(m_sm)) / rng
    pk, _ = find_peaks(mn, height=0.3)
    return {"peaks": kpar[pk] if pk.size else np.array([])}


def _snap_cached(cache: dict, x_target: float, window: float) -> float | None:
    peaks = np.asarray(cache.get("peaks", []), dtype=float)
    if peaks.size == 0:
        return None
    d = np.abs(peaks - float(x_target))
    i = int(np.argmin(d))
    return float(peaks[i]) if d[i] <= float(window) else None


def _apply_center_drag(ctrl, x_new: float) -> None:
    """Commit a dragged Γ/MDC center line to the UI and current entry."""
    try:
        raw = getattr(ctrl._parent, "_raw_data", None) or {}
        kpar = np.asarray(raw.get("kpar", []), dtype=float)
        if kpar.size:
            x_new = float(np.clip(x_new, np.nanmin(kpar), np.nanmax(kpar)))
    except Exception:
        x_new = float(x_new)
    sp = ctrl._params.sp_cx
    try:
        x_new = float(np.clip(x_new, float(sp.minimum()), float(sp.maximum())))
    except Exception:
        pass
    sp.setValue(float(x_new))
    try:
        entry = ctrl._parent._current_entry()
        if entry is not None:
            entry.fit_params.center_init = float(x_new)
            ctrl._parent._session.save()
    except Exception:
        pass
    try:
        ctrl._draw_mdc_edc()
    except Exception:
        pass
    try:
        ctrl._parent._schedule_live_guess()
    except Exception:
        pass


def snap_to_mdc_peak(ctrl, x_target: float) -> float | None:
    res = ctrl._get_mdc()
    if res is None:
        return None
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d

    kpar, mdc = res
    s = max(1, int(ctrl._params.sp_sfd.value()))
    m_sm = gaussian_filter1d(np.nan_to_num(mdc), sigma=s)
    gi = float(ctrl._params.sp_gi.value())
    window = max(gi * 2.0, 0.05)
    mask = (kpar >= x_target - window) & (kpar <= x_target + window)
    if int(mask.sum()) < 3:
        return None
    kw = kpar[mask]
    mw = m_sm[mask]
    rng = float(np.nanmax(mw) - np.nanmin(mw))
    if rng <= 0:
        return None
    mn = (mw - np.nanmin(mw)) / rng
    pks, _ = find_peaks(mn, height=0.3)
    if pks.size == 0:
        return None
    i = int(np.argmin(np.abs(kw[pks] - x_target)))
    return float(kw[pks[i]])


def on_kf_init_drag(ctrl, pair_idx: int, sign: int, kf_new: float) -> None:
    try:
        params = ctrl._params._pair_params
        while len(params) <= pair_idx:
            params.append({"kF_init": 0.30,
                           "gamma_init": ctrl._params.sp_gi.value(),
                           "gamma_max": ctrl._params.sp_gm.value()})
        params[pair_idx]["kF_init"] = float(kf_new)
        cur = getattr(ctrl._params, "_current_pair", 0)
        if cur == pair_idx:
            ctrl._params.sp_kfi.blockSignals(True)
            ctrl._params.sp_kfi.setValue(float(kf_new))
            ctrl._params.sp_kfi.blockSignals(False)
    except Exception:
        pass
    try:
        ctrl._draw_mdc_edc()
    except Exception:
        pass
    try:
        ctrl._parent._schedule_live_guess()
    except Exception:
        pass
