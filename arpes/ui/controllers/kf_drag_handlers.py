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


def on_kf_motion(ctrl, event) -> None:
    active = getattr(ctrl._parent, "_kf_drag_active", None)
    if not active or event.inaxes is None or event.xdata is None:
        return
    _pi, _sign, line = active
    x = float(event.xdata)
    line.set_xdata([x, x])
    ctrl._mdc_edc.canvas.draw_idle()


def on_kf_release(ctrl, event) -> None:
    active = getattr(ctrl._parent, "_kf_drag_active", None)
    if not active:
        return
    pi, sign, line = active
    ctrl._parent._kf_drag_active = None
    if event.inaxes is None or event.xdata is None:
        return
    cx = float(ctrl._params.sp_cx.value())
    x_new = float(event.xdata)
    x_snap = snap_to_mdc_peak(ctrl, x_new)
    if x_snap is not None:
        x_new = x_snap
    line.set_xdata([x_new, x_new])
    kf_new = max(0.0, sign * (x_new - cx))
    ctrl._params.kf_init_drag_changed.emit(int(pi), int(sign), float(kf_new))


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
