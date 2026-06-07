"""Signal-window helpers for distortion overlays."""
from __future__ import annotations

import numpy as np


def signal_bbox(
    data: np.ndarray, kpar, ev,
    *, intensity_percentile: float = 50.0, finite_only: bool = True,
) -> dict:
    """Detect the signal bounding box (intensity > threshold)."""
    arr = np.asarray(data, dtype=float)
    kpar_axis = np.asarray(kpar, dtype=float)
    ev_axis = np.asarray(ev, dtype=float)
    fallback = {
        "k_min": float(np.nanmin(kpar_axis)) if kpar_axis.size else 0.0,
        "k_max": float(np.nanmax(kpar_axis)) if kpar_axis.size else 0.0,
        "ev_min": float(np.nanmin(ev_axis)) if ev_axis.size else 0.0,
        "ev_max": float(np.nanmax(ev_axis)) if ev_axis.size else 0.0,
        "valid": False,
    }
    if arr.ndim != 2 or arr.shape != (kpar_axis.size, ev_axis.size):
        return fallback
    finite = np.isfinite(arr)
    if not finite.any():
        return fallback
    base = arr[finite] if finite_only else arr.ravel()
    try:
        threshold = float(np.nanpercentile(base, intensity_percentile))
    except Exception:
        return fallback
    above = (arr > threshold) & finite
    if not above.any():
        return fallback
    rows_ok = above.any(axis=1)
    cols_ok = above.any(axis=0)
    k_lo = int(np.argmax(rows_ok))
    k_hi = int(rows_ok.size - np.argmax(rows_ok[::-1]) - 1)
    e_lo = int(np.argmax(cols_ok))
    e_hi = int(cols_ok.size - np.argmax(cols_ok[::-1]) - 1)
    if k_hi <= k_lo or e_hi <= e_lo:
        return fallback
    return {
        "k_min": float(kpar_axis[k_lo]),
        "k_max": float(kpar_axis[k_hi]),
        "ev_min": float(ev_axis[e_lo]),
        "ev_max": float(ev_axis[e_hi]),
        "valid": True,
        "threshold": threshold,
    }
