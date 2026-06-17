"""Live, lag-free EF-offset adjustment on the band map.

The EF offset is a *pure scalar shift* of the binding-energy axis: k// is
computed from the kinetic energy (``ef_kinetic``), independent of the offset, and
the intensity matrix never changes. So raising/lowering EF does NOT need a
reload or a display recompute — we rigidly shift the in-memory energy axis of the
loaded data (a clone, so the load cache is untouched) plus the cached display
axis, then redraw. Redraw and session save are debounced so dragging the
spinbox stays fluid.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QTimer

_REDRAW_MS = 40
_SAVE_MS = 600


def apply_live_ef_offset(window, value: float) -> None:
    """Shift the displayed binding-energy axis to the requested EF offset.

    ``window._ef_offset_applied`` tracks the offset currently baked into the
    in-memory axis (set to ``entry.ef_offset`` on load). Only the delta since
    then is applied, so this is idempotent and safe to call repeatedly.
    """
    d = getattr(window, "_raw_data", None)
    if d is None or d.get("ev_arr") is None:
        return
    try:
        new = float(value)
    except (TypeError, ValueError):
        return
    baked = getattr(window, "_ef_offset_applied", None)
    delta = new - float(baked if baked is not None else 0.0)
    if abs(delta) < 1e-12:
        return

    # Rigid shift of the binding-energy axis. New arrays → the (cloned) load
    # cache entry is never mutated. The intensity matrix is left untouched.
    d["ev_arr"] = np.asarray(d["ev_arr"], dtype=float) + delta
    disp_ev = getattr(window, "_data_disp_ev", None)
    if disp_ev is not None:
        window._data_disp_ev = np.asarray(disp_ev, dtype=float) + delta
    window._ef_offset_applied = new

    path = getattr(window, "_current_path", None)
    if path:
        try:
            entry = window._session.get_or_create(window._session.key_for_path(path))
            entry.ef_offset = new
        except Exception:
            pass

    _schedule(window, "_ef_redraw_timer", _REDRAW_MS, lambda: window._draw_current_view())
    _schedule(window, "_ef_save_timer", _SAVE_MS, lambda: _save(window))


def _schedule(window, attr: str, ms: int, fn) -> None:
    t = getattr(window, attr, None)
    if t is None:
        t = QTimer()  # parentless; the ref stored on window keeps it alive
        t.setSingleShot(True)
        t.timeout.connect(fn)
        setattr(window, attr, t)
    t.start(ms)


def _save(window) -> None:
    try:
        window._session.save()
    except Exception:
        pass
