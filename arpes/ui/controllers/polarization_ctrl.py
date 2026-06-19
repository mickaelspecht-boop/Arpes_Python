"""P(k,E) polarization-contrast orchestration (free functions, window-first).

Strategy: try to auto-find the orthogonal-polarization partner of the current
band map in the session (same direction / azimuth / hν, opposite polarization,
nearest measurement number) and load it headlessly via ``load_arpes_file``. If no
partner is found or the load fails, fall back to a robust two-click capture
workflow (capture the current map as reference, switch file, click again).

Wired via a closure in panels.py — no PROXY_MAP entry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _cur_entry(window):
    path = getattr(window, "_current_path", None)
    if not path:
        return None, None
    key = window._session.key_for_path(path)
    return window._session.get_or_create(key), key


def _key_to_path(window, key: str) -> str:
    folder = getattr(window._session, "folder", None)
    p = Path(key)
    if p.is_absolute() or not folder:
        return str(p)
    return str(Path(folder) / p)


def _pol(meta) -> str:
    return str(getattr(meta, "polarization", "") or "").strip().upper()


def find_partner(window):
    """Opposite-polarization band-map key matching geometry, or None."""
    cur, ckey = _cur_entry(window)
    if cur is None:
        return None
    cm = cur.meta
    cpol = _pol(cm)
    if not cpol:
        return None
    best, best_d = None, 1 << 30
    for key, e in getattr(window._session, "files", {}).items():
        if key == ckey:
            continue
        m = e.meta
        pol = _pol(m)
        if not pol or pol == cpol:
            continue
        if (m.direction or "") != (cm.direction or ""):
            continue
        if cm.hv and m.hv and abs(float(m.hv) - float(cm.hv)) > 0.05 * max(float(cm.hv), 1.0):
            continue
        if cm.azi is not None and m.azi is not None and abs(float(m.azi) - float(cm.azi)) > 3.0:
            continue
        d = abs(int(getattr(m, "meas_no", 0)) - int(getattr(cm, "meas_no", 0)))
        if d < best_d:
            best, best_d = key, d
    return best


def _load_partner_map(window, key):
    from arpes.core.sample import sample_for_entry
    from arpes.io.loaders.common import load_arpes_file
    e = window._session.get_or_create(key)
    s = sample_for_entry(window._session, e, entry_key=key)
    return load_arpes_file(
        _key_to_path(window, key),
        work_func=float(getattr(s, "work_function_eV", 0.0) or 0.0),
        ef_offset=float(getattr(e, "ef_offset", 0.0) or 0.0),
        a_lattice=float(getattr(s, "a_angstrom", 0.0) or 0.0),
        hv=(float(e.meta.hv) if e.meta.hv else None),
        temperature=(float(e.meta.temperature) if e.meta.temperature else None),
        azi=e.meta.azi,
        pol=(e.meta.polarization or ""),
    )


def _valid_bm(d) -> bool:
    return bool(d) and d.get("kpar") is not None and d.get("data") is not None


def _open(window, a, a_pol, b, b_pol):
    from arpes.ui.widgets.dialogs.polarization_dialog import PolarizationDialog
    dlg = PolarizationDialog(
        window,
        np.asarray(a["kpar"], float), np.asarray(a["ev_arr"], float),
        np.asarray(a["data"], float), str(a_pol),
        np.asarray(b["kpar"], float), np.asarray(b["ev_arr"], float),
        np.asarray(b["data"], float), str(b_pol),
    )
    dlg.show()


def show_pkE(window) -> None:
    raw = getattr(window, "_raw_data", None)
    if not _valid_bm(raw):
        window._status("Load a band map first (P(k,E) needs k// vs E).")
        return
    cur, ckey = _cur_entry(window)
    cur_pol = (cur.meta.polarization if cur is not None else "") or "?"

    # 1) Auto partner.
    pk = find_partner(window)
    if pk is not None:
        try:
            pd = _load_partner_map(window, pk)
        except Exception:
            pd = None
        if _valid_bm(pd):
            ppol = window._session.get_or_create(pk).meta.polarization or "?"
            _open(window, raw, cur_pol, pd, ppol)
            window._status(f"P(k,E): {cur_pol} vs {ppol} (auto-paired).")
            return

    # 2) Capture fallback.
    ref = getattr(window, "_pkE_ref", None)
    if ref is None:
        window._pkE_ref = {
            "data": {"kpar": np.asarray(raw["kpar"], float),
                     "ev_arr": np.asarray(raw["ev_arr"], float),
                     "data": np.asarray(raw["data"], float)},
            "pol": cur_pol, "key": ckey,
        }
        window._status(
            f"P(k,E): reference captured (pol={cur_pol}). Load the orthogonal-"
            "polarization partner and click P(k,E) again."
        )
        return
    if ref.get("key") == ckey:
        window._status("P(k,E): load the OTHER polarization — current file is the captured one.")
        return
    _open(window, raw, cur_pol, ref["data"], ref["pol"])
    window._pkE_ref = None
    window._status(f"P(k,E): {cur_pol} vs {ref['pol']} (manual capture).")
