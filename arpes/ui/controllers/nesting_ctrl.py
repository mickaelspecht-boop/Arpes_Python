"""Open the C(q) Fermi-surface autocorrelation viewer for the current FS.

Free function wired via a closure in panels.py (no PROXY_MAP entry). Pulls the
current FS map through the same ``extract_fs_map`` the FS canvas uses, then hands
it to ``NestingDialog`` which computes and displays C(q).
"""
from __future__ import annotations

import numpy as np

from arpes.physics.fs import extract_fs_map


def show_cq(window) -> None:
    raw = getattr(window, "_raw_data", None)
    if raw is None:
        window._status("Load a Fermi surface first.")
        return
    meta = (raw.get("metadata", {}) or {})
    if meta.get("fs_kind") != "kxky":
        window._status("C(q) needs a 2D FS map in (kx, ky) — set φ, a and hν first.")
        return
    controls = getattr(window, "_fs_controls", None)
    if controls is None:
        return
    try:
        params = controls.params()
        kx, ky, fs, title = extract_fs_map(raw, params)
    except Exception as exc:
        window._status(f"C(q): {exc}")
        return
    fs = np.asarray(fs, dtype=float)
    if fs.ndim != 2 or min(fs.shape) < 4:
        window._status("C(q): FS map too small (need a 2D volume).")
        return
    from arpes.ui.widgets.dialogs.nesting_dialog import NestingDialog
    dlg = NestingDialog(window, kx, ky, fs, title=str(title or ""))
    dlg.show()
    window._status("C(q) autocorrelation computed.")
