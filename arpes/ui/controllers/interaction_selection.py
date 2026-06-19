"""Fit-point selection helpers for InteractionController."""
from __future__ import annotations

import numpy as np


def handle_single_click_selection(ctrl, ax, click_disp, *, additive: bool) -> None:
    p = ctrl._parent
    if p._fit_res is None:
        return
    e_fit = np.asarray(p._fit_res.get("e_fitted", []), dtype=float)
    if e_fit.size == 0:
        return
    nearest = ctrl._find_nearest_kf_point(
        p._fit_res, e_fit, ax, click_disp, pixel_radius=ctrl._PICK_RADIUS_PX,
    )
    if nearest is None:
        if not additive and p._fit_selected:
            p._fit_selected = []
            ctrl._status("Selection cleared.")
            _sync_results_link(ctrl)
        return
    if not additive:
        if nearest in p._fit_selected and len(p._fit_selected) == 1:
            p._fit_selected = []
        else:
            p._fit_selected = [nearest]
    else:
        if nearest in p._fit_selected:
            p._fit_selected.remove(nearest)
        else:
            p._fit_selected.append(nearest)
    ctrl._status(f"{len(p._fit_selected)} point(s) selected. Press Delete to remove.")
    _sync_results_link(ctrl)


def handle_rect_selection(ctrl, ax, xs, ys, *, additive: bool) -> None:
    p = ctrl._parent
    if p._fit_res is None:
        return
    fr = p._fit_res
    e_fit = np.asarray(fr.get("e_fitted", []), dtype=float)
    if e_fit.size == 0:
        return
    x0, x1 = xs
    y0, y1 = ys
    hits: list[tuple[str, int, int]] = []
    for branch in ("kF_minus", "kF_plus"):
        for pair_idx, raw in enumerate(fr.get(branch) or []):
            arr = np.asarray(raw, dtype=float)
            n = min(arr.size, e_fit.size)
            if n == 0:
                continue
            k_arr = arr[:n]
            e_arr = e_fit[:n]
            mask = (
                np.isfinite(k_arr) & np.isfinite(e_arr)
                & (k_arr >= x0) & (k_arr <= x1)
                & (e_arr >= y0) & (e_arr <= y1)
            )
            for idx in np.flatnonzero(mask):
                hits.append((branch, pair_idx, int(idx)))
    if not additive:
        p._fit_selected = list(hits)
    else:
        existing = set(p._fit_selected)
        for h in hits:
            if h in existing:
                existing.remove(h)
            else:
                existing.add(h)
        p._fit_selected = list(existing)
    ctrl._status(f"{len(p._fit_selected)} point(s) selected. Press Delete to remove.")
    _sync_results_link(ctrl)


def _sync_results_link(ctrl) -> None:
    p = ctrl._parent
    results = getattr(p, "_results", None)
    current = getattr(p, "_current_path", None)
    if results is None or not current:
        return
    try:
        filename = p._session.key_for_path(current)
        sel = (p._fit_selected[0] if len(p._fit_selected) == 1 else None)
        results.sync_linked_fit_selection(filename, sel)
    except Exception:
        pass
