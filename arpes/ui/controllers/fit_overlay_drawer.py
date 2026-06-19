"""Fit-overlay drawing helpers extracted from plot_controller.

Free functions that take the PlotController as ``ctrl`` so they can reach
``ctrl._parent``, ``ctrl._params``, ``ctrl._fit_res`` without subclassing.
Keeps plot_controller.py under the architect's 700-LOC ceiling.
"""
from __future__ import annotations

import numpy as np

PAIR_COLORS = (
    "#60a5fa", "#fbbf24", "#34d399", "#f87171",
    "#a78bfa", "#fb923c", "#22d3ee", "#f472b6",
)


def axis_state_mismatch(ctrl, fr: dict) -> bool:
    """True if fr was fitted under different grid/distortion state than now."""
    if not isinstance(fr, dict):
        return False
    fit_dist = fr.get("distorted")
    fit_grid = fr.get("grid_active")
    if fit_dist is None and fit_grid is None:
        return False
    p = ctrl._parent
    if not getattr(p, "_current_path", None):
        return False
    entry = p._session.get_or_create(p._session.key_for_path(p._current_path))
    from arpes.physics.distortion import is_distortion_active
    cur_dist = bool(
        getattr(entry, "bm_distortion", None)
        and is_distortion_active(entry.bm_distortion)
    )
    cur_grid = bool((getattr(entry, "grid_correction", None) or {}).get("enabled"))
    if fit_dist is not None and bool(fit_dist) != cur_dist:
        return True
    if fit_grid is not None and bool(fit_grid) != cur_grid:
        return True
    return False


def scatter_kf_with_chi2(ax, k_values, ev_f, bad_mask, color, marker,
                         *, kf_std=None) -> None:
    n = min(len(k_values), len(ev_f), len(bad_mask))
    if n == 0:
        return
    k = np.asarray(k_values[:n], dtype=float)
    e = np.asarray(ev_f[:n], dtype=float)
    bad = np.asarray(bad_mask[:n], dtype=bool)
    valid = np.isfinite(k) & np.isfinite(e)
    good = valid & ~bad
    if kf_std is not None and good.any():
        xerr = np.asarray(kf_std[:n], dtype=float)
        xerr_good = xerr[good]
        mask_finite_err = np.isfinite(xerr_good) & (xerr_good > 0)
        if mask_finite_err.any():
            idx = subtle_uncertainty_indices(np.where(good)[0][mask_finite_err])
            ax.errorbar(
                k[idx], e[idx], xerr=xerr[idx],
                fmt="none", ecolor=color, elinewidth=0.45,
                capsize=0.0, alpha=0.24, zorder=3,
            )
    if good.any():
        ax.scatter(k[good], e[good], s=7, color=color, marker=marker,
                   zorder=5, alpha=0.85)
    if (valid & bad).any():
        ax.scatter(k[valid & bad], e[valid & bad], s=20, color="#fb923c",
                   marker=marker, edgecolors="black", linewidths=0.35,
                   zorder=6, alpha=0.95)


def subtle_uncertainty_indices(indices, *, max_bars: int = 18) -> np.ndarray:
    """Return a sparse, stable subset of indices for subtle BM uncertainty bars."""
    idx = np.asarray(indices, dtype=int)
    if idx.size <= int(max_bars):
        return idx
    keep = np.linspace(0, idx.size - 1, int(max_bars), dtype=int)
    return idx[keep]


def kf_sigma_arrays(fr: dict, branch: str):
    """Return kF sigma arrays for a branch, supporting full and ensemble fits."""
    direct = fr.get(f"sigma_{branch}") or []
    if direct:
        return direct
    ensemble = fr.get("ensemble") or {}
    return ensemble.get(f"{branch}_std") or []


def smooth_kf_for_display(arr, *, smooth_on: bool, sigma: float):
    """Smooth kF for display while keeping deleted/outlier gaps hidden."""
    a = np.asarray(arr, dtype=float)
    if not smooth_on or sigma <= 0 or a.size < 3:
        return a
    mask = np.isfinite(a)
    if not mask.any():
        return a
    from scipy.ndimage import gaussian_filter1d
    x = np.arange(a.size, dtype=float)
    a_interp = np.interp(x, x[mask], a[mask])
    smoothed = gaussian_filter1d(a_interp, sigma=float(sigma))
    return np.where(mask, smoothed, np.nan)


def _mdc_map_view_window(ctrl):
    """View limits for the MDC Fit mini-map.

    Returns ``(k0, k1, e0, e1, show_info)`` or ``None``.

    With ≥1 fit zone the window is the **full data extent** so every zone
    rectangle stays visible (show_info=False). With no zone it falls back to the
    legacy single-range ROI zoom (show_info=True). ``None`` when no data.
    """
    p = ctrl._parent
    d = getattr(p, "_raw_data", None)
    if d is None:
        return None
    entry = None
    if getattr(p, "_current_path", None):
        entry = p._session.get_or_create(p._session.key_for_path(p._current_path))
    zones = getattr(entry, "fit_zones", None) or []
    if zones:
        kpar = np.asarray(d["kpar"], dtype=float)
        ev = np.asarray(d["ev_arr"], dtype=float)
        return (float(np.nanmin(kpar)), float(np.nanmax(kpar)),
                float(np.nanmin(ev)), float(np.nanmax(ev)), False)
    b = ctrl._fit_roi_bounds()
    if b is None:
        return None
    return (b[0], b[1], b[2], b[3], True)


def apply_mdc_map_view(ctrl, ax) -> None:
    """Set the MDC mini-map limits + info text per ``_mdc_map_view_window``."""
    win = _mdc_map_view_window(ctrl)
    if win is None:
        return
    k0, k1, e0, e1, show_info = win
    ax.set_xlim(k0, k1)
    ax.set_ylim(e0, e1)
    if show_info:
        ax.text(
            0.01, 0.02,
            f"Fenetre fit: k {k0:+.3f} -> {k1:+.3f} pi/a | E {e0:+.3f} -> {e1:+.3f} eV",
            transform=ax.transAxes, ha="left", va="bottom",
            color="white", fontsize=7,
            bbox={"facecolor": "#111827", "edgecolor": "#38bdf8", "alpha": 0.72, "pad": 3},
            zorder=30,
        )
    lbl = getattr(ctrl, "_lbl_fit_view_info", None)
    if lbl is not None:
        lbl.setText("Plage d'analyse" if show_info else "Toutes les zones")


def draw_zone_overlays(ctrl, ax) -> None:
    """Overlay kF for every zone's fit_result + colored rectangles."""
    p = ctrl._parent
    if not getattr(p, "_current_path", None):
        return
    entry = p._session.get_or_create(p._session.key_for_path(p._current_path))
    zones = getattr(entry, "fit_zones", None) or []
    if not zones:
        return
    from arpes.ui.controllers.fit_zones_controller import ZONE_PALETTE
    active_id = entry.active_zone_id
    for z in zones:
        zid = z.get("id")
        fr = z.get("fit_result")
        if not fr or zid == active_id:
            continue
        color = ZONE_PALETTE[int(z.get("color_idx", 0)) % len(ZONE_PALETTE)]
        try:
            ev_f = np.asarray(fr["e_fitted"], dtype=float)
        except Exception:
            continue
        for branch, marker in (("kF_minus", "o"), ("kF_plus", "^")):
            arrays = fr.get(branch) or []
            for arr in arrays:
                a = np.asarray(arr, dtype=float)
                n = min(a.size, ev_f.size)
                if n == 0:
                    continue
                valid = np.isfinite(a[:n]) & np.isfinite(ev_f[:n])
                if valid.any():
                    # No legend label: the rectangle corner text already names
                    # each zone, and per-scatter labels spam the legend.
                    ax.scatter(
                        a[:n][valid], ev_f[:n][valid], s=5, marker=marker,
                        color=color, alpha=0.65, zorder=4,
                    )
    try:
        from matplotlib.patches import Rectangle
    except ImportError:
        return
    # The active zone's window is edited live through the panel spinboxes, so
    # draw its rectangle from the live ROI bounds (the panel IS its editor).
    # Other zones use their stored snapshot. This makes the active rectangle
    # track parameter edits immediately instead of waiting for the debounced
    # auto-bind to persist them.
    try:
        active_bounds = ctrl._fit_roi_bounds()
    except Exception:
        active_bounds = None
    for z in zones:
        if z.get("id") == active_id and active_bounds is not None:
            k0, k1, e0, e1 = active_bounds
        else:
            fp = z.get("fit_params", {})
            try:
                k0 = float(fp.get("k_min")); k1 = float(fp.get("k_max"))
                e0 = float(fp.get("ev_start")); e1 = float(fp.get("ev_end"))
            except Exception:
                continue
        color = ZONE_PALETTE[int(z.get("color_idx", 0)) % len(ZONE_PALETTE)]
        lw = 1.8 if z.get("id") == active_id else 1.0
        ax.add_patch(Rectangle(
            (k0, e0), k1 - k0, e1 - e0,
            fill=False, edgecolor=color, linewidth=lw, alpha=0.9,
            linestyle="-" if z.get("active", True) else "--",
            zorder=3,
        ))
        ax.text(
            k0, e1, f" {z.get('label')}",
            color=color, fontsize=7, va="bottom", ha="left",
            alpha=0.95, zorder=4,
        )


def draw_fit_annotations(ctrl, ax, fr: dict) -> None:
    p = ctrl._parent
    if not getattr(p, "_current_path", None):
        return
    entry = p._session.get_or_create(p._session.key_for_path(p._current_path))
    annotations = entry.annotations or {}
    if not annotations:
        return
    ev_f = np.asarray(fr.get("e_fitted", []), dtype=float)
    if ev_f.size == 0:
        return
    for branch in ("kF_minus", "kF_plus"):
        arrays = fr.get(branch) or []
        for note in annotations.get(branch, []):
            try:
                pair_idx = int(note.get("pair", -1))
                point_idx = int(note.get("index", -1))
            except Exception:
                continue
            if not (0 <= pair_idx < len(arrays)):
                continue
            arr = np.asarray(arrays[pair_idx], dtype=float)
            if not (0 <= point_idx < arr.size and point_idx < ev_f.size):
                continue
            k_val = arr[point_idx]
            e_val = ev_f[point_idx]
            if not (np.isfinite(k_val) and np.isfinite(e_val)):
                continue
            text = str(note.get("text", "")).strip()
            ax.plot(
                k_val, e_val, marker="*", markersize=12,
                color="#fde047", markeredgecolor="black", zorder=8,
            )
            if text:
                ax.annotate(
                    text[:30], xy=(k_val, e_val), xytext=(5, 5),
                    textcoords="offset points", fontsize=7,
                    color="#fde047", zorder=9,
                )


def draw_kf_overlay(ctrl, ax):
    """Main kF overlay routine — delegates to zone draw when no active fit."""
    if ctrl._fit_res is None:
        draw_zone_overlays(ctrl, ax)
        return
    fr = ctrl._fit_res
    if axis_state_mismatch(ctrl, fr):
        ax.text(
            0.5, 0.02,
            "⚠ fit_result axes mismatch (grid/distortion changed) — "
            "relancer le fit MDC",
            transform=ax.transAxes, ha="center", va="bottom",
            color="#fb923c", fontsize=8, alpha=0.95,
        )
        draw_zone_overlays(ctrl, ax)
        return
    params = ctrl._params
    n = params.sp_np.value()
    ev_f = np.asarray(fr["e_fitted"])
    chi2 = np.asarray(fr.get("chi2_red", []), dtype=float)
    threshold = float(getattr(params, "sp_chi2_threshold", None).value()) \
        if hasattr(params, "sp_chi2_threshold") else np.inf
    bad_mask = chi2 > threshold if chi2.size == ev_f.size \
        else np.zeros(ev_f.size, dtype=bool)
    km_std_all = kf_sigma_arrays(fr, "kF_minus")
    kp_std_all = kf_sigma_arrays(fr, "kF_plus")
    smooth_on = bool(getattr(params, "chk_smooth_kf", None)
                     and params.chk_smooth_kf.isChecked())
    sm_sigma = float(getattr(params, "sp_smooth_kf_sigma", None).value()
                     if hasattr(params, "sp_smooth_kf_sigma") else 0.0)

    for i in range(n):
        c = PAIR_COLORS[i % len(PAIR_COLORS)]
        if i < len(fr.get("kF_minus", [])):
            xerr_m = (np.asarray(km_std_all[i], dtype=float)
                      if i < len(km_std_all) else None)
            scatter_kf_with_chi2(
                ax,
                smooth_kf_for_display(
                    fr["kF_minus"][i], smooth_on=smooth_on, sigma=sm_sigma),
                ev_f, bad_mask, c, "o",
                kf_std=xerr_m,
            )
        if i < len(fr.get("kF_plus", [])):
            xerr_p = (np.asarray(kp_std_all[i], dtype=float)
                      if i < len(kp_std_all) else None)
            scatter_kf_with_chi2(
                ax,
                smooth_kf_for_display(
                    fr["kF_plus"][i], smooth_on=smooth_on, sigma=sm_sigma),
                ev_f, bad_mask, c, "^",
                kf_std=xerr_p,
            )
    selected = list(getattr(ctrl._parent, "_fit_selected", []) or [])
    if selected:
        ev_f = np.asarray(fr["e_fitted"])
        sel_k: list[float] = []
        sel_e: list[float] = []
        for branch, pair_idx, point_idx in selected:
            arrays = fr.get(branch) or []
            if not (0 <= pair_idx < len(arrays)):
                continue
            arr = np.asarray(arrays[pair_idx], dtype=float)
            if not (0 <= point_idx < arr.size and point_idx < ev_f.size):
                continue
            k_val = arr[point_idx]
            e_val = ev_f[point_idx]
            if not (np.isfinite(k_val) and np.isfinite(e_val)):
                continue
            sel_k.append(float(k_val))
            sel_e.append(float(e_val))
        if sel_k:
            ax.scatter(sel_k, sel_e, s=70, facecolors="none",
                       edgecolors="#fbbf24", linewidths=1.6, zorder=7)
    draw_fit_annotations(ctrl, ax, fr)
    draw_zone_overlays(ctrl, ax)
