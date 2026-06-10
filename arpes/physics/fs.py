#!/usr/bin/env python3
"""FS pure-physics module — FSParams + extract_fs_map + cache helpers.

PyQt widgets (FSControlPanel, FermiSurfaceCanvas) moved to
``arpes/ui/widgets/fs_panel.py`` to respect the layering rule
(no PyQt6 in arpes/physics/).
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Any
import hashlib

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

from arpes.physics.norm import apply_fs_flux_factors_to_map, fs_flux_profile_factors
from arpes.physics.bz import bz_high_symmetry_points, bz_polygon, resolve_bz_preset
from arpes.physics.fs_gamma import detect_gamma_from_fs_map

@dataclass
class FSParams:
    a_lattice: float = 0.0
    b_lattice: float = 0.0
    ef_window: float = 0.030
    ef_resolution_meV: float = 0.0
    temperature_K: float = 0.0
    norm_ref_lo: float = -0.60
    norm_ref_hi: float = -0.20
    smooth_sigma: float = 1.0
    klim: float = 1.3
    kx_center: float = 0.0
    ky_center: float = 0.0
    bz_shape: str = "rectangle"
    bz_half_x: float = 1.0
    bz_half_y: float = 1.0
    bz_angle_deg: float = 90.0
    normalize_profile: bool = True
    overlay_bz: bool = True
    show_hsym: bool = True
    cmap: str = "inferno"
    # --- Real crystal BZ overlay (MP lattice) ----------------------------
    v0_eV: float = 12.0                # inner potential for kz calculation
    kz_plane: str = "Auto"             # "Gamma" | "Z" | "Auto"
    phi_c_deg: float = 0.0             # crystal/detector rotation (deg)
    overlay_bz_crystal: bool = False   # show crystal BZ polygon
    overlay_hs_crystal: bool = False   # show crystal HS labels
    mp_id: str = ""                    # Materials Project ID (read-only here)


def _ef_window_mask_and_warning(ev: np.ndarray, ef_window: float) -> tuple[np.ndarray, str]:
    e = np.asarray(ev, dtype=float)
    mask = np.abs(e) <= float(ef_window)
    if mask.sum() == 0:
        mask[np.argmin(np.abs(e))] = True
        return mask, "empty EF window: nearest energy point used"
    e_sel = e[mask]
    has_below = bool(np.any(e_sel < 0))
    has_above = bool(np.any(e_sel > 0))
    lo = float(np.nanmin(e_sel))
    hi = float(np.nanmax(e_sel))
    tol = max(1e-12, 0.25 * float(ef_window))
    if not has_below or not has_above or abs(lo + hi) > tol:
        return mask, f"asymmetric EF window [{lo:+.3f},{hi:+.3f}] eV"
    return mask, ""


def _ef_integrate(data: np.ndarray, ev: np.ndarray, mask: np.ndarray, params: FSParams, axis: int):
    """Integrate near EF; optionally use Fermi-Dirac/resolution weighting."""
    subset = np.take(data, np.where(mask)[0], axis=axis)
    e = np.asarray(ev, dtype=float)[mask]
    sigma_res = max(0.0, float(params.ef_resolution_meV)) / 1000.0
    kbt = 8.617333262e-5 * max(0.0, float(params.temperature_K))
    sigma = float(np.hypot(sigma_res, 3.5 * kbt))
    if sigma <= 0:
        return np.nanmean(subset, axis=axis), "boxcar EF"
    weights = np.exp(-0.5 * (e / sigma) ** 2)
    if not np.isfinite(weights).any() or float(np.nansum(weights)) <= 0:
        return np.nanmean(subset, axis=axis), "boxcar EF"
    shape = [1] * subset.ndim
    shape[axis] = weights.size
    w = weights.reshape(shape)
    num = np.nansum(subset * w, axis=axis)
    den = np.nansum(np.where(np.isfinite(subset), w, 0.0), axis=axis)
    out = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out, "Fermi/resolution weighted EF"

def _robust_norm(img: np.ndarray) -> np.ndarray:
    """Normalize a 2D image to [0,1].

    The scale is preferentially computed on the central region (80% of kx
    columns) to keep detector edges from dominating contrast. If the center
    lacks variation (empty data or flat background), fall back to the full
    image with 1-99 percentiles.
    """
    arr = np.asarray(img, dtype=float)
    if not np.isfinite(arr).any():
        return arr
    lo, hi = None, None
    if arr.ndim == 2 and arr.shape[1] >= 10:
        margin = max(1, arr.shape[1] // 10)
        ref = arr[:, margin: arr.shape[1] - margin]
        valid_c = ref[np.isfinite(ref)]
        if valid_c.size >= 4:
            lo_c, hi_c = np.percentile(valid_c, [2, 98])
            if hi_c - lo_c > 1e-6:
                lo, hi = lo_c, hi_c
    if lo is None:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return arr
        lo, hi = np.percentile(valid, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def extract_fs_map(raw_data: dict[str, Any], params: FSParams):
    """Return kx, ky, fs_norm, title from the explorer legacy dict."""
    if raw_data is None:
        raise ValueError("No data loaded")
    meta = raw_data.get("metadata", {}) or {}
    fs_data = meta.get("fs_data")
    if fs_data is None:
        # 2D BM fallback: show an MDC integrated around EF as a one-line image.
        data = np.asarray(raw_data["data"], dtype=float)
        ev = np.asarray(raw_data["ev_arr"], dtype=float)
        kx = np.asarray(raw_data["kpar"], dtype=float)
        mask, warn = _ef_window_mask_and_warning(ev, params.ef_window)
        mdc, ef_note = _ef_integrate(data, ev, mask, params, axis=1)
        suffix = f" | {ef_note}" + (f" | {warn}" if warn else "")
        return kx, np.array([0.0]), _robust_norm(mdc[None, :]), "No FS volume: EF MDC only" + suffix

    fs_data = np.asarray(fs_data, dtype=float)  # expected (ny, nx, ne)
    kx = np.asarray(meta.get("fs_kx"), dtype=float)
    ky = np.asarray(meta.get("fs_ky"), dtype=float)
    ev = np.asarray(meta.get("fs_energy"), dtype=float)
    if fs_data.ndim != 3:
        raise ValueError(f"Invalid FS volume: shape={fs_data.shape}")
    if fs_data.shape[-1] != len(ev):
        raise ValueError("FS volume: last axis ≠ energy")

    mask, ef_warn = _ef_window_mask_and_warning(ev, params.ef_window)
    fs, ef_note = _ef_integrate(fs_data, ev, mask, params, axis=2)

    norm_note = "no norm"
    if params.normalize_profile:
        safe_y, safe_x, norm_note = fs_flux_profile_factors(
            fs_data,
            ev,
            ref_range=(params.norm_ref_lo, params.norm_ref_hi),
            normalize_y=True,
            normalize_x=True,
        )
        fs = apply_fs_flux_factors_to_map(fs, safe_y, safe_x)

    if params.smooth_sigma > 0 and gaussian_filter is not None:
        nan = ~np.isfinite(fs)
        tmp = np.where(nan, np.nanmedian(fs[np.isfinite(fs)]) if np.isfinite(fs).any() else 0, fs)
        fs = gaussian_filter(tmp, sigma=params.smooth_sigma)
        fs[nan] = np.nan
    fs_n = _robust_norm(fs)

    # CLS: ky is often tilt in degrees. Display it as-is unless the user recenters.
    source = meta.get("fs_source", raw_data.get("source_format", ""))
    warn_note = f" | {ef_warn}" if ef_warn else ""
    return kx, ky, fs_n, f"FS {source} — ±{params.ef_window*1000:.0f} meV | {ef_note}{warn_note} | {norm_note}"


def _axis_signature(axis: Any) -> tuple:
    arr = np.asarray(axis, dtype=float)
    if arr.size == 0:
        return (0,)
    payload = np.ascontiguousarray(arr, dtype=np.float64)
    digest = hashlib.sha256(payload.tobytes()).hexdigest()
    return (tuple(payload.shape), digest)


def _data_signature(arr: np.ndarray) -> tuple:
    """Content-sensitive, cheap signature for large data arrays.

    ``id()`` is unsafe as a cache key: after a reload, CPython can hand the new
    array the address of the garbage-collected old one; with identical axes the
    whole key then collides and a stale cached map is served. Hash a sparse
    strided sample (≤65536 elements) instead of the full buffer so the cost
    stays negligible on (700, 700, 400) volumes.
    """
    a = np.asarray(arr)
    flat = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
    step = max(1, flat.size // 65536)
    sample = np.ascontiguousarray(flat[::step])
    digest = hashlib.sha256(sample.tobytes()).hexdigest()[:16]
    return (tuple(a.shape), str(a.dtype), digest)


def _fs_cache_key(raw_data: dict[str, Any], params: FSParams) -> tuple:
    meta = raw_data.get("metadata", {}) or {}
    fs_data = meta.get("fs_data")
    if fs_data is None:
        data = np.asarray(raw_data.get("data"))
        return (
            "bm-fallback",
            _data_signature(data),
            tuple(data.shape),
            _axis_signature(raw_data.get("kpar")),
            _axis_signature(raw_data.get("ev_arr")),
            round(float(params.ef_window), 8),
        )
    fs_arr = np.asarray(fs_data)
    return (
        "fs-volume",
        _data_signature(fs_arr),
        tuple(fs_arr.shape),
        _axis_signature(meta.get("fs_kx")),
        _axis_signature(meta.get("fs_ky")),
        _axis_signature(meta.get("fs_energy")),
        round(float(params.ef_window), 8),
        round(float(params.norm_ref_lo), 8),
        round(float(params.norm_ref_hi), 8),
        round(float(params.smooth_sigma), 8),
        bool(params.normalize_profile),
    )
