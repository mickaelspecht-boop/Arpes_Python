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
    # --- Overlay BZ cristal réel (lattice MP) ----------------------------
    v0_eV: float = 12.0                # inner potential pour calcul kz
    kz_plane: str = "Auto"             # "Gamma" | "Z" | "Auto"
    phi_c_deg: float = 0.0             # rotation cristal/détecteur (deg)
    overlay_bz_crystal: bool = False   # afficher polygone BZ cristal
    overlay_hs_crystal: bool = False   # afficher labels HS cristal
    mp_id: str = ""                    # Materials Project ID (read-only ici)

def _robust_norm(img: np.ndarray) -> np.ndarray:
    """Normalise une image 2D vers [0,1].

    L'échelle est calculée préférentiellement sur la région centrale (80 % des
    colonnes kx) pour éviter que les bords du détecteur dominent le contraste.
    Si le centre manque de variation (données vides ou fond plat), repli sur
    l'image complète avec percentiles 1-99.
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
    """Retourne kx, ky, fs_norm, titre à partir du dict legacy de l'explorer."""
    if raw_data is None:
        raise ValueError("Aucune donnée chargée")
    meta = raw_data.get("metadata", {}) or {}
    fs_data = meta.get("fs_data")
    if fs_data is None:
        # Fallback BM 2D: montre une MDC intégrée autour EF comme image 1 ligne.
        data = np.asarray(raw_data["data"], dtype=float)
        ev = np.asarray(raw_data["ev_arr"], dtype=float)
        kx = np.asarray(raw_data["kpar"], dtype=float)
        mask = np.abs(ev) <= params.ef_window
        if mask.sum() == 0: mask[np.argmin(np.abs(ev))] = True
        mdc = np.nanmean(data[:, mask], axis=1)
        return kx, np.array([0.0]), _robust_norm(mdc[None, :]), "Pas de volume FS: MDC à EF seulement"

    fs_data = np.asarray(fs_data, dtype=float)  # attendu (ny, nx, ne)
    kx = np.asarray(meta.get("fs_kx"), dtype=float)
    ky = np.asarray(meta.get("fs_ky"), dtype=float)
    ev = np.asarray(meta.get("fs_energy"), dtype=float)
    if fs_data.ndim != 3:
        raise ValueError(f"Volume FS invalide: shape={fs_data.shape}")
    if fs_data.shape[-1] != len(ev):
        raise ValueError("Volume FS: dernier axe ≠ énergie")

    mask = np.abs(ev) <= params.ef_window
    if mask.sum() == 0: mask[np.argmin(np.abs(ev))] = True
    fs = np.nanmean(fs_data[:, :, mask], axis=2)

    norm_note = "sans norm"
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

    # CLS: ky est souvent tilt en degrés. On l'affiche tel quel sauf si l'utilisateur recentre.
    source = meta.get("fs_source", raw_data.get("source_format", ""))
    return kx, ky, fs_n, f"FS {source} — ±{params.ef_window*1000:.0f} meV | {norm_note}"


def _axis_signature(axis: Any) -> tuple:
    arr = np.asarray(axis, dtype=float)
    if arr.size == 0:
        return (0,)
    payload = np.ascontiguousarray(arr, dtype=np.float64)
    digest = hashlib.sha256(payload.tobytes()).hexdigest()
    return (tuple(payload.shape), digest)


def _fs_cache_key(raw_data: dict[str, Any], params: FSParams) -> tuple:
    meta = raw_data.get("metadata", {}) or {}
    fs_data = meta.get("fs_data")
    if fs_data is None:
        data = np.asarray(raw_data.get("data"))
        return (
            "bm-fallback",
            id(data),
            tuple(data.shape),
            _axis_signature(raw_data.get("kpar")),
            _axis_signature(raw_data.get("ev_arr")),
            round(float(params.ef_window), 8),
        )
    fs_arr = np.asarray(fs_data)
    return (
        "fs-volume",
        id(fs_arr),
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

