"""Projection d'une BM dans le repère (kx, ky) d'une FS — pur, sans Qt.

B.1 du plan BM↔FS (cf BM_FS_ORGANIZATION_PLAN.md). Calcule la ligne
correspondant à une BM (cut à polar fixe) dans le repère 2D d'une FS map.

Principe physique :
- Une BM mesurée à `polar = P_bm` correspond géométriquement à une coupe
  horizontale dans (kx, ky) à une ordonnée fixe ky_in_fs déterminée par
  la différence (P_bm − P_fs_center).
- Si l'azi diffère entre la FS et la BM, la coupe est tournée par
  Δazi = azi_fs − azi_bm autour de Γ.
- Si l'hv diffère, le facteur d'échelle k change → projection extrapolée
  (qualité dégradée).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


C_ARPES = 0.51233          # cf arpes/physics/gamma.py
A_LATTICE_DEFAULT = 3.96   # Å (BaNi₂As₂, override via paramètre)


Quality = Literal["exact", "rotated", "scaled", "incompatible"]


@dataclass(frozen=True)
class BMCutLine:
    """Représentation d'une coupe BM projetée dans le repère d'une FS.

    `kx_points` et `ky_points` sont des arrays de même longueur définissant
    le segment à tracer dans le panneau FS. `quality` indique la fiabilité
    physique de la projection.
    """
    label: str                 # nom court pour affichage / pick
    bm_path: str               # path complet pour interaction
    polar_bm: float            # angle moteur BM (deg)
    azi_bm: float | None
    hv_bm: float
    kx_points: np.ndarray
    ky_points: np.ndarray
    quality: Quality
    warning: str = ""


def _scale_factor(hv: float, work_func: float, a_lattice: float) -> float | None:
    """C·√(Ek)·a/π — facteur de conversion sin(θ) → k(π/a).

    Renvoie None si Ek = hv − φ non valide.
    """
    try:
        ek = float(hv) - float(work_func)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ek) or ek <= 0:
        return None
    s = C_ARPES * np.sqrt(ek) * float(a_lattice) / np.pi
    if not np.isfinite(s) or s <= 0:
        return None
    return float(s)


def _polar_fs_center(fs_metadata: dict, fs_entry) -> float:
    """Polar central du scan FS (deg).

    Priorité : `fs_metadata["fs_scan_axis_deg"]["center"]`, puis
    `fs_entry.meta.polar`, puis 0.0.
    """
    axis = (fs_metadata or {}).get("fs_scan_axis_deg")
    if isinstance(axis, dict):
        center = axis.get("center")
        if center is not None:
            try:
                v = float(center)
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
    p = getattr(fs_entry.meta, "polar", None) if fs_entry is not None else None
    try:
        v = float(p) if p is not None else 0.0
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _classify_quality(
    hv_bm: float, hv_fs: float, azi_bm, azi_fs,
    *, hv_tol_rel: float, azi_tol_deg: float,
) -> tuple[Quality, str]:
    if hv_bm <= 0 or hv_fs <= 0:
        return "incompatible", "hv invalide"
    hv_diff_rel = abs(hv_bm - hv_fs) / max(hv_bm, hv_fs)
    hv_close = hv_diff_rel <= hv_tol_rel
    if azi_bm is None or azi_fs is None:
        azi_diff = 0.0  # bénéfice du doute si non renseigné
    else:
        try:
            azi_diff = abs(_angle_delta_deg(float(azi_fs), float(azi_bm)))
        except (TypeError, ValueError):
            azi_diff = 0.0
    azi_close = azi_diff <= azi_tol_deg

    if hv_close and azi_close:
        return "exact", ""
    if hv_close and not azi_close:
        return "rotated", f"Δazi={azi_diff:+.1f}° → rotation appliquée"
    if not hv_close and azi_close:
        return "scaled", f"Δhv={hv_bm - hv_fs:+.1f} eV → échelle extrapolée"
    return "scaled", (
        f"Δhv={hv_bm - hv_fs:+.1f} eV, Δazi={azi_diff:+.1f}° → "
        "projection composite (à interpréter avec prudence)"
    )


def _angle_delta_deg(dst: float, src: float) -> float:
    """Signed shortest angular delta dst-src in degrees, in [-180, 180)."""
    return (float(dst) - float(src) + 180.0) % 360.0 - 180.0


def compute_bm_cut_in_fs_frame(
    bm_entry,
    bm_path: str,
    fs_entry,
    fs_path: str,
    fs_metadata: dict,
    *,
    work_func: float,
    a_lattice: float = A_LATTICE_DEFAULT,
    kpar_range: tuple[float, float] = (-1.5, 1.5),
    n_points: int = 80,
    azi_tolerance_deg: float = 0.5,
    hv_tolerance_rel: float = 0.02,
) -> BMCutLine | None:
    """Projette une BM dans le repère (kx, ky) d'une FS.

    Args:
        bm_entry: FileEntry de la BM (lit meta.hv, meta.polar, meta.azi).
        bm_path: clé/path de la BM dans session.files.
        fs_entry: FileEntry de la FS de référence.
        fs_path: clé/path de la FS.
        fs_metadata: dict raw_data["metadata"] de la FS (pour fs_scan_axis_deg).
        work_func: φ (eV) pour la conversion angle↔k.
        a_lattice: paramètre de maille (Å), défaut 3.96.
        kpar_range: bornes du segment kpar à tracer (en π/a), défaut (-1.5, 1.5).
        n_points: nombre de points le long du segment.
        azi_tolerance_deg: au-delà → quality="rotated".
        hv_tolerance_rel: au-delà → quality="scaled".

    Returns:
        BMCutLine ou None si BM incomplète (pas un BM, pas de polar, etc.).
    """
    if bm_entry is None or fs_entry is None:
        return None
    if getattr(bm_entry.meta, "scan_kind", "") != "BM":
        return None
    polar_bm_raw = getattr(bm_entry.meta, "polar", None)
    if polar_bm_raw is None:
        return None
    try:
        polar_bm = float(polar_bm_raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(polar_bm):
        return None
    try:
        hv_bm = float(getattr(bm_entry.meta, "hv", 0.0) or 0.0)
        hv_fs = float(getattr(fs_entry.meta, "hv", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    azi_bm = getattr(bm_entry.meta, "azi", None)
    azi_fs = getattr(fs_entry.meta, "azi", None)

    scale_fs = _scale_factor(hv_fs, work_func, a_lattice)
    if scale_fs is None:
        return BMCutLine(
            label=_short_label(bm_path), bm_path=bm_path,
            polar_bm=polar_bm, azi_bm=azi_bm, hv_bm=hv_bm,
            kx_points=np.array([]), ky_points=np.array([]),
            quality="incompatible",
            warning="hv FS invalide → projection impossible",
        )

    polar_fs_c = _polar_fs_center(fs_metadata, fs_entry)
    ky_in_fs_local = scale_fs * np.sin(np.radians(polar_bm - polar_fs_c))

    # Segment kx dans le repère LOCAL de la BM (avant rotation azi)
    # Si hv diffère, scale kx pour rester comparable à la FS
    scale_bm = _scale_factor(hv_bm, work_func, a_lattice)
    t = np.linspace(float(kpar_range[0]), float(kpar_range[1]), int(n_points))
    if scale_bm is None or scale_bm <= 0:
        kx_local = t.copy()
    elif abs(scale_fs - scale_bm) / max(scale_fs, scale_bm) > 1e-6:
        kx_local = t * (scale_fs / scale_bm)
    else:
        kx_local = t.copy()
    ky_local = np.full_like(kx_local, ky_in_fs_local)

    # Rotation par delta_azi autour de Γ si azi diffère
    if azi_bm is not None and azi_fs is not None:
        try:
            delta_azi_rad = np.radians(_angle_delta_deg(float(azi_fs), float(azi_bm)))
        except (TypeError, ValueError):
            delta_azi_rad = 0.0
    else:
        delta_azi_rad = 0.0
    if abs(delta_azi_rad) > 1e-12:
        c, s = np.cos(delta_azi_rad), np.sin(delta_azi_rad)
        kx_out = kx_local * c - ky_local * s
        ky_out = kx_local * s + ky_local * c
    else:
        kx_out = kx_local
        ky_out = ky_local

    quality, warning = _classify_quality(
        hv_bm, hv_fs, azi_bm, azi_fs,
        hv_tol_rel=hv_tolerance_rel,
        azi_tol_deg=azi_tolerance_deg,
    )

    return BMCutLine(
        label=_short_label(bm_path),
        bm_path=bm_path,
        polar_bm=polar_bm,
        azi_bm=(float(azi_bm) if azi_bm is not None else None),
        hv_bm=hv_bm,
        kx_points=kx_out,
        ky_points=ky_out,
        quality=quality,
        warning=warning,
    )


def _short_label(path: str) -> str:
    """Nom court pour affichage légende (basename sans extension)."""
    from pathlib import Path
    try:
        return Path(path).stem
    except Exception:
        return str(path)
