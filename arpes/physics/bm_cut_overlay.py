"""Project a BM into the (kx, ky) frame of an FS, pure and Qt-free.

B.1 of the BM↔FS plan (cf BM_FS_ORGANIZATION_PLAN.md). Computes the line
corresponding to a BM (cut at fixed polar) in the 2D frame of an FS map.

Physical principle:
- A BM measured at `polar = P_bm` geometrically corresponds to a horizontal
  cut in (kx, ky) at a fixed ordinate ky_in_fs determined by the difference
  (P_bm − P_fs_center).
- If azi differs between the FS and BM, the cut is rotated by
  Δazi = azi_fs − azi_bm autour de Γ.
- If hv differs, the k scale factor changes → extrapolated projection
  (degraded quality).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Constant + scale factor: single source (P2.1a). Tilt corrected in P2.1b.
from arpes.physics.kpar_geometry import (
    C_ARPES,
    kpar_scale,
)
from arpes.physics.hs_directions import normalize_direction_label


Quality = Literal["exact", "rotated", "scaled", "incompatible"]


@dataclass(frozen=True)
class BMCutLine:
    """Representation of a BM cut projected into an FS frame.

    `kx_points` and `ky_points` are equal-length arrays defining the segment
    to draw in the FS panel. `quality` indicates the physical reliability of
    the projection.
    """
    label: str                 # short display / pick label
    bm_path: str               # full path for interaction
    polar_bm: float            # BM motor angle (deg)
    azi_bm: float | None
    hv_bm: float
    kx_points: np.ndarray
    ky_points: np.ndarray
    quality: Quality
    warning: str = ""


def _scale_factor(hv: float, work_func: float, a_lattice: float) -> float | None:
    """C·√(Ek)·a/π — conversion factor sin(θ) → k(π/a).

    Thin wrapper over ``kpar_geometry.kpar_scale`` (single source). Returns None
    if Ek = hv − φ is invalid.
    """
    return kpar_scale(hv, work_func, a_lattice)


def _safe_float(value, default: float = 0.0) -> float:
    """Read a float while tolerating None/non-finite values → ``default`` (missing tilt = 0°)."""
    if value is None:
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _polar_fs_center(fs_metadata: dict, fs_entry) -> float:
    """Central polar of the FS scan (deg).

    Priority: `fs_metadata["fs_scan_axis_deg"]["center"]`, then
    `fs_entry.meta.polar`, then 0.0.
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
        return "incompatible", "invalid hv"
    hv_diff_rel = abs(hv_bm - hv_fs) / max(hv_bm, hv_fs)
    hv_close = hv_diff_rel <= hv_tol_rel
    if azi_bm is None or azi_fs is None:
        azi_diff = 0.0  # benefit of the doubt if unspecified
    else:
        try:
            azi_diff = abs(_angle_delta_deg(float(azi_fs), float(azi_bm)))
        except (TypeError, ValueError):
            azi_diff = 0.0
    azi_close = azi_diff <= azi_tol_deg

    if hv_close and azi_close:
        return "exact", ""
    if hv_close and not azi_close:
        return "rotated", f"Δazi={azi_diff:+.1f}° → rotation applied"
    if not hv_close and azi_close:
        return "scaled", f"Δhv={hv_bm - hv_fs:+.1f} eV → extrapolated scale"
    return "scaled", (
        f"Δhv={hv_bm - hv_fs:+.1f} eV, Δazi={azi_diff:+.1f}° → "
        "composite projection (interpret with caution)"
    )


def _angle_delta_deg(dst: float, src: float) -> float:
    """Signed shortest angular delta dst-src in degrees, in [-180, 180)."""
    return (float(dst) - float(src) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class BZGeometry:
    """BZ parameters of the FS panel, used to resolve direction labels.

    Defaults match the historical hardcoded behaviour (square zone).
    `label_overrides` is the user label convention (e.g. {"M": "Σ"}): the
    logbook direction is matched against the RENAMED labels, so a "Γ-Σ" cut
    keeps working after the user switches convention.
    """
    shape: str = "square"
    half_x: float = 1.0
    half_y: float = 1.0
    angle_deg: float = 90.0
    label_overrides: dict | None = None


def _hs_point_coords(label: str, geom: BZGeometry) -> tuple[float, float] | None:
    """Coordinates of the first listed HS point named `label` (post-convention).

    Equivalent points (4 X's on a square…) are listed positive-x first in
    bz_high_symmetry_points, so the choice is deterministic: Γ-X → 0°,
    Γ-M → 45° on the default square zone.

    Logbook alias: on a square zone the vertical axis point (0, b) is also
    labelled X (it is symmetry-equivalent), yet experimenters write "Γ-Y"
    for the vertical cut. A request for an absent "Y" therefore resolves to
    the most vertical X-equivalent point when one exists.
    """
    from arpes.physics.bz import bz_high_symmetry_points
    points = bz_high_symmetry_points(
        geom.shape, geom.half_x, geom.half_y, geom.angle_deg,
        label_overrides=geom.label_overrides,
    )
    for x, y, name, _color in points:
        if name == label:
            return float(x), float(y)
    if label == "Y":
        candidates = [(x, y) for x, y, name, _ in points
                      if name == "X" and (abs(x) > 1e-12 or abs(y) > 1e-12)]
        if candidates:
            # Most vertical X-equivalent (line angles are 180°-periodic).
            def _vert(c):
                ang = np.degrees(np.arctan2(c[1], c[0])) % 180.0
                return abs(ang - 90.0)
            best = min(candidates, key=_vert)
            if _vert(best) < 45.0:
                return float(best[0]), float(best[1])
    return None


def _direction_angle_deg(value, geom: BZGeometry) -> tuple[float | None, str]:
    """Angle (deg, 0 = +kx) of a high-symmetry line label in the chosen BZ.

    Data-driven: the angle comes from the actual BZ geometry selected in the
    FS panel (shape, half_x/half_y, angle, label convention), not from a
    hardcoded table — Γ-M is 45° on a square zone but not on a rectangle.

    Returns (angle, "") on success, (None, reason) when the label cannot be
    resolved (no direction, oblique zone, label absent from this BZ).
    """
    label = normalize_direction_label(value)
    if not label:
        return None, ""
    parts = [p for p in label.split("-") if p]
    if len(parts) < 2:
        return None, f"direction '{label}' incomplete"
    start, end = parts[0], parts[-1]
    p_start = (0.0, 0.0) if start == "Γ" else _hs_point_coords(start, geom)
    p_end = (0.0, 0.0) if end == "Γ" else _hs_point_coords(end, geom)
    if p_start is None or p_end is None:
        missing = start if p_start is None else end
        return None, (
            f"direction {label}: point '{missing}' not in the current BZ "
            f"({geom.shape}) — check BZ shape / label conventions"
        )
    dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None, f"direction {label}: degenerate (zero-length line)"
    return float(np.degrees(np.arctan2(dy, dx))), ""


def _direction_delta_deg(bm_entry, fs_entry, geom: BZGeometry) -> tuple[float | None, str]:
    """Rotation FS−BM from logbook direction labels, resolved on `geom`.

    The FS direction defaults to the 0° axis of the zone when unspecified
    (the FS kx axis is taken as the reference horizontal direction).
    Unresolvable BM labels return (None, loud reason) — never a silent 0°.
    """
    bm_dir = getattr(getattr(bm_entry, "meta", None), "direction", "")
    fs_dir = getattr(getattr(fs_entry, "meta", None), "direction", "")
    bm_ang, bm_reason = _direction_angle_deg(bm_dir, geom)
    if bm_ang is None:
        return None, bm_reason
    fs_ang, _ = _direction_angle_deg(fs_dir, geom)
    fs_label = normalize_direction_label(str(fs_dir))
    if fs_ang is None:
        fs_ang = 0.0
        fs_label = "FS axis (0°)"
    # BM line angle in the FS frame = θ_bm − θ_fs (the FS kx axis lies along
    # the FS's own direction). The historical fs−bm order rotated Γ-M cuts
    # to 135° instead of 45° — wrong side of the zone diagonal.
    return _angle_delta_deg(bm_ang, fs_ang), (
        f"direction {normalize_direction_label(str(bm_dir))} vs "
        f"{fs_label} → rotation applied"
    )


def compute_bm_cut_in_fs_frame(
    bm_entry,
    bm_path: str,
    fs_entry,
    fs_path: str,
    fs_metadata: dict,
    *,
    work_func: float,
    a_lattice: float = 0.0,
    kpar_range: tuple[float, float] = (-1.5, 1.5),
    n_points: int = 80,
    azi_tolerance_deg: float = 0.5,
    hv_tolerance_rel: float = 0.02,
    overlay_max_hv_rel: float = 0.05,
    bz_geometry: BZGeometry | None = None,
) -> BMCutLine | None:
    """Project a BM into the (kx, ky) frame of an FS.

    Args:
        bm_entry: BM FileEntry (reads meta.hv, meta.polar, meta.azi,
            meta.direction).
        bm_path: BM key/path in session.files.
        fs_entry: reference FS FileEntry.
        fs_path: FS key/path.
        fs_metadata: FS raw_data["metadata"] dict (for fs_scan_axis_deg).
        work_func: φ (eV) for angle↔k conversion.
        a_lattice: lattice parameter (Å). 0 = unknown, projection disabled.
        kpar_range: bounds of the kpar segment to draw (in π/a), default (-1.5, 1.5).
        n_points: number of points along the segment.
        azi_tolerance_deg: beyond this → quality="rotated".
        hv_tolerance_rel: beyond this → quality="scaled".
        bz_geometry: BZ shape/size/convention selected in the FS panel; used
            to resolve logbook direction labels into angles. None = default
            square zone (historical behaviour).

    Orientation priority (user decision, 2026-06): the logbook `direction`
    label WINS over the motor azimuth when both are present — experimenters
    record the crystal direction deliberately while the azi motor zero is
    often uncalibrated. A visible warning reports any disagreement.

    Returns:
        BMCutLine or None if the BM is incomplete (not a BM, no polar, etc.).
    """
    if bm_entry is None or fs_entry is None:
        return None
    if getattr(bm_entry.meta, "scan_kind", "") != "BM":
        return None
    polar_bm_raw = getattr(bm_entry.meta, "polar", None)
    hv_bm_raw = getattr(bm_entry.meta, "hv", None)
    try:
        hv_bm = float(hv_bm_raw or 0.0)
        hv_fs = float(getattr(fs_entry.meta, "hv", 0.0) or 0.0)
    except (TypeError, ValueError):
        hv_bm = 0.0; hv_fs = 0.0
    polar_bm = None
    if polar_bm_raw is not None:
        try:
            v = float(polar_bm_raw)
            if np.isfinite(v):
                polar_bm = v
        except (TypeError, ValueError):
            pass
    if polar_bm is None:
        return BMCutLine(
            label=_short_label(bm_path), bm_path=bm_path,
            polar_bm=float("nan"),
            azi_bm=getattr(bm_entry.meta, "azi", None),
            hv_bm=hv_bm,
            kx_points=np.array([]), ky_points=np.array([]),
            quality="incompatible",
            warning="BM polar missing (logbook or metadata) → no overlay",
        )
    azi_bm = getattr(bm_entry.meta, "azi", None)
    azi_fs = getattr(fs_entry.meta, "azi", None)

    # P2.1b — CORRECTED tilt (Ishida & Shin 2018). The app maps polar→ky (the
    # FS is a polar scan); tilt (rotation around the slit axis) shifts ky
    # ADDITIVELY with polar. The BM is drawn as a ky=const line, and the tilt
    # offset is exact at the cut center. Correct instead of rejecting (old
    # P2.1a guard). Extreme tilt is still reported: the ky=const line deviates
    # from the true cut far from the center.
    tilt_bm = _safe_float(getattr(bm_entry.meta, "tilt", None))
    tilt_fs = _safe_float(getattr(fs_entry.meta, "tilt", None))
    tilt_rel = tilt_bm - tilt_fs

    scale_fs = _scale_factor(hv_fs, work_func, a_lattice)
    if scale_fs is None:
        return BMCutLine(
            label=_short_label(bm_path), bm_path=bm_path,
            polar_bm=polar_bm, azi_bm=azi_bm, hv_bm=hv_bm,
            kx_points=np.array([]), ky_points=np.array([]),
            quality="incompatible",
            warning="invalid FS hv → projection impossible",
        )

    polar_fs_c = _polar_fs_center(fs_metadata, fs_entry)
    # ky = polar contribution (FS scan) + tilt contribution (Ishida & Shin,
    # exact at the cut center; both shift ky in this frame).
    ky_in_fs_local = scale_fs * (
        np.sin(np.radians(polar_bm - polar_fs_c)) + np.sin(np.radians(tilt_rel))
    )
    # Report when the ky=const line deviates noticeably far from the center
    # (unplotted tilt cos(α) term): only for large tilts.
    tilt_note = ""
    if abs(tilt_rel) > 10.0:
        tilt_note = (
            f"tilt Δ{tilt_rel:+.1f}° corrected at center (1st-order Ishida); "
            "ky line approx. far from cut center"
        )

    # kx segment in the LOCAL BM frame (before azi rotation)
    # If hv differs, scale kx to remain comparable to the FS
    scale_bm = _scale_factor(hv_bm, work_func, a_lattice)
    t = np.linspace(float(kpar_range[0]), float(kpar_range[1]), int(n_points))
    if scale_bm is None or scale_bm <= 0:
        kx_local = t.copy()
    elif abs(scale_fs - scale_bm) / max(scale_fs, scale_bm) > 1e-6:
        kx_local = t * (scale_fs / scale_bm)
    else:
        kx_local = t.copy()
    ky_local = np.full_like(kx_local, ky_in_fs_local)

    # Orientation: logbook direction label FIRST (resolved on the chosen BZ
    # geometry), motor azi as fallback. When both exist and disagree, the
    # direction wins and the conflict is reported loudly.
    geom = bz_geometry if bz_geometry is not None else BZGeometry()
    direction_note = ""
    used_direction_rotation = False
    delta_azi: float | None = None
    if azi_bm is not None and azi_fs is not None:
        try:
            delta_azi = _angle_delta_deg(float(azi_fs), float(azi_bm))
        except (TypeError, ValueError):
            delta_azi = None
    delta_dir, direction_note = _direction_delta_deg(bm_entry, fs_entry, geom)
    if delta_dir is not None:
        delta_deg = float(delta_dir)
        used_direction_rotation = abs(delta_deg) > 1e-12
        if delta_azi is not None and abs(_angle_delta_deg(delta_azi, delta_deg)) > azi_tolerance_deg:
            direction_note = (
                f"{direction_note} | ⚠ logbook direction ({delta_deg:+.1f}°) "
                f"overrides motor azi (Δazi={delta_azi:+.1f}°) — verify"
            )
        elif not used_direction_rotation:
            direction_note = ""  # same direction, no rotation: nothing to report
    elif direction_note:
        # The BM has a direction label that could not be resolved on the
        # current BZ: never rotate silently, surface the reason.
        delta_deg = delta_azi if delta_azi is not None else 0.0
        direction_note = f"{direction_note} → direction ignored"
        used_direction_rotation = False
    elif delta_azi is not None:
        delta_deg = delta_azi
    else:
        delta_deg = 0.0
    delta_azi_rad = np.radians(delta_deg)
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
    if used_direction_rotation and quality == "exact":
        quality = "rotated"
    if direction_note:
        # Always surfaced: rotation applied, conflict with motor azi, or
        # unresolvable label — none of these may stay silent.
        warning = f"{warning} | {direction_note}" if warning else direction_note
    if tilt_note:
        warning = f"{warning} | {tilt_note}" if warning else tilt_note
    # Strict overlay guard: if Δhv/max > overlay_max_hv_rel, list the BM
    # but suppress the projection (different kz → misleading projection).
    hv_diff_rel = (abs(hv_bm - hv_fs) / max(hv_bm, hv_fs)) if (hv_bm > 0 and hv_fs > 0) else float("inf")
    if hv_diff_rel > float(overlay_max_hv_rel):
        return BMCutLine(
            label=_short_label(bm_path), bm_path=bm_path,
            polar_bm=polar_bm,
            azi_bm=(float(azi_bm) if azi_bm is not None else None),
            hv_bm=hv_bm,
            kx_points=np.array([]), ky_points=np.array([]),
            quality="incompatible",
            warning=f"Δhv/max = {hv_diff_rel*100:.1f}% > {overlay_max_hv_rel*100:.0f}% → overlay hidden (different kz)",
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
    """Short name for legend display (basename without extension)."""
    from pathlib import Path
    try:
        return Path(path).stem
    except Exception:
        return str(path)
