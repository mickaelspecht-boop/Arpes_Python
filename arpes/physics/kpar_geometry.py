"""ARPES k-parallel geometry: pure functions, no PyQt.

Single source of truth for the angle→k conversion constant and scale factor
``C·√(Ek)·a/π``. Includes guard rails for tilt-sensitive BM↔FS overlays and
the full angle-to-k conversion used by calibrated views.
"""
from __future__ import annotations

import numpy as np

# ARPES constant (identical to the historical value). √(2 m_e)/ħ in units such
# that k[Å⁻¹] = C_ARPES · √(Ek[eV]) · sin(θ).
C_ARPES = 0.51233

# Beyond this tilt (degrees), the 1-angle angle→k projection (polar only) is
# too wrong: disable the BM↔FS overlay instead of producing biased k.
TILT_GUARD_DEG = 2.0


def kpar_scale(hv: float, work_func: float, a_lattice: float) -> float | None:
    """Factor ``C·√(Ek)·a/π`` converting ``sin(θ)`` to ``k`` (π/a).

    Returns ``None`` if ``Ek = hv − φ`` is invalid or if ``a_lattice`` is
    unknown (0). Single source: every other module should call this.
    """
    try:
        ek = float(hv) - float(work_func)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ek) or ek <= 0:
        return None
    try:
        a = float(a_lattice)
    except (TypeError, ValueError):
        return None
    scale = C_ARPES * np.sqrt(ek) * a / np.pi
    if not np.isfinite(scale) or scale <= 0:
        return None
    return float(scale)


def _coerce_tilt(tilt_deg) -> float:
    """Read a tilt (deg), tolerating ``None``/missing field → 0.0.

    Missing tilt means 0° (standard scan without tilt). Loaders must populate
    ``FileMeta.tilt`` when the tilt motor is non-zero.
    """
    if tilt_deg is None:
        return 0.0
    try:
        v = float(tilt_deg)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def tilt_within_guard(tilt_deg, guard_deg: float = TILT_GUARD_DEG) -> bool:
    """``True`` if ``|tilt| ≤ guard`` (1-angle angle→k projection usable)."""
    return abs(_coerce_tilt(tilt_deg)) <= float(guard_deg)


def kpar_from_angles(
    slit_deg,
    polar_deg: float = 0.0,
    tilt_deg: float = 0.0,
    azi_deg: float = 0.0,
    *,
    ek: float,
    slit_axis: str = "x",
) -> tuple[np.ndarray, np.ndarray]:
    """6-axis angle→(kx, ky) conversion in Å⁻¹ — Ishida & Shin RSI 89,043903 (2018).

    Uses the cumulative rotation-matrix formula: n̂_det rotated by
    R = R_φ·R_β·R_θ (azimuth·tilt·polar), then projected.

    Args:
        slit_deg: emission angle along the slit (α), vectorizable.
        polar_deg, tilt_deg, azi_deg: manipulator rotations (θ, β, φ), scalars.
        ek: kinetic energy (eV), > 0.
        slit_axis: 'x' (Ishida convention, slit ∥ x → α in kx) or 'y'
            (Scienta convention, slit ∥ y). Default 'x': reduces exactly to
            the historical 1-angle formula (kx=C·√Ek·sinα, ky=0) when
            tilt=azi=polar=0.

    Returns:
        (kx, ky) in Å⁻¹, same shape as ``slit_deg``.

    Angle signs depend on the lab: DO NOT hard-code them, calibrate from the
    data (UNCALIBRATED register, cf. P2.6).
    """
    a = np.radians(np.asarray(slit_deg, dtype=float))
    th = float(np.radians(polar_deg))
    be = float(np.radians(tilt_deg))
    ph = float(np.radians(azi_deg))
    ek_val = float(ek)
    if not np.isfinite(ek_val) or ek_val <= 0:
        raise ValueError(f"kpar_from_angles: Ek={ek_val} eV invalid (>0 required).")

    if slit_axis not in ("x", "y"):
        raise ValueError(f"unknown slit_axis: {slit_axis!r} (expected 'x' or 'y').")
    # n̂_det: slit ∥ x → (sinα,0,cosα); slit ∥ y → (0,sinα,cosα).
    sa, ca = np.sin(a), np.cos(a)
    if slit_axis == "x":
        nx, ny, nz = sa, np.zeros_like(sa), ca
    else:
        nx, ny, nz = np.zeros_like(sa), sa, ca

    # M = R_β · R_θ applied to n̂_det (before azimuth).
    cth, sth = np.cos(th), np.sin(th)
    cbe, sbe = np.cos(be), np.sin(be)
    mx = cth * nx - sth * ny
    my = cbe * (sth * nx + cth * ny) - sbe * nz
    # R_φ: rotation of plane (mx, my) by φ.
    cph, sph = np.cos(ph), np.sin(ph)
    scale = C_ARPES * np.sqrt(ek_val)
    kx = scale * (cph * mx - sph * my)
    ky = scale * (sph * mx + cph * my)
    return kx, ky


def ky_residual_pi_a(
    tilt_deg, *, hv: float, work_func: float, a_lattice: float
) -> float:
    """Uncorrected ky error (π/a) induced by non-zero tilt.

    First-order Ishida & Shin estimate: ``Δky ≈ scale · sin(tilt)``. Used to
    display residual uncertainty in the gray zone ``0 < |tilt| ≤ 2°`` where
    the overlay remains allowed but uncorrected for tilt. Returns 0.0 if the
    scale factor is undetermined.
    """
    scale = kpar_scale(hv, work_func, a_lattice)
    if scale is None:
        return 0.0
    return float(abs(scale) * abs(np.sin(np.radians(_coerce_tilt(tilt_deg)))))
