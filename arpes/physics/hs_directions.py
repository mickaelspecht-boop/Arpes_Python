"""High-symmetry cut directions: label normalization + per-BZ-shape registry.

A cut direction is the in-plane crystal direction a band map slices along, e.g.
``Γ-X``. ARPES logbooks write these many ways ("GX", "G-X", "Gamma to X",
"ΓX", "Γ→Σ"); this module canonicalizes them to ``A-B`` with proper Greek
point labels, and lists the standard directions per Brillouin-zone shape so the
UI can offer/validate them.

Convention note: the single letter ``S`` is treated as ``Σ`` (Sigma) on *input*
because users write ``GS`` for the Γ-Σ cut. The rectangle BZ corner point is
also called ``S``; it stays literal in the registry below (registry strings are
canonical outputs and are not passed through the input normalizer). Square /
hexagonal zones — including BaNi2As2 — have no ``S`` point, so there is no
clash there.
"""
from __future__ import annotations

import re

import numpy as np

# Input letter -> canonical high-symmetry point label.
_LETTER_TO_POINT = {
    "g": "Γ", "x": "X", "m": "M", "k": "K", "y": "Y", "z": "Z", "s": "Σ",
}


def normalize_direction_label(value) -> str:
    """Canonicalize a free-text cut direction to ``A-B`` (e.g. "GX" -> "Γ-X").

    Returns "" if nothing parseable. Accepts arrows (``->``, ``→``), the words
    ``to``/``vers``, slashes/underscores/spaces as separators, the words
    ``gamma``/``sigma``, contiguous codes ("gx", "xm"), and multi-point paths
    ("Γ-X-M").
    """
    if value is None:
        return ""
    low = str(value).strip().lower()
    if not low:
        return ""
    low = low.replace("γ", "g").replace("σ", "s")
    low = low.replace("gamma", "g").replace("sigma", "s")
    # Unify every separator (arrows, "to"/"vers", slash/underscore/space) to '-'.
    low = re.sub(r"(?:->|=>|→|to|vers|[\s/_>])+", "-", low)
    low = re.sub(r"-{2,}", "-", low).strip("-")
    if not low:
        return ""
    points: list[str] = []
    for part in low.split("-"):
        if not part:
            continue
        chars = [part] if len(part) == 1 else list(part)
        for ch in chars:
            points.append(_LETTER_TO_POINT.get(ch, ch.upper()))
    points = [p for p in points if p]
    return "-".join(points)


# Standard cut directions per canonical BZ shape (names match bz.py polygons).
# Registry strings are authoritative outputs — do NOT run them through the input
# normalizer (it would turn the rectangle corner "S" into "Σ").
BZ_DIRECTIONS: dict[str, list[str]] = {
    "square": ["Γ-X", "Γ-M", "X-M"],
    "rectangle": ["Γ-X", "Γ-Y", "Γ-S", "X-S", "Y-S"],
    "hexagon": ["Γ-M", "Γ-K", "M-K"],
    "centered_rect": ["Γ-X", "Γ-S"],
}


def bz_directions(shape: str) -> list[str]:
    """Standard cut directions for a BZ shape (empty for unknown/oblique)."""
    return list(BZ_DIRECTIONS.get(str(shape or ""), []))


def _angle_diff_mod180(a: float, b: float) -> float:
    """Smallest absolute angle difference (deg) treating a line as 180°-periodic."""
    d = (float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def direction_from_azimuth(
    azi_deg: float,
    azi_ref_deg: float | None,
    shape: str,
    *,
    bx: float = 1.0,
    by: float = 1.0,
    angle_deg: float = 90.0,
    tol_deg: float = 12.0,
) -> tuple[str, str]:
    """Map a measured azimuth to the nearest Γ-based high-symmetry cut direction.

    Data-driven and freezable, per the project's calibration rule: if
    ``azi_ref_deg`` is None/NaN the azimuth reference is *not* calibrated, so we
    return ``("", "azi=…° (UNCALIBRATED)")`` and never invent a crystal
    direction. Otherwise the cut angle in the crystal frame is
    ``azi_deg − azi_ref_deg`` and we return ``(label, "")`` for the nearest
    Γ-direction within ``tol_deg`` (lines are 180°-periodic), or
    ``("", "azi=…° (no Γ-dir within …°)")`` if none is close.

    Direction angles are derived from the BZ geometry (``bz_high_symmetry_points``)
    so multiple ZDB conventions are handled by the shape/lattice, not hardcoded.
    """
    if azi_ref_deg is None or not np.isfinite(float(azi_ref_deg)):
        return "", f"azi={float(azi_deg):.1f}° (UNCALIBRATED)"
    from arpes.physics.bz import bz_high_symmetry_points

    crystal_angle = float(azi_deg) - float(azi_ref_deg)
    best_label, best_diff = "", float("inf")
    for x, y, label, _color in bz_high_symmetry_points(shape, bx, by, angle_deg):
        if not label or label == "Γ":
            continue
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            continue
        ang = np.degrees(np.arctan2(float(y), float(x)))
        diff = _angle_diff_mod180(crystal_angle, ang)
        if diff < best_diff:
            best_label, best_diff = f"Γ-{label}", diff
    if best_label and best_diff <= float(tol_deg):
        return best_label, ""
    return "", f"azi={float(azi_deg):.1f}° (no Γ-dir within {tol_deg:.0f}°)"
