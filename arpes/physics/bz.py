"""Simple 2D Brillouin zone presets for FS overlays."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.spatial import Voronoi
except Exception:  # pragma: no cover - exact half-plane fallback
    Voronoi = None


@dataclass(frozen=True)
class BZPreset:
    key: str
    label: str
    shape: str
    half_x: float
    half_y: float
    angle_deg: float = 90.0
    note: str = ""


BZ_PRESETS: dict[str, BZPreset] = {
    "square": BZPreset("square", "Square", "square", 1.0, 1.0, 90.0, "Γ-X-M, a=b, angle 90°"),
    "rectangle": BZPreset("rectangle", "Rectangle", "rectangle", 1.0, 0.75, 90.0, "Γ-X/Y-S, a≠b"),
    "hexagonal": BZPreset("hexagonal", "Hexagonal", "hexagon", 1.0, 0.866, 60.0, "Γ-M-K"),
    "centered_rect": BZPreset("centered_rect", "Centered rectangle", "centered_rect", 1.0, 0.75, 90.0, "2D centered cell, Wigner-Seitz BZ"),
    "oblique": BZPreset("oblique", "Oblique", "oblique", 1.0, 0.85, 75.0, "Free angle between reciprocal vectors"),
}

BZ_PRESET_ALIASES = {
    "tetragonal": "square",
    "orthorhombic": "rectangle",
}


def resolve_bz_preset(key: str) -> BZPreset:
    """Resolve legacy preset keys to the current 5 2D lattice presets."""
    resolved = BZ_PRESET_ALIASES.get(str(key), str(key))
    return BZ_PRESETS[resolved]


def _closed_wigner_seitz(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """2D Wigner-Seitz cell around Γ for a reciprocal lattice."""
    if Voronoi is None:
        return _closed_wigner_seitz_halfplanes(g1, g2)
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pts.append(i * g1 + j * g2)
    pts_arr = np.asarray(pts, dtype=float)
    origin_idx = int(np.argmin(np.sum(pts_arr * pts_arr, axis=1)))
    vor = Voronoi(pts_arr)
    region_idx = vor.point_region[origin_idx]
    region = vor.regions[region_idx]
    if not region or any(idx < 0 for idx in region):
        raise ValueError("Voronoi BZ is unbounded")
    poly = np.asarray([vor.vertices[idx] for idx in region], dtype=float)
    center = poly.mean(axis=0)
    order = np.argsort(np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0]))
    poly = poly[order]
    return np.vstack([poly, poly[0]])


def _closed_wigner_seitz_halfplanes(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Intersection of bisectors between Γ and its lattice neighbors."""
    scale = 8.0 * max(float(np.linalg.norm(g1)), float(np.linalg.norm(g2)), 1.0)
    poly = np.asarray(
        [[-scale, -scale], [scale, -scale], [scale, scale], [-scale, scale]],
        dtype=float,
    )
    neigh = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            r = i * g1 + j * g2
            neigh.append(r)
    neigh.sort(key=lambda v: float(np.dot(v, v)))
    for r in neigh:
        limit = 0.5 * float(np.dot(r, r))
        poly = _clip_polygon_halfplane(poly, r, limit)
        if poly.size == 0:
            raise ValueError("Wigner-Seitz BZ is empty")
    center = poly.mean(axis=0)
    order = np.argsort(np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0]))
    poly = poly[order]
    return np.vstack([poly, poly[0]])


def _clip_polygon_halfplane(poly: np.ndarray, normal: np.ndarray, limit: float) -> np.ndarray:
    if poly.size == 0:
        return poly
    out = []
    prev = poly[-1]
    prev_val = float(np.dot(prev, normal) - limit)
    for cur in poly:
        cur_val = float(np.dot(cur, normal) - limit)
        if cur_val <= 1e-10:
            if prev_val > 1e-10:
                t = prev_val / (prev_val - cur_val)
                out.append(prev + t * (cur - prev))
            out.append(cur)
        elif prev_val <= 1e-10:
            t = prev_val / (prev_val - cur_val)
            out.append(prev + t * (cur - prev))
        prev = cur
        prev_val = cur_val
    if not out:
        return np.empty((0, 2), dtype=float)
    return np.asarray(out, dtype=float)


# P4.8: trigonal/triangular lattices (Bi2Se3 & topological materials) → hexagonal BZ.
_SHAPE_ALIASES = {"triangular": "hexagon", "trigonal": "hexagon"}


def _norm_shape(shape: str) -> str:
    return _SHAPE_ALIASES.get(str(shape), str(shape))


def _reciprocal_basis(shape: str, half_x: float, half_y: float, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    shape = _norm_shape(shape)
    bx = max(float(half_x), 1e-9)
    by = max(float(half_y), 1e-9)
    if shape == "square":
        by = bx
    if shape in {"square", "rectangle"}:
        return np.array([2.0 * bx, 0.0]), np.array([0.0, 2.0 * by])
    if shape == "hexagon":
        return np.array([1.5 * bx, by]), np.array([1.5 * bx, -by])
    if shape == "centered_rect":
        return np.array([2.0 * bx, 0.0]), np.array([bx, 2.0 * by])
    angle = float(np.clip(angle_deg, 20.0, 160.0))
    theta = np.deg2rad(angle)
    return np.array([2.0 * bx, 0.0]), np.array([2.0 * by * np.cos(theta), 2.0 * by * np.sin(theta)])


def bz_polygon(shape: str, half_x: float, half_y: float, angle_deg: float = 90.0) -> np.ndarray:
    """Return the closed vertices of a normalized 2D BZ."""
    g1, g2 = _reciprocal_basis(str(shape), half_x, half_y, angle_deg)
    return _closed_wigner_seitz(g1, g2)


def bz_high_symmetry_points(
    shape: str,
    half_x: float,
    half_y: float,
    angle_deg: float = 90.0,
    label_overrides: dict[str, str] | None = None,
) -> list[tuple[float, float, str, str]]:
    """Visual aid points: (x, y, label, color).

    Labels follow standard 2D crystallographic conventions:
    - square (tetragonal)      : Γ, X (edge center), M (corner)
    - hexagonal                : Γ, K (vertex), M (edge center)
    - rectangle (orthorhombic) : Γ, X=(π/a,0), Y=(0,π/b), S (corner)
    - centered rectangle       : Γ, X (edge center), S (BZ vertex)
    - oblique                  : Γ only; vertices marked without label
      (no standard high-symmetry points beyond Γ).

    ``label_overrides`` renames the canonical labels above on output
    (e.g. ``{"M": "Σ"}`` for the I4/mmm 122-pnictide convention). It is the
    single remap point: every consumer (BZ overlay, direction matching)
    sees the renamed labels, so a logbook "Γ-Σ" cut keeps matching the
    geometry.
    """
    if label_overrides:
        base = bz_high_symmetry_points(shape, half_x, half_y, angle_deg)
        src = list(label_overrides.keys())
        dst = [str(label_overrides[k]) for k in src]
        return _label_remap(base, src, dst)
    shape = _norm_shape(shape)
    bx = max(float(half_x), 1e-9)
    by = max(float(half_y), 1e-9)
    points = [(0.0, 0.0, "Γ", "white")]
    if shape == "square":
        by = bx
    if shape == "hexagon":
        poly = bz_polygon("hexagon", bx, by, angle_deg)[:-1]
        for x, y in poly:
            points.append((float(x), float(y), "K", "lime"))
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            mid = 0.5 * (a + b)
            points.append((float(mid[0]), float(mid[1]), "M", "cyan"))
        return points
    if shape == "square":
        for x, y in [(bx, 0), (-bx, 0), (0, by), (0, -by)]:
            points.append((x, y, "X", "cyan"))
        for x, y in [(bx, by), (bx, -by), (-bx, by), (-bx, -by)]:
            points.append((x, y, "M", "lime"))
        return points
    if shape == "rectangle":
        # Orthorhombic: inequivalent edges X≠Y, corner = S (≠ M).
        for x, y in [(bx, 0), (-bx, 0)]:
            points.append((x, y, "X", "cyan"))
        for x, y in [(0, by), (0, -by)]:
            points.append((x, y, "Y", "deepskyblue"))
        for x, y in [(bx, by), (bx, -by), (-bx, by), (-bx, -by)]:
            points.append((x, y, "S", "lime"))
        return points
    poly = bz_polygon(shape, bx, by, angle_deg)[:-1]
    if shape == "centered_rect":
        # Hexagonal BZ of the centered lattice: edge centers → X,
        # vertices → S. No M (reserved for square/hexagonal).
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            mid = 0.5 * (a + b)
            points.append((float(mid[0]), float(mid[1]), "X", "cyan"))
        for x, y in poly:
            points.append((float(x), float(y), "S", "lime"))
        return points
    # oblique: no standard high-symmetry label beyond Γ. Mark the
    # unnamed vertices (visual reference, no misleading X/M label).
    for x, y in poly:
        points.append((float(x), float(y), "", "#9ca3af"))
    return points


# --------------------------------------------------------------------------
# Real 3D crystal mapping → 2D preset + HS labels for the current kz plane.
# Serves the BZ overlay in the FS window (cf. Fig. 4 Ideta 2014, BaNi2P2).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Lattice3D:
    """Crystallographic unit cell parameters (3D Bravais lattice)."""
    a: float                       # Å
    b: float                       # Å
    c: float                       # Å
    alpha_deg: float = 90.0
    beta_deg: float = 90.0
    gamma_deg: float = 90.0
    bravais: str = "tetragonal"    # tetragonal | orthorhombic | hexagonal | cubic
    space_group: str | int = ""
    mp_id: str = ""

    def preset_key(self) -> str:
        bv = (self.bravais or "").lower()
        if bv in ("cubic", "tetragonal"):
            return "square"
        if bv == "orthorhombic":
            return "rectangle"
        if bv == "hexagonal":
            return "hexagonal"
        return "square"  # reasonable fallback


# Display conventions for the theoretical BZ overlay. Pure label renames —
# the geometry never changes. Papers disagree on naming (e.g. 122-pnictide
# I4/mmm articles label the square-zone corner Σ where the P-tetragonal
# convention says M); the user picks the convention matching the article, or
# edits labels freely on top of a preset.
BZ_LABEL_CONVENTION_PRESETS: dict[str, dict[str, str]] = {
    "standard": {},
    "i4mmm_sigma_corner": {"M": "Σ"},
    "pnictide_1fe_2fe_swap": {"X": "M", "M": "X"},
}

BZ_LABEL_CONVENTION_TITLES: dict[str, str] = {
    "standard": "Standard (X faces, M corners)",
    "i4mmm_sigma_corner": "I4/mmm 122 (Σ corners)",
    "pnictide_1fe_2fe_swap": "Pnictide 1-Fe ↔ 2-Fe (swap X/M)",
}


_HS_LABELS_BY_PLANE: dict[str, dict[str, list[str]]] = {
    # kz=0 plane (Γ-plane) → center = Γ; kz=π/c plane (Z-plane) → center = Z.
    # Tetragonal I4/mmm: Γ-X-M ↔ Z-R-A (cf. Bilbao server).
    "tetragonal": {"Gamma": ["Γ", "X", "M"], "Z": ["Z", "R", "A"]},
    "orthorhombic": {"Gamma": ["Γ", "X", "Y", "S"], "Z": ["Z", "U", "T", "R"]},
    "hexagonal": {"Gamma": ["Γ", "K", "M"], "Z": ["A", "H", "L"]},
    "cubic": {"Gamma": ["Γ", "X", "M"], "Z": ["Γ", "X", "M"]},  # kz=2π/a equiv. Γ
}


def _label_remap(points_2d: list[tuple[float, float, str, str]],
                 src_labels: list[str], dst_labels: list[str]
                 ) -> list[tuple[float, float, str, str]]:
    """Rename HS labels according to the src→dst table, preserving order/color."""
    if not src_labels or not dst_labels or len(src_labels) != len(dst_labels):
        return points_2d
    rename = dict(zip(src_labels, dst_labels))
    out = []
    for x, y, lab, col in points_2d:
        out.append((x, y, rename.get(lab, lab), col))
    return out


def bz_points_for_lattice_plane(
    lattice: Lattice3D,
    plane: str = "Gamma",
) -> tuple[np.ndarray, list[tuple[float, float, str, str]], str]:
    """2D BZ polygon and labelled HS points for a 3D crystal in a kz plane.

    Returns ``(polygon_xy, hs_points, preset_key)``.

    - ``plane`` ∈ {"Gamma", "Z"}. Selects the appropriate HS label table.
    - Polygon and coordinates in units of **π/a** along kx and **π/b** along ky
      (consistent with the ARPES loader convention).
    - ``preset_key`` returned for debug / persistence.

    V1 limitation: hexagonal-centered and triclinic BZs are not handled.
    For unrecognized Bravais types: fallback to "square" + Γ-plane labels.
    """
    preset_key = lattice.preset_key()
    # Half-extents in units consistent with presets (1.0 = π/a).
    if preset_key == "square":
        half_x = 1.0
        half_y = 1.0
    elif preset_key == "rectangle":
        half_x = 1.0
        half_y = float(lattice.a) / max(float(lattice.b), 1e-9)
    elif preset_key == "hexagonal":
        half_x = 1.0
        half_y = float(np.sqrt(3.0) / 2.0)
    else:
        preset_key = "square"
        half_x = 1.0
        half_y = 1.0

    # bz_polygon/bz_high_symmetry_points expect the `shape` string
    # (`square`, `hexagon`, ...), not the preset key (`hexagonal`).
    shape = resolve_bz_preset(preset_key).shape
    poly = bz_polygon(shape, half_x, half_y, lattice.gamma_deg)
    pts_raw = bz_high_symmetry_points(shape, half_x, half_y, lattice.gamma_deg)

    bv = (lattice.bravais or "tetragonal").lower()
    table = _HS_LABELS_BY_PLANE.get(bv, _HS_LABELS_BY_PLANE["tetragonal"])
    src = table.get("Gamma", [])
    dst = table.get(plane, src)
    pts = _label_remap(pts_raw, src, dst)
    return poly, pts, preset_key
