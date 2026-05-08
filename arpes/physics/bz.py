"""Presets simples de zone de Brillouin 2D pour overlays FS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import Voronoi
except Exception:  # pragma: no cover - fallback exact par demi-plans
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
    "square": BZPreset("square", "Carré", "square", 1.0, 1.0, 90.0, "Γ-X-M, a=b, angle 90°"),
    "rectangle": BZPreset("rectangle", "Rectangle", "rectangle", 1.0, 0.75, 90.0, "Γ-X/Y-S, a≠b"),
    "hexagonal": BZPreset("hexagonal", "Hexagonal", "hexagon", 1.0, 0.866, 60.0, "Γ-M-K"),
    "centered_rect": BZPreset("centered_rect", "Rectangle centré", "centered_rect", 1.0, 0.75, 90.0, "Maille centrée 2D, ZDB Wigner-Seitz"),
    "oblique": BZPreset("oblique", "Oblique", "oblique", 1.0, 0.85, 75.0, "Angle libre entre vecteurs réciproques"),
}

BZ_PRESET_ALIASES = {
    "tetragonal": "square",
    "orthorhombic": "rectangle",
}


def resolve_bz_preset(key: str) -> BZPreset:
    """Résout les anciennes clés de preset vers les 5 réseaux 2D actuels."""
    resolved = BZ_PRESET_ALIASES.get(str(key), str(key))
    return BZ_PRESETS[resolved]


def _closed_wigner_seitz(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Cellule de Wigner-Seitz 2D autour de Γ pour un réseau réciproque."""
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
        raise ValueError("ZDB Voronoi non bornée")
    poly = np.asarray([vor.vertices[idx] for idx in region], dtype=float)
    center = poly.mean(axis=0)
    order = np.argsort(np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0]))
    poly = poly[order]
    return np.vstack([poly, poly[0]])


def _closed_wigner_seitz_halfplanes(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Intersection des médiatrices entre Γ et ses voisins de réseau."""
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
            raise ValueError("ZDB Wigner-Seitz vide")
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


def _reciprocal_basis(shape: str, half_x: float, half_y: float, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
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
    """Retourne les sommets fermés d'une ZDB 2D normalisée."""
    g1, g2 = _reciprocal_basis(str(shape), half_x, half_y, angle_deg)
    return _closed_wigner_seitz(g1, g2)


def bz_high_symmetry_points(
    shape: str,
    half_x: float,
    half_y: float,
    angle_deg: float = 90.0,
) -> list[tuple[float, float, str, str]]:
    """Points d'aide visuelle: (x, y, label, color)."""
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
    if shape in {"square", "rectangle"}:
        for x, y in [(bx, 0), (-bx, 0), (0, by), (0, -by)]:
            points.append((x, y, "X", "cyan"))
        for x, y in [(bx, by), (bx, -by), (-bx, by), (-bx, -by)]:
            points.append((x, y, "M", "lime"))
        return points
    poly = bz_polygon(shape, bx, by, angle_deg)[:-1]
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        mid = 0.5 * (a + b)
        points.append((float(mid[0]), float(mid[1]), "X", "cyan"))
    for x, y in poly:
        points.append((float(x), float(y), "M", "lime"))
    return points
