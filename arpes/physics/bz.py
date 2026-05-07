"""Presets simples de zone de Brillouin 2D pour overlays FS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BZPreset:
    key: str
    label: str
    shape: str
    half_x: float
    half_y: float
    note: str = ""


BZ_PRESETS: dict[str, BZPreset] = {
    "tetragonal": BZPreset("tetragonal", "Tétragonal / carré", "rectangle", 1.0, 1.0, "Γ-X-M, a≈b"),
    "orthorhombic": BZPreset("orthorhombic", "Orthorhombique / rectangle", "rectangle", 1.0, 0.75, "Γ-X/Y-S, a≠b"),
    "hexagonal": BZPreset("hexagonal", "Hexagonal", "hexagon", 1.0, 0.866, "Γ-M-K"),
}


def bz_polygon(shape: str, half_x: float, half_y: float) -> np.ndarray:
    """Retourne les sommets fermés d'une ZDB 2D normalisée."""
    bx = max(float(half_x), 1e-9)
    by = max(float(half_y), 1e-9)
    if shape == "hexagon":
        angles = np.deg2rad([0, 60, 120, 180, 240, 300, 0])
        return np.column_stack([bx * np.cos(angles), by * np.sin(angles)])
    return np.asarray([[-bx, -by], [bx, -by], [bx, by], [-bx, by], [-bx, -by]], dtype=float)


def bz_high_symmetry_points(shape: str, half_x: float, half_y: float) -> list[tuple[float, float, str, str]]:
    """Points d'aide visuelle: (x, y, label, color)."""
    bx = max(float(half_x), 1e-9)
    by = max(float(half_y), 1e-9)
    points = [(0.0, 0.0, "Γ", "white")]
    if shape == "hexagon":
        poly = bz_polygon("hexagon", bx, by)[:-1]
        for x, y in poly:
            points.append((float(x), float(y), "K", "lime"))
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            mid = 0.5 * (a + b)
            points.append((float(mid[0]), float(mid[1]), "M", "cyan"))
        return points
    for x, y in [(bx, 0), (-bx, 0), (0, by), (0, -by)]:
        points.append((x, y, "X", "cyan"))
    for x, y in [(bx, by), (bx, -by), (-bx, by), (-bx, -by)]:
        points.append((x, y, "M", "lime"))
    return points
