"""Human-in-the-loop pocket seeding from a rectangular lasso selection.

The user drags a box around ONE pocket on the FS map; this module derives the
seed point and iso-level automatically so the wizard's smoothing/level knobs
become unnecessary for the common case. Pure numpy — no PyQt (layering rule).

Every degenerate input returns a loud ``ValueError`` with a user-displayable
message (never a silent fallback), per the project rule against silent
corrections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LassoSeed:
    """Seed + iso-level derived from a lasso selection (raw FS coordinates)."""
    seed_kx: float
    seed_ky: float
    level: float
    n_px: int


def lasso_to_seed(kx, ky, fs, rect) -> LassoSeed:
    """Derive (seed, level) from a rectangular selection on the FS map.

    Parameters
    ----------
    kx, ky : 1D axes of ``fs`` (raw, uncentered coordinates).
    fs : 2D map, shape (len(ky), len(kx)), expected normalized [0, 1].
    rect : (kx0, kx1, ky0, ky1) selection bounds, any order per axis.

    Seed = selection center; level = 40th percentile of the finite intensity
    inside the box (a closed pocket wall sits above it, background below).
    """
    kx = np.asarray(kx, dtype=float).ravel()
    ky = np.asarray(ky, dtype=float).ravel()
    fs = np.asarray(fs, dtype=float)
    x0, x1 = sorted((float(rect[0]), float(rect[1])))
    y0, y1 = sorted((float(rect[2]), float(rect[3])))

    mx = (kx >= x0) & (kx <= x1)
    my = (ky >= y0) & (ky <= y1)
    n_px = int(mx.sum()) * int(my.sum())
    if n_px < 16:
        raise ValueError("Pocket lasso: selection too small — drag a box around one pocket.")

    block = fs[np.ix_(my, mx)]
    vals = block[np.isfinite(block)]
    if vals.size < 16:
        raise ValueError("Pocket lasso: no data in the selection (NaN region).")
    vmin, vmax = float(vals.min()), float(vals.max())
    if not np.isfinite(vmax - vmin) or (vmax - vmin) < 1e-6:
        raise ValueError("Pocket lasso: selection has no intensity contrast.")

    level = float(np.percentile(vals, 40))
    return LassoSeed(
        seed_kx=0.5 * (x0 + x1),
        seed_ky=0.5 * (y0 + y1),
        level=level,
        n_px=int(vals.size),
    )


def contour_convexity_ratio(contour) -> float:
    """convex_hull_area / contour_area — ≈1 for a convex pocket, ≫1 for a
    figure-eight / double pocket (lasso probably caught two bands)."""
    from scipy.spatial import ConvexHull

    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or len(pts) < 4:
        return float("nan")
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area <= 1e-12:
        return float("inf")
    try:
        hull = ConvexHull(pts)
        return float(hull.volume) / area  # 2D: .volume == area
    except Exception:
        return float("nan")


def contour_touches_boundary(contour, kx, ky, tol_frac: float = 0.02) -> bool:
    """True if the contour runs along the scan edge (pocket cut by the
    acquisition window → Arc mode is the honest representation)."""
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or pts.size == 0:
        return False
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    tx = tol_frac * (float(kx.max()) - float(kx.min()))
    ty = tol_frac * (float(ky.max()) - float(ky.min()))
    near = (
        (pts[:, 0] <= kx.min() + tx) | (pts[:, 0] >= kx.max() - tx)
        | (pts[:, 1] <= ky.min() + ty) | (pts[:, 1] >= ky.max() - ty)
    )
    return int(near.sum()) > 3
