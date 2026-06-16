"""Manual FS pocket contour helpers."""
from __future__ import annotations

import numpy as np

from arpes.physics.pocket import (
    _close_contour,
    _properties_from_contour,
    simplify_closed_contour,
    smooth_closed_contour,
)


def snap_manual_contour_points(
    image,
    kx,
    ky,
    points,
    *,
    radius_px: int = 2,
) -> np.ndarray:
    """Move hand-picked contour points to the strongest local FS edge."""
    z = np.asarray(image, dtype=float)
    x = np.asarray(kx, dtype=float)
    y = np.asarray(ky, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("manual contour points must have shape (N,2).")
    if z.ndim != 2 or z.shape != (y.size, x.size):
        raise ValueError("FS image must be 2D with shape (ky,kx).")
    if pts.shape[0] < 5:
        raise ValueError("manual contour needs at least 5 points.")
    fill = float(np.nanmedian(z[np.isfinite(z)])) if np.isfinite(z).any() else 0.0
    work = np.where(np.isfinite(z), z, fill)
    gy, gx = np.gradient(work)
    strength = np.hypot(gx, gy)
    r = max(0, int(radius_px))
    if r == 0:
        return pts.copy()
    out = []
    for px, py in pts:
        ix = int(np.argmin(np.abs(x - float(px))))
        iy = int(np.argmin(np.abs(y - float(py))))
        x0, x1 = max(0, ix - r), min(x.size, ix + r + 1)
        y0, y1 = max(0, iy - r), min(y.size, iy + r + 1)
        window = strength[y0:y1, x0:x1]
        if window.size == 0 or not np.isfinite(window).any():
            out.append((float(px), float(py)))
            continue
        rel = int(np.nanargmax(window))
        wy, wx = np.unravel_index(rel, window.shape)
        out.append((float(x[x0 + wx]), float(y[y0 + wy])))
    return np.asarray(out, dtype=float)


def characterize_manual_contour(
    image,
    kx,
    ky,
    points,
    *,
    bz_polygon,
    hs_points,
    n_bands: int = 1,
    spin: int = 2,
    hs_dir_x_deg: float = 0.0,
    hs_dir_m_deg: float = 45.0,
    hs_dir_tol_deg: float = 10.0,
    contour_window: int = 5,
    simplify_step: float = 0.0,
):
    """Characterize a user-drawn closed contour without fitting hidden kF."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 5:
        raise ValueError("manual contour needs at least 5 points.")
    contour = _close_contour(pts)
    if abs(float(np.nan_to_num(np.linalg.norm(contour[0] - contour[-2])))) < 1e-12:
        raise ValueError("manual contour has duplicate first/last point.")
    contour = smooth_closed_contour(contour, window=max(3, int(contour_window)))
    if float(simplify_step) > 0:
        contour = simplify_closed_contour(contour, min_step=float(simplify_step))
    props = _properties_from_contour(
        image,
        kx,
        ky,
        contour,
        bz_polygon=bz_polygon,
        hs_points=hs_points,
        n_bands=n_bands,
        spin=spin,
        hs_dir_x_deg=hs_dir_x_deg,
        hs_dir_m_deg=hs_dir_m_deg,
        hs_dir_tol_deg=hs_dir_tol_deg,
        analysis_mode="manual_contour",
    )
    return props, contour
