"""Compare an experimental pocket contour to a DFT pocket contour.

All inputs are 2D contours ``(N, 2)`` in matched units (caller responsibility).
Returns relative shape metrics, robust to small N-point differences.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PocketCompareResult:
    delta_area_pct: float           # 100 × (A_exp - A_dft) / A_dft
    delta_kF_mean_pct: float        # 100 × (kF_exp - kF_dft) / kF_dft
    hausdorff: float                # max one-sided point-to-point distance (units of contour)
    centroid_shift: float           # ||c_exp - c_dft||
    n_exp: int
    n_dft: int

    def asdict(self) -> dict:
        return asdict(self)


def _close(contour: np.ndarray) -> np.ndarray:
    c = np.asarray(contour, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 3:
        raise ValueError("invalid contour: shape (N≥3, 2) expected.")
    if np.linalg.norm(c[0] - c[-1]) > 1e-10:
        c = np.vstack([c, c[0]])
    return c


def _signed_area(c: np.ndarray) -> float:
    return float(0.5 * np.sum(c[:-1, 0] * c[1:, 1] - c[1:, 0] * c[:-1, 1]))


def _centroid(c: np.ndarray) -> np.ndarray:
    a = _signed_area(c)
    if abs(a) < 1e-14:
        return np.nanmean(c[:-1], axis=0)
    x = c[:, 0]; y = c[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    cx = np.sum((x[:-1] + x[1:]) * cross) / (6.0 * a)
    cy = np.sum((y[:-1] + y[1:]) * cross) / (6.0 * a)
    return np.array([cx, cy], dtype=float)


def _mean_radius(c: np.ndarray, center: np.ndarray) -> float:
    return float(np.nanmean(np.linalg.norm(c[:-1] - center, axis=1)))


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two point sets in 2D."""
    pa = _close(a)[:-1]
    pb = _close(b)[:-1]
    d_ab = np.max(np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2), axis=1))
    d_ba = np.max(np.min(np.linalg.norm(pb[:, None, :] - pa[None, :, :], axis=2), axis=1))
    return float(max(d_ab, d_ba))


def compare_pocket_contours(
    contour_exp: np.ndarray,
    contour_dft: np.ndarray,
) -> PocketCompareResult:
    """Compare two closed contours expressed in the same units (e.g. π/a).

    Caller must align the contours into the same reference frame (same Γ, same
    orientation) before calling. Hausdorff is sensitive to alignment.
    """
    a = _close(contour_exp)
    b = _close(contour_dft)
    area_a = abs(_signed_area(a))
    area_b = abs(_signed_area(b))
    if area_b <= 0.0:
        raise ValueError("DFT contour has zero area.")
    ca = _centroid(a)
    cb = _centroid(b)
    r_a = _mean_radius(a, ca)
    r_b = _mean_radius(b, cb)
    if r_b <= 0.0:
        raise ValueError("DFT contour has zero mean radius.")
    return PocketCompareResult(
        delta_area_pct=float(100.0 * (area_a - area_b) / area_b),
        delta_kF_mean_pct=float(100.0 * (r_a - r_b) / r_b),
        hausdorff=hausdorff_distance(a, b),
        centroid_shift=float(np.linalg.norm(ca - cb)),
        n_exp=int(a.shape[0] - 1),
        n_dft=int(b.shape[0] - 1),
    )
