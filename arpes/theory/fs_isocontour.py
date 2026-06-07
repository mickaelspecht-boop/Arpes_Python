"""Fermi-surface iso-contours at fixed kz from a 3D DFT grid.

Input: uniform grid ``E_n(kx, ky, kz)`` (n_bands, n_kx, n_ky, n_kz).
Output: for each band, list of closed (or open) contours in the detector
plane (kx, ky), selected at the ``E = EF`` iso-level.

Backend: prefer ``contourpy`` (matplotlib dependency); fallback to
``matplotlib.contour`` through an Agg figure closed immediately (no display).

Pure module: no PyQt. pymatgen is not required here (input is raw numeric data;
the caller performs the DFT BS -> 3D grid conversion).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FsContour:
    band_index: int
    points: np.ndarray  # shape (M, 2), columns (kx, ky)
    closed: bool


def _interp_band_to_kz(
    band_3d: np.ndarray,
    kz_axis: np.ndarray,
    kz_value: float,
) -> np.ndarray:
    """Linearly interpolate E(kx, ky, kz) at kz=kz_value -> 2D array (n_kx, n_ky).

    Outside the kz range: clamp to bounds (flat extrapolation).
    """
    kz_axis = np.asarray(kz_axis, dtype=float)
    if kz_axis.ndim != 1 or kz_axis.size < 2:
        raise ValueError("kz_axis must be 1D with at least 2 points")
    if band_3d.ndim != 3 or band_3d.shape[2] != kz_axis.size:
        raise ValueError(
            f"band_3d shape {band_3d.shape} incompatible kz_axis ({kz_axis.size})"
        )
    kz = float(kz_value)
    if kz <= kz_axis[0]:
        return np.asarray(band_3d[:, :, 0], dtype=float)
    if kz >= kz_axis[-1]:
        return np.asarray(band_3d[:, :, -1], dtype=float)
    idx_hi = int(np.searchsorted(kz_axis, kz))
    idx_lo = idx_hi - 1
    kz_lo = float(kz_axis[idx_lo])
    kz_hi = float(kz_axis[idx_hi])
    w = (kz - kz_lo) / max(kz_hi - kz_lo, 1e-12)
    return (1.0 - w) * band_3d[:, :, idx_lo] + w * band_3d[:, :, idx_hi]


def _contour_lines(
    kx_axis: np.ndarray,
    ky_axis: np.ndarray,
    energy_2d: np.ndarray,
    level: float,
) -> list[np.ndarray]:
    """Extract iso-lines at `level` from a 2D field. Use contourpy if available."""
    try:
        import contourpy
        gen = contourpy.contour_generator(
            x=kx_axis, y=ky_axis, z=energy_2d.T,  # contourpy expects (ny, nx)
        )
        return [np.asarray(line, dtype=float) for line in gen.lines(float(level))]
    except Exception:
        pass

    # matplotlib fallback (force Agg, close the figure immediately).
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cs = ax.contour(kx_axis, ky_axis, energy_2d.T, levels=[float(level)])
        segs = []
        if len(cs.allsegs) > 0:
            for seg in cs.allsegs[0]:
                segs.append(np.asarray(seg, dtype=float))
        plt.close(fig)
        return segs
    except Exception as exc:
        raise RuntimeError(
            "fs_isocontour: neither contourpy nor matplotlib is available."
        ) from exc


def _is_closed(points: np.ndarray, tol: float = 1e-9) -> bool:
    if points.shape[0] < 3:
        return False
    return bool(np.allclose(points[0], points[-1], atol=tol))


def extract_fs_isocontour(
    bands_3d: np.ndarray,
    kx_axis: np.ndarray,
    ky_axis: np.ndarray,
    kz_axis: np.ndarray,
    *,
    kz_value: float,
    ef: float = 0.0,
    band_indices: list[int] | None = None,
    min_points: int = 6,
) -> list[FsContour]:
    """E=ef iso-contours in the kz=kz_value plane for a 3D DFT grid.

    - ``bands_3d``: shape (n_bands, n_kx, n_ky, n_kz). Energies are E - EF if
      already referenced (then ``ef=0``).
    - ``band_indices``: subset to process; None = all.
    - ``min_points``: ignore contours that are too short (< min_points).

    Return a list of ``FsContour(band_index, points, closed)``.
    """
    bands_3d = np.asarray(bands_3d, dtype=float)
    if bands_3d.ndim != 4:
        raise ValueError(f"bands_3d must be 4D, got ndim={bands_3d.ndim}")
    n_b, n_kx, n_ky, n_kz = bands_3d.shape
    kx = np.asarray(kx_axis, dtype=float)
    ky = np.asarray(ky_axis, dtype=float)
    kz = np.asarray(kz_axis, dtype=float)
    if kx.size != n_kx or ky.size != n_ky or kz.size != n_kz:
        raise ValueError(
            f"incompatible axes: ({kx.size},{ky.size},{kz.size}) "
            f"vs bands {(n_kx, n_ky, n_kz)}"
        )
    indices = list(range(n_b)) if band_indices is None else [int(i) for i in band_indices]

    out: list[FsContour] = []
    for b in indices:
        if b < 0 or b >= n_b:
            continue
        e2d = _interp_band_to_kz(bands_3d[b], kz, float(kz_value))
        if not np.isfinite(e2d).all():
            e2d = np.where(np.isfinite(e2d), e2d, float(ef) + 1e9)
        lines = _contour_lines(kx, ky, e2d, float(ef))
        for line in lines:
            if line.shape[0] < int(min_points):
                continue
            out.append(FsContour(
                band_index=int(b),
                points=line,
                closed=_is_closed(line),
            ))
    return out


def isocontour_at_planes(
    bands_3d: np.ndarray,
    kx_axis: np.ndarray,
    ky_axis: np.ndarray,
    kz_axis: np.ndarray,
    *,
    kz_values: list[float],
    ef: float = 0.0,
) -> dict[float, list[FsContour]]:
    """Helper: iso-contours at multiple kz values (for example Γ-plane + Z-plane)."""
    return {
        float(kz): extract_fs_isocontour(
            bands_3d, kx_axis, ky_axis, kz_axis, kz_value=float(kz), ef=ef
        )
        for kz in kz_values
    }
