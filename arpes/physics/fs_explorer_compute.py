"""FS Explorer compute: arbitrary BM cuts through an FS volume (headless).

ARPEST-style slicing of `metadata["fs_data"]` (n_ky, n_kx, n_E):
- iso-energy slice for the left map (E−EF slider),
- band map extracted along an arbitrary line (center + angle + length) via
  trilinear interpolation (scipy.ndimage.map_coordinates, float32 in/out),
- snap to the native measured cuts (fs_ky steps) for the "Native BMs" mode.

No PyQt here. Out-of-volume samples come back as NaN, never as silent zeros.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

# Free-angle cuts mix the two k axes: only meaningful when both are in the
# same units. Loaders tag that with fs_kind == "kxky" (CLS, BESSY, Solaris);
# anything else (e.g. ALLS "scan-kx-energy") only allows native cuts.
FREE_CUT_KINDS = {"kxky"}


@dataclass
class CutResult:
    """BM extracted along a line: image (n_pts, n_E) + axis along the line."""
    image: np.ndarray          # float32, NaN outside the volume
    k_along: np.ndarray        # signed distance from the line center
    nan_fraction: float        # 0..1, fraction of out-of-volume samples


def volume_from_meta(meta: dict) -> tuple | None:
    """Return (vol float32, kx, ky, e_ax) from metadata, or None if absent."""
    vol = meta.get("fs_data")
    if vol is None:
        return None
    vol = np.asarray(vol)
    if vol.dtype != np.float32:
        vol = vol.astype(np.float32)
    kx = np.asarray(meta.get("fs_kx"), dtype=float)
    ky = np.asarray(meta.get("fs_ky"), dtype=float)
    e_ax = np.asarray(meta.get("fs_energy"), dtype=float)
    if vol.ndim != 3 or vol.shape != (ky.size, kx.size, e_ax.size):
        raise ValueError(
            f"FS volume axes mismatch: fs_data={vol.shape}, "
            f"len(fs_ky)={ky.size}, len(fs_kx)={kx.size}, len(fs_energy)={e_ax.size}"
        )
    return vol, kx, ky, e_ax


def free_cut_allowed(meta: dict) -> bool:
    return str(meta.get("fs_kind", "") or "") in FREE_CUT_KINDS


def downsample_volume(vol, kx, ky, factor: int):
    """Strided view (no copy) for fast cuts during drag/animation."""
    f = max(1, int(factor))
    if f == 1:
        return vol, kx, ky
    return vol[::f, ::f, :], kx[::f], ky[::f]


def extract_iso_e_slice(vol, e_ax, e_target: float, width: float = 0.0):
    """Map at E−EF = e_target (mean over ±width if width > 0): (n_ky, n_kx)."""
    e_ax = np.asarray(e_ax, dtype=float)
    if width > 0:
        mask = np.abs(e_ax - float(e_target)) <= float(width)
        if not mask.any():
            mask[int(np.argmin(np.abs(e_ax - float(e_target))))] = True
        return np.nanmean(vol[:, :, mask], axis=2)
    idx = int(np.argmin(np.abs(e_ax - float(e_target))))
    return vol[:, :, idx]


def _axis_to_index(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Data coords → fractional indices; out-of-range → NaN (not clamped)."""
    n = axis.size
    idx = np.interp(values, axis, np.arange(n, dtype=float))
    lo, hi = (axis[0], axis[-1]) if axis[0] <= axis[-1] else (axis[-1], axis[0])
    idx[(values < lo) | (values > hi)] = np.nan
    return idx


def extract_bm_cut(
    vol, kx, ky, e_ax, *,
    cx: float, cy: float, angle_deg: float, length: float,
    n_pts: int = 400,
) -> CutResult | None:
    """BM along the line center=(cx,cy), angle (deg, 0 = +kx), given length.

    Returns None when the line has no extent (guard against div-by-zero).
    Samples falling outside the (kx, ky) footprint are NaN in the image.
    """
    if not np.isfinite(length) or float(length) < 1e-6:
        return None
    t = np.linspace(-0.5 * float(length), 0.5 * float(length), int(n_pts))
    ang = np.deg2rad(float(angle_deg))
    xs = float(cx) + t * np.cos(ang)
    ys = float(cy) + t * np.sin(ang)
    ix = _axis_to_index(xs, np.asarray(kx, dtype=float))
    iy = _axis_to_index(ys, np.asarray(ky, dtype=float))
    outside = ~(np.isfinite(ix) & np.isfinite(iy))
    n_e = vol.shape[2]
    # coords (3, n_pts, n_E): full energy column at every point of the line.
    ie = np.arange(n_e, dtype=np.float32)
    coords = np.empty((3, t.size, n_e), dtype=np.float32)
    coords[0] = np.nan_to_num(iy, nan=0.0)[:, None]
    coords[1] = np.nan_to_num(ix, nan=0.0)[:, None]
    coords[2] = ie[None, :]
    # In-volume NaN propagates locally through the trilinear weights: a
    # missing voxel shows as a NaN pixel, which is the honest display.
    image = map_coordinates(vol, coords, order=1, mode="nearest",
                            output=np.float32, prefilter=False)
    image[outside, :] = np.nan
    return CutResult(
        image=image,
        k_along=t,
        nan_fraction=float(np.count_nonzero(outside)) / float(t.size),
    )


def snap_to_native(ky_ax, cy: float) -> int:
    """Index of the measured fs_ky step closest to the line center."""
    return int(np.argmin(np.abs(np.asarray(ky_ax, dtype=float) - float(cy))))


def native_cut(vol, kx, e_ax, idx: int) -> CutResult:
    """The exact measured BM at fs_ky[idx]: no interpolation at all."""
    idx = int(np.clip(idx, 0, vol.shape[0] - 1))
    return CutResult(
        image=np.asarray(vol[idx], dtype=np.float32),
        k_along=np.asarray(kx, dtype=float),
        nan_fraction=0.0,
    )
