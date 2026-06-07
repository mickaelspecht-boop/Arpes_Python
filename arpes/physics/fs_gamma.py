"""Pure FS Γ detection helpers.

This module deliberately has no Qt/matplotlib dependency. UI code builds the
FS map elsewhere, then calls `detect_gamma_from_fs_map`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FSGammaDetection:
    kx: float
    ky: float
    gamma_kx_list: list[float]
    gamma_ky_list: list[float]
    symmetry_score: float
    symmetry_fraction: float
    mad_kx: float
    mad_ky: float
    quality: str
    kx_axis_center: float
    ky_axis_center: float
    gamma_delta_kx: float
    gamma_delta_ky: float
    method: str = "profiles+symmetry2d"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kx": self.kx,
            "ky": self.ky,
            "gamma_kx_list": self.gamma_kx_list,
            "gamma_ky_list": self.gamma_ky_list,
            "symmetry_score": self.symmetry_score,
            "symmetry_fraction": self.symmetry_fraction,
            "mad_kx": self.mad_kx,
            "mad_ky": self.mad_ky,
            "quality": self.quality,
            "kx_axis_center": self.kx_axis_center,
            "ky_axis_center": self.ky_axis_center,
            "gamma_delta_kx": self.gamma_delta_kx,
            "gamma_delta_ky": self.gamma_delta_ky,
            "method": self.method,
        }


def _profile_gamma_center(axis, prof, center_guess, *, klim: float) -> float:
    y = np.asarray(prof, dtype=float)
    if not np.isfinite(y).any():
        return np.nan
    lo, hi = np.nanpercentile(y, [5, 99])
    if hi - lo <= 1e-12:
        return np.nan
    y = np.clip((y - lo) / (hi - lo), 0, None)

    axis = np.asarray(axis, dtype=float)
    finite = np.isfinite(axis) & np.isfinite(y)
    if finite.sum() < 5:
        return np.nan
    a = axis[finite]
    yy = y[finite]
    if a[0] > a[-1]:
        a = a[::-1]
        yy = yy[::-1]

    if yy.size >= 3:
        peak_mask = np.r_[False, (yy[1:-1] >= yy[:-2]) & (yy[1:-1] >= yy[2:]), False]
    else:
        peak_mask = np.ones_like(yy, dtype=bool)
    if peak_mask.sum() < 2:
        strongest = np.argsort(yy)[-min(6, yy.size):]
    else:
        strongest = np.where(peak_mask)[0]
        strongest = strongest[np.argsort(yy[strongest])[-min(12, strongest.size):]]

    span = max(float(np.nanmax(a) - np.nanmin(a)), 1e-12)
    min_sep = max(0.08, 0.08 * span)
    max_mid_gap = max(float(klim), 0.35)
    candidates = []
    for ii, il in enumerate(strongest):
        for ir in strongest[ii + 1:]:
            kl, kr = float(a[il]), float(a[ir])
            sep = abs(kr - kl)
            if sep < min_sep:
                continue
            center = 0.5 * (kl + kr)
            if abs(center - center_guess) > max_mid_gap:
                continue
            balance = abs(float(yy[il]) - float(yy[ir]))
            amp = float(yy[il] + yy[ir])
            candidates.append((amp - 0.4 * balance - 0.05 * abs(center - center_guess), center))
    if candidates:
        return float(max(candidates, key=lambda item: item[0])[1])

    left = a < center_guess
    right = a > center_guess
    if not left.any() or not right.any():
        return np.nan
    kl = a[left][int(np.nanargmax(yy[left]))]
    kr = a[right][int(np.nanargmax(yy[right]))]
    return float((kl + kr) / 2)


def _bilinear_sample_on_grid(x_axis, y_axis, img, xq, yq):
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_axis, dtype=float)
    z = np.asarray(img, dtype=float)
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)
    out = np.full(xq.shape, np.nan, dtype=float)
    if x.size < 2 or y.size < 2 or z.shape != (y.size, x.size):
        return out
    ix = np.searchsorted(x, xq, side="right") - 1
    iy = np.searchsorted(y, yq, side="right") - 1
    valid = (ix >= 0) & (iy >= 0) & (ix < x.size - 1) & (iy < y.size - 1)
    if not valid.any():
        return out
    ixv = ix[valid]
    iyv = iy[valid]
    x0 = x[ixv]
    x1 = x[ixv + 1]
    y0 = y[iyv]
    y1 = y[iyv + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = (xq[valid] - x0) / (x1 - x0)
        ty = (yq[valid] - y0) / (y1 - y0)
    z00 = z[iyv, ixv]
    z10 = z[iyv, ixv + 1]
    z01 = z[iyv + 1, ixv]
    z11 = z[iyv + 1, ixv + 1]
    vals = (
        (1.0 - tx) * (1.0 - ty) * z00
        + tx * (1.0 - ty) * z10
        + (1.0 - tx) * ty * z01
        + tx * ty * z11
    )
    vals[~np.isfinite(vals)] = np.nan
    out[valid] = vals
    return out


def _symmetry_score_2d(kx, ky, img, cx: float, cy: float, *, klim: float) -> tuple[float, float]:
    kx_arr = np.asarray(kx, dtype=float)
    ky_arr = np.asarray(ky, dtype=float)
    vals = np.asarray(img, dtype=float)
    KX, KY = np.meshgrid(kx_arr, ky_arr)
    window = (
        (np.abs(KX - cx) <= float(klim))
        & (np.abs(KY - cy) <= float(klim))
        & np.isfinite(vals)
    )
    if int(window.sum()) < 16:
        return float("nan"), 0.0
    mirrored = _bilinear_sample_on_grid(kx_arr, ky_arr, vals, 2.0 * cx - KX, 2.0 * cy - KY)
    valid = window & np.isfinite(mirrored)
    if int(valid.sum()) < 16:
        return float("nan"), 0.0
    a = vals[valid]
    b = mirrored[valid]
    fraction = float(valid.sum() / max(window.sum(), 1))
    a = a - float(np.nanmean(a))
    b = b - float(np.nanmean(b))
    denom = float(np.sqrt(np.nansum(a * a) * np.nansum(b * b)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return float("nan"), fraction
    corr = float(np.nansum(a * b) / denom)
    return corr, fraction


def detect_gamma_from_fs_map(kx, ky, fs, params) -> FSGammaDetection:
    """Detect Γ on a kx/ky FS map using 1D peak pairs plus 2D mirror symmetry."""
    img = np.asarray(fs, dtype=float)
    kx_arr = np.asarray(kx, dtype=float)
    ky_arr = np.asarray(ky, dtype=float)
    if img.shape != (ky_arr.size, kx_arr.size):
        raise ValueError("FS map is incompatible with kx/ky axes.")
    if kx_arr.size < 5 or ky_arr.size < 5:
        raise ValueError("Cannot detect FS Γ without a sufficient kx/ky grid.")
    if kx_arr[0] > kx_arr[-1]:
        kx_arr = kx_arr[::-1]
        img = img[:, ::-1]
    if ky_arr[0] > ky_arr[-1]:
        ky_arr = ky_arr[::-1]
        img = img[::-1, :]

    kx_centers: list[float] = []
    ky_centers: list[float] = []
    y_samples = np.linspace(max(float(ky_arr.min()), -params.klim), min(float(ky_arr.max()), params.klim), 15)
    for y0 in y_samples:
        iy = int(np.argmin(np.abs(ky_arr - y0)))
        c = _profile_gamma_center(kx_arr, img[iy, :], params.kx_center, klim=params.klim)
        if np.isfinite(c):
            kx_centers.append(float(c))

    x_samples = np.linspace(max(float(kx_arr.min()), -params.klim), min(float(kx_arr.max()), params.klim), 15)
    for x0 in x_samples:
        ix = int(np.argmin(np.abs(kx_arr - x0)))
        c = _profile_gamma_center(ky_arr, img[:, ix], params.ky_center, klim=params.klim)
        if np.isfinite(c):
            ky_centers.append(float(c))

    gx0 = float(np.nanmedian(kx_centers)) if kx_centers else float(params.kx_center)
    gy0 = float(np.nanmedian(ky_centers)) if ky_centers else float(params.ky_center)
    if not (np.isfinite(gx0) and np.isfinite(gy0)):
        raise ValueError("Not enough finite signal to detect FS Γ.")

    dx = float(np.nanmedian(np.abs(np.diff(kx_arr)))) if kx_arr.size > 1 else 0.02
    dy = float(np.nanmedian(np.abs(np.diff(ky_arr)))) if ky_arr.size > 1 else 0.02
    radius_x = max(3.0 * dx, min(0.25, 0.20 * float(params.klim)))
    radius_y = max(3.0 * dy, min(0.25, 0.20 * float(params.klim)))
    xs = np.linspace(gx0 - radius_x, gx0 + radius_x, 17)
    ys = np.linspace(gy0 - radius_y, gy0 + radius_y, 17)
    xs = xs[(xs >= float(kx_arr.min())) & (xs <= float(kx_arr.max()))]
    ys = ys[(ys >= float(ky_arr.min())) & (ys <= float(ky_arr.max()))]

    best = (float("-inf"), gx0, gy0, 0.0)
    for cx in xs:
        for cy in ys:
            corr, fraction = _symmetry_score_2d(kx_arr, ky_arr, img, float(cx), float(cy), klim=params.klim)
            if not np.isfinite(corr):
                continue
            score = corr - 0.15 * max(0.0, 0.75 - fraction)
            if score > best[0]:
                best = (score, float(cx), float(cy), float(fraction))

    score, gx, gy, fraction = best
    if not np.isfinite(score):
        if len(kx_centers) < 3 or len(ky_centers) < 3:
            raise ValueError("Not enough symmetric pairs detected on the FS.")
        gx, gy, score, fraction = gx0, gy0, float("nan"), 0.0

    mad_kx = float(1.4826 * np.nanmedian(np.abs(np.asarray(kx_centers) - gx))) if kx_centers else float("nan")
    mad_ky = float(1.4826 * np.nanmedian(np.abs(np.asarray(ky_centers) - gy))) if ky_centers else float("nan")
    enough_profiles = len(kx_centers) >= 3 and len(ky_centers) >= 3
    if np.isfinite(score) and score >= 0.85 and fraction >= 0.55 and enough_profiles:
        quality = "high"
    elif np.isfinite(score) and score >= 0.45 and fraction >= 0.35:
        quality = "medium"
    else:
        quality = "low"
    kx_axis_center = float(0.5 * (float(np.nanmin(kx_arr)) + float(np.nanmax(kx_arr))))
    ky_axis_center = float(0.5 * (float(np.nanmin(ky_arr)) + float(np.nanmax(ky_arr))))
    return FSGammaDetection(
        kx=float(gx),
        ky=float(gy),
        gamma_kx_list=kx_centers,
        gamma_ky_list=ky_centers,
        symmetry_score=float(score),
        symmetry_fraction=float(fraction),
        mad_kx=mad_kx,
        mad_ky=mad_ky,
        quality=quality,
        kx_axis_center=kx_axis_center,
        ky_axis_center=ky_axis_center,
        gamma_delta_kx=float(gx - kx_axis_center),
        gamma_delta_ky=float(gy - ky_axis_center),
    )
