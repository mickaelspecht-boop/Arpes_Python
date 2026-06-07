"""FFT2 grid-artifact removal helpers."""
from __future__ import annotations

import numpy as np


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    src = np.asarray(mask, dtype=bool)
    out = src.copy()
    ny, nx = src.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            sy0 = max(0, -dy)
            sy1 = min(ny, ny - dy)
            sx0 = max(0, -dx)
            sx1 = min(nx, nx - dx)
            dy0 = max(0, dy)
            dy1 = min(ny, ny + dy)
            dx0 = max(0, dx)
            dx1 = min(nx, nx + dx)
            out[dy0:dy1, dx0:dx1] |= src[sy0:sy1, sx0:sx1]
    return out


def remove_grid_artifact_fft2_mask(
    data_2d: np.ndarray,
    *,
    center_radius: float = 8.0,
    peak_sensitivity: float = 8.0,
    mask_radius: int = 2,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Correct a grid artifact via automatic 2D FFT magnitude masking."""
    arr = np.asarray(data_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"2D FFT grid correction: expected 2D data, shape={arr.shape}")
    if min(arr.shape) < 4:
        return arr.copy(), {"method": "fft2mask", "removed_peak_count": 0}

    strength = float(np.clip(strength, 0.0, 1.0))
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    arr_filled = np.where(finite, arr, fill)

    fft_shift = np.fft.fftshift(np.fft.fft2(arr_filled))
    mag = np.abs(fft_shift)
    phase = np.angle(fft_shift)

    ny, nx = arr.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    protected_center = rr <= max(float(center_radius), 0.0)
    candidates = ~protected_center & np.isfinite(mag)
    if candidates.sum() < 16:
        return arr.copy(), {"method": "fft2mask", "removed_peak_count": 0}

    log_mag = np.log1p(mag)
    vals = log_mag[candidates]
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = float(np.nanstd(vals))
    if not np.isfinite(mad) or mad <= 1e-12:
        return arr.copy(), {"method": "fft2mask", "removed_peak_count": 0}

    threshold = med + max(float(peak_sensitivity), 0.1) * 1.4826 * mad
    mask = candidates & (log_mag > threshold)
    mask = _dilate_mask(mask, int(mask_radius))
    mask &= ~protected_center
    removed = int(mask.sum())
    if removed == 0:
        return arr.copy(), {
            "method": "fft2mask",
            "removed_peak_count": 0,
            "fft2_center_radius": float(center_radius),
            "fft2_peak_sensitivity": float(peak_sensitivity),
            "fft2_mask_radius": int(mask_radius),
            "strength": strength,
        }

    replacement = float(np.nanmean(mag[candidates & ~mask]))
    if not np.isfinite(replacement):
        replacement = float(np.nanmean(mag[candidates]))
    mag_clean = mag.copy()
    mag_clean[mask] = replacement

    fft_clean = mag_clean * np.exp(1j * phase)
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fft_clean)))
    out = arr_filled + strength * (filtered - arr_filled)

    med_in = float(np.nanmedian(arr_filled[finite])) if finite.any() else np.nan
    med_out = float(np.nanmedian(out[finite])) if finite.any() else np.nan
    if np.isfinite(med_in) and np.isfinite(med_out) and abs(med_out) > 1e-12:
        out *= med_in / med_out
    out[~finite] = np.nan
    scale = float(np.nanpercentile(arr_filled[finite], 95) - np.nanpercentile(arr_filled[finite], 5)) if finite.any() else np.nan
    rms_delta = float(np.sqrt(np.nanmean((out[finite] - arr_filled[finite]) ** 2))) if finite.any() else np.nan
    delta_percent = 0.0
    if np.isfinite(scale) and scale > 1e-12 and np.isfinite(rms_delta):
        delta_percent = float(100.0 * rms_delta / scale)
    return out, {
        "method": "fft2mask",
        "removed_peak_count": removed,
        "fft2_center_radius": float(center_radius),
        "fft2_peak_sensitivity": float(peak_sensitivity),
        "fft2_mask_radius": int(mask_radius),
        "fft2_threshold_log": float(threshold),
        "rms_delta_percent": delta_percent,
        "strength": strength,
    }
