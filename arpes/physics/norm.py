#!/usr/bin/env python3
"""Shared normalizations for ARPES maps.

Loaders must remain responsible for units/axes. Intensity normalizations live
here so Solaris and CLS use exactly the same display/analysis rules.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - scipy is a normal app dependency
    gaussian_filter1d = None


def _finite_ref_mask(energy: np.ndarray, ref_range: Sequence[float]) -> tuple[np.ndarray, tuple[float, float]]:
    ev = np.asarray(energy, dtype=float)
    ref_lo, ref_hi = sorted((float(ref_range[0]), float(ref_range[1])))
    finite_ev = ev[np.isfinite(ev)]
    if finite_ev.size == 0:
        return np.zeros(ev.shape, dtype=bool), (ref_lo, ref_hi)
    ref_lo = max(ref_lo, float(np.nanmin(finite_ev)))
    ref_hi = min(ref_hi, float(np.nanmax(finite_ev)))
    mask = (ev >= ref_lo) & (ev <= ref_hi)
    return mask, (ref_lo, ref_hi)


_PROFILE_FACTOR_MIN = 0.5
_PROFILE_FACTOR_MAX = 2.0


def _safe_profile(profile: np.ndarray, min_valid: int) -> np.ndarray | None:
    """Flux profile normalized to its median, with anti-artifact clamping.

    Factors are bounded to [_PROFILE_FACTOR_MIN, _PROFILE_FACTOR_MAX] to avoid
    a nearly empty column/slice in the reference window producing an extreme
    correction (for example CLS at angular limits where the detector receives
    almost nothing: without clamping, noise is amplified by 100×+).
    """
    p = np.asarray(profile, dtype=float)
    finite = np.isfinite(p) & (np.abs(p) > 1e-12)
    if finite.sum() < max(2, min_valid):
        return None
    fill = float(np.nanmedian(p[finite]))
    safe = np.where(finite, p, fill)
    med = float(np.nanmedian(safe[np.isfinite(safe)]))
    if not np.isfinite(med) or abs(med) <= 1e-12:
        return None
    factors = safe / med
    return np.clip(factors, _PROFILE_FACTOR_MIN, _PROFILE_FACTOR_MAX)


def normalize_bandmap_flux_profile(
    data: np.ndarray,
    energy: np.ndarray,
    ref_range: tuple[float, float] = (-0.60, -0.20),
) -> tuple[np.ndarray, str]:
    """Normalize a BM `(nk, nE)` by its flux profile along k.

    The reference window must be below EF and wide enough to measure the flux
    background without depending too strongly on details near EF.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[-1] != len(energy) or arr.shape[0] <= 1:
        return arr, "without flux norm"
    mask, (ref_lo, ref_hi) = _finite_ref_mask(np.asarray(energy, dtype=float), ref_range)
    if mask.sum() == 0:
        return arr, "empty flux norm ref"
    profile = np.nanmean(arr[:, mask], axis=1)
    safe = _safe_profile(profile, min_valid=arr.shape[0] // 4)
    if safe is None:
        return arr, "invalid flux norm profile"
    return arr / safe[:, None], f"norm flux k [{ref_lo:.2f},{ref_hi:.2f}] eV"


def normalize_bandmap_above_ef(
    data: np.ndarray,
    energy: np.ndarray,
    ev_above_range: tuple[float, float] = (0.05, 0.20),
    smooth_k_sigma: float = 3.0,
    ef_calibrated: bool = False,
) -> tuple[np.ndarray, str]:
    """Normalize a BM `(nk, nE)` by a profile integrated above EF.

    Assumption: the `energy` axis is calibrated so `E = 0` corresponds to EF.
    `data` is integrated over `[EF + ev_above_min, EF + ev_above_max]` to
    estimate the instrumental background + flux per k column, then divided.

    Preserves `|M|²(k)` (unlike normalization by a below-EF profile).

    If `ef_calibrated=False`, emits a warning: the above-EF window is
    physically meaningful only if EF has already been calibrated.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[-1] != len(energy) or arr.shape[0] <= 1:
        return arr, "without above-EF norm"

    ev = np.asarray(energy, dtype=float)
    finite_ev = ev[np.isfinite(ev)]
    if finite_ev.size == 0:
        return arr, "above-EF norm empty energy axis"

    if not ef_calibrated:
        warnings.warn(
            "normalize_bandmap_above_ef: EF has not been calibrated (ef_calibrated=False). "
            "The above-EF window may cut into occupied states and the correction "
            "will be wrong. Calibrate EF before applying this normalization.",
            RuntimeWarning,
            stacklevel=2,
        )

    if float(np.nanmax(finite_ev)) < float(ev_above_range[0]):
        return arr, (
            f"above-EF norm impossible: energy axis max={np.nanmax(finite_ev):.3f} eV "
            f"< lower bound {ev_above_range[0]:.3f} eV"
        )

    mask, (ref_lo, ref_hi) = _finite_ref_mask(ev, ev_above_range)
    if mask.sum() < 2:
        return arr, "above-EF norm window too narrow"

    profile = np.nanmean(arr[:, mask], axis=1)

    if smooth_k_sigma and smooth_k_sigma > 0:
        profile = _grid_profile_smooth(profile, sigma=float(smooth_k_sigma))

    safe = _safe_profile(profile, min_valid=arr.shape[0] // 4)
    if safe is None:
        return arr, "invalid above-EF norm profile"

    label = f"norm above-EF [{ref_lo:.2f},{ref_hi:.2f}] eV (σk={smooth_k_sigma:.0f}px)"
    if not ef_calibrated:
        label += " [EF not calibrated]"
    return arr / safe[:, None], label


def normalize_fs_flux_profiles(
    fs_data: np.ndarray,
    energy: np.ndarray,
    ref_range: tuple[float, float] = (-0.60, -0.20),
    normalize_y: bool = True,
    normalize_x: bool = True,
) -> tuple[np.ndarray, str]:
    """Normalize an FS volume `(ny, nx, nE)` by y then x flux profiles.

    The same treatment is deliberately applied regardless of loader. `y`
    corrects slice-to-slice variations; `x` corrects a remaining detector
    profile along theta/kx.
    """
    arr = np.asarray(fs_data, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != len(energy):
        return arr, "without flux norm"
    mask, (ref_lo, ref_hi) = _finite_ref_mask(np.asarray(energy, dtype=float), ref_range)
    if mask.sum() == 0:
        return arr, "empty flux norm ref"

    out = arr.copy()
    notes: list[str] = []
    if normalize_y and out.shape[0] > 1:
        profile_y = np.nanmean(out[:, :, mask], axis=(1, 2))
        safe_y = _safe_profile(profile_y, min_valid=out.shape[0] // 4)
        if safe_y is not None:
            out = out / safe_y[:, None, None]
            notes.append("y")
    if normalize_x and out.shape[1] > 1:
        profile_x = np.nanmean(out[:, :, mask], axis=(0, 2))
        safe_x = _safe_profile(profile_x, min_valid=out.shape[1] // 4)
        if safe_x is not None:
            out = out / safe_x[None, :, None]
            notes.append("x")

    if not notes:
        return arr, "invalid flux norm profile"
    axes = "+".join(notes)
    return out, f"norm flux {axes} [{ref_lo:.2f},{ref_hi:.2f}] eV"


def fs_flux_profile_factors(
    fs_data: np.ndarray,
    energy: np.ndarray,
    ref_range: tuple[float, float] = (-0.60, -0.20),
    normalize_y: bool = True,
    normalize_x: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return FS normalization factors without copying the 3D volume."""
    arr = np.asarray(fs_data, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != len(energy):
        return None, None, "without flux norm"
    mask, (ref_lo, ref_hi) = _finite_ref_mask(np.asarray(energy, dtype=float), ref_range)
    if mask.sum() == 0:
        return None, None, "empty flux norm ref"

    safe_y = None
    safe_x = None
    notes: list[str] = []
    if normalize_y and arr.shape[0] > 1:
        profile_y = np.nanmean(arr[:, :, mask], axis=(1, 2))
        safe_y = _safe_profile(profile_y, min_valid=arr.shape[0] // 4)
        if safe_y is not None:
            notes.append("y")
    if normalize_x and arr.shape[1] > 1:
        base = arr
        if safe_y is not None:
            base = arr / safe_y[:, None, None]
        profile_x = np.nanmean(base[:, :, mask], axis=(0, 2))
        safe_x = _safe_profile(profile_x, min_valid=arr.shape[1] // 4)
        if safe_x is not None:
            notes.append("x")

    if not notes:
        return None, None, "invalid flux norm profile"
    axes = "+".join(notes)
    return safe_y, safe_x, f"norm flux {axes} [{ref_lo:.2f},{ref_hi:.2f}] eV"


def apply_fs_flux_factors_to_map(
    fs_map: np.ndarray,
    safe_y: np.ndarray | None,
    safe_x: np.ndarray | None,
) -> np.ndarray:
    """Apply `(ny,)` and `(nx,)` factors to a 2D FS map."""
    out = np.asarray(fs_map, dtype=float)
    if safe_y is not None:
        out = out / safe_y[:, None]
    if safe_x is not None:
        out = out / safe_x[None, :]
    return out


def _grid_profile_smooth(profile: np.ndarray, sigma: float) -> np.ndarray:
    p = np.asarray(profile, dtype=float)
    finite = np.isfinite(p)
    fill = float(np.nanmedian(p[finite])) if finite.any() else 1.0
    p = np.where(finite, p, fill)
    if gaussian_filter1d is not None:
        return gaussian_filter1d(p, sigma=max(float(sigma), 0.0))
    width = max(3, int(round(float(sigma) * 4.0)) | 1)
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(p, kernel, mode="same")


from arpes.physics.norm_grid_fft import (  # noqa: E402
    _dilate_mask,
    remove_grid_artifact_fft2_mask as _remove_grid_artifact_fft2_mask,
)


def _remove_grid_artifact_2d(
    data_2d: np.ndarray,
    *,
    method: str = "profile",
    grid_freq: float | None = None,
    grid_period_px: float | None = None,
    notch_width: int = 2,
    notch_sigma: float = 0.8,
    strength: float = 0.85,
    fft2_center_radius: float = 8.0,
    fft2_peak_sensitivity: float = 8.0,
    fft2_plane: str = "detector",
) -> tuple[np.ndarray, dict]:
    """Correct a periodic artifact along axis 0 of a 2D matrix."""
    arr = np.asarray(data_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Grid correction: expected 2D data, shape={arr.shape}")
    n_axis = arr.shape[0]
    if n_axis < 4 or arr.shape[1] == 0:
        return arr.copy(), {"method": "none", "grid_freq": None, "grid_period_px": None}

    method = (method or "profile").lower()
    if method not in {"profile", "fft", "fft2mask"}:
        raise ValueError("Grid correction: expected method 'profile', 'fft', or 'fft2mask'.")
    if grid_freq is not None and float(grid_freq) <= 0:
        grid_freq = None
    if grid_period_px is not None and float(grid_period_px) <= 0:
        grid_period_px = None
    if grid_freq is None and grid_period_px is not None:
        grid_freq = 1.0 / float(grid_period_px)
    strength = float(np.clip(strength, 0.0, 1.0))

    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    arr_filled = np.where(finite, arr, fill)

    if method == "fft2mask":
        return _remove_grid_artifact_fft2_mask(
            arr_filled,
            center_radius=fft2_center_radius,
            peak_sensitivity=fft2_peak_sensitivity,
            mask_radius=notch_width,
            strength=strength,
        )

    if method == "profile":
        ref_profile = np.nanmean(arr_filled, axis=1).astype(float)
        smooth_px = max(n_axis // 20, 3)
        ref_smooth = _grid_profile_smooth(ref_profile, sigma=smooth_px)
        smooth_med = float(np.nanmedian(np.abs(ref_smooth[np.isfinite(ref_smooth)])))
        if not np.isfinite(smooth_med) or smooth_med <= 1e-12:
            return arr.copy(), {"method": "profile", "grid_freq": None, "grid_period_px": None}

        safe_smooth = np.where(
            np.isfinite(ref_smooth) & (np.abs(ref_smooth) > 1e-12),
            ref_smooth,
            smooth_med,
        )
        grid_gain = ref_profile / safe_smooth
        valid_gain = grid_gain[np.isfinite(grid_gain) & (grid_gain > 0)]
        if valid_gain.size < 4:
            return arr.copy(), {"method": "profile", "grid_freq": None, "grid_period_px": None}

        gain_center = float(np.nanmedian(valid_gain))
        if not np.isfinite(gain_center) or gain_center <= 1e-12:
            gain_center = 1.0
        grid_gain = grid_gain / gain_center
        valid_gain = grid_gain[np.isfinite(grid_gain) & (grid_gain > 0)]
        lo_g, hi_g = np.nanpercentile(valid_gain, [1, 99])
        lo_g = max(float(lo_g), 0.2)
        hi_g = min(float(hi_g), 5.0)
        if hi_g <= lo_g:
            lo_g, hi_g = 0.2, 5.0
        grid_gain = np.clip(grid_gain, lo_g, hi_g)

        correction = 1.0 + strength * (grid_gain - 1.0)
        correction = np.where(
            np.isfinite(correction) & (correction > 1e-6),
            correction,
            1.0,
        )
        out = arr_filled / correction[:, None]
        med_in = float(np.nanmedian(arr_filled[finite])) if finite.any() else np.nan
        med_out = float(np.nanmedian(out[finite])) if finite.any() else np.nan
        if np.isfinite(med_in) and np.isfinite(med_out) and abs(med_out) > 1e-12:
            out *= med_in / med_out
        out[~finite] = np.nan
        p5, p95 = np.nanpercentile(valid_gain, [5, 95])
        ripple = float(max(0.0, p95 - p5) * 100.0)
        return out, {
            "method": "profile",
            "grid_freq": None,
            "grid_period_px": None,
            "profile_smooth_px": float(smooth_px),
            "strength": strength,
            "grid_ripple_percent": ripple,
        }

    ref_profile = np.nanmean(arr_filled, axis=1).astype(float)
    hp_sigma = max(n_axis // 10, 3)
    ref_highpass = ref_profile - _grid_profile_smooth(ref_profile, sigma=hp_sigma)
    fft_ref = np.fft.rfft(ref_highpass)
    freqs = np.fft.rfftfreq(n_axis)
    power = np.abs(fft_ref) ** 2
    if power.size:
        power[0] = 0.0
    if grid_freq is None:
        min_period_px = 3.0
        max_period_px = max(4.0, min(80.0, n_axis / 2.0))
        valid = (freqs >= 1.0 / max_period_px) & (freqs <= 1.0 / min_period_px)
        if not valid.any():
            valid = freqs > 0
        masked_power = np.where(valid, power, 0.0)
        if masked_power.size <= 1 or not np.isfinite(masked_power[1:]).any() or np.nanmax(masked_power) <= 0:
            return arr.copy(), {"method": "fft", "grid_freq": None, "grid_period_px": None}
        idx_peak = int(np.nanargmax(masked_power))
        grid_freq = float(freqs[idx_peak])
    else:
        idx_peak = int(np.argmin(np.abs(freqs - float(grid_freq))))
        grid_freq = float(freqs[idx_peak])

    filt = np.ones(len(freqs), dtype=float)
    half_width = max(0, int(notch_width))
    lo = max(1, idx_peak - half_width)
    hi = min(len(freqs) - 1, idx_peak + half_width)
    if lo <= hi:
        if notch_sigma > 0:
            sigma = max(float(notch_sigma), 1e-9)
            for i in range(lo, hi + 1):
                filt[i] = 1.0 - np.exp(-0.5 * ((i - idx_peak) / sigma) ** 2)
            filt[idx_peak] = 0.0
        else:
            filt[lo:hi + 1] = 0.0

    out = np.empty_like(arr_filled)
    # Block processing to avoid a large complex array on fast maps.
    block_cols = max(256, min(4096, arr_filled.shape[1]))
    for start in range(0, arr_filled.shape[1], block_cols):
        stop = min(start + block_cols, arr_filled.shape[1])
        fft_block = np.fft.rfft(arr_filled[:, start:stop], axis=0)
        filtered = np.fft.irfft(fft_block * filt[:, None], n=n_axis, axis=0)
        out[:, start:stop] = arr_filled[:, start:stop] + strength * (filtered - arr_filled[:, start:stop])
    med_in = float(np.nanmedian(arr_filled[finite])) if finite.any() else np.nan
    med_out = float(np.nanmedian(out[finite])) if finite.any() else np.nan
    if np.isfinite(med_in) and np.isfinite(med_out) and abs(med_out) > 1e-12:
        out *= med_in / med_out
    out[~finite] = np.nan
    period = 1.0 / grid_freq if grid_freq and grid_freq > 0 else None
    return out, {
        "method": "fft",
        "grid_freq": grid_freq,
        "grid_period_px": period,
        "grid_period_px_input": grid_period_px,
        "grid_period_px_detected": period,
        "strength": strength,
    }


def remove_grid_artifact(
    data: np.ndarray,
    *,
    axis: int = 0,
    method: str = "profile",
    grid_freq: float | None = None,
    grid_period_px: float | None = None,
    notch_width: int = 2,
    notch_sigma: float = 0.8,
    strength: float = 0.85,
    fft2_center_radius: float = 8.0,
    fft2_peak_sensitivity: float = 8.0,
    fft2_plane: str = "detector",
) -> tuple[np.ndarray, dict]:
    """Remove the Solaris/DA30 grid effect along a detector axis.

    For a Solaris FS, historical V5/V6 usage corrected the `beta` axis before
    k-space conversion. In the interface, the same principle is applied to the
    already standardized volume by correcting the slow FS axis `(ny, nx, nE)`.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 2:
        return arr.copy(), {"method": "none", "grid_freq": None, "grid_period_px": None}
    method = (method or "profile").lower()
    if method == "fft2mask":
        if arr.ndim == 2:
            clean, info = _remove_grid_artifact_fft2_mask(
                arr,
                center_radius=fft2_center_radius,
                peak_sensitivity=fft2_peak_sensitivity,
                mask_radius=notch_width,
                strength=strength,
            )
            info.update({"axis": None, "shape": tuple(arr.shape), "fft2_plane": "image_2d"})
            return clean, info
        if arr.ndim == 3:
            clean = np.empty_like(arr, dtype=float)
            infos = []
            plane = (fft2_plane or "detector").lower()
            if plane in {"detector", "kx_energy", "theta_energy"}:
                # Standardized FS: (beta/ky, theta/kx, E). The detector grid
                # lives on theta-E images, so correct each beta slice
                # separately, like FFT_gridrem3D in Igor.
                for i in range(arr.shape[0]):
                    clean[i, :, :], info_i = _remove_grid_artifact_fft2_mask(
                        arr[i, :, :],
                        center_radius=fft2_center_radius,
                        peak_sensitivity=fft2_peak_sensitivity,
                        mask_radius=notch_width,
                        strength=strength,
                    )
                    infos.append(info_i)
                slice_axis = 0
                corrected_plane = "detector_kx_energy"
            elif plane in {"map", "kxky", "energy"}:
                for i in range(arr.shape[2]):
                    clean[:, :, i], info_i = _remove_grid_artifact_fft2_mask(
                        arr[:, :, i],
                        center_radius=fft2_center_radius,
                        peak_sensitivity=fft2_peak_sensitivity,
                        mask_radius=notch_width,
                        strength=strength,
                    )
                    infos.append(info_i)
                slice_axis = 2
                corrected_plane = "map_kx_ky"
            else:
                raise ValueError("expected fft2_plane: 'detector' or 'map'.")
            removed = int(sum(int(info.get("removed_peak_count", 0)) for info in infos))
            active = [info for info in infos if int(info.get("removed_peak_count", 0)) > 0]
            deltas = [float(info.get("rms_delta_percent", 0.0) or 0.0) for info in infos]
            info = {
                "method": "fft2mask",
                "removed_peak_count": removed,
                "corrected_slices": len(active),
                "slice_count": len(infos),
                "slice_axis": slice_axis,
                "fft2_plane": corrected_plane,
                "fft2_center_radius": float(fft2_center_radius),
                "fft2_peak_sensitivity": float(fft2_peak_sensitivity),
                "fft2_mask_radius": int(notch_width),
                "rms_delta_percent": float(np.nanmean(deltas)) if deltas else 0.0,
                "strength": float(np.clip(strength, 0.0, 1.0)),
                "axis": None,
                "shape": tuple(arr.shape),
            }
            return clean, info
        raise ValueError("2D FFT grid correction is available only for 2D data or FS volumes (ny, nx, E).")
    axis = int(axis) % arr.ndim
    moved = np.moveaxis(arr, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    flat_clean, info = _remove_grid_artifact_2d(
        flat,
        method=method,
        grid_freq=grid_freq,
        grid_period_px=grid_period_px,
        notch_width=notch_width,
        notch_sigma=notch_sigma,
        strength=strength,
        fft2_center_radius=fft2_center_radius,
        fft2_peak_sensitivity=fft2_peak_sensitivity,
        fft2_plane=fft2_plane,
    )
    clean = np.moveaxis(flat_clean.reshape(moved.shape), 0, axis)
    info.update({"axis": axis, "shape": tuple(arr.shape)})
    return clean, info
