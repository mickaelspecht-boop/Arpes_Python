"""Geometric BM distortion corrections (θ trapezoid + E parabola).

Pure module (numpy + scipy only, no PyQt). Conventions:
- ``data`` shape ``(n_kpar, n_e)`` (consistent with ``arpes.physics.norm``).
- ``kpar`` shape ``(n_kpar,)`` strictly increasing.
- ``ev``   shape ``(n_e,)``    strictly increasing.

Trapezoid model: for each energy row ``E``, the effective kpar is stretched
(``slope_left``, ``slope_right`` in ``Δkpar/ΔE``) around ``pivot_ev``.
Parabola model: warp the energy axis by ``E_corr = E + a·(kpar−k0)²``
(Scienta non-isochromaticity, *NOT* an intensity subtraction, which would crush
the physical dispersion).

All functions return a *new* image (axes remain identical); no mutation of the
input. NaNs at edges after warp.

Guards:
- ``clamp_params`` bounds parameters to avoid warps extending >50% outside the
  domain.
- ``auto_detect_*`` rejects BMs that are too narrow or flat (returns ``None``).
- ``apply_distortion`` guarantees bit-exact identity if the dict is empty or
  disabled (toggle-off reversibility test).
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.ndimage import map_coordinates
except ImportError:  # pragma: no cover
    map_coordinates = None  # type: ignore


_MIN_KPAR_FOR_AUTO = 16
_MIN_DISPERSION_EV = 0.010  # max 10 meV variation → flat BM, auto refusal
_PARAM_CACHE_PRECISION = 6


def is_distortion_active(cfg: dict | None) -> bool:
    """True if the config requests at least one correction."""
    if not cfg or not cfg.get("enabled", False):
        return False
    trap = (cfg.get("trapezoid") or {})
    para = (cfg.get("parabola") or {})
    if trap.get("enabled") and (
        abs(float(trap.get("slope_left", 0.0) or 0.0)) > 0.0
        or abs(float(trap.get("slope_right", 0.0) or 0.0)) > 0.0
    ):
        return True
    if para.get("enabled") and abs(float(para.get("a", 0.0) or 0.0)) > 0.0:
        return True
    return False


def _safe_axis(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1 or out.size < 2:
        raise ValueError("distortion : axis must be 1D with at least 2 points")
    return out


def _grid_step(axis: np.ndarray) -> float:
    """Mean grid step (assumes a regular increasing grid)."""
    return float(axis[-1] - axis[0]) / float(axis.size - 1)


def clamp_params(cfg: dict | None, kpar, ev) -> dict:
    """Bound parameters to avoid degenerate warps.

    Returns a clamped *copy* of ``cfg`` (without mutating the original).
    Limits:
    - ``|slope|·Δev_total ≤ 0.5·Δkpar_total`` (the trapezoidal shift at the
      extremes of the energy window stays below half the kpar domain).
    - ``|a|·max(|kpar−k0|)² ≤ 0.5·Δev_total``.
    """
    if not cfg:
        return {}
    out = {
        "enabled": bool(cfg.get("enabled", False)),
        "trapezoid": dict(cfg.get("trapezoid") or {}),
        "parabola": dict(cfg.get("parabola") or {}),
    }
    for key in ("calib_key", "source", "angle_offsets_hash", "gamma_shift_at_calib"):
        if key in cfg:
            out[key] = cfg[key]

    kpar_arr = np.asarray(kpar, dtype=float)
    ev_arr = np.asarray(ev, dtype=float)
    if kpar_arr.size < 2 or ev_arr.size < 2:
        return out
    dk_total = float(np.nanmax(kpar_arr) - np.nanmin(kpar_arr))
    de_total = float(np.nanmax(ev_arr) - np.nanmin(ev_arr))
    if dk_total <= 0 or de_total <= 0:
        return out

    trap = out["trapezoid"]
    if trap:
        max_slope = 0.5 * dk_total / max(de_total, 1e-12)
        for key in ("slope_left", "slope_right"):
            val = float(trap.get(key, 0.0) or 0.0)
            if abs(val) > max_slope:
                trap[key] = float(np.sign(val) * max_slope)
                trap.setdefault("clamped", []).append(key)

    para = out["parabola"]
    if para:
        k0 = float(para.get("k0", 0.0) or 0.0)
        max_dk = max(abs(float(np.nanmax(kpar_arr)) - k0),
                     abs(float(np.nanmin(kpar_arr)) - k0))
        if max_dk > 0:
            max_a = 0.5 * de_total / (max_dk * max_dk)
            a_val = float(para.get("a", 0.0) or 0.0)
            if abs(a_val) > max_a:
                para["a"] = float(np.sign(a_val) * max_a)
                para.setdefault("clamped", []).append("a")
    return out


def _source_coords(
    kpar_out: np.ndarray,
    ev_out: np.ndarray,
    *,
    slope_left: float,
    slope_right: float,
    pivot_ev: float,
    a_para: float,
    k0_para: float,
    apply_trap: bool,
    apply_para: bool,
    trap_mode: str = "symmetric",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute source coordinates (k_src, e_src) in one pass.

    Convention: output (kpar, ev) → (k_src, e_src) to sample on the raw grid.
    Trapezoid: kpar stretched around pivot_ev in E. Parabola: energy warped by
    a·(kpar−k0)².
    """
    K, E = np.meshgrid(kpar_out, ev_out, indexing="ij")
    if apply_trap:
        d_e = E - float(pivot_ev)
        k_min = float(kpar_out[0])
        k_max = float(kpar_out[-1])
        span = max(k_max - k_min, 1e-12)
        u = (K - k_min) / span
        left_src = k_min - float(slope_left) * d_e
        right_src = k_max + float(slope_right) * d_e
        k_src = (1.0 - u) * left_src + u * right_src
    else:
        k_src = K
    if apply_para:
        e_src = E + float(a_para) * (K - float(k0_para)) ** 2
    else:
        e_src = E
    return k_src, e_src


def apply_distortion(
    data: np.ndarray,
    kpar,
    ev,
    cfg: dict | None,
) -> tuple[np.ndarray, dict]:
    """Apply configured distortion to ``data`` (shape ``(n_kpar, n_e)``).

    Returns ``(data_corrected, info)``. If ``cfg`` is empty / disabled / has no
    effective correction, returns **the input itself** (bit-exact identity, no
    copy), which is important for toggle reversibility.
    """
    if not is_distortion_active(cfg):
        return np.asarray(data), {"applied": False, "reason": "disabled"}
    if map_coordinates is None:
        return np.asarray(data), {
            "applied": False,
            "reason": "scipy.ndimage.map_coordinates unavailable",
        }

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"apply_distortion: data must be 2D (n_kpar, n_e), got shape {arr.shape}"
        )
    kpar_axis = _safe_axis(kpar)
    ev_axis = _safe_axis(ev)
    if arr.shape != (kpar_axis.size, ev_axis.size):
        raise ValueError(
            f"apply_distortion: shape mismatch data={arr.shape} vs "
            f"(kpar={kpar_axis.size}, ev={ev_axis.size})"
        )

    cfg_clamped = clamp_params(cfg, kpar_axis, ev_axis)
    trap = cfg_clamped.get("trapezoid") or {}
    para = cfg_clamped.get("parabola") or {}
    apply_trap = bool(trap.get("enabled")) and (
        abs(float(trap.get("slope_left", 0.0) or 0.0)) > 0.0
        or abs(float(trap.get("slope_right", 0.0) or 0.0)) > 0.0
    )
    apply_para = bool(para.get("enabled")) and abs(float(para.get("a", 0.0) or 0.0)) > 0.0
    if not apply_trap and not apply_para:
        return arr, {"applied": False, "reason": "no_effective_change"}

    pivot_ev = trap.get("pivot_ev")
    if pivot_ev is None:
        pivot_ev = float(0.5 * (ev_axis[0] + ev_axis[-1]))

    k_src, e_src = _source_coords(
        kpar_axis, ev_axis,
        slope_left=float(trap.get("slope_left", 0.0) or 0.0),
        slope_right=float(trap.get("slope_right", 0.0) or 0.0),
        pivot_ev=float(pivot_ev),
        a_para=float(para.get("a", 0.0) or 0.0),
        k0_para=float(para.get("k0", 0.0) or 0.0),
        apply_trap=apply_trap,
        apply_para=apply_para,
        trap_mode=str(trap.get("mode") or "symmetric"),
    )

    dk = _grid_step(kpar_axis)
    de = _grid_step(ev_axis)
    i_src = (k_src - kpar_axis[0]) / dk
    j_src = (e_src - ev_axis[0]) / de
    out = map_coordinates(
        arr, np.stack([i_src, j_src]),
        order=1, mode="constant", cval=np.nan, prefilter=False,
    )
    info = {
        "applied": True,
        "trapezoid_applied": apply_trap,
        "parabola_applied": apply_para,
        "pivot_ev": float(pivot_ev),
        "clamped_trapezoid": list(trap.get("clamped", [])),
        "clamped_parabola": list(para.get("clamped", [])),
        "kpar_axis": kpar_axis,
        "ev_axis": ev_axis,
    }
    out = out.astype(np.float32, copy=False)
    if cfg_clamped.get("crop_to_signal") or (cfg or {}).get("crop_to_signal"):
        out, info["kpar_axis"], info["ev_axis"], info["crop_slices"] = _crop_to_signal(
            out, kpar_axis, ev_axis,
        )
    return out, info


def _ky_drift_metric(
    volume: np.ndarray,
    *,
    finite_only: bool = True,
) -> float:
    """σ(BM_par_ky) / ⟨BM⟩: detector drift metric along ky.

    ``volume`` shape ``(n_ky, n_kx, n_e)``. Reduces to (n_ky, n_kx), then
    measures the standard deviation of profiles by ky, normalized by global
    mean.

    Returns 0.0 if the volume is empty / flat / mostly NaN.
    Redteam guard: refuse propagation if > 0.15.
    """
    v = np.asarray(volume, dtype=float)
    if v.ndim != 3 or v.shape[0] < 2:
        return 0.0
    profiles = np.nanmean(v, axis=2)  # (n_ky, n_kx)
    if finite_only and not np.isfinite(profiles).any():
        return 0.0
    mean_per_ky = np.nanmean(profiles, axis=1)  # (n_ky,)
    finite = mean_per_ky[np.isfinite(mean_per_ky)]
    if finite.size < 2:
        return 0.0
    mu = float(np.nanmean(finite))
    if abs(mu) < 1e-12:
        return 0.0
    return float(np.nanstd(finite) / abs(mu))


def fs_domain_checksum(kpar, ev) -> tuple[float, float, float, float]:
    """Checksum (kpar_min, kpar_max, ev_min, ev_max) — invalidate calibration on shift."""
    k = _safe_axis(kpar)
    e = _safe_axis(ev)
    return (
        float(k[0]) if k.size else 0.0,
        float(k[-1]) if k.size else 0.0,
        float(e[0]) if e.size else 0.0,
        float(e[-1]) if e.size else 0.0,
    )


def apply_distortion_to_fs_volume(
    volume: np.ndarray,
    kx,
    ky,
    ev,
    cfg: dict | None,
    *,
    drift_threshold: float = 0.15,
    bm_checksum: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Propagate BM distortion (trapezoid only) to each ky cut of an FS volume.

    ``volume`` shape ``(n_ky, n_kx, n_e)`` (convention loaders FS).
    Calls ``apply_distortion(slice_kx_ev, kx, ev, cfg_trapezoid_only)`` for
    each ``ky``. **Parabola forbidden** on FS volume (risk of capturing real
    Dirac/Shockley dispersion, per arpes-physicist advice).

    Guards:
    - σ(BM_par_ky)/⟨BM⟩ > ``drift_threshold`` → raise ``ValueError`` (ky-
      dependent detector tilt, BM calibration invalid away from center).
    - ``bm_checksum`` ≠ ``fs_domain_checksum(kx, ev)`` → raise (calibration
      outside FS domain, e.g. copy-pasted params from another run).
    - Volume with NaNs → preserved (slice by slice, no propagation).

    Returns ``(volume_corrected, info)`` with
    ``info = {"applied": bool, "n_slices", "drift_ratio", "reason"?}``.

    Bit-exact identity if ``cfg`` is disabled.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(
            f"apply_distortion_to_fs_volume: volume must be 3D (n_ky,n_kx,n_e), got {vol.shape}"
        )
    n_ky, n_kx, n_e = vol.shape
    kx_axis = _safe_axis(kx)
    ky_axis = _safe_axis(ky)
    ev_axis = _safe_axis(ev)
    if (kx_axis.size, ev_axis.size) != (n_kx, n_e):
        raise ValueError(
            f"apply_distortion_to_fs_volume: axes (kx={kx_axis.size}, ev={ev_axis.size}) "
            f"incompatible volume shape={vol.shape}"
        )
    if ky_axis.size != n_ky:
        raise ValueError(
            f"apply_distortion_to_fs_volume: ky.size={ky_axis.size} ≠ n_ky={n_ky}"
        )

    if not is_distortion_active(cfg):
        return vol, {"applied": False, "reason": "disabled", "n_slices": 0}

    # GF Redteam #4 — kpar/ev domain checksum.
    fs_check = fs_domain_checksum(kx_axis, ev_axis)
    if bm_checksum is not None:
        for a, b in zip(bm_checksum, fs_check):
            if abs(float(a) - float(b)) > max(1e-3, 0.05 * max(abs(a), abs(b))):
                raise ValueError(
                    f"apply_distortion_to_fs_volume: BM calibration outside FS domain "
                    f"(BM={bm_checksum} vs FS={fs_check}). Recalibrate."
                )

    # GF Redteam #1 — σ(BM_par_ky)/⟨BM⟩: ky-dependent detector tilt.
    drift = _ky_drift_metric(vol)
    if drift > float(drift_threshold):
        raise ValueError(
            f"apps_distortion_to_fs_volume: drift ky/⟨BM⟩={drift:.3f} > "
            f"{drift_threshold:.3f}. Mean BM calibration invalid away from center."
        )

    # GF Physicist — parabola forbidden on FS volume.
    cfg_trap_only = dict(cfg or {})
    if cfg_trap_only.get("parabola"):
        cfg_trap_only["parabola"] = dict(cfg_trap_only["parabola"])
        cfg_trap_only["parabola"]["enabled"] = False

    out = np.empty_like(vol)
    n_applied = 0
    for iy in range(n_ky):
        slice_2d = vol[iy]  # (n_kx, n_e)
        slice_corr, info_slice = apply_distortion(
            slice_2d, kx_axis, ev_axis, cfg_trap_only,
        )
        out[iy] = slice_corr
        if info_slice.get("applied"):
            n_applied += 1

    return out, {
        "applied": n_applied > 0,
        "n_slices": int(n_applied),
        "drift_ratio": float(drift),
        "fs_checksum": fs_check,
        "parabola_skipped": bool((cfg or {}).get("parabola", {}).get("enabled", False)),
    }


def signal_bbox(
    data: np.ndarray, kpar, ev,
    *, intensity_percentile: float = 50.0, finite_only: bool = True,
) -> dict:
    from arpes.physics.distortion_signal import signal_bbox as _signal_bbox
    return _signal_bbox(
        data, kpar, ev,
        intensity_percentile=intensity_percentile,
        finite_only=finite_only,
    )


def _crop_to_signal(
    data: np.ndarray, kpar: np.ndarray, ev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove fully-NaN rows/columns at the edges.

    Keeps the largest rectangular sub-window with *at least one* finite value.
    Returns (data_crop, kpar_crop, ev_crop, slices).
    """
    finite = np.isfinite(data)
    if not finite.any():
        return data, kpar, ev, {"k": (0, kpar.size), "e": (0, ev.size)}
    rows_ok = finite.any(axis=1)
    cols_ok = finite.any(axis=0)
    k_lo, k_hi = int(np.argmax(rows_ok)), int(rows_ok.size - np.argmax(rows_ok[::-1]))
    e_lo, e_hi = int(np.argmax(cols_ok)), int(cols_ok.size - np.argmax(cols_ok[::-1]))
    if k_hi - k_lo < 2 or e_hi - e_lo < 2:
        return data, kpar, ev, {"k": (0, kpar.size), "e": (0, ev.size)}
    return (
        data[k_lo:k_hi, e_lo:e_hi],
        kpar[k_lo:k_hi],
        ev[e_lo:e_hi],
        {"k": (k_lo, k_hi), "e": (e_lo, e_hi)},
    )


def cache_signature(cfg: dict | None) -> tuple:
    """Lightweight signature for ``_disp_cache_key`` (4 rounded floats + flags)."""
    if not is_distortion_active(cfg):
        return ("distortion", False)
    trap = cfg.get("trapezoid") or {}
    para = cfg.get("parabola") or {}
    return (
        "distortion", True,
        round(float(trap.get("slope_left", 0.0) or 0.0), _PARAM_CACHE_PRECISION),
        round(float(trap.get("slope_right", 0.0) or 0.0), _PARAM_CACHE_PRECISION),
        round(float(trap.get("pivot_ev") if trap.get("pivot_ev") is not None else 0.0),
              _PARAM_CACHE_PRECISION),
        round(float(para.get("a", 0.0) or 0.0), _PARAM_CACHE_PRECISION),
        round(float(para.get("k0", 0.0) or 0.0), _PARAM_CACHE_PRECISION),
        bool(trap.get("enabled", False)),
        bool(para.get("enabled", False)),
    )


def auto_detect_trapezoid(
    data: np.ndarray, kpar, ev, *, percentile: float = 80.0,
) -> dict | None:
    """Detect (slope_left, slope_right, pivot_ev) by intensity envelope.

    For each energy row: find left/right edges of the > p_threshold area;
    linear fit of kpar_edge(E). Slopes = angular coefficients.

    Returns ``None`` if the BM is too narrow (n_kpar < 16) or if the fit is not
    robust (R² < 0.5 on at least one edge).
    """
    arr = np.asarray(data, dtype=float)
    kpar_axis = _safe_axis(kpar)
    ev_axis = _safe_axis(ev)
    if arr.shape != (kpar_axis.size, ev_axis.size):
        return None
    if kpar_axis.size < _MIN_KPAR_FOR_AUTO:
        return None

    finite = np.isfinite(arr)
    if not finite.any():
        return None
    threshold = float(np.nanpercentile(arr[finite], percentile))
    mask = arr > threshold
    left_k = np.full(ev_axis.size, np.nan)
    right_k = np.full(ev_axis.size, np.nan)
    for j in range(ev_axis.size):
        idx = np.where(mask[:, j])[0]
        if idx.size >= 3:
            left_k[j] = float(kpar_axis[idx[0]])
            right_k[j] = float(kpar_axis[idx[-1]])

    valid = np.isfinite(left_k) & np.isfinite(right_k)
    if valid.sum() < max(8, ev_axis.size // 4):
        return None
    e_v = ev_axis[valid]
    pivot = float(0.5 * (e_v[0] + e_v[-1]))
    de = e_v - pivot

    def _slope_r2(y):
        if np.allclose(de, 0):
            return 0.0, 0.0
        slope, intercept = np.polyfit(de, y, 1)
        pred = slope * de + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return float(slope), float(r2)

    slope_l_raw, r2_l = _slope_r2(left_k[valid])
    slope_r_raw, r2_r = _slope_r2(right_k[valid])
    if r2_l < 0.5 and r2_r < 0.5:
        return None

    # The left edge moves left when the trapezoid widens in E
    # → positive slope_left if the base moves away (k decreases with increasing E).
    return {
        "slope_left": -slope_l_raw,
        "slope_right": slope_r_raw,
        "pivot_ev": pivot,
        "r2_left": r2_l,
        "r2_right": r2_r,
    }


def auto_detect_parabola(
    data: np.ndarray, kpar, ev, *, smooth_sigma_k: float = 3.0,
) -> dict | None:
    """Detect (a, k0) by fitting a degree-2 polynomial on ``argmax_k`` by E row.

    Returns ``None`` if the BM is too narrow, lacks clear dispersion, or if the
    fit converges to extreme parameters.
    """
    arr = np.asarray(data, dtype=float)
    kpar_axis = _safe_axis(kpar)
    ev_axis = _safe_axis(ev)
    if arr.shape != (kpar_axis.size, ev_axis.size):
        return None
    if kpar_axis.size < _MIN_KPAR_FOR_AUTO:
        return None

    if smooth_sigma_k > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            arr = gaussian_filter1d(arr, sigma=float(smooth_sigma_k), axis=0,
                                    mode="nearest")
        except ImportError:  # pragma: no cover
            pass

    arg = np.argmax(np.where(np.isfinite(arr), arr, -np.inf), axis=0)
    k_max = kpar_axis[arg]
    valid = np.isfinite(k_max) & (np.ptp(arr, axis=0) > 1e-6)
    if valid.sum() < max(8, ev_axis.size // 4):
        return None
    k_v = k_max[valid]
    e_v = ev_axis[valid]
    if (np.ptp(e_v) < _MIN_DISPERSION_EV) and (np.ptp(k_v) < 0.05 * np.ptp(kpar_axis)):
        return None

    # Fit E = a*(k - k0)^2 + c → expanded: E = a·k² − 2a·k0·k + (a·k0² + c)
    coeffs = np.polyfit(k_v, e_v, 2)
    a_fit = float(coeffs[0])
    if abs(a_fit) < 1e-6:
        return None
    k0_fit = float(-coeffs[1] / (2.0 * a_fit))
    return {
        "a": a_fit,
        "k0": k0_fit,
        "n_points": int(valid.sum()),
    }


from arpes.physics.distortion_meta import (  # noqa: E402
    angle_offsets_hash,
    calib_key_for_meta,
    gamma_shift_signature,
    is_fs_data,
)


__all__: list[str] = [
    "is_distortion_active",
    "apply_distortion",
    "clamp_params",
    "cache_signature",
    "auto_detect_trapezoid",
    "auto_detect_parabola",
    "angle_offsets_hash",
    "gamma_shift_signature",
    "calib_key_for_meta",
    "is_fs_data",
    "signal_bbox",
]


def get_cfg_summary(cfg: dict | None) -> str:
    from arpes.physics.distortion_meta import get_cfg_summary as _get_cfg_summary
    return _get_cfg_summary(cfg, is_active=is_distortion_active)
