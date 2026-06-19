"""Polarization-contrast (linear dichroism) maps for ARPES band maps.

P(k,E) = (I_pi - I_sigma) / (I_pi + I_sigma) compares two band maps taken in the
same geometry with orthogonal light polarizations (LH/LV ≈ pi/sigma). It is a
robust, geometry-fixed proxy for orbital symmetry: a band bright in one
polarization and dark in the other has a strong |P|.

Pure numpy/scipy — no PyQt, headless-testable. The two maps must share the same
(k, E) grid; use :func:`resample_map` to put a partner map onto the reference
grid before calling :func:`pkE_contrast`.
"""
from __future__ import annotations

import numpy as np


def resample_map(
    values: np.ndarray,
    src_k: np.ndarray,
    src_e: np.ndarray,
    dst_k: np.ndarray,
    dst_e: np.ndarray,
) -> np.ndarray:
    """Bilinearly resample a band map ``values[e, k]`` onto ``(dst_e, dst_k)``.

    Out-of-range points are NaN. Axes may be given in any monotonic order; they
    are sorted internally. Returns an array of shape ``(len(dst_e), len(dst_k))``.
    """
    from scipy.interpolate import RegularGridInterpolator

    v = np.asarray(values, dtype=float)
    se = np.asarray(src_e, dtype=float)
    sk = np.asarray(src_k, dtype=float)
    if v.shape != (se.size, sk.size):
        raise ValueError(f"values shape {v.shape} != (E={se.size}, k={sk.size})")
    # RegularGridInterpolator requires strictly ascending axes.
    e_order = np.argsort(se)
    k_order = np.argsort(sk)
    interp = RegularGridInterpolator(
        (se[e_order], sk[k_order]),
        v[np.ix_(e_order, k_order)],
        bounds_error=False,
        fill_value=np.nan,
    )
    de = np.asarray(dst_e, dtype=float)
    dk = np.asarray(dst_k, dtype=float)
    kk, ee = np.meshgrid(dk, de)
    out = interp(np.column_stack([ee.ravel(), kk.ravel()]))
    return out.reshape(ee.shape)


def pkE_contrast(
    i_pi: np.ndarray,
    i_sigma: np.ndarray,
    *,
    denom_floor_frac: float = 0.02,
    clip: float = 1.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Linear-dichroism map ``(I_pi - I_sigma) / (I_pi + I_sigma)``.

    Both maps must share shape. ``denom_floor_frac`` masks pixels where the sum
    ``I_pi + I_sigma`` is below that fraction of its max (low-statistics regions
    → NaN instead of exploding). ``smooth_sigma`` (px) optionally Gaussian-smooths
    each map first; ``clip`` bounds the result to ``[-clip, clip]``.
    """
    a = np.asarray(i_pi, dtype=float)
    b = np.asarray(i_sigma, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"map shapes differ: {a.shape} vs {b.shape}")
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(np.nan_to_num(a), float(smooth_sigma))
        b = gaussian_filter(np.nan_to_num(b), float(smooth_sigma))
    denom = a + b
    max_denom = np.nanmax(denom) if np.isfinite(np.nanmax(denom)) else 0.0
    floor = float(denom_floor_frac) * float(max_denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (a - b) / np.where(np.abs(denom) <= floor, np.nan, denom)
    if clip and clip > 0:
        p = np.clip(p, -float(clip), float(clip))
    return p
