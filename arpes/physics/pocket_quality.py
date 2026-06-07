"""Physical guard-rails for pocket characterization.

Three checks, motivated by the physicist review:
- local SNR around the seed (reject if noise > 33% of signal)
- contour touches the FS map border (kF undefined on border)
- smoothing σ too large vs pocket size (smooths the pocket itself)

No PyQt. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityCheck:
    ok: bool                # False => blocking
    code: str               # "snr_low", "border", "smooth_excess", "ok"
    message: str            # short text to show to the user
    metric: float = float("nan")


def local_snr(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    seed_point: tuple[float, float],
    radius: float,
) -> float:
    """Median / std ratio of pixels in the disk with radius ``radius`` around
    ``seed_point`` (coordinates in the same frame as kx/ky).

    Returns NaN if fewer than 8 finite pixels are in the zone.
    """
    x = np.asarray(kx, dtype=float)
    y = np.asarray(ky, dtype=float)
    z = np.asarray(image, dtype=float)
    cx, cy = float(seed_point[0]), float(seed_point[1])
    xg, yg = np.meshgrid(x, y, indexing="xy")
    mask = (xg - cx) ** 2 + (yg - cy) ** 2 <= float(radius) ** 2
    vals = z[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size < 8:
        return float("nan")
    med = float(np.nanmedian(finite))
    std = float(np.nanstd(finite, ddof=1))
    if std <= 0.0:
        return float("inf")
    return float(abs(med) / std)


def contour_touches_border(
    contour: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    tol_pixels: int = 2,
) -> bool:
    """True if a contour point is within ``tol_pixels`` of the border."""
    c = np.asarray(contour, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] == 0:
        return False
    x = np.asarray(kx, dtype=float)
    y = np.asarray(ky, dtype=float)
    dx = float(x[1] - x[0]) if x.size >= 2 else 0.0
    dy = float(y[1] - y[0]) if y.size >= 2 else 0.0
    tol_x = abs(dx) * int(max(0, tol_pixels))
    tol_y = abs(dy) * int(max(0, tol_pixels))
    return bool(
        np.any(c[:, 0] <= float(x[0]) + tol_x)
        or np.any(c[:, 0] >= float(x[-1]) - tol_x)
        or np.any(c[:, 1] <= float(y[0]) + tol_y)
        or np.any(c[:, 1] >= float(y[-1]) - tol_y)
    )


def smoothing_warning(sigma_pixels: tuple[float, float],
                      pixel_size: tuple[float, float],
                      kf_mean: float) -> QualityCheck:
    """Warn if max(σ_y·dy, σ_x·dx) > 0.5·kF_mean: the pocket is being smoothed."""
    sy, sx = float(sigma_pixels[0]), float(sigma_pixels[1])
    dy, dx = float(pixel_size[0]), float(pixel_size[1])
    sigma_k = max(sy * abs(dy), sx * abs(dx))
    if not np.isfinite(kf_mean) or kf_mean <= 0.0:
        return QualityCheck(ok=True, code="ok", message="", metric=float(sigma_k))
    ratio = sigma_k / float(kf_mean)
    if ratio > 0.5:
        return QualityCheck(
            ok=False,
            code="smooth_excess",
            message=(
                f"σ_smooth·dk = {sigma_k:.3f} π/a > 0.5·kF ({kf_mean:.3f}) — "
                "the pocket is smoothed by the filter, kF biased."
            ),
            metric=ratio,
        )
    return QualityCheck(ok=True, code="ok", message="", metric=ratio)


def run_pocket_guards(
    *,
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    seed_point: tuple[float, float],
    contour: np.ndarray,
    sigma_pixels: tuple[float, float],
    kf_mean: float,
    snr_min: float = 5.0,
    snr_radius_factor: float = 2.0,
    border_tol_pixels: int | None = None,
) -> list[QualityCheck]:
    """Run all guards and return per-check results.

    A check with ``ok=False`` should block publication; the caller decides
    whether to also block persistence.
    """
    results: list[QualityCheck] = []
    dx = float(kx[1] - kx[0]) if kx.size >= 2 else 0.0
    dy = float(ky[1] - ky[0]) if ky.size >= 2 else 0.0
    pixel = max(abs(dx), abs(dy))
    # P4.5: adaptive border tolerance — 2 px or 2% of the smallest dimension.
    if border_tol_pixels is None:
        border_tol_pixels = max(2, int(0.02 * min(int(kx.size), int(ky.size))))
    snr = local_snr(image, kx, ky, seed_point, radius=float(snr_radius_factor) * pixel)
    if not np.isfinite(snr):
        results.append(QualityCheck(
            ok=False, code="snr_low",
            message="Local SNR cannot be computed (seed zone too small).",
            metric=float(snr),
        ))
    elif snr < float(snr_min):
        results.append(QualityCheck(
            ok=False, code="snr_low",
            message=f"Local SNR = {snr:.2f} < {snr_min:.1f}. Noise too strong around seed.",
            metric=float(snr),
        ))
    else:
        results.append(QualityCheck(ok=True, code="snr_ok",
                                    message=f"SNR={snr:.2f}", metric=float(snr)))

    if contour_touches_border(contour, kx, ky, tol_pixels=border_tol_pixels):
        results.append(QualityCheck(
            ok=False, code="border",
            message="Contour touches the FS map border: kF undefined, widen the scan.",
        ))
    else:
        results.append(QualityCheck(ok=True, code="border_ok", message=""))

    results.append(smoothing_warning(sigma_pixels, (dy, dx), kf_mean))
    return results
