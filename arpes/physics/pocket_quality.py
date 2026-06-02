"""Physical guard-rails for pocket characterization.

Three checks, motivés par la review physicien :
- local SNR autour du seed (rejet si bruit > 33 % du signal)
- contour touche le bord de la FS map (kF non défini sur bord)
- smoothing σ trop grand vs taille de la poche (on lisse la poche elle-même)

Aucun PyQt. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityCheck:
    ok: bool                # False => bloquant
    code: str               # "snr_low", "border", "smooth_excess", "ok"
    message: str            # texte court à afficher à l'user
    metric: float = float("nan")


def local_snr(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    seed_point: tuple[float, float],
    radius: float,
) -> float:
    """Ratio median / std des pixels dans le disque de rayon ``radius``
    autour de ``seed_point`` (coordonnées du même repère que kx/ky).

    Renvoie NaN si moins de 8 pixels finis dans la zone.
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
    """True si un point du contour est à moins de ``tol_pixels`` du bord."""
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
    """Warning si max(σ_y·dy, σ_x·dx) > 0.5·kF_mean : on lisse la poche."""
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
                "la poche est lissée par le filtre, kF biaisé."
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
    snr_min: float = 3.0,
    snr_radius_factor: float = 2.0,
    border_tol_pixels: int = 2,
) -> list[QualityCheck]:
    """Run all guards and return per-check results.

    A check with ``ok=False`` should block publication; the caller decides
    whether to also block persistence.
    """
    results: list[QualityCheck] = []
    dx = float(kx[1] - kx[0]) if kx.size >= 2 else 0.0
    dy = float(ky[1] - ky[0]) if ky.size >= 2 else 0.0
    pixel = max(abs(dx), abs(dy))
    snr = local_snr(image, kx, ky, seed_point, radius=float(snr_radius_factor) * pixel)
    if not np.isfinite(snr):
        results.append(QualityCheck(
            ok=False, code="snr_low",
            message="SNR local non calculable (zone seed trop petite).",
            metric=float(snr),
        ))
    elif snr < float(snr_min):
        results.append(QualityCheck(
            ok=False, code="snr_low",
            message=f"SNR local = {snr:.2f} < {snr_min:.1f}. Bruit trop fort autour du seed.",
            metric=float(snr),
        ))
    else:
        results.append(QualityCheck(ok=True, code="snr_ok",
                                    message=f"SNR={snr:.2f}", metric=float(snr)))

    if contour_touches_border(contour, kx, ky, tol_pixels=border_tol_pixels):
        results.append(QualityCheck(
            ok=False, code="border",
            message="Contour touche le bord de la FS map : kF non défini, élargis le scan.",
        ))
    else:
        results.append(QualityCheck(ok=True, code="border_ok", message=""))

    results.append(smoothing_warning(sigma_pixels, (dy, dx), kf_mean))
    return results
