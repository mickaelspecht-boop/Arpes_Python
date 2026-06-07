"""MDC-radial kF extraction (publication-grade).

For each angular direction from an estimated center, extract the MDC along the
radius, fit a Lorentzian, and return the Fermi radius ``kF(theta)`` with
uncertainty from the fit covariance matrix.

Reference: Damascelli, Hussain, Shen, RMP 75 (2003): kF = max of A(k, omega=EF),
so fit the MDC at EF, not an iso-intensity contour.

No PyQt. Pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class RadialMDCKf:
    theta_deg: float
    kF: float
    kF_std: float
    fit_r2: float
    ok: bool
    n_samples: int

    def asdict(self) -> dict:
        return asdict(self)


def _bilinear(image: np.ndarray, kx: np.ndarray, ky: np.ndarray,
              point: tuple[float, float]) -> float:
    x, y = float(point[0]), float(point[1])
    if x < kx[0] or x > kx[-1] or y < ky[0] or y > ky[-1]:
        return float("nan")
    ix = int(np.searchsorted(kx, x) - 1)
    iy = int(np.searchsorted(ky, y) - 1)
    ix = max(0, min(ix, kx.size - 2))
    iy = max(0, min(iy, ky.size - 2))
    x0, x1 = float(kx[ix]), float(kx[ix + 1])
    y0, y1 = float(ky[iy]), float(ky[iy + 1])
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    z00 = image[iy, ix]; z10 = image[iy, ix + 1]
    z01 = image[iy + 1, ix]; z11 = image[iy + 1, ix + 1]
    return float((1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10
                 + (1 - tx) * ty * z01 + tx * ty * z11)


def _lorentzian(r, A, r0, gamma, bg):
    return A * gamma * gamma / ((r - r0) ** 2 + gamma * gamma) + bg


def sample_radial_mdc(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    center: tuple[float, float],
    theta_deg: float,
    r_max: float,
    n_points: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Echantillonne intensity le long du rayon (center, θ) jusqu'à r_max.

    Retourne (radii, intensities) avec intensités via interpolation bilinéaire.
    NaN où la sonde sort de la grille.
    """
    if r_max <= 0 or n_points < 4:
        raise ValueError("sample_radial_mdc: r_max>0 and n_points>=4 required.")
    theta = np.radians(float(theta_deg))
    radii = np.linspace(0.0, float(r_max), int(n_points))
    cx, cy = float(center[0]), float(center[1])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    intens = np.array([
        _bilinear(image, kx, ky, (cx + r * cos_t, cy + r * sin_t))
        for r in radii
    ], dtype=float)
    return radii, intens


def fit_lorentzian_mdc(
    radii: np.ndarray,
    intensities: np.ndarray,
    *,
    r_min: float = 0.0,
) -> tuple[float, float, float, int]:
    """Ajuste un Lorentzien sur (radii, intensities). Retourne
    ``(kF, kF_std, r2, n_used)``. NaN si fit échoue.
    """
    r = np.asarray(radii, dtype=float)
    y = np.asarray(intensities, dtype=float)
    mask = np.isfinite(r) & np.isfinite(y) & (r >= float(r_min))
    n = int(np.sum(mask))
    if n < 6:
        return float("nan"), float("nan"), float("nan"), n
    r_use = r[mask]; y_use = y[mask]
    y_min = float(np.nanmin(y_use))
    y_max = float(np.nanmax(y_use))
    if y_max - y_min <= 1e-12:
        return float("nan"), float("nan"), float("nan"), n
    r0_guess = float(r_use[int(np.argmax(y_use))])
    half = 0.5 * (y_max + y_min)
    above = r_use[y_use >= half]
    gamma_guess = float(0.5 * (above.max() - above.min())) if above.size > 1 else float(0.05 * (r_use[-1] - r_use[0]))
    if gamma_guess <= 0:
        gamma_guess = 0.01
    p0 = [y_max - y_min, r0_guess, gamma_guess, y_min]
    lo = [0.0, float(r_use[0]), 1e-6, -np.inf]
    hi = [np.inf, float(r_use[-1]), float(r_use[-1] - r_use[0]) * 2.0, np.inf]
    try:
        popt, pcov = curve_fit(_lorentzian, r_use, y_use, p0=p0,
                               bounds=(lo, hi), maxfev=4000)
    except Exception:
        return float("nan"), float("nan"), float("nan"), n
    kF = float(popt[1])
    std = float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else float("nan")
    pred = _lorentzian(r_use, *popt)
    ss_res = float(np.sum((y_use - pred) ** 2))
    ss_tot = float(np.sum((y_use - np.nanmean(y_use)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return kF, std, r2, n


def kf_radial_mdc(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    center: tuple[float, float],
    theta_deg: float,
    *,
    r_max: float,
    n_points: int = 64,
    r2_min: float = 0.5,
) -> RadialMDCKf:
    """Extract kF along one radial direction via MDC Lorentzian fit."""
    radii, ints = sample_radial_mdc(image, kx, ky, center, theta_deg, r_max, n_points)
    kF, std, r2, n_used = fit_lorentzian_mdc(radii, ints)
    ok = bool(np.isfinite(kF) and np.isfinite(r2) and r2 >= float(r2_min))
    return RadialMDCKf(
        theta_deg=float(theta_deg),
        kF=float(kF), kF_std=float(std), fit_r2=float(r2),
        ok=ok, n_samples=int(n_used),
    )


def characterize_pocket_mdc_radial(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    seed_point: tuple[float, float],
    n_directions: int = 36,
    r_max: float | None = None,
    n_points: int = 64,
    r2_min: float = 0.5,
    refine_center: bool = True,
) -> tuple[np.ndarray, list[RadialMDCKf], tuple[float, float]]:
    """Build a Fermi pocket contour by radial MDC fits.

    Returns ``(contour (N, 2), per_direction_results, final_center)``.
    Contour est fermé (premier point répété en fin). Directions ratées
    (``ok=False``) sont skipées du contour.

    ``refine_center`` : après une 1re passe, recalcule le centre = moyenne
    pondérée des kF·(cosθ, sinθ) et refait une seconde passe.
    """
    x = np.asarray(kx, dtype=float); y = np.asarray(ky, dtype=float)
    z = np.asarray(image, dtype=float)
    if z.ndim != 2 or z.shape != (y.size, x.size):
        raise ValueError("FS image must be 2D with shape (ky, kx).")
    if r_max is None or r_max <= 0:
        rx = min(x[-1] - float(seed_point[0]), float(seed_point[0]) - x[0])
        ry = min(y[-1] - float(seed_point[1]), float(seed_point[1]) - y[0])
        r_max = float(max(0.05, 0.9 * min(rx, ry)))

    def _one_pass(center):
        thetas = np.linspace(0.0, 360.0, int(n_directions), endpoint=False)
        return [
            kf_radial_mdc(z, x, y, center, t,
                          r_max=r_max, n_points=n_points, r2_min=r2_min)
            for t in thetas
        ]

    results = _one_pass(seed_point)
    center = (float(seed_point[0]), float(seed_point[1]))
    if refine_center:
        ok_pts = [(np.cos(np.radians(r.theta_deg)) * r.kF + center[0],
                   np.sin(np.radians(r.theta_deg)) * r.kF + center[1])
                  for r in results if r.ok]
        if len(ok_pts) >= 8:
            arr = np.asarray(ok_pts, dtype=float)
            center = (float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1])))
            results = _one_pass(center)
    pts = []
    for r in results:
        if not r.ok:
            continue
        th = np.radians(r.theta_deg)
        pts.append((center[0] + r.kF * np.cos(th),
                    center[1] + r.kF * np.sin(th)))
    if len(pts) < 2:
        raise ValueError(
            f"MDC-radial: only {len(pts)} valid directions; "
            "increase n_directions or lower r2_min."
        )
    arr = np.asarray(pts, dtype=float)
    arr = np.vstack([arr, arr[0]])
    return arr, results, center


def arc_coverage_deg(results: list[RadialMDCKf]) -> float:
    """Couverture angulaire valide (deg) = N_ok / N_total * 360."""
    if not results:
        return 0.0
    n_ok = sum(1 for r in results if r.ok)
    return float(360.0 * n_ok / len(results))


def is_arc_gap(results: list[RadialMDCKf], max_gap_deg: float = 30.0) -> bool:
    """True si la séquence de fits valides présente un trou angulaire > max_gap_deg.

    Trou contigu de directions ratées indique poche tronquée par bord scan.
    """
    if not results:
        return False
    sorted_r = sorted(results, key=lambda r: r.theta_deg)
    n = len(sorted_r)
    step = 360.0 / float(n)
    gap = 0
    max_run = 0
    for r in sorted_r + sorted_r[:1]:  # wrap
        if not r.ok:
            gap += 1
            max_run = max(max_run, gap)
        else:
            gap = 0
    return bool(max_run * step > float(max_gap_deg))
