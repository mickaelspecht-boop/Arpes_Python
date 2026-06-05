"""Fermi-pocket characterization helpers.

Pure numerical layer: no Qt, no Materials Project dependency.  The module
extracts an iso-intensity contour from an experimental FS map and derives
simple geometric descriptors useful before comparing to DFT/MP.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Literal

import contourpy
import numpy as np
from matplotlib.path import Path as MplPath

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - scipy is present in the app env
    gaussian_filter = None


PocketTopology = Literal["electron", "hole", "unclear"]
HsPointMap = dict[str, tuple[float, float] | list[tuple[float, float]]]


@dataclass(frozen=True)
class PocketProperties:
    centroid_kx: float
    centroid_ky: float
    area_inv_a2: float
    area_pct_bz: float
    kF_mean: float
    kF_a: float
    kF_b: float
    ellipse_angle_deg: float
    topology: PocketTopology
    topology_confidence: float
    hs_label_nearest: str
    hs_distance: float
    kF_gamma_x: float = float("nan")
    kF_gamma_m: float = float("nan")
    aspect_ratio: float = float("nan")
    eccentricity: float = float("nan")
    curvature_mean: float = float("nan")
    curvature_var: float = float("nan")
    n_carriers_2D: float = float("nan")
    topology_rays_used: int = 0
    analysis_mode: str = "unknown"
    mdc_valid_directions: int = 0
    mdc_total_directions: int = 0

    def asdict(self) -> dict:
        """Return a JSON-ready representation for session persistence."""
        return asdict(self)


def _as_axes(kx, ky, image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(kx, dtype=float)
    y = np.asarray(ky, dtype=float)
    z = np.asarray(image, dtype=float)
    if z.ndim != 2:
        raise ValueError("image FS doit être 2D.")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("kx et ky doivent être des axes 1D.")
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"shape image {z.shape} incompatible avec ky/kx {(y.size, x.size)}."
        )
    if x.size < 2 or y.size < 2:
        raise ValueError("axes kx/ky trop courts.")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("kx et ky doivent être strictement croissants.")
    if not np.isfinite(z).any():
        raise ValueError("image FS sans valeurs finies.")
    return x, y, z


def _close_contour(contour: np.ndarray) -> np.ndarray:
    c = np.asarray(contour, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 4:
        return c
    if np.linalg.norm(c[0] - c[-1]) > 1e-10:
        c = np.vstack([c, c[0]])
    return c


def smooth_fs_image(image: np.ndarray, sigma: tuple[float, float] = (1.0, 3.0)) -> np.ndarray:
    """Return a denoised FS map for contour extraction.

    Experimental FS maps often have many detector/normalization stripes.  The
    displayed image can stay sharp, but pocket contours need a more stable
    scalar field or contourpy follows pixel noise.
    """
    z = np.asarray(image, dtype=float)
    if gaussian_filter is None or z.ndim != 2:
        return z
    finite = np.isfinite(z)
    if not finite.any():
        return z
    fill = float(np.nanmedian(z[finite]))
    work = np.where(finite, z, fill)
    smoothed = gaussian_filter(work, sigma=sigma, mode="nearest")
    return np.where(finite, smoothed, np.nan)


def smooth_closed_contour(contour: np.ndarray, window: int = 9) -> np.ndarray:
    """Smooth a closed contour with circular moving average."""
    c = _close_contour(contour)
    if c.shape[0] < max(8, window + 2):
        return c
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    pts = c[:-1]
    pad = w // 2
    ext = np.vstack([pts[-pad:], pts, pts[:pad]])
    kernel = np.ones(w, dtype=float) / float(w)
    xs = np.convolve(ext[:, 0], kernel, mode="valid")
    ys = np.convolve(ext[:, 1], kernel, mode="valid")
    out = np.column_stack([xs, ys])
    return _close_contour(out)


def simplify_closed_contour(contour: np.ndarray, min_step: float = 0.015) -> np.ndarray:
    """Drop near-duplicate contour points while preserving closure."""
    c = _close_contour(contour)
    if c.shape[0] < 5:
        return c
    kept = [c[0]]
    for p in c[1:-1]:
        if np.linalg.norm(p - kept[-1]) >= float(min_step):
            kept.append(p)
    if len(kept) < 4:
        return c
    return _close_contour(np.asarray(kept, dtype=float))


def extract_fs_contour(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    level: float,
    seed_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return the closed iso-contour containing ``seed_point``.

    If ``seed_point`` is None, the largest closed contour is returned.  The
    contour has shape ``(N, 2)`` with columns ``kx, ky``.
    """
    x, y, z = _as_axes(kx, ky, image)
    lvl = float(level)
    finite = z[np.isfinite(z)]
    if not np.isfinite(lvl) or lvl < float(np.nanmin(finite)) or lvl > float(np.nanmax(finite)):
        raise ValueError("level hors plage intensité FS.")
    z_work = np.where(np.isfinite(z), z, float(np.nanmin(finite)))
    gen = contourpy.contour_generator(x=x, y=y, z=z_work, name="serial")
    contours = [_close_contour(c) for c in gen.lines(lvl)]
    contours = [
        c for c in contours
        if c.shape[0] >= 4 and np.linalg.norm(c[0] - c[-1]) <= 1e-8
    ]
    if not contours:
        raise ValueError("aucun contour fermé trouvé à ce niveau.")

    if seed_point is not None:
        seed = tuple(map(float, seed_point))
        containing = [c for c in contours if MplPath(c).contains_point(seed)]
        if not containing:
            raise ValueError("aucun contour fermé ne contient le seed_point.")
        contours = containing

    return max(contours, key=lambda c: abs(pocket_area(c)))


def pocket_area(contour: np.ndarray) -> float:
    """Signed area using the shoelace formula, in axis units squared."""
    c = _close_contour(contour)
    if c.shape[0] < 4:
        return 0.0
    x = c[:, 0]
    y = c[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _centroid(contour: np.ndarray) -> np.ndarray:
    c = _close_contour(contour)
    x = c[:, 0]
    y = c[:, 1]
    a = pocket_area(c)
    if abs(a) < 1e-14:
        return np.nanmean(c[:-1], axis=0)
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    cx = np.sum((x[:-1] + x[1:]) * cross) / (6.0 * a)
    cy = np.sum((y[:-1] + y[1:]) * cross) / (6.0 * a)
    return np.array([cx, cy], dtype=float)


def _interp_image(image: np.ndarray, kx: np.ndarray, ky: np.ndarray, point: np.ndarray) -> float:
    x, y = float(point[0]), float(point[1])
    if x < kx[0] or x > kx[-1] or y < ky[0] or y > ky[-1]:
        return float("nan")
    ix = int(np.searchsorted(kx, x) - 1)
    iy = int(np.searchsorted(ky, y) - 1)
    ix = max(0, min(ix, kx.size - 2))
    iy = max(0, min(iy, ky.size - 2))
    x0, x1 = kx[ix], kx[ix + 1]
    y0, y1 = ky[iy], ky[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    q00 = image[iy, ix]
    q10 = image[iy, ix + 1]
    q01 = image[iy + 1, ix]
    q11 = image[iy + 1, ix + 1]
    return float(
        (1 - tx) * (1 - ty) * q00
        + tx * (1 - ty) * q10
        + (1 - tx) * ty * q01
        + tx * ty * q11
    )


def kf_along_direction(
    contour: np.ndarray,
    center: tuple[float, float],
    theta_deg: float,
    tol_deg: float = 10.0,
) -> float:
    """Median radius of contour points in angular sector ``theta_deg ± tol_deg``.

    Returns NaN if no contour points fall in the sector.
    """
    c = _close_contour(contour)[:-1]
    if c.shape[0] < 3:
        return float("nan")
    vec = c - np.asarray(center, dtype=float)
    angles = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    delta = (angles - float(theta_deg) + 180.0) % 360.0 - 180.0
    mask = np.abs(delta) <= float(tol_deg)
    if not np.any(mask):
        return float("nan")
    radii = np.linalg.norm(vec[mask], axis=1)
    return float(np.nanmedian(radii))


def pocket_curvature(contour: np.ndarray) -> tuple[float, float]:
    """Mean and variance of unsigned local curvature ``|κ|`` along the contour.

    Uses the discrete formula ``κ_i = 2·area(p_{i-1}, p_i, p_{i+1}) /
    (|p_i-p_{i-1}| · |p_{i+1}-p_i| · |p_{i+1}-p_{i-1}|)``.
    Units: ``(π/a)^{-1}`` if the contour is in ``π/a``.
    """
    c = _close_contour(contour)[:-1]
    if c.shape[0] < 4:
        return float("nan"), float("nan")
    prev_pts = np.roll(c, 1, axis=0)
    next_pts = np.roll(c, -1, axis=0)
    a = np.linalg.norm(c - prev_pts, axis=1)
    b = np.linalg.norm(next_pts - c, axis=1)
    d = np.linalg.norm(next_pts - prev_pts, axis=1)
    cross = (c[:, 0] - prev_pts[:, 0]) * (next_pts[:, 1] - prev_pts[:, 1]) - \
            (c[:, 1] - prev_pts[:, 1]) * (next_pts[:, 0] - prev_pts[:, 0])
    denom = a * b * d
    valid = denom > 1e-14
    kappa = np.zeros(c.shape[0], dtype=float)
    kappa[valid] = np.abs(cross[valid]) * 2.0 / denom[valid]
    if not np.any(valid):
        return float("nan"), float("nan")
    return float(np.nanmean(kappa[valid])), float(np.nanvar(kappa[valid]))


def luttinger_count(
    area_inv_a2: float,
    bz_area_inv_a2: float,
    *,
    n_bands: int = 1,
    spin: int = 2,
) -> float:
    """Carriers per unit cell from pocket area (Luttinger).

    ``n = (A_pocket / A_BZ) × n_bands × spin``. For a hole pocket, caller
    should negate the result if signed counts are required.
    """
    if bz_area_inv_a2 <= 0:
        return float("nan")
    return float(area_inv_a2 / bz_area_inv_a2 * float(n_bands) * float(spin))


def pocket_topology(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    contour: np.ndarray,
    n_rays: int = 8,
    *,
    neighbor_brightness_ratio: float = 0.85,
) -> tuple[PocketTopology, float, int]:
    """Classify pocket as electron/hole from inside-vs-outside intensity.

    ``electron`` means intensity is stronger inside the contour than outside;
    ``hole`` means the inverse.  Confidence is the vote imbalance in ``[0, 1]``.
    Rays whose outside probe is as bright as the inside probe
    (``i_out >= neighbor_brightness_ratio × i_in``) are dropped: the probe
    likely fell on a neighbor pocket so it cannot vote reliably.
    Returns ``(label, confidence, rays_used)``.
    """
    x, y, z = _as_axes(kx, ky, image)
    c = _close_contour(contour)
    center = _centroid(c)
    vecs = c[:-1] - center
    radii = np.linalg.norm(vecs, axis=1)
    r0 = float(np.nanmedian(radii))
    if not np.isfinite(r0) or r0 <= 0:
        return "unclear", 0.0, 0
    ratio = float(neighbor_brightness_ratio)
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_rays), endpoint=False)
    samples: list[tuple[float, float]] = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        inside = center + direction * (0.55 * r0)
        outside = center + direction * (1.35 * r0)
        i_in = _interp_image(z, x, y, inside)
        i_out = _interp_image(z, x, y, outside)
        if np.isfinite(i_in) and np.isfinite(i_out):
            samples.append((float(i_in), float(i_out)))
    if not samples:
        return "unclear", 0.0, 0
    outs = np.array([s[1] for s in samples], dtype=float)
    out_med = float(np.median(outs))
    out_mad = float(np.median(np.abs(outs - out_med))) or 1e-12
    neighbor_cutoff = out_med + 4.0 * out_mad

    votes: list[int] = []
    for i_in, i_out in samples:
        if i_out > neighbor_cutoff and i_out > ratio * i_in:
            continue
        delta = i_in - i_out
        if abs(delta) <= 1e-12:
            continue
        votes.append(1 if delta > 0 else -1)

    if not votes:
        return "unclear", 0.0, 0
    score = float(np.mean(votes))
    confidence = abs(score)
    rays_used = len(votes)
    if confidence < 0.25:
        return "unclear", confidence, rays_used
    return ("electron" if score > 0 else "hole"), confidence, rays_used


def fit_pocket_ellipse(contour: np.ndarray) -> tuple[float, float, float]:
    """Fit an ellipse approximation by PCA: ``(major, minor, angle_deg)``."""
    c = _close_contour(contour)[:-1]
    if c.shape[0] < 4:
        return 0.0, 0.0, 0.0
    centered = c - np.nanmean(c, axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    axes = np.sqrt(np.maximum(vals, 0.0) * 2.0)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    angle = (angle + 90.0) % 180.0 - 90.0
    return float(axes[0]), float(axes[1]), float(angle)


def assign_hs_label(
    centroid: tuple[float, float],
    hs_points: HsPointMap | list[tuple[str, float, float] | tuple[str, tuple[float, float]]],
) -> tuple[str, float]:
    """Return nearest high-symmetry label and distance.

    ``hs_points`` accepte 3 formats :
    - dict[label, (kx, ky)]                : 1 position par label
    - dict[label, list[(kx, ky)]]          : plusieurs copies du même label
    - list/iterable of (label, kx, ky)     : forme étendue (sans dédup)

    Indispensable pour les BZ carrée/hexagonale où il y a 4 X, 4 M, etc.
    """
    if not hs_points:
        return "", float("nan")
    c = np.asarray(centroid, dtype=float)
    candidates: list[tuple[str, tuple[float, float]]] = []
    if isinstance(hs_points, dict):
        for label, val in hs_points.items():
            if val is None:
                continue
            arr = np.asarray(val, dtype=float)
            if arr.size == 0:
                continue
            if arr.ndim == 1 and arr.size == 2:
                candidates.append((str(label), (float(arr[0]), float(arr[1]))))
            elif arr.ndim == 2 and arr.shape[1] == 2:
                for row in arr:
                    candidates.append((str(label), (float(row[0]), float(row[1]))))
    else:
        for item in hs_points:
            if item is None:
                continue
            if len(item) == 3:
                lab, x, y = item
                candidates.append((str(lab), (float(x), float(y))))
            elif len(item) == 2:
                pt = np.asarray(item[1], dtype=float)
                if pt.shape == (2,):
                    candidates.append((str(item[0]), (float(pt[0]), float(pt[1]))))
    best_label = ""
    best_dist = float("inf")
    for label, point in candidates:
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            continue
        d = float(np.linalg.norm(c - p))
        if d < best_dist:
            best_label = label
            best_dist = d
    return best_label, best_dist


def _properties_from_contour(
    image,
    kx,
    ky,
    contour: np.ndarray,
    *,
    bz_polygon,
    hs_points: HsPointMap,
    n_bands: int,
    spin: int,
    hs_dir_x_deg: float,
    hs_dir_m_deg: float,
    hs_dir_tol_deg: float,
    analysis_mode: str,
    mdc_valid_directions: int = 0,
    mdc_total_directions: int = 0,
) -> PocketProperties:
    """Compute pocket metrics once a physical contour has been selected."""
    contour = _close_contour(np.asarray(contour, dtype=float))
    center = _centroid(contour)
    area = abs(pocket_area(contour))
    bz = np.asarray(bz_polygon, dtype=float)
    if bz.ndim != 2 or bz.shape[1] != 2 or bz.shape[0] < 3:
        raise ValueError("bz_polygon doit contenir au moins 3 points (kx, ky).")
    bz_area = abs(pocket_area(bz))
    if bz_area <= 0.0:
        raise ValueError("bz_polygon a une aire nulle.")
    area_pct = float(100.0 * area / bz_area) if bz_area > 0 else float("nan")
    radii = np.linalg.norm(_close_contour(contour)[:-1] - center, axis=1)
    kf_mean = float(np.nanmean(radii)) if radii.size else 0.0
    kf_a, kf_b, angle = fit_pocket_ellipse(contour)
    topology, confidence, rays_used = pocket_topology(image, kx, ky, contour)
    hs_label, hs_dist = assign_hs_label((float(center[0]), float(center[1])), hs_points)
    center_t = (float(center[0]), float(center[1]))
    kf_gx = kf_along_direction(contour, center_t, hs_dir_x_deg, hs_dir_tol_deg)
    kf_gm = kf_along_direction(contour, center_t, hs_dir_m_deg, hs_dir_tol_deg)
    if kf_a > 0 and kf_b >= 0:
        ratio = max(kf_a, kf_b) / max(min(kf_a, kf_b), 1e-12)
        ecc = float(np.sqrt(max(0.0, 1.0 - (min(kf_a, kf_b) / max(kf_a, kf_b)) ** 2)))
    else:
        ratio = float("nan")
        ecc = float("nan")
    curv_mean, curv_var = pocket_curvature(contour)
    n_carriers = luttinger_count(area, bz_area, n_bands=n_bands, spin=spin)
    if topology == "hole" and np.isfinite(n_carriers):
        n_carriers = -abs(n_carriers)
    return PocketProperties(
        centroid_kx=float(center[0]),
        centroid_ky=float(center[1]),
        area_inv_a2=float(area),
        area_pct_bz=area_pct,
        kF_mean=kf_mean,
        kF_a=kf_a,
        kF_b=kf_b,
        ellipse_angle_deg=float(angle),
        topology=topology,
        topology_confidence=float(confidence),
        hs_label_nearest=hs_label,
        hs_distance=float(hs_dist),
        kF_gamma_x=float(kf_gx),
        kF_gamma_m=float(kf_gm),
        aspect_ratio=float(ratio),
        eccentricity=float(ecc),
        curvature_mean=float(curv_mean),
        curvature_var=float(curv_var),
        n_carriers_2D=float(n_carriers),
        topology_rays_used=int(rays_used),
        analysis_mode=str(analysis_mode),
        mdc_valid_directions=int(mdc_valid_directions),
        mdc_total_directions=int(mdc_total_directions),
    )


def characterize_pocket(
    image,
    kx,
    ky,
    *,
    seed_point: tuple[float, float],
    level: float,
    bz_polygon,
    hs_points: HsPointMap,
    contour_window: int = 9,
    n_bands: int = 1,
    spin: int = 2,
    hs_dir_x_deg: float = 0.0,
    hs_dir_m_deg: float = 45.0,
    hs_dir_tol_deg: float = 10.0,
    publication: bool = True,
    mdc_n_directions: int = 36,
    mdc_r_max: float | None = None,
    mdc_n_points: int = 64,
    mdc_r2_min: float = 0.5,
) -> PocketProperties:
    """Complete pocket characterization pipeline.

    In publication mode, kF is extracted by radial MDC fits. The old
    iso-contour path remains as a preview fallback because it depends on an
    arbitrary intensity level.
    """
    if publication:
        try:
            from arpes.physics.pocket_mdc_radial import characterize_pocket_mdc_radial

            contour, mdc_results, _center = characterize_pocket_mdc_radial(
                image, kx, ky,
                seed_point=seed_point,
                n_directions=mdc_n_directions,
                r_max=mdc_r_max,
                n_points=mdc_n_points,
                r2_min=mdc_r2_min,
                refine_center=True,
            )
            props = _properties_from_contour(
                image, kx, ky, contour,
                bz_polygon=bz_polygon,
                hs_points=hs_points,
                n_bands=n_bands,
                spin=spin,
                hs_dir_x_deg=hs_dir_x_deg,
                hs_dir_m_deg=hs_dir_m_deg,
                hs_dir_tol_deg=hs_dir_tol_deg,
                analysis_mode="mdc_radial",
                mdc_valid_directions=sum(1 for r in mdc_results if r.ok),
                mdc_total_directions=len(mdc_results),
            )
            dx = float(np.nanmedian(np.diff(np.asarray(kx, dtype=float))))
            dy = float(np.nanmedian(np.diff(np.asarray(ky, dtype=float))))
            if props.kF_mean <= 3.0 * max(abs(dx), abs(dy)):
                raise ValueError("MDC-radial : rayon trop proche du pas de grille.")
            return props
        except Exception:
            pass

    contour = smooth_closed_contour(
        extract_fs_contour(image, kx, ky, level, seed_point=seed_point),
        window=contour_window,
    )
    return _properties_from_contour(
        image, kx, ky, contour,
        bz_polygon=bz_polygon,
        hs_points=hs_points,
        n_bands=n_bands,
        spin=spin,
        hs_dir_x_deg=hs_dir_x_deg,
        hs_dir_m_deg=hs_dir_m_deg,
        hs_dir_tol_deg=hs_dir_tol_deg,
        analysis_mode="isocontour_preview" if publication else "isocontour_preview",
    )


_SCALAR_FIELDS = (
    "centroid_kx", "centroid_ky", "area_inv_a2", "area_pct_bz",
    "kF_mean", "kF_a", "kF_b", "ellipse_angle_deg",
    "topology_confidence", "hs_distance",
    "kF_gamma_x", "kF_gamma_m", "aspect_ratio", "eccentricity",
    "curvature_mean", "curvature_var", "n_carriers_2D",
)


@dataclass(frozen=True)
class PocketBootstrap:
    central: PocketProperties
    std: dict[str, float] = field(default_factory=dict)
    n_valid: int = 0
    n_total: int = 0

    def asdict(self) -> dict:
        out = self.central.asdict()
        out["uncertainty"] = dict(self.std)
        out["n_bootstrap_valid"] = int(self.n_valid)
        out["n_bootstrap_total"] = int(self.n_total)
        return out


def characterize_pocket_bootstrap(
    image,
    kx,
    ky,
    *,
    seed_point: tuple[float, float],
    level: float,
    bz_polygon,
    hs_points: HsPointMap,
    smooth_sigma: tuple[float, float] = (1.0, 3.0),
    n_bootstrap: int = 20,
    level_rel_jitter: float = 0.10,
    smooth_rel_jitter: float = 0.25,
    rng=None,
    **characterize_kwargs,
) -> PocketBootstrap:
    """Repeat ``characterize_pocket`` with jittered level and smoothing.

    Each iteration draws ``level' = level × (1 + U[-r, r])`` and
    ``σ' = σ × (1 + U[-s, s])`` independently per axis. Returns the median
    pocket as ``central`` and per-scalar standard deviation as ``std``.
    Categorical fields (``topology``, ``hs_label_nearest``) take the mode.
    Iterations that fail (level out of range, no closed contour) are skipped.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_total = int(max(1, n_bootstrap))
    lvl_r = float(max(0.0, level_rel_jitter))
    sm_r = float(max(0.0, smooth_rel_jitter))
    sigma_y0, sigma_x0 = float(smooth_sigma[0]), float(smooth_sigma[1])
    runs: list[PocketProperties] = []
    for _ in range(n_total):
        l_jit = float(level) * (1.0 + rng.uniform(-lvl_r, lvl_r))
        sy = max(0.0, sigma_y0 * (1.0 + rng.uniform(-sm_r, sm_r)))
        sx = max(0.0, sigma_x0 * (1.0 + rng.uniform(-sm_r, sm_r)))
        try:
            smoothed = smooth_fs_image(image, sigma=(sy, sx))
            props = characterize_pocket(
                smoothed, kx, ky,
                seed_point=seed_point,
                level=l_jit,
                bz_polygon=bz_polygon,
                hs_points=hs_points,
                **characterize_kwargs,
            )
        except (ValueError, RuntimeError):
            continue
        runs.append(props)
    if not runs:
        empty = characterize_pocket(
            smooth_fs_image(image, sigma=(sigma_y0, sigma_x0)), kx, ky,
            seed_point=seed_point, level=float(level),
            bz_polygon=bz_polygon, hs_points=hs_points,
            **characterize_kwargs,
        )
        return PocketBootstrap(central=empty, std={}, n_valid=0, n_total=n_total)

    medians: dict[str, float] = {}
    stds: dict[str, float] = {}
    for f in _SCALAR_FIELDS:
        vals = np.array([getattr(p, f) for p in runs], dtype=float)
        finite = vals[np.isfinite(vals)]
        medians[f] = float(np.nanmedian(finite)) if finite.size else float("nan")
        stds[f] = float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0
    topology_mode = Counter(p.topology for p in runs).most_common(1)[0][0]
    hs_mode = Counter(p.hs_label_nearest for p in runs).most_common(1)[0][0]
    rays_med = int(np.median([p.topology_rays_used for p in runs]))
    central = PocketProperties(
        centroid_kx=medians["centroid_kx"],
        centroid_ky=medians["centroid_ky"],
        area_inv_a2=medians["area_inv_a2"],
        area_pct_bz=medians["area_pct_bz"],
        kF_mean=medians["kF_mean"],
        kF_a=medians["kF_a"],
        kF_b=medians["kF_b"],
        ellipse_angle_deg=medians["ellipse_angle_deg"],
        topology=topology_mode,
        topology_confidence=medians["topology_confidence"],
        hs_label_nearest=hs_mode,
        hs_distance=medians["hs_distance"],
        kF_gamma_x=medians["kF_gamma_x"],
        kF_gamma_m=medians["kF_gamma_m"],
        aspect_ratio=medians["aspect_ratio"],
        eccentricity=medians["eccentricity"],
        curvature_mean=medians["curvature_mean"],
        curvature_var=medians["curvature_var"],
        n_carriers_2D=medians["n_carriers_2D"],
        topology_rays_used=rays_med,
    )
    return PocketBootstrap(central=central, std=stds, n_valid=len(runs), n_total=n_total)
