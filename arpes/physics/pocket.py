"""Fermi-pocket characterization helpers.

Pure numerical layer: no Qt, no Materials Project dependency.  The module
extracts an iso-intensity contour from an experimental FS map and derives
simple geometric descriptors useful before comparing to DFT/MP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import contourpy
import numpy as np
from matplotlib.path import Path as MplPath

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - scipy is present in the app env
    gaussian_filter = None


PocketTopology = Literal["electron", "hole", "unclear"]


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


def pocket_topology(
    image: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    contour: np.ndarray,
    n_rays: int = 8,
) -> tuple[PocketTopology, float]:
    """Classify pocket as electron/hole from inside-vs-outside intensity.

    ``electron`` means intensity is stronger inside the contour than outside;
    ``hole`` means the inverse.  Confidence is the vote imbalance in ``[0, 1]``.
    """
    x, y, z = _as_axes(kx, ky, image)
    c = _close_contour(contour)
    center = _centroid(c)
    vecs = c[:-1] - center
    radii = np.linalg.norm(vecs, axis=1)
    r0 = float(np.nanmedian(radii))
    if not np.isfinite(r0) or r0 <= 0:
        return "unclear", 0.0

    votes: list[int] = []
    for angle in np.linspace(0.0, 2.0 * np.pi, int(n_rays), endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        inside = center + direction * (0.55 * r0)
        outside = center + direction * (1.35 * r0)
        i_in = _interp_image(z, x, y, inside)
        i_out = _interp_image(z, x, y, outside)
        if not (np.isfinite(i_in) and np.isfinite(i_out)):
            continue
        delta = i_in - i_out
        if abs(delta) <= 1e-12:
            continue
        votes.append(1 if delta > 0 else -1)

    if not votes:
        return "unclear", 0.0
    score = float(np.mean(votes))
    confidence = abs(score)
    if confidence < 0.25:
        return "unclear", confidence
    return ("electron" if score > 0 else "hole"), confidence


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
    hs_points: dict[str, tuple[float, float]],
) -> tuple[str, float]:
    """Return nearest high-symmetry label and distance."""
    if not hs_points:
        return "", float("nan")
    c = np.asarray(centroid, dtype=float)
    best_label = ""
    best_dist = float("inf")
    for label, point in hs_points.items():
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            continue
        d = float(np.linalg.norm(c - p))
        if d < best_dist:
            best_label = str(label)
            best_dist = d
    return best_label, best_dist


def characterize_pocket(
    image,
    kx,
    ky,
    *,
    seed_point: tuple[float, float],
    level: float,
    bz_polygon,
    hs_points: dict[str, tuple[float, float]],
) -> PocketProperties:
    """Complete pocket characterization pipeline."""
    contour = smooth_closed_contour(
        extract_fs_contour(image, kx, ky, level, seed_point=seed_point)
    )
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
    topology, confidence = pocket_topology(image, kx, ky, contour)
    hs_label, hs_dist = assign_hs_label((float(center[0]), float(center[1])), hs_points)
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
    )
