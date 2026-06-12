"""Bootstrap pocket characterization by jittering level and smoothing.

Replays ``characterize_pocket`` with jittered iso-level and smoothing σ, then
aggregates median and σ for scalar outputs. Boolean and categorical flags are
aggregated separately.

No PyQt. Pure numpy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from arpes.physics.pocket import (
    HsPointMap,
    PocketProperties,
    characterize_pocket,
    smooth_fs_image,
)

# Scalar fields aggregated by median/σ. Bool and string fields are aggregated
# separately because a median is not meaningful for categorical values.
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
    Iterations that fail (level out of range, no closed contour, arc too
    open → ``PocketFitRefusedError``) are skipped.
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
    # P2.4 — flags outside _SCALAR_FIELDS: explicit aggregation (redteam).
    extrap_any = bool(any(p.is_extrapolated for p in runs))
    ellipse_valid_all = bool(all(p.ellipse_fit_valid for p in runs))
    coverage_med = float(np.nanmedian([p.arc_coverage_deg for p in runs]))
    ka_sig_med = float(np.nanmedian([p.kF_a_sigma for p in runs]))
    kb_sig_med = float(np.nanmedian([p.kF_b_sigma for p in runs]))
    method_mode = Counter(p.fit_method for p in runs).most_common(1)[0][0]
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
        is_extrapolated=extrap_any,
        ellipse_fit_valid=ellipse_valid_all,
        arc_coverage_deg=coverage_med,
        kF_a_sigma=ka_sig_med,
        kF_b_sigma=kb_sig_med,
        fit_method=method_mode,
    )
    return PocketBootstrap(central=central, std=stds, n_valid=len(runs), n_total=n_total)
