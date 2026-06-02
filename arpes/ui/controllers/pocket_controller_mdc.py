"""MDC-radial pocket characterization, free-function pipeline.

Extracted from ``pocket_controller.py`` to keep that file under the 700 LOC cap.
``ctrl`` is the ``PocketController`` instance (proxies to the parent window).
"""
from __future__ import annotations

import numpy as np

from arpes.physics.fs import extract_fs_map
from arpes.physics.pocket import (
    _close_contour,
    assign_hs_label,
    fit_pocket_ellipse,
    kf_along_direction,
    luttinger_count,
    pocket_area,
    pocket_curvature,
    pocket_topology,
    smooth_fs_image,
)
from arpes.physics.pocket_mdc_radial import (
    arc_coverage_deg,
    characterize_pocket_mdc_radial,
)
from arpes.physics.pocket_quality import contour_touches_border
from arpes.physics.pocket_quality import run_pocket_guards
from arpes.ui.widgets.dialogs.pocket_result import PocketResultDialog


def characterize_mdc_at(ctrl, payload: dict):
    if ctrl._raw_data is None or not ctrl._current_is_fs():
        ctrl._status("Poche MDC : charge une carte FS d'abord.")
        return None
    entry = ctrl._current_entry()
    if entry is None:
        return None
    try:
        seed_plot = (float(payload["kx"]), float(payload["ky"]))
        params = ctrl._fs_controls.params()
        settings = ctrl._pocket_settings()
        seed_raw = (seed_plot[0] + float(params.kx_center),
                    seed_plot[1] + float(params.ky_center))
        kx, ky, fs, _ = extract_fs_map(ctrl._raw_data, params)
        sigma = (settings["smooth_sigma_y"], settings["smooth_sigma_x"])
        fs_pocket = smooth_fs_image(fs, sigma=sigma)
        n_dirs = int(payload.get("n_directions", settings.get("mdc_n_directions", 36)))
        r2_min = float(payload.get("r2_min", settings.get("mdc_r2_min", 0.5)))
        contour_raw, results, center = characterize_pocket_mdc_radial(
            fs_pocket, kx, ky,
            seed_point=seed_raw, n_directions=n_dirs, r2_min=r2_min,
        )
        bz_raw = ctrl._bz_polygon_raw(params)
        hs_raw = ctrl._hs_points_raw(params)
        contour_closed = _close_contour(contour_raw)
        area = abs(pocket_area(contour_closed))
        bz_area = abs(pocket_area(_close_contour(bz_raw)))
        kf_a, kf_b, angle = fit_pocket_ellipse(contour_closed)
        kf_per_dir = [r.kF for r in results if r.ok]
        kf_std_per_dir = [r.kF_std for r in results if r.ok and np.isfinite(r.kF_std)]
        kf_mean = float(np.nanmean(kf_per_dir)) if kf_per_dir else float("nan")
        kf_std_agg = (
            float(np.sqrt(np.nanmean(np.asarray(kf_std_per_dir) ** 2)))
            if kf_std_per_dir else float("nan")
        )
        topology, conf, rays = pocket_topology(fs_pocket, kx, ky, contour_closed)
        hs_label, hs_dist = assign_hs_label(center, hs_raw)
        curv_mean, curv_var = pocket_curvature(contour_closed)
        tol = float(settings.get("hs_dir_tol_deg", 10.0))
        kf_gx = kf_along_direction(contour_closed, center,
                                   float(settings.get("hs_dir_x_deg", 0.0)), tol)
        kf_gm = kf_along_direction(contour_closed, center,
                                   float(settings.get("hs_dir_m_deg", 45.0)), tol)
        if kf_a > 0 and kf_b > 0:
            ratio = max(kf_a, kf_b) / max(min(kf_a, kf_b), 1e-12)
            ecc = float(np.sqrt(max(0.0, 1.0 - (min(kf_a, kf_b) / max(kf_a, kf_b)) ** 2)))
        else:
            ratio = float("nan"); ecc = float("nan")
        coverage_deg = arc_coverage_deg(results)
        on_border = contour_touches_border(contour_closed, kx, ky)
        force_arc = bool(payload.get("force_arc", False))
        closed = (not on_border) and (coverage_deg >= 355.0) and (not force_arc)
        n_carriers = (
            luttinger_count(area, bz_area,
                            n_bands=int(settings.get("n_bands", 1)),
                            spin=int(settings.get("spin", 2)))
            if closed else float("nan")
        )
        shifted = contour_closed.copy()
        shifted[:, 0] -= float(params.kx_center); shifted[:, 1] -= float(params.ky_center)
        pocket = {
            "centroid_kx": float(center[0]), "centroid_ky": float(center[1]),
            "area_inv_a2": float(area) if closed else float("nan"),
            "area_pct_bz": (float(100.0 * area / bz_area) if (bz_area > 0 and closed) else float("nan")),
            "closed": bool(closed),
            "arc_coverage_deg": float(coverage_deg),
            "kF_mean": kf_mean, "kF_mean_std": kf_std_agg,
            "kF_a": kf_a, "kF_b": kf_b, "ellipse_angle_deg": float(angle),
            "topology": topology, "topology_confidence": float(conf),
            "topology_rays_used": int(rays),
            "hs_label_nearest": hs_label, "hs_distance": float(hs_dist),
            "kF_gamma_x": float(kf_gx), "kF_gamma_m": float(kf_gm),
            "aspect_ratio": float(ratio), "eccentricity": float(ecc),
            "curvature_mean": float(curv_mean), "curvature_var": float(curv_var),
            "n_carriers_2D": float(n_carriers),
            "contour": shifted.tolist(), "level": float("nan"),
            "algo": "mdc_radial",
            "n_directions": int(n_dirs),
            "n_valid_directions": int(sum(1 for r in results if r.ok)),
            "per_direction": [r.asdict() for r in results],
            "processing": {"smooth_sigma_yx": [sigma[0], sigma[1]], "r2_min": r2_min},
        }
        guards = run_pocket_guards(
            image=fs_pocket, kx=kx, ky=ky, seed_point=seed_raw,
            contour=contour_closed, sigma_pixels=sigma, kf_mean=kf_mean,
        )
        if not closed:
            # Arc mode : border is expected, downgrade to non-blocking.
            guards = [
                (
                    g.__class__(ok=True, code=g.code + "_arc",
                                message="(arc mode) " + g.message, metric=g.metric)
                    if g.code == "border" else g
                )
                for g in guards
            ]
        pocket["quality_checks"] = [g.__dict__ for g in guards]
        blocking = [g for g in guards if not g.ok]
        if blocking:
            msgs = " | ".join(g.message for g in blocking if g.message)
            ctrl._status(f"Poche MDC rejetée : {msgs}")
            return None
        ctrl._attach_dft_compare(pocket, entry, params)
        entry.fs_pockets = list(getattr(entry, "fs_pockets", []) or []) + [pocket]
        ctrl._session.save()
        ctrl._draw_fs_tab()
        idx = len(entry.fs_pockets) - 1
        dialog = PocketResultDialog(ctrl._parent, pocket, allow_delete=True)
        dialog.exec()
        if getattr(dialog, "delete_requested", False):
            ctrl._delete_pocket({"index": idx})
            return None
        badge = "fermée" if closed else f"ARC {coverage_deg:.0f}°"
        ctrl._status(
            f"Poche MDC [{badge}] : {hs_label or '?'} "
            f"kF={kf_mean:.3f}±{kf_std_agg:.3f} "
            f"({pocket['n_valid_directions']}/{n_dirs} dirs), {topology}."
        )
        return pocket
    except Exception as exc:
        ctrl._status(f"Poche MDC : {exc}")
        return None
