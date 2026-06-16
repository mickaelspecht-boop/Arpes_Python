"""Manual FS pocket contour controller path."""
from __future__ import annotations

import numpy as np

from arpes.physics.fs import extract_fs_map
from arpes.physics.pocket import smooth_fs_image
from arpes.physics.pocket_manual import (
    characterize_manual_contour,
    snap_manual_contour_points,
)
from arpes.physics.pocket_quality import run_pocket_guards


def characterize_manual_contour_at(ctrl, payload: dict):
    if ctrl._raw_data is None or not ctrl._current_is_fs():
        ctrl._status("Manual contour: load an FS map first.")
        return None
    entry = ctrl._current_entry()
    if entry is None:
        return None
    try:
        points_plot = np.asarray(payload.get("points") or [], dtype=float)
        if points_plot.ndim != 2 or points_plot.shape[1] != 2 or points_plot.shape[0] < 5:
            ctrl._status("Manual contour: place at least 5 crosses.")
            return None
        params = ctrl._fs_controls.params()
        settings = ctrl._pocket_settings()
        center = np.array([float(params.kx_center), float(params.ky_center)])
        points_raw = points_plot + center
        kx, ky, fs, _ = extract_fs_map(ctrl._raw_data, params)
        sigma = (settings["smooth_sigma_y"], settings["smooth_sigma_x"])
        fs_pocket = smooth_fs_image(fs, sigma=sigma)
        snap = bool(payload.get("snap", True))
        used_raw = (
            snap_manual_contour_points(fs_pocket, kx, ky, points_raw, radius_px=2)
            if snap else np.asarray(points_raw, dtype=float)
        )
        props, contour_raw = characterize_manual_contour(
            fs_pocket,
            kx,
            ky,
            used_raw,
            bz_polygon=ctrl._bz_polygon_raw(params),
            hs_points=ctrl._hs_points_raw(params),
            n_bands=int(settings.get("n_bands", 1)),
            spin=int(settings.get("spin", 2)),
            hs_dir_x_deg=float(settings.get("hs_dir_x_deg", 0.0)),
            hs_dir_m_deg=float(settings.get("hs_dir_m_deg", 45.0)),
            hs_dir_tol_deg=float(settings.get("hs_dir_tol_deg", 10.0)),
            contour_window=min(7, int(settings["contour_window"])),
            simplify_step=0.0,
        )
        pocket = props.asdict()
        contour_shifted = np.asarray(contour_raw, dtype=float).copy()
        contour_shifted[:, 0] -= center[0]
        contour_shifted[:, 1] -= center[1]
        pocket["contour"] = contour_shifted.tolist()
        pocket["manual_points"] = (used_raw - center).tolist()
        pocket["level"] = float("nan")
        pocket["algo"] = "manual_contour"
        pocket["processing"] = {
            "quality": settings.get("quality", "Standard"),
            "smooth_sigma_yx": [settings["smooth_sigma_y"], settings["smooth_sigma_x"]],
            "snap": snap,
            "n_points": int(points_plot.shape[0]),
        }
        seed_raw = (
            float(np.nanmean(used_raw[:, 0])),
            float(np.nanmean(used_raw[:, 1])),
        )
        guards = run_pocket_guards(
            image=fs_pocket,
            kx=kx,
            ky=ky,
            seed_point=seed_raw,
            contour=contour_raw,
            sigma_pixels=sigma,
            kf_mean=float(pocket.get("kF_mean") or 0.0),
        )
        pocket["quality_checks"] = [g.__dict__ for g in guards]
        blocking = [g for g in guards if not g.ok]
        if blocking:
            msgs = " | ".join(g.message for g in blocking if g.message)
            ctrl._status(f"Manual contour rejected: {msgs}")
            return None
        if float(pocket.get("area_pct_bz", 0.0) or 0.0) < settings["min_area_pct_bz"]:
            ctrl._status(
                f"Manual contour rejected: area {pocket['area_pct_bz']:.2f}% BZ "
                f"< min {settings['min_area_pct_bz']:.2f}%."
            )
            return None
        mp_label = ctrl._mp_label_for(pocket)
        if mp_label:
            pocket["mp_label"] = mp_label
        ctrl._attach_dft_compare(pocket, entry, params)
        entry.fs_pockets = list(getattr(entry, "fs_pockets", []) or []) + [pocket]
        ctrl._session.save()
        ctrl._draw_fs_tab()
        idx = len(entry.fs_pockets) - 1
        from arpes.ui.controllers import pocket_controller as pc_mod

        dialog = pc_mod.PocketResultDialog(ctrl._parent, pocket, allow_delete=True)
        dialog.exec()
        if getattr(dialog, "delete_requested", False):
            ctrl._delete_pocket({"index": idx})
            return None
        ctrl._status(
            f"Manual contour: {pocket['hs_label_nearest'] or '?'} "
            f"{pocket['area_pct_bz']:.2f}% BZ, {pocket['topology']}."
        )
        return pocket
    except Exception as exc:
        ctrl._status(f"Manual contour: {exc}")
        return None
