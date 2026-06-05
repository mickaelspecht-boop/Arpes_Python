"""Angle-offset loading helpers for ArpesExplorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from arpes.io.loaders import load_arpes_file
from arpes.physics.gamma import (
    angle_offset_candidates_for_load as _gamma_angle_offset_candidates,
    score_bm_gamma_residual as _gamma_score_bm_residual,
)


def angle_offsets_for_load(win, path: str | Path, entry, hv: float | None) -> dict:
    """Retourne les offsets angulaires a injecter dans le loader CLS."""
    ref = win._stored_gamma_reference()
    if not ref:
        return win._session.angle_offsets or {}

    p = Path(path)
    is_cls_bm_file = p.is_file() and (p.parent / f"{p.name}_param.txt").exists()
    is_cls_fs_dir = p.is_dir()
    geom = win._cls_geometry_for_path(p, entry)
    if is_cls_bm_file:
        azi_bm = geom.get("azi", entry.meta.azi if (entry and entry.meta.azi is not None) else None)
        gamma_bm, _ = win._project_gamma_by_azi(
            ref, azi_bm, warn_label="Γ référence → BM"
        )
        if not np.isfinite(gamma_bm):
            return {}
        offsets = win._angle_offsets_from_k_center(
            float(gamma_bm), 0.0,
            hv=hv,
            source="gamma_reference_projected_to_bm",
            ref_path=ref.get("path"),
            azi=azi_bm,
        )
        if offsets:
            offsets["gamma_bm_pi_over_a"] = float(gamma_bm)
            offsets["gamma_ref_source"] = ref.get("source", "")
            offsets["target_polar"] = geom.get("polar")
            offsets["target_tilt"] = geom.get("tilt")
            p_ref = ref.get("polar")
            p_target = geom.get("polar")
            if p_ref is not None and p_target is not None:
                offsets["theta0_deg"] = (
                    float(offsets.get("theta0_deg", 0.0) or 0.0)
                    + float(p_ref)
                    - float(p_target)
                )
                offsets["source"] = "gamma_reference_projected_to_bm_raw_polar"
                offsets["ref_polar"] = float(p_ref)
            return offsets

    if is_cls_fs_dir:
        azi_fs = geom.get("azi", entry.meta.azi if (entry and entry.meta.azi is not None) else None)
        gamma_kx, gamma_ky = win._project_gamma_by_azi(
            ref, azi_fs, warn_label="Γ référence → FS"
        )
        if not np.isfinite(gamma_kx) or not np.isfinite(gamma_ky):
            return {}
        offsets = win._angle_offsets_from_k_center(
            float(gamma_kx), float(gamma_ky),
            hv=hv,
            source="gamma_reference_projected_to_fs",
            ref_path=ref.get("path"),
            azi=azi_fs,
        )
        if offsets:
            offsets["gamma_fs_kx_pi_over_a"] = float(gamma_kx)
            offsets["gamma_fs_ky_pi_over_a"] = float(gamma_ky)
            offsets["gamma_ref_source"] = ref.get("source", "")
            offsets["target_polar"] = geom.get("polar")
            offsets["target_tilt"] = geom.get("tilt")
            return offsets

    return {}


def angle_offset_candidates_for_load(
    win,
    path: str | Path,
    entry,
    hv: float | None,
    primary: dict,
) -> list[dict]:
    """Wrapper UI : délègue à `arpes_gamma.angle_offset_candidates_for_load`."""
    target_geom = (
        win._cls_geometry_for_path(path, entry)
        if (entry is not None and Path(path).is_file()) else None
    )
    target_azi_fallback = (
        entry.meta.azi if (entry is not None and entry.meta.azi is not None) else None
    )
    return _gamma_angle_offset_candidates(
        primary=primary,
        is_file=Path(path).is_file(),
        ref=win._stored_gamma_reference() or None,
        target_geom=target_geom,
        target_azi_fallback=target_azi_fallback,
        hv=hv,
        work_func=float(win._params.sp_phi.value()),
    )


def score_bm_gamma_residual(win, d: dict) -> float:
    """Wrapper UI : délègue à `arpes_gamma.score_bm_gamma_residual`."""
    if win.ap is None:
        return float("inf")
    return _gamma_score_bm_residual(
        d,
        ev_range=(win._params.sp_evs.value(), win._params.sp_eve.value()),
        k_range=(win._params.sp_kmin.value(), win._params.sp_kmax.value()),
        center_window=win._params.sp_xg.value() * 2.0,
        smooth_sigma=win._params.sp_sfd.value(),
        estimate_fn=win.ap.estimate_gamma_bm_mdc,
    )


def load_with_best_angle_offsets(
    win,
    path: str,
    entry,
    hv_for_load: float,
    angle_offsets: dict,
    a_lattice: float | None = None,
) -> tuple[dict | None, dict]:
    """Charge une BM CLS avec la convention d'offset qui centre le mieux Γ."""
    candidates = win._angle_offset_candidates_for_load(path, entry, hv_for_load, angle_offsets)
    if len(candidates) <= 1:
        d = _load_file_with_offsets(win, path, entry, hv_for_load, angle_offsets, a_lattice)
        return d, angle_offsets

    best_d = None
    best_cfg = candidates[0]
    best_score = float("inf")
    for cfg in candidates:
        d_try = _load_file_with_offsets(win, path, entry, hv_for_load, cfg, a_lattice)
        if d_try is None:
            continue
        score = win._score_bm_gamma_residual(d_try)
        if score < best_score:
            best_score = score
            best_d = d_try
            best_cfg = cfg

    if best_d is not None and np.isfinite(best_score):
        try:
            md = best_d.get("metadata", {}) or {}
            md["angle_offset_candidate_score"] = float(best_score)
            md["angle_offset_candidate"] = best_cfg.get("candidate", "")
            best_d["metadata"] = md
        except Exception:
            pass
        return best_d, best_cfg

    d = _load_file_with_offsets(win, path, entry, hv_for_load, angle_offsets, a_lattice)
    return d, angle_offsets


def _load_file_with_offsets(
    win,
    path: str,
    entry,
    hv_for_load: float,
    angle_offsets: dict,
    a_lattice: float | None = None,
):
    lattice_kwargs = {}
    if a_lattice is not None and float(a_lattice) > 0:
        lattice_kwargs["a_lattice"] = float(a_lattice)
    return load_arpes_file(
        path, win._params.sp_phi.value(), win._params.sp_ef.value(),
        **lattice_kwargs,
        hv=hv_for_load,
        temperature=entry.meta.temperature if entry.meta.temperature > 0 else None,
        azi=entry.meta.azi,
        pol=entry.meta.polarization,
        angle_offsets=angle_offsets,
        bessy_energy_reference=win._bessy_energy_reference_mode(),
    )
