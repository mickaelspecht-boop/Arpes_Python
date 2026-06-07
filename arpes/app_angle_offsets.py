"""Angle-offset loading helpers for ArpesExplorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from arpes.io.loaders import load_arpes_file
from arpes.physics.gamma import (
    angle_offset_candidates_for_load as _gamma_angle_offset_candidates,
    score_bm_gamma_residual as _gamma_score_bm_residual,
    score_bm_gamma_residual_detail as _gamma_score_bm_residual_detail,
)
from arpes.physics.angle_convention import (
    filter_candidates as _filter_candidates,
    select_best_candidate as _select_best_candidate,
    evaluate_confidence as _evaluate_confidence,
)


def angle_offsets_for_load(win, path: str | Path, entry, hv: float | None) -> dict:
    """Return the angular offsets to inject into the CLS loader."""
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
            ref, azi_bm, warn_label="Γ reference -> BM"
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
            ref, azi_fs, warn_label="Γ reference -> FS"
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
    work_func: float | None = None,
) -> list[dict]:
    """UI wrapper: delegate to `arpes_gamma.angle_offset_candidates_for_load`."""
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
        work_func=float(work_func if work_func is not None else win._params.sp_phi.value()),
    )


def score_bm_gamma_residual(win, d: dict) -> float:
    """UI wrapper: delegate to `arpes_gamma.score_bm_gamma_residual`."""
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


def _score_detail(win, d: dict, gamma_expected: float = 0.0) -> dict:
    """Raw score details (gamma/mad/n/gamma_residual_after) for confidence.

    ``gamma_expected`` (π/a): expected band center (P2.6c, default 0=Γ).
    """
    if win.ap is None:
        return {"score": float("inf"), "gamma": float("nan"), "mad": 0.0,
                "n": 0, "gamma_residual_after": float("nan")}
    return _gamma_score_bm_residual_detail(
        d,
        ev_range=(win._params.sp_evs.value(), win._params.sp_eve.value()),
        k_range=(win._params.sp_kmin.value(), win._params.sp_kmax.value()),
        center_window=win._params.sp_xg.value() * 2.0,
        smooth_sigma=win._params.sp_sfd.value(),
        estimate_fn=win.ap.estimate_gamma_bm_mdc,
        gamma_expected=float(gamma_expected),
    )


def load_with_best_angle_offsets(
    win,
    path: str,
    entry,
    hv_for_load: float,
    angle_offsets: dict,
    work_func: float | None = None,
    a_lattice: float | None = None,
) -> tuple[dict | None, dict]:
    """Load a CLS BM with the offset convention that best centers Γ."""
    resolved_work_func = float(
        work_func if work_func is not None else win._params.sp_phi.value()
    )
    candidates = win._angle_offset_candidates_for_load(
        path,
        entry,
        hv_for_load,
        angle_offsets,
        work_func=resolved_work_func,
    )
    # P2.6a — restrict to the frozen sign if a beamline convention exists
    # (UNCALIBRATED -> unchanged list = data-driven mode).
    registry = dict(getattr(win._session, "convention_registry", {}) or {})
    beamline = str(getattr(entry.meta, "source_format", "") or "")
    candidates = _filter_candidates(
        candidates, registry,
        beamline=beamline, hv=hv_for_load,
        azi=getattr(entry.meta, "azi", None),
        polar=getattr(entry.meta, "polar", None),
    )
    if len(candidates) <= 1:
        d = _load_file_with_offsets(
            win, path, entry, hv_for_load, angle_offsets, resolved_work_func, a_lattice
        )
        return d, angle_offsets

    # Load and score each candidate once; cache data and raw details.
    loaded_cache: dict[int, dict] = {}
    detail_cache: dict[int, dict] = {}
    # P2.6c — expected band center (π/a): 0 by default (Γ), but a known
    # off-Γ band can set it through meta_gamma_state so the scorer does not
    # choose the wrong sign by assuming Γ_true=0.
    try:
        g_expected = float(entry.meta_gamma_state.get("gamma_expected", 0.0) or 0.0)
    except Exception:
        g_expected = 0.0

    def _score(cfg: dict) -> float:
        key = id(cfg)
        d_try = _load_file_with_offsets(
            win, path, entry, hv_for_load, cfg, resolved_work_func, a_lattice
        )
        loaded_cache[key] = d_try
        if d_try is None:
            detail_cache[key] = {"score": float("inf"), "gamma": float("nan"),
                                 "mad": 0.0, "n": 0, "gamma_residual_after": float("nan")}
            return float("inf")
        det = _score_detail(win, d_try, gamma_expected=g_expected)
        detail_cache[key] = det
        return float(det.get("score", float("inf")))

    sel = _select_best_candidate(candidates, _score)
    best_cfg = sel["best"]
    best_score = sel["best_score"]
    best_d = loaded_cache.get(id(best_cfg)) if best_cfg is not None else None

    if best_d is not None and np.isfinite(best_score):
        det = detail_cache.get(id(best_cfg), {})
        verdict = _evaluate_confidence(
            confidence=sel["confidence"],
            gamma_best=det.get("gamma", float("nan")),
            mad_best=det.get("mad", 0.0),
            gamma_residual_after=det.get("gamma_residual_after", float("nan")),
            tie=sel["tie"],
        )
        new_candidate = best_cfg.get("candidate", "")
        _emit_sign_warnings(win, entry, sel, verdict, det, new_candidate)
        try:
            md = best_d.get("metadata", {}) or {}
            md["angle_offset_candidate_score"] = float(best_score)
            md["angle_offset_candidate"] = new_candidate
            md["angle_offset_candidate_score_2nd"] = float(sel.get("second_score", float("inf")))
            md["angle_offset_confidence"] = float(sel["confidence"])
            md["angle_offset_ambiguous"] = bool(verdict["ambiguous"])
            md["angle_offset_gamma_residual_after"] = float(
                det.get("gamma_residual_after", float("nan"))
            )
            best_d["metadata"] = md
        except Exception:
            pass
        # Persist the selected candidate to detect changes on the next load.
        try:
            entry.meta_gamma_state["angle_offset_candidate"] = new_candidate
        except Exception:
            pass
        return best_d, best_cfg

    d = _load_file_with_offsets(
        win, path, entry, hv_for_load, angle_offsets, resolved_work_func, a_lattice
    )
    return d, angle_offsets


def _emit_sign_warnings(win, entry, sel: dict, verdict: dict, det: dict, new_candidate: str) -> None:
    """Report ambiguity / tie / refusal / convention changes on reload."""
    status = getattr(win, "_status", None)
    if not callable(status):
        return
    if sel.get("tie"):
        status(
            f"Warning: angle sign is undecided (scores {sel['best_score']:.4f} vs "
            f"{sel.get('opposite_score', sel.get('second_score')):.4f}) — "
            "first candidate kept, low-reliability result."
        )
    if verdict.get("refuse"):
        status(
            f"Warning: angle-sign detection refused (confidence "
            f"{sel['confidence']:.3f}, |Γ| {abs(det.get('gamma', float('nan'))):.3f} π/a) — "
            "manual override required."
        )
    elif verdict.get("ambiguous"):
        status(
            "Warning: ambiguous angle-sign convention — "
            + "; ".join(verdict.get("reasons", []))
        )
    # CASE6: convention differs from the previous session.
    prev = ""
    try:
        prev = str(entry.meta_gamma_state.get("angle_offset_candidate", "") or "")
    except Exception:
        prev = ""
    if prev and prev != new_candidate:
        status(
            f"Warning: Γ convention changed since the previous session "
            f"({prev} -> {new_candidate}) — kF NOT updated, refit to apply."
        )


def _load_file_with_offsets(
    win,
    path: str,
    entry,
    hv_for_load: float,
    angle_offsets: dict,
    work_func: float,
    a_lattice: float | None = None,
):
    lattice_kwargs = {}
    if a_lattice is not None and float(a_lattice) > 0:
        lattice_kwargs["a_lattice"] = float(a_lattice)
    return load_arpes_file(
        path, work_func, win._params.sp_ef.value(),
        **lattice_kwargs,
        hv=hv_for_load,
        temperature=entry.meta.temperature if entry.meta.temperature > 0 else None,
        azi=entry.meta.azi,
        pol=entry.meta.polarization,
        angle_offsets=angle_offsets,
        bessy_energy_reference=win._bessy_energy_reference_mode(),
    )
