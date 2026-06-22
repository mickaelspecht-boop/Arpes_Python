"""Run curvature-based dispersion extraction as a cross-check of MDC fits.

Free-function runner so it adds no lines to FitRunnerController (already near
the 700-LOC cap) and needs no PROXY_MAP entry (wired directly to a panel
signal). Reuses the controller's ``_get_work_data`` so the curvature sees the
same corrected data (distortion / grid / EDCnorm) as the Lorentzian fit, then
stores positions-only in ``entry.curvature_dispersion`` and refreshes Results.
"""
from __future__ import annotations

import traceback

from arpes.core import processing_history as ph
from arpes.physics.curvature_dispersion import extract_curvature_dispersion


def run_curvature_dispersion(window) -> None:
    """Compute kF(E) from curvature maxima for the current file (cross-check)."""
    ctrl = getattr(window, "_fit_runner_ctrl", None)
    if ctrl is None:
        return
    data, kpar, ev = ctrl._get_work_data()
    if data is None:
        return
    fp = window._params.get_fit_params()

    sp_c0 = getattr(window, "_sp_deriv_c0", None)
    try:
        c0 = float(sp_c0.value()) if sp_c0 is not None else 0.05
    except Exception:
        c0 = 0.05

    try:
        cd = extract_curvature_dispersion(
            data, kpar, ev,
            ev_start=fp.ev_start, ev_end=fp.ev_end,
            k_min=fp.k_min, k_max=fp.k_max, center_init=fp.center_init,
            n_pairs=fp.n_pairs, c0_alpha=c0,
        )
    except Exception as exc:
        window._status(f"Warning: curvature dispersion: {exc}")
        traceback.print_exc()
        return

    n_slices = len(cd.get("e_fitted") or [])
    entry = None
    if getattr(window, "_current_path", None):
        key = window._session.key_for_path(window._current_path)
        entry = window._session.get_or_create(key)
        entry.curvature_dispersion = cd
        window._session.save()

    ph.log_action(
        window, ph.CAT_FIT, "curvature dispersion (cross-check)",
        entry=entry, summary=f"{n_slices} slices, C0α={c0:.3f}",
    )

    res = getattr(window, "_results", None)
    if res is not None and hasattr(res, "refresh"):
        try:
            res.refresh()
        except Exception:
            pass

    if n_slices == 0:
        window._status(
            "Curvature dispersion: no peak found — widen the k window or lower C0."
        )
    else:
        window._status(
            f"Curvature dispersion: {n_slices} slices — cross-check overlay in Results "
            "(enable 'Curvature cross-check')."
        )
