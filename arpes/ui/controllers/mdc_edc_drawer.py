"""MDC + EDC drawing routine extracted from plot_controller.

Single free function ``draw_mdc_edc(ctrl)`` rendering the bottom canvas.
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from arpes.ui.controllers.plot_model_helpers import build_model_pairs
from arpes.ui.controllers.fit_overlay_drawer import PAIR_COLORS


def _num_array(v):
    """Coerce to a float ndarray, or None if not numeric (e.g. a stringified
    legacy array saved by an old ensemble fit). Never raises."""
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=float)
    except (ValueError, TypeError):
        return None
    return a if a.ndim == 1 and a.size else None


def _has_postfit_model(ctrl) -> bool:
    """True if a per-slice fit model exists (Full fit). Ensemble fits store no
    fit_curves → False (post-fit view then unavailable)."""
    try:
        entry = ctrl._current_entry()
        fr = getattr(entry, "fit_result", None)
    except Exception:
        fr = getattr(ctrl, "_fit_res", None)
    return bool(fr and (fr.get("fit_curves") or []))


def _postfit_slice(ctrl):
    """(fit_kpar, total, bg, residual, E) for the fitted slice nearest _sel_ev.

    None if no fit / no usable per-slice model (ensemble fits store no
    fit_curves; legacy fits may have stringified arrays). Data is reconstructed
    as total+residual so it shares the fit's normalisation exactly.
    """
    fr = None
    try:
        entry = ctrl._current_entry()
        fr = getattr(entry, "fit_result", None)
    except Exception:
        fr = getattr(ctrl, "_fit_res", None)
    if not fr:
        return None
    ev = _num_array(fr.get("e_fitted"))
    kf = _num_array(fr.get("fit_kpar"))
    fc = fr.get("fit_curves") or []
    if ev is None or kf is None or not fc:
        return None
    idx = int(np.argmin(np.abs(ev - ctrl._sel_ev)))
    if not (0 <= idx < len(fc)):
        return None
    total = _num_array(fc[idx])
    if total is None or total.size != kf.size:
        return None

    def _at(key):
        arr = fr.get(key) or []
        if 0 <= idx < len(arr):
            a = _num_array(arr[idx])
            return a if (a is not None and a.size == kf.size) else None
        return None

    return kf, total, _at("fit_bg"), _at("residuals"), float(ev[idx])


def draw_postfit_mdc(ctrl, ax_mdc) -> bool:
    """Draw the stored fit at the current slice: data, total, background, peaks,
    residual. Always returns True when called (the toggle owns the view): shows
    a clear note when no per-slice model is available (e.g. ensemble fit)."""
    out = _postfit_slice(ctrl)
    if out is None:
        ax_mdc.text(
            0.5, 0.5,
            "Vue post-fit indisponible pour cette tranche.\n"
            "Lance un « Full fit » (l'ensemble ne stocke pas le modèle par tranche),\n"
            "ou re-fitte ce fichier.",
            transform=ax_mdc.transAxes, ha="center", va="center",
            color="#e6b35a", fontsize=8.5)
        ax_mdc.set_title("Post-fit MDC — indisponible", fontsize=8, color="w")
        ax_mdc.tick_params(colors="w", labelsize=7)
        for sp in ax_mdc.spines.values():
            sp.set_edgecolor("#555")
        return True
    kf, total, bg, resid, e_used = out
    data = total + resid if (resid is not None) else None
    if data is not None:
        ax_mdc.plot(kf, data, color="white", lw=1.3, label="MDC data", zorder=3)
        ax_mdc.fill_between(kf, np.nanmin(data), data, alpha=0.06, color="white", zorder=1)
    ax_mdc.plot(kf, total, color="#ffa040", lw=1.8, label="fit total", zorder=5)
    if bg is not None and np.isfinite(bg).any():
        ax_mdc.plot(kf, bg, color="#9aa0a6", lw=1.0, ls="--", label="background", zorder=4)
        ax_mdc.plot(kf, total - bg, color="#7dd3fc", lw=1.0, ls=":", label="peaks (fit−bg)", zorder=4)
    if resid is not None and np.isfinite(resid).any():
        base = float(np.nanmin(data if data is not None else total)) - 0.18
        ax_mdc.axhline(base, color="#555", lw=0.6, ls="-", alpha=0.6, zorder=2)
        ax_mdc.plot(kf, resid + base, color="#f87171", lw=0.8, alpha=0.85,
                    label="residual", zorder=4)
    cx = ctrl._params.sp_cx.value()
    ax_mdc.axvline(cx, color="cyan", lw=0.8, ls=":", alpha=0.6, zorder=2)
    ax_mdc.set_xlabel("k// (π/a)", fontsize=8, color="w")
    ax_mdc.set_ylabel("I (norm. fit)", fontsize=8, color="w")
    ax_mdc.set_title(f"Post-fit MDC  E={e_used:+.3f} eV  (data + modèle + fond)",
                     fontsize=8, color="w")
    ax_mdc.tick_params(colors="w", labelsize=7)
    ax_mdc.legend(fontsize=7, facecolor="#333", labelcolor="w",
                  loc="upper right", framealpha=0.7, ncol=2)
    for sp in ax_mdc.spines.values():
        sp.set_edgecolor("#555")
    return True


def draw_mdc_edc(ctrl) -> None:
    from scipy.ndimage import gaussian_filter1d

    ax_mdc = ctrl._mdc_edc.axes[0]
    ax_edc = ctrl._edc_canvas.axes[0] if hasattr(ctrl, "_edc_canvas") else None
    ax_mdc.cla()
    if ax_edc is not None:
        ax_edc.cla()
    ctrl._mdc_edc._dark()
    if hasattr(ctrl, "_edc_canvas"):
        ctrl._edc_canvas._dark()

    params = ctrl._params
    show_logic = bool(
        getattr(params, "chk_fit_slice_inspector", None) is None
        or params.chk_fit_slice_inspector.isChecked()
    )

    # Post-fit view (toggle button above the MDC cut): show the stored fit
    # decomposition for this slice instead of the live guess preview. The button
    # is enabled only when a per-slice model exists (a Full fit, not ensemble).
    _pf = getattr(ctrl, "_btn_postfit_view", None)
    has_model = _has_postfit_model(ctrl)
    if _pf is not None:
        _pf.setEnabled(has_model)
        if not has_model and _pf.isChecked():
            _pf.blockSignals(True)
            _pf.setChecked(False)
            _pf.blockSignals(False)
    mdc_done = False
    if _pf is not None and _pf.isChecked() and has_model:
        mdc_done = draw_postfit_mdc(ctrl, ax_mdc)
    res = None if mdc_done else ctrl._get_mdc()
    if res is not None:
        kpar, mdc = res
        lo, hi = np.nanpercentile(mdc, [1, 99])
        mdc_n = np.clip((mdc - lo) / (hi - lo + 1e-12), 0, 1)

        ax_mdc.plot(kpar, mdc_n, color="white", lw=1.2, label="MDC", zorder=3)
        ax_mdc.fill_between(kpar, 0, mdc_n, alpha=0.08, color="white", zorder=1)

        kmin = params.sp_kmin.value()
        kmax = params.sp_kmax.value()
        ax_mdc.axvspan(kpar.min(), kmin, alpha=0.15, color="gray", zorder=0)
        ax_mdc.axvspan(kmax, kpar.max(), alpha=0.15, color="gray", zorder=0)

        pairs = []
        if show_logic:
            pairs, mdc_smooth = build_model_pairs(
                kpar, mdc_n,
                n_pairs=params.sp_np.value(),
                gamma_init=params.sp_gi.value(),
                k_min=kmin, k_max=kmax,
                center_init=params.sp_cx.value(),
                smooth_sigma=params.sp_sfd.value(),
            )
            ax_mdc.plot(kpar, mdc_smooth, color="#aaa", lw=0.8, ls="-",
                        alpha=0.55,
                        label=f"smoothed-det (sigma={params.sp_sfd.value():.1f})",
                        zorder=2)

        sff = params.sp_sff.value()
        sfd = params.sp_sfd.value()
        if show_logic and sff > 0.5 and abs(sff - sfd) > 0.3:
            _mdc_fit_sm = gaussian_filter1d(np.nan_to_num(mdc_n.copy()),
                                             sigma=max(0.5, sff))
            ax_mdc.plot(kpar, _mdc_fit_sm, color="#ffa040", lw=0.8, ls="-",
                        alpha=0.55, zorder=2,
                        label=f"smoothed-fit (sigma={sff:.1f})")

        cx = params.sp_cx.value()
        xgr = params.sp_xg.value()
        ax_mdc.axvspan(cx - xgr, cx + xgr, alpha=0.08, color="cyan",
                       zorder=0, label=f"Gamma window +/-{xgr:.2f}")
        center_line = ax_mdc.axvline(
            cx, color="cyan", lw=1.0, ls=":", alpha=0.75, zorder=5, picker=7,
        )
        center_line._kf_meta = ("center", 0)

        if not params.chk_k0a.isChecked():
            k0m = params.sp_k0m.value()
            ax_mdc.axvline(cx + k0m, color="plum", lw=0.9, ls=":", alpha=0.7,
                           zorder=1, label=f"|kF|<{k0m:.2f}")
            ax_mdc.axvline(cx - k0m, color="plum", lw=0.9, ls=":", alpha=0.7,
                           zorder=1)

        n_p = params.sp_np.value()
        ctrl._kf_drag_lines = []
        # Draggable kF handles are staggered into vertical bands so that two
        # lines at the same k (small kF → +/- nearly coincide, or two pairs with
        # similar kF) no longer fully overlap: each handle keeps its own grab
        # zone and stays pickable. A faint full-height guide still marks the
        # exact k position for reading.
        blend = ax_mdc.get_xaxis_transform()
        n_handles = max(1, 2 * n_p)
        y_centers = np.linspace(0.90, 0.12, n_handles)
        handle_h = min(0.13, 0.78 / n_handles + 0.02)
        slot = 0
        logic_lines = [
            f"slice E={ctrl._sel_ev:+.3f} eV  int=±{params.sp_int_win.value()*1000:.0f} meV",
            f"fit k=[{kmin:+.3f},{kmax:+.3f}]  scan E=[{params.sp_evs.value():+.3f},{params.sp_eve.value():+.3f}]",
            f"xg={cx:+.3f}±{xgr:.3f}  γ0={params.sp_gi.value():.3f}  γmax={params.sp_gm.value():.3f}",
            f"σfit={params.sp_sff.value():.1f}  σdetect={params.sp_sfd.value():.1f}"
            f"  Amin={params.sp_ma.value():.2f}  jump={params.sp_mj.value():.2f}",
        ]
        pair_line_parts = []
        for pi, pp in enumerate(params._pair_params[:n_p]):
            kf = pp.get("kF_init", 0.30)
            pc = PAIR_COLORS[pi % len(PAIR_COLORS)]
            for sign in (+1, -1):
                xk = cx + sign * kf
                guide = ax_mdc.axvline(xk, color=pc, lw=0.7, ls=":",
                                       alpha=0.30, zorder=2)
                yc = float(y_centers[slot % n_handles]); slot += 1
                handle = Line2D(
                    [xk, xk], [yc - handle_h / 2.0, yc + handle_h / 2.0],
                    transform=blend, color=pc, lw=3.0, ls="-", alpha=0.95,
                    solid_capstyle="round", marker="o", markersize=4,
                    zorder=6, picker=6,
                )
                handle._kf_meta = (pi, sign)
                handle._kf_guide = guide
                ax_mdc.add_line(handle)
                ctrl._kf_drag_lines.append((pi, sign, handle))
            pair_line_parts.append(f"P{pi+1}:kF0={float(kf):.3f}")
        ctrl._install_kf_drag_handlers()
        if pair_line_parts:
            logic_lines.append("  ".join(pair_line_parts))
        plot_logic_lines = [
            logic_lines[0],
            f"k=[{kmin:+.2f},{kmax:+.2f}]  Γ={cx:+.3f}±{xgr:.3f}",
        ]
        if pair_line_parts:
            plot_logic_lines.append("  ".join(pair_line_parts))

        gmax = params.sp_gm.value()
        total = np.zeros_like(mdc_n)
        for i, (curve, km, kp, cl, cr) in enumerate(pairs):
            c = PAIR_COLORS[i % len(PAIR_COLORS)]
            for k0 in (km, kp):
                ax_mdc.axvspan(k0 - gmax, k0 + gmax, alpha=0.05, color=c,
                               zorder=0)
            valid = np.isfinite(curve)
            if valid.any():
                ax_mdc.plot(kpar, np.where(valid, curve, np.nan),
                            color=c, lw=1.3, ls="--", zorder=4,
                            label=f"P{i+1}  kF≈{abs(kp - km) / 2:.3f}")
                for comp in (cl, cr):
                    vc = np.isfinite(comp)
                    if vc.any():
                        ax_mdc.plot(kpar, np.where(vc, comp, np.nan),
                                    color=c, lw=0.7, ls=":", alpha=0.55,
                                    zorder=3)
                total += np.where(valid, curve, 0.)
        if show_logic and n_p > 1:
            ax_mdc.plot(kpar, total, color="white", lw=0.8, ls=":",
                        alpha=0.45, label="Σ", zorder=4)

        ax_mdc.axvline(0, color="w", lw=0.5, ls="--", alpha=0.3)
        int_win = params.sp_int_win.value()
        ax_mdc.set_xlabel("k// (π/a)", fontsize=8, color="w")
        ax_mdc.set_ylabel("I (norm.)", fontsize=8, color="w")
        ax_mdc.set_title(
            f"MDC  E={ctrl._sel_ev:.3f} eV  ±{int_win * 1000:.0f} meV"
            f"  |  {ctrl._ef_offset_text()}",
            fontsize=8, color="w")
        if show_logic:
            ax_mdc.text(
                0.015, 0.965, "\n".join(plot_logic_lines),
                transform=ax_mdc.transAxes,
                va="top", ha="left", fontsize=6.5, color="#dbeafe",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#111827",
                          edgecolor="#334155", alpha=0.82),
                zorder=10,
            )
        lbl_logic = getattr(params, "lbl_fit_slice_logic", None)
        if lbl_logic is not None:
            lbl_logic.setText("\n".join(logic_lines))
        ax_mdc.tick_params(colors="w", labelsize=7)
        if show_logic:
            ax_mdc.legend(fontsize=7, facecolor="#333", labelcolor="w",
                          loc="upper right", framealpha=0.7, ncol=2)
        for sp in ax_mdc.spines.values():
            sp.set_edgecolor("#555")
    else:
        lbl_logic = getattr(params, "lbl_fit_slice_logic", None)
        if lbl_logic is not None:
            lbl_logic.setText("")

    res2 = ctrl._get_edc()
    if ax_edc is not None and res2 is not None:
        ev_arr, edc = res2
        lo, hi = np.nanpercentile(edc, [1, 99])
        edc_n = np.clip((edc - lo) / (hi - lo + 1e-12), 0, 1)
        ax_edc.plot(ev_arr, edc_n, color="#7dd3fc", lw=1.2)
        ax_edc.fill_between(ev_arr, 0, edc_n, alpha=0.15, color="#7dd3fc")
        ax_edc.axvline(0, color="cyan", lw=0.8, ls="--", alpha=0.7)
        ax_edc.axvline(ctrl._sel_ev, color="lime", lw=1.0, ls=":")
        ctrl._draw_ef_label(ax_edc, horizontal=False)
        ax_edc.set_xlabel("E − EF (eV)", fontsize=8, color="w")
        ax_edc.set_ylabel("I (norm.)", fontsize=8, color="w")
        ax_edc.set_title(
            f"EDC  k={ctrl._sel_k:.3f} π/a  |  {ctrl._ef_offset_text()}",
            fontsize=8, color="w")
        ax_edc.tick_params(colors="w", labelsize=7)
        for sp in ax_edc.spines.values():
            sp.set_edgecolor("#555")

    ctrl._mdc_edc.fig.tight_layout(pad=0.5)
    ctrl._mdc_edc.redraw()
    if hasattr(ctrl, "_edc_canvas"):
        ctrl._edc_canvas.fig.tight_layout(pad=0.5)
        ctrl._edc_canvas.redraw()
