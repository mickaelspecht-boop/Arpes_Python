"""UI plot controller for ArpesExplorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from arpes.physics.norm import remove_grid_artifact as remove_detector_grid_artifact
from arpes.physics.plot_compute import (
    compute_bandmap_display,
    draw_bandmap_axes as _plot_draw_bandmap_axes,
    draw_ef_label as _plot_draw_ef_label,
    draw_fit_roi_overlay as _plot_draw_fit_roi_overlay,
    draw_waterfall_axes as _plot_draw_waterfall_axes,
    edc_curve as _plot_edc_curve,
    fit_roi_bounds as _plot_fit_roi_bounds,
    fit_roi_data as _plot_fit_roi_data,
    map_color_kwargs as _plot_map_color_kwargs,
    mdc_curve as _plot_mdc_curve,
    scroll_zoom_limits as _plot_scroll_zoom_limits,
)

PAIR_COLORS = ["#ff8c00", "#00e5ff", "#7fff00", "#ff44cc"]


def _lorentzian(k, k0, gamma, A):
    return A * gamma**2 / ((k - k0)**2 + gamma**2)


def build_model_pairs(k_arr, mdc, n_pairs, gamma_init,
                      k_min, k_max, center_init, smooth_sigma,
                      spacing=0.25):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    mask = (k_arr >= k_min) & (k_arr <= k_max)
    k_w = k_arr[mask]
    m_w = mdc[mask]

    s_full = max(0.5, float(smooth_sigma))
    m_sm_full = gaussian_filter1d(np.nan_to_num(mdc.copy()), sigma=s_full)
    lo_f, hi_f = m_sm_full.min(), m_sm_full.max()
    mdc_smooth_norm = (m_sm_full - lo_f) / (hi_f - lo_f + 1e-12)

    if k_w.size < 10:
        return [], mdc_smooth_norm

    s = max(1, int(smooth_sigma))
    m_sm = gaussian_filter1d(np.nan_to_num(m_w), sigma=s)
    lo, hi = m_sm.min(), m_sm.max()
    if hi - lo < 1e-10:
        return [], mdc_smooth_norm
    m_n = (m_sm - lo) / (hi - lo)
    bg = float(np.nanpercentile(m_sm, 10))
    A0 = float(hi - lo)

    pks, _ = find_peaks(m_n, height=0.10, distance=max(3, s))
    if len(pks):
        pks = pks[np.argsort(m_n[pks])[::-1]]

    params = []
    if len(pks) >= 2:
        k_pks = k_w[pks]
        A_pks = m_sm[pks] - bg
        pos = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp >= center_init]
        neg = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp < center_init]
        for i in range(min(n_pairs, max(len(pos), len(neg)))):
            km = neg[i][0] if i < len(neg) else center_init - spacing * (i + 1)
            kp = pos[i][0] if i < len(pos) else center_init + spacing * (i + 1)
            params.append((km, kp, A0))
    elif len(pks) == 1:
        k0 = float(k_w[pks[0]])
        d = abs(k0 - center_init)
        params.append((center_init - d, center_init + d, A0))

    while len(params) < n_pairs:
        i = len(params)
        params.append((center_init - spacing * (i + 1),
                       center_init + spacing * (i + 1), A0 * 0.6))

    pairs = []
    for km, kp, A in params[:n_pairs]:
        cl = _lorentzian(k_arr, km, gamma_init, A)
        cr = _lorentzian(k_arr, kp, gamma_init, A)
        curve = cl + cr + bg
        curve = (curve - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)
        cln = (cl - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)
        crn = (cr - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)
        pairs.append((curve, km, kp, cln, crn))
    return pairs, mdc_smooth_norm


class PlotController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    def _on_scroll_zoom(self, event):
        """Zoom molette centré sur la position du curseur dans un axe matplotlib."""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        try:
            xlim, ylim = _plot_scroll_zoom_limits(
                ax.get_xlim(),
                ax.get_ylim(),
                xdata=float(event.xdata),
                ydata=float(event.ydata),
                step=float(getattr(event, "step", 0.0) or 0.0),
                button=getattr(event, "button", ""),
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            event.canvas.draw_idle()
        except Exception:
            return

    def _update_display_data(self):
        if self._raw_data is None:
            return
        d    = self._raw_data
        raw  = d["data"]
        mode = self._cmb_view.currentText()

        entry = self._current_entry()
        grid_cfg_active = entry.grid_correction if entry and entry.grid_correction.get("enabled") else None
        grid_key = (
            grid_cfg_active.get("strength"),
            grid_cfg_active.get("center_radius"),
            grid_cfg_active.get("peak_sensitivity"),
            grid_cfg_active.get("notch_width"),
        ) if grid_cfg_active else None
        cache_key = (id(raw), mode, grid_key)
        if cache_key == self._disp_cache_key and self._data_disp is not None:
            return  # rien n'a changé qui affecte l'affichage BM

        result = compute_bandmap_display(
            d,
            mode=mode,
            edc_norm_enabled=mode == "EDCnorm",
            grid_correction=grid_cfg_active,
            grid_artifact_fn=remove_detector_grid_artifact,
        )
        self._data_disp = result.data
        self._grid_display_info = result.grid_info
        self._disp_cache_key = cache_key

    def _fit_roi_bounds(self) -> tuple[float, float, float, float] | None:
        if self._raw_data is None:
            return None
        d = self._raw_data
        return _plot_fit_roi_bounds(
            d["kpar"], d["ev_arr"],
            k_min=self._params.sp_kmin.value(),
            k_max=self._params.sp_kmax.value(),
            ev_start=self._params.sp_evs.value(),
            ev_end=self._params.sp_eve.value(),
        )

    def _fit_roi_data(self, disp: np.ndarray, kpar: np.ndarray, ev: np.ndarray) -> np.ndarray:
        return _plot_fit_roi_data(disp, kpar, ev, self._fit_roi_bounds())

    def _map_color_kwargs(self, disp: np.ndarray, mode: str, *, roi_scale: bool = False) -> tuple[str, dict]:
        d = self._raw_data
        ref = self._fit_roi_data(disp, d["kpar"], d["ev_arr"]) if roi_scale and d is not None else disp
        return _plot_map_color_kwargs(disp, mode=mode, roi_ref=ref)

    def _draw_fit_roi_overlay(self, ax):
        _plot_draw_fit_roi_overlay(ax, self._fit_roi_bounds())

    def _ef_offset_text(self) -> str:
        return f"EF offset={self._params.sp_ef.value()*1000:+.0f} meV"

    def _draw_ef_label(self, ax, *, horizontal: bool = True):
        txt = f"EF  {self._ef_offset_text()}"
        _plot_draw_ef_label(ax, txt, horizontal=horizontal)

    # ─────────────────────────────────────────────────────────────────────────
    # Band map
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_bm(self):
        if self._data_disp is None:
            return
        d    = self._raw_data
        disp = self._data_disp
        mode = self._cmb_view.currentText()
        kpar = d["kpar"]; ev = d["ev_arr"]

        ax = self._bm_canvas.ax
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=False)
        int_win = self._params.sp_int_win.value()
        fname = Path(d["path"]).name
        _plot_draw_bandmap_axes(
            ax,
            kpar=kpar, ev=ev, disp=disp,
            cmap=cmap, color_kwargs=ckw,
            gamma=self._sp_gamma.value(),
            sel_ev=self._sel_ev, sel_k=self._sel_k, int_win=int_win,
            title=f"{fname}  [{mode}]  {self._ef_offset_text()}",
            title_size=9, label_size=10,
            show_k_zero=True,
        )

        self._draw_fit_roi_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)
        self._bm_canvas.redraw()
        self._draw_mdc_energy_map()

    def _draw_mdc_energy_map(self):
        """Mini BM visible dans l'onglet MDC Fit pour choisir E,k sans revenir à BM."""
        if not hasattr(self, "_mdc_map_canvas") or self._data_disp is None:
            return
        d = self._raw_data
        disp = self._data_disp
        mode = self._cmb_view.currentText()
        kpar = d["kpar"]
        ev = d["ev_arr"]
        ax = self._mdc_map_canvas.ax
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=True)
        int_win = self._params.sp_int_win.value()
        _plot_draw_bandmap_axes(
            ax,
            kpar=kpar, ev=ev, disp=disp,
            cmap=cmap, color_kwargs=ckw,
            gamma=1.0,
            sel_ev=self._sel_ev, sel_k=self._sel_k, int_win=int_win,
            title=f"Plage d'analyse [{mode}]",
            title_size=8, label_size=8, tick_label_size=8,
            show_k_zero=False,
        )
        bounds = self._fit_roi_bounds()
        if bounds is not None:
            k0, k1, e0, e1 = bounds
            ax.set_xlim(k0, k1)
            ax.set_ylim(e0, e1)
            info = f"Fenetre fit: k {k0:+.3f} -> {k1:+.3f} pi/a | E {e0:+.3f} -> {e1:+.3f} eV"
            ax.text(
                0.01, 0.02, info,
                transform=ax.transAxes,
                ha="left", va="bottom",
                color="white", fontsize=7,
                bbox={"facecolor": "#111827", "edgecolor": "#38bdf8", "alpha": 0.72, "pad": 3},
                zorder=30,
            )
            if hasattr(self, "_lbl_fit_view_info"):
                self._lbl_fit_view_info.setText("Plage d'analyse")
        self._draw_fit_roi_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)
        self._mdc_map_canvas.redraw()

    def _draw_mdc_waterfall(self):
        if not hasattr(self, "_waterfall_canvas") or self._raw_data is None:
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return

        ax = self._waterfall_canvas.ax
        self._waterfall_canvas.fig.set_facecolor("#2b2b2b")

        bounds = self._fit_roi_bounds() or (
            float(self._params.sp_kmin.value()),
            float(self._params.sp_kmax.value()),
            float(self._params.sp_evs.value()),
            float(self._params.sp_eve.value()),
        )
        n_target = int(self._params.sp_wf_n.value()) if hasattr(self._params, "sp_wf_n") else 32
        amp_scale = float(self._params.sp_wf_relief.value()) if hasattr(self._params, "sp_wf_relief") else 1.8
        _plot_draw_waterfall_axes(
            ax, data, kpar, ev,
            bounds=bounds,
            n_target=n_target,
            amp_scale=amp_scale,
            smooth_sigma=self._params.sp_sff.value(),
            fit_result=self._fit_res,
            n_pairs=self._params.sp_np.value(),
            pair_colors=PAIR_COLORS,
            gamma_center=self._params.sp_cx.value(),
        )
        self._waterfall_canvas.fig.tight_layout(pad=0.6)
        self._waterfall_canvas.redraw()

    def _draw_kf_overlay(self, ax):
        if self._fit_res is None:
            return
        fr = self._fit_res
        n  = self._params.sp_np.value()
        for i in range(n):
            c = PAIR_COLORS[i % len(PAIR_COLORS)]
            ev_f = np.asarray(fr["e_fitted"])
            if i < len(fr.get("kF_minus", [])):
                ax.scatter(np.asarray(fr["kF_minus"][i]), ev_f,
                           s=7, color=c, marker="o", zorder=5, alpha=0.85)
            if i < len(fr.get("kF_plus", [])):
                ax.scatter(np.asarray(fr["kF_plus"][i]), ev_f,
                           s=7, color=c, marker="^", zorder=5, alpha=0.85)

    # ─────────────────────────────────────────────────────────────────────────
    # MDC + EDC
    # ─────────────────────────────────────────────────────────────────────────

    def _get_mdc(self):
        if self._raw_data is None: return None
        return _plot_mdc_curve(
            self._raw_data,
            selected_ev=self._sel_ev,
            int_window=self._params.sp_int_win.value(),
            edc_norm_enabled=self._cmb_view.currentText() == "EDCnorm",
        )

    def _get_edc(self):
        if self._raw_data is None: return None
        return _plot_edc_curve(
            self._raw_data,
            selected_k=self._sel_k,
            edc_norm_enabled=self._cmb_view.currentText() == "EDCnorm",
        )

    def _draw_mdc_edc(self):
        from scipy.ndimage import gaussian_filter1d

        ax_mdc = self._mdc_edc.axes[0]
        ax_edc = self._edc_canvas.axes[0] if hasattr(self, "_edc_canvas") else None
        ax_mdc.cla()
        if ax_edc is not None:
            ax_edc.cla()
        self._mdc_edc._dark()
        if hasattr(self, "_edc_canvas"):
            self._edc_canvas._dark()

        # ── MDC ──────────────────────────────────────────────────────────────
        res = self._get_mdc()
        if res is not None:
            kpar, mdc = res
            lo, hi = np.nanpercentile(mdc, [1, 99])
            mdc_n = np.clip((mdc - lo) / (hi - lo + 1e-12), 0, 1)

            ax_mdc.plot(kpar, mdc_n, color="white", lw=1.2, label="MDC", zorder=3)
            ax_mdc.fill_between(kpar, 0, mdc_n, alpha=0.08, color="white", zorder=1)

            kmin = self._params.sp_kmin.value()
            kmax = self._params.sp_kmax.value()
            ax_mdc.axvspan(kpar.min(), kmin, alpha=0.15, color="gray", zorder=0)
            ax_mdc.axvspan(kmax, kpar.max(), alpha=0.15, color="gray", zorder=0)

            pairs, mdc_smooth = build_model_pairs(
                kpar, mdc_n,
                n_pairs      = self._params.sp_np.value(),
                gamma_init   = self._params.sp_gi.value(),
                k_min        = kmin, k_max = kmax,
                center_init  = self._params.sp_cx.value(),
                smooth_sigma = self._params.sp_sfd.value(),
            )

            # ── courbe lissée détection (comme Igor "smooth before detect") ──
            ax_mdc.plot(kpar, mdc_smooth, color="#aaa", lw=0.8, ls="-",
                        alpha=0.55, label=f"lissé-det (σ={self._params.sp_sfd.value():.1f})", zorder=2)

            # ── courbe lissée ajustement (utilisée par l'optimiseur scipy) ────
            sff = self._params.sp_sff.value()
            sfd = self._params.sp_sfd.value()
            if sff > 0.5 and abs(sff - sfd) > 0.3:
                _mdc_fit_sm = gaussian_filter1d(np.nan_to_num(mdc_n.copy()), sigma=max(0.5, sff))
                ax_mdc.plot(kpar, _mdc_fit_sm, color="#ffa040", lw=0.8, ls="-",
                            alpha=0.55, zorder=2, label=f"lissé-fit (σ={sff:.1f})")

            # ── zone de contrainte xg (center_init ± xg_range) ───────────────
            cx  = self._params.sp_cx.value()
            xgr = self._params.sp_xg.value()
            ax_mdc.axvspan(cx - xgr, cx + xgr, alpha=0.08, color="cyan",
                           zorder=0, label=f"Fenêtre Γ ±{xgr:.2f}")
            ax_mdc.axvline(cx, color="cyan", lw=0.6, ls=":", alpha=0.45, zorder=1)

            # ── contrainte kF max (si active) ─────────────────────────────────
            if not self._params.chk_k0a.isChecked():
                k0m = self._params.sp_k0m.value()
                ax_mdc.axvline(cx + k0m, color="plum", lw=0.9, ls=":", alpha=0.7, zorder=1,
                               label=f"|kF|<{k0m:.2f}")
                ax_mdc.axvline(cx - k0m, color="plum", lw=0.9, ls=":", alpha=0.7, zorder=1)

            # ── marqueurs kF_init par paire ───────────────────────────────────
            n_p = self._params.sp_np.value()
            for pi, pp in enumerate(self._params._pair_params[:n_p]):
                kf = pp.get("kF_init", 0.30)
                pc = PAIR_COLORS[pi % len(PAIR_COLORS)]
                ax_mdc.axvline(cx + kf, color=pc, lw=0.8, ls="-.", alpha=0.7, zorder=2)
                ax_mdc.axvline(cx - kf, color=pc, lw=0.8, ls="-.", alpha=0.7, zorder=2)

            # ── modèle Lorentzien décomposé ───────────────────────────────────
            gmax = self._params.sp_gm.value()
            total = np.zeros_like(mdc_n)
            for i, (curve, km, kp, cl, cr) in enumerate(pairs):
                c = PAIR_COLORS[i % len(PAIR_COLORS)]
                # zones γ_max autour des pics détectés (largeur maximale autorisée)
                for k0 in (km, kp):
                    ax_mdc.axvspan(k0 - gmax, k0 + gmax, alpha=0.05, color=c, zorder=0)
                valid = np.isfinite(curve)
                if valid.any():
                    # courbe totale de la paire
                    ax_mdc.plot(kpar, np.where(valid, curve, np.nan),
                                color=c, lw=1.3, ls="--", zorder=4,
                                label=f"P{i+1}  kF≈{abs(kp-km)/2:.3f}")
                    # pics individuels (gauche / droite) — modèle Igor lor_pair
                    for comp in (cl, cr):
                        vc = np.isfinite(comp)
                        if vc.any():
                            ax_mdc.plot(kpar, np.where(vc, comp, np.nan),
                                        color=c, lw=0.7, ls=":", alpha=0.55, zorder=3)
                    total += np.where(valid, curve, 0.)
            if n_p > 1:
                ax_mdc.plot(kpar, total, color="white", lw=0.8, ls=":",
                            alpha=0.45, label="Σ", zorder=4)

            ax_mdc.axvline(0, color="w", lw=0.5, ls="--", alpha=0.3)
            int_win = self._params.sp_int_win.value()
            ax_mdc.set_xlabel("k// (π/a)", fontsize=8, color="w")
            ax_mdc.set_ylabel("I (norm.)", fontsize=8, color="w")
            ax_mdc.set_title(
                f"MDC  E={self._sel_ev:.3f} eV  ±{int_win*1000:.0f} meV  |  {self._ef_offset_text()}",
                fontsize=8, color="w")
            ax_mdc.tick_params(colors="w", labelsize=7)
            ax_mdc.legend(fontsize=7, facecolor="#333", labelcolor="w",
                          loc="upper right", framealpha=0.7, ncol=2)
            for sp in ax_mdc.spines.values(): sp.set_edgecolor("#555")

        # ── EDC ──────────────────────────────────────────────────────────────
        res2 = self._get_edc()
        if ax_edc is not None and res2 is not None:
            ev_arr, edc = res2
            lo, hi = np.nanpercentile(edc, [1, 99])
            edc_n = np.clip((edc - lo) / (hi - lo + 1e-12), 0, 1)

            ax_edc.plot(ev_arr, edc_n, color="#7dd3fc", lw=1.2)
            ax_edc.fill_between(ev_arr, 0, edc_n, alpha=0.15, color="#7dd3fc")
            ax_edc.axvline(0, color="cyan", lw=0.8, ls="--", alpha=0.7)
            ax_edc.axvline(self._sel_ev, color="lime", lw=1.0, ls=":")
            self._draw_ef_label(ax_edc, horizontal=False)
            ax_edc.set_xlabel("E − EF (eV)", fontsize=8, color="w")
            ax_edc.set_ylabel("I (norm.)", fontsize=8, color="w")
            ax_edc.set_title(f"EDC  k={self._sel_k:.3f} π/a  |  {self._ef_offset_text()}",
                             fontsize=8, color="w")
            ax_edc.tick_params(colors="w", labelsize=7)
            for sp in ax_edc.spines.values(): sp.set_edgecolor("#555")

        self._mdc_edc.fig.tight_layout(pad=0.5)
        self._mdc_edc.redraw()
        if hasattr(self, "_edc_canvas"):
            self._edc_canvas.fig.tight_layout(pad=0.5)
            self._edc_canvas.redraw()

    # ─────────────────────────────────────────────────────────────────────────
    # Interactions carte
    # ─────────────────────────────────────────────────────────────────────────
