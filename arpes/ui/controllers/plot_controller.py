"""UI plot controller for ArpesExplorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from arpes.physics.norm import remove_grid_artifact as remove_detector_grid_artifact
from arpes.physics.plot_compute import (
    BandmapAxesState,
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


def _axis_cache_signature(axis) -> tuple:
    arr = np.asarray(axis, dtype=float)
    if arr.size == 0:
        return (0,)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (arr.size, "all-nan")
    return (
        int(arr.size),
        float(finite[0]),
        float(finite[-1]),
        float(np.nanmin(finite)),
        float(np.nanmax(finite)),
    )


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
            # aspect 'auto' : sinon (FS en aspect 'equal') le cadre rétrécit au
            # lieu de simplement adapter les bornes d'axes.
            ax.set_aspect("auto")
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
        raw_key = getattr(self, "_current_raw_load_cache_key", None)
        cache_key = (
            raw_key,
            id(raw),
            tuple(np.asarray(raw).shape),
            mode,
            grid_key,
            _axis_cache_signature(d["kpar"]),
            _axis_cache_signature(d["ev_arr"]),
        )
        if cache_key == self._disp_cache_key and self._data_disp is not None:
            return  # rien n'a changé qui affecte l'affichage BM

        display_cache = getattr(self, "_display_cache", None)
        if display_cache is not None and cache_key in display_cache:
            disp_cached, info_cached = display_cache.pop(cache_key)
            display_cache[cache_key] = (disp_cached, info_cached)
            self._data_disp = disp_cached
            self._grid_display_info = dict(info_cached or {})
            self._disp_cache_key = cache_key
            return

        result = compute_bandmap_display(
            d,
            mode=mode,
            edc_norm_enabled=mode in ("EDCnorm", "SecDev", "Curvature"),
            grid_correction=grid_cfg_active,
            grid_artifact_fn=remove_detector_grid_artifact,
        )
        self._data_disp = result.data
        self._grid_display_info = result.grid_info
        self._disp_cache_key = cache_key
        if display_cache is not None:
            display_cache[cache_key] = (result.data, dict(result.grid_info or {}))
            max_items = int(getattr(self, "_display_cache_max", 12) or 12)
            while len(display_cache) > max_items:
                display_cache.popitem(last=False)

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

    def _draw_current_view(self, *, include_curves: bool = True):
        tabs = getattr(self, "_tabs", None)
        index = tabs.currentIndex() if tabs is not None else 0
        if index == 0:
            self._draw_bm()
        elif index == 1:
            self._draw_mdc_energy_map()
            if include_curves:
                if hasattr(self, "_mdc_fit_tabs") and self._mdc_fit_tabs.currentIndex() == 1:
                    self._draw_mdc_waterfall()
                else:
                    self._draw_mdc_edc()
        elif index == 3:
            self._draw_fs_tab()

    def _draw_bm(self):
        if self._data_disp is None:
            return
        d    = self._raw_data
        disp = self._data_disp
        mode = self._cmb_view.currentText()
        kpar = d["kpar"]; ev = d["ev_arr"]

        ax = self._bm_canvas.ax
        self._clear_plot_overlays(ax)
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=False)
        int_win = self._params.sp_int_win.value()
        fname = Path(d["path"]).name
        state = getattr(self._parent, "_bm_plot_state", None)
        if state is None:
            state = BandmapAxesState()
        plot_key = getattr(self._parent, "_disp_cache_key", None)
        reset_limits = plot_key != getattr(self._parent, "_bm_plot_data_key", None)
        self._parent._bm_plot_state = _plot_draw_bandmap_axes(
            ax,
            kpar=kpar, ev=ev, disp=disp,
            cmap=cmap, color_kwargs=ckw,
            gamma=self._sp_gamma.value(),
            sel_ev=self._sel_ev, sel_k=self._sel_k, int_win=int_win,
            title=f"{fname}  [{mode}]  {self._ef_offset_text()}",
            title_size=9, label_size=10,
            show_k_zero=True,
            state=state,
            reset_limits=reset_limits,
        )
        self._parent._bm_plot_data_key = plot_key

        before = self._axis_artist_snapshot(ax)
        self._draw_fit_roi_overlay(ax)
        self._draw_theory_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_gamma_preview_axvline(ax)
        self._draw_ef_label(ax, horizontal=True)
        self._tag_new_plot_overlays(ax, before)
        self._bm_canvas.redraw()

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
        self._clear_plot_overlays(ax)
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=True)
        int_win = self._params.sp_int_win.value()
        state = getattr(self._parent, "_mdc_map_plot_state", None)
        if state is None:
            state = BandmapAxesState()
        plot_key = getattr(self._parent, "_disp_cache_key", None)
        reset_limits = plot_key != getattr(self._parent, "_mdc_map_plot_data_key", None)
        self._parent._mdc_map_plot_state = _plot_draw_bandmap_axes(
            ax,
            kpar=kpar, ev=ev, disp=disp,
            cmap=cmap, color_kwargs=ckw,
            gamma=1.0,
            sel_ev=self._sel_ev, sel_k=self._sel_k, int_win=int_win,
            title=f"Plage d'analyse [{mode}]",
            title_size=8, label_size=8, tick_label_size=8,
            show_k_zero=False,
            state=state,
            reset_limits=reset_limits,
        )
        self._parent._mdc_map_plot_data_key = plot_key
        before = self._axis_artist_snapshot(ax)
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
        self._draw_theory_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)
        self._tag_new_plot_overlays(ax, before)
        self._mdc_map_canvas.redraw()

    @staticmethod
    def _axis_artist_snapshot(ax) -> set:
        artists = list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts) + list(ax.images)
        return {id(artist) for artist in artists}

    def _clear_plot_overlays(self, ax) -> None:
        active = {
            id(rect)
            for rect in (
                getattr(self._parent, "_fit_roi_rect", None),
                getattr(self._parent, "_fit_select_rect", None),
            )
            if rect is not None
        }
        artists = list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts) + list(ax.images)
        for artist in artists:
            if id(artist) in active:
                continue
            if getattr(artist, "_arpes_plot_overlay", False) or artist.get_gid() == "arpes_overlay":
                try:
                    artist.remove()
                except Exception:
                    pass

    @staticmethod
    def _tag_new_plot_overlays(ax, before: set) -> None:
        artists = list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts) + list(ax.images)
        for artist in artists:
            if id(artist) in before:
                continue
            try:
                artist.set_gid("arpes_overlay")
            except Exception:
                pass
            try:
                setattr(artist, "_arpes_plot_overlay", True)
            except Exception:
                pass

    def _draw_mdc_waterfall(self):
        if not hasattr(self, "_waterfall_canvas") or self._raw_data is None:
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return

        has_residuals = bool(self._fit_res and self._fit_res.get("residuals"))
        fig = self._waterfall_canvas.fig
        if has_residuals:
            fig.clear()
            gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08)
            ax = fig.add_subplot(gs[0, 0])
            residual_ax = fig.add_subplot(gs[1, 0], sharex=ax)
            self._waterfall_canvas.axes = [ax, residual_ax]
            self._waterfall_canvas.ax = ax
        elif len(getattr(self._waterfall_canvas, "axes", [])) != 1:
            fig.clear()
            ax = fig.add_subplot(111)
            self._waterfall_canvas.axes = [ax]
            self._waterfall_canvas.ax = ax
            residual_ax = None
        else:
            ax = self._waterfall_canvas.ax
            residual_ax = None
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
            residual_ax=residual_ax,
        )
        self._waterfall_canvas.fig.tight_layout(pad=0.6)
        self._waterfall_canvas.redraw()

    def _draw_kf_overlay(self, ax):
        if self._fit_res is None:
            return
        fr = self._fit_res
        n  = self._params.sp_np.value()
        ev_f = np.asarray(fr["e_fitted"])
        chi2 = np.asarray(fr.get("chi2_red", []), dtype=float)
        threshold = float(getattr(self._params, "sp_chi2_threshold", None).value()) if hasattr(self._params, "sp_chi2_threshold") else np.inf
        bad_mask = chi2 > threshold if chi2.size == ev_f.size else np.zeros(ev_f.size, dtype=bool)
        for i in range(n):
            c = PAIR_COLORS[i % len(PAIR_COLORS)]
            if i < len(fr.get("kF_minus", [])):
                self._scatter_kf_with_chi2(ax, np.asarray(fr["kF_minus"][i]), ev_f, bad_mask, c, "o")
            if i < len(fr.get("kF_plus", [])):
                self._scatter_kf_with_chi2(ax, np.asarray(fr["kF_plus"][i]), ev_f, bad_mask, c, "^")
        selected = list(getattr(self._parent, "_fit_selected", []) or [])
        if selected:
            ev_f = np.asarray(fr["e_fitted"])
            sel_k: list[float] = []
            sel_e: list[float] = []
            for branch, pair_idx, point_idx in selected:
                arrays = fr.get(branch) or []
                if not (0 <= pair_idx < len(arrays)):
                    continue
                arr = np.asarray(arrays[pair_idx], dtype=float)
                if not (0 <= point_idx < arr.size and point_idx < ev_f.size):
                    continue
                k_val = arr[point_idx]
                e_val = ev_f[point_idx]
                if not (np.isfinite(k_val) and np.isfinite(e_val)):
                    continue
                sel_k.append(float(k_val))
                sel_e.append(float(e_val))
            if sel_k:
                ax.scatter(sel_k, sel_e, s=70, facecolors="none",
                           edgecolors="#fbbf24", linewidths=1.6, zorder=7)
        self._draw_fit_annotations(ax, fr)

    def _scatter_kf_with_chi2(self, ax, k_values, ev_f, bad_mask, color, marker) -> None:
        n = min(len(k_values), len(ev_f), len(bad_mask))
        if n == 0:
            return
        k = np.asarray(k_values[:n], dtype=float)
        e = np.asarray(ev_f[:n], dtype=float)
        bad = np.asarray(bad_mask[:n], dtype=bool)
        valid = np.isfinite(k) & np.isfinite(e)
        good = valid & ~bad
        if good.any():
            ax.scatter(k[good], e[good], s=7, color=color, marker=marker,
                       zorder=5, alpha=0.85)
        if (valid & bad).any():
            ax.scatter(k[valid & bad], e[valid & bad], s=20, color="#fb923c",
                       marker=marker, edgecolors="black", linewidths=0.35,
                       zorder=6, alpha=0.95)

    def _draw_fit_annotations(self, ax, fr: dict) -> None:
        p = self._parent
        if not getattr(p, "_current_path", None):
            return
        entry = p._session.get_or_create(p._session.key_for_path(p._current_path))
        annotations = entry.annotations or {}
        if not annotations:
            return
        ev_f = np.asarray(fr.get("e_fitted", []), dtype=float)
        if ev_f.size == 0:
            return
        for branch in ("kF_minus", "kF_plus"):
            arrays = fr.get(branch) or []
            for note in annotations.get(branch, []):
                try:
                    pair_idx = int(note.get("pair", -1))
                    point_idx = int(note.get("index", -1))
                except Exception:
                    continue
                if not (0 <= pair_idx < len(arrays)):
                    continue
                arr = np.asarray(arrays[pair_idx], dtype=float)
                if not (0 <= point_idx < arr.size and point_idx < ev_f.size):
                    continue
                k_val = arr[point_idx]
                e_val = ev_f[point_idx]
                if not (np.isfinite(k_val) and np.isfinite(e_val)):
                    continue
                text = str(note.get("text", "")).strip()
                ax.plot(
                    k_val, e_val, marker="*", markersize=12,
                    color="#fde047", markeredgecolor="black", zorder=8,
                )
                if text:
                    ax.annotate(
                        text[:30], xy=(k_val, e_val), xytext=(5, 5),
                        textcoords="offset points", fontsize=7,
                        color="#fde047", zorder=9,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # MDC + EDC
    # ─────────────────────────────────────────────────────────────────────────

    def _get_mdc(self):
        if self._raw_data is None: return None
        edc_norm = self._cmb_view.currentText() == "EDCnorm"
        raw_data = self._raw_data
        if edc_norm:
            if self._data_disp is None or np.shape(self._data_disp) != np.shape(self._raw_data["data"]):
                self._update_display_data()
            if self._data_disp is not None and np.shape(self._data_disp) == np.shape(self._raw_data["data"]):
                raw_data = dict(self._raw_data)
                raw_data["data"] = self._data_disp
                edc_norm = False
        return _plot_mdc_curve(
            raw_data,
            selected_ev=self._sel_ev,
            int_window=self._params.sp_int_win.value(),
            edc_norm_enabled=edc_norm,
        )

    def _get_edc(self):
        if self._raw_data is None: return None
        edc_norm = self._cmb_view.currentText() == "EDCnorm"
        raw_data = self._raw_data
        if edc_norm:
            if self._data_disp is None or np.shape(self._data_disp) != np.shape(self._raw_data["data"]):
                self._update_display_data()
            if self._data_disp is not None and np.shape(self._data_disp) == np.shape(self._raw_data["data"]):
                raw_data = dict(self._raw_data)
                raw_data["data"] = self._data_disp
                edc_norm = False
        return _plot_edc_curve(
            raw_data,
            selected_k=self._sel_k,
            edc_norm_enabled=edc_norm,
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

    def _draw_gamma_preview_axvline(self, ax) -> None:
        sp_cx = getattr(self._params, "sp_cx", None)
        if sp_cx is None:
            return
        try:
            value = float(sp_cx.value())
        except Exception:
            return
        line = ax.axvline(
            value, color="#67e8f9", lw=1.0, ls="-.", alpha=0.55, zorder=4,
        )
        try:
            line.set_gid("arpes_gamma_preview")
        except Exception:
            pass

    def _update_gamma_preview(self, value: float) -> None:
        canvas = getattr(self._parent, "_bm_canvas", None)
        if canvas is None or not canvas.fig.axes:
            return
        ax = canvas.fig.axes[0]
        found = None
        for line in ax.lines:
            if getattr(line, "get_gid", lambda: None)() == "arpes_gamma_preview":
                found = line
                break
        try:
            if found is not None:
                found.set_xdata([float(value), float(value)])
            else:
                new = ax.axvline(
                    float(value), color="#67e8f9", lw=1.0, ls="-.", alpha=0.55, zorder=4,
                )
                new.set_gid("arpes_gamma_preview")
                try:
                    setattr(new, "_arpes_plot_overlay", True)
                except Exception:
                    pass
        except Exception:
            return
        try:
            canvas.fig.canvas.draw_idle()
        except Exception:
            pass
