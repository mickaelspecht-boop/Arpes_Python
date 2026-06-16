"""UI plot controller for ArpesExplorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from arpes.physics.norm import remove_grid_artifact as remove_detector_grid_artifact
from arpes.physics.plot_compute import (
    BandmapAxesState,
    DerivParams,
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
from arpes.ui.controllers.plot_model_helpers import (
    axis_cache_signature as _axis_cache_signature,
    build_model_pairs,
)

PAIR_COLORS = ["#ff8c00", "#00e5ff", "#7fff00", "#ff44cc"]


class PlotController:
    # P3.1: writes through to parent are allow-listed (fail-loud on typo).
    # _kf_drag_lines is written by the mdc_edc_drawer helper.
    _OWN_ATTRS = frozenset({"_parent"})
    _PARENT_WRITES = frozenset({
        "_data_disp", "_data_disp_ev", "_data_disp_kpar", "_disp_cache_key",
        "_distortion_display_info", "_grid_display_info", "_kf_drag_lines",
    })

    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
        elif name in self._PARENT_WRITES:
            setattr(self._parent, name, value)
        else:
            raise AttributeError(
                f"{type(self).__name__} refuses to write '{name}': missing from "
                "_PARENT_WRITES (typo?). Add it to _PARENT_WRITES "
                "if the parent attribute is legitimate."
            )

    def _on_scroll_zoom(self, event):
        """Mouse-wheel zoom centered on the cursor position in a matplotlib axis."""
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
            # aspect 'auto' : sinon (FS en aspect 'equal') le cadre rétrécit.
            ax.set_aspect("auto")
            # fige la mise en page : sinon tight_layout se relance à chaque
            # draw et redimensionne le cadre (longueur des labels qui change).
            try:
                event.canvas.figure.set_layout_engine("none")
            except Exception:
                pass
            event.canvas.draw_idle()
        except Exception:
            return

    def _deriv_params(self) -> DerivParams:
        """Read the SecDev/Curvature tuning spinboxes (fall back to defaults)."""
        p = self._parent
        dp = DerivParams()
        try:
            dp.sigma_e_eV = float(p._sp_deriv_sigma_e.value())
            dp.sigma_k_inv_a = float(p._sp_deriv_sigma_k.value())
            dp.c0_alpha = float(p._sp_deriv_c0.value())
        except (AttributeError, RuntimeError):
            pass  # widgets not built yet (headless / early load)
        return dp

    def _update_display_data(self):
        if self._raw_data is None:
            return
        d    = self._raw_data
        raw  = d["data"]
        mode = self._cmb_view.currentText()
        if mode in ("SecDev", "Curvature") and (d.get("metadata", {}) or {}).get("axes_raw_view"):
            # Derivative smoothing scales are in eV / Å⁻¹: meaningless on raw
            # θ/E axes. Refuse loudly and fall back to Raw display.
            self._status(
                f"{mode} not available in browse-only mode (raw θ/E axes) — showing Raw."
            )
            mode = "Raw"

        entry = self._current_entry()
        grid_cfg_active = entry.grid_correction if entry and entry.grid_correction.get("enabled") else None
        grid_key = (
            grid_cfg_active.get("strength"),
            grid_cfg_active.get("center_radius"),
            grid_cfg_active.get("peak_sensitivity"),
            grid_cfg_active.get("notch_width"),
        ) if grid_cfg_active else None
        from arpes.physics.distortion import (
            cache_signature as _distortion_cache_sig,
            is_distortion_active as _distortion_active,
        )
        bm_dist = getattr(entry, "bm_distortion", None) if entry else None
        distortion_cfg_active = bm_dist if (bm_dist and _distortion_active(bm_dist)) else None
        distortion_key = _distortion_cache_sig(distortion_cfg_active)
        raw_key = getattr(self, "_current_raw_load_cache_key", None)
        deriv_params = self._deriv_params()
        deriv_key = (
            (deriv_params.sigma_e_eV, deriv_params.sigma_k_inv_a,
             deriv_params.c0_alpha, deriv_params.ef_margin_eV)
            if mode in ("SecDev", "Curvature") else None
        )
        cache_key = (
            raw_key,
            id(raw),
            tuple(np.asarray(raw).shape),
            mode,
            grid_key,
            distortion_key,
            deriv_key,
            _axis_cache_signature(d["kpar"]),
            _axis_cache_signature(d["ev_arr"]),
        )
        if cache_key == self._disp_cache_key and self._data_disp is not None:
            return  # nothing changed that affects BM display

        display_cache = getattr(self, "_display_cache", None)
        if display_cache is not None and cache_key in display_cache:
            cached = display_cache.pop(cache_key)
            if len(cached) == 2:
                disp_cached, info_cached = cached
                distortion_info_cached = {}
                kpar_cached = None
                ev_cached = None
            else:
                disp_cached, info_cached, distortion_info_cached, kpar_cached, ev_cached = cached
            display_cache[cache_key] = cached
            self._data_disp = disp_cached
            self._grid_display_info = dict(info_cached or {})
            self._distortion_display_info = dict(distortion_info_cached or {})
            self._data_disp_kpar = np.asarray(kpar_cached) if kpar_cached is not None else None
            self._data_disp_ev = np.asarray(ev_cached) if ev_cached is not None else None
            self._disp_cache_key = cache_key
            return

        result = compute_bandmap_display(
            d,
            mode=mode,
            edc_norm_enabled=mode in ("EDCnorm", "SecDev", "Curvature"),
            grid_correction=grid_cfg_active,
            grid_artifact_fn=remove_detector_grid_artifact,
            distortion_correction=distortion_cfg_active,
            deriv_params=deriv_params,
        )
        self._data_disp = result.data
        self._grid_display_info = result.grid_info
        self._distortion_display_info = getattr(result, "distortion_info", {})
        rk = getattr(result, "kpar", None)
        re_ = getattr(result, "ev", None)
        self._data_disp_kpar = np.asarray(rk) if rk is not None else None
        self._data_disp_ev = np.asarray(re_) if re_ is not None else None
        self._disp_cache_key = cache_key
        if display_cache is not None:
            display_cache[cache_key] = (
                result.data,
                dict(result.grid_info or {}),
                dict(getattr(result, "distortion_info", {}) or {}),
                np.asarray(rk) if rk is not None else None,
                np.asarray(re_) if re_ is not None else None,
            )
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
        roi_bounds = self._fit_roi_bounds() if roi_scale and d is not None else None
        cache_key = (
            getattr(self._parent, "_disp_cache_key", None),
            mode,
            bool(roi_scale),
            roi_bounds,
            tuple(np.asarray(disp).shape),
        )
        cache = getattr(self._parent, "_color_kwargs_cache", None)
        if cache is not None and cache_key in cache:
            cmap, ckw = cache.pop(cache_key)
            cache[cache_key] = (cmap, dict(ckw))
            return cmap, dict(ckw)

        ref = self._fit_roi_data(disp, d["kpar"], d["ev_arr"]) if roi_bounds is not None else disp
        cmap, ckw = _plot_map_color_kwargs(disp, mode=mode, roi_ref=ref)
        if cache is not None:
            cache[cache_key] = (cmap, dict(ckw))
            max_items = int(getattr(self._parent, "_color_kwargs_cache_max", 48) or 48)
            while len(cache) > max_items:
                cache.popitem(last=False)
        return cmap, ckw

    @staticmethod
    def _raw_axes_info(d) -> tuple[tuple[str, str] | None, str | None]:
        """Browse-only raw view: (xlabel, ylabel) + physicist note, else (None, None)."""
        meta = (d or {}).get("metadata", {}) or {}
        if not meta.get("axes_raw_view"):
            return None, None
        labels = (
            str(meta.get("axes_raw_xlabel") or "θ (°) [raw]"),
            str(meta.get("axes_raw_ylabel") or "E (eV) [raw]"),
        )
        return labels, "raw axes — θ is non-linear in k// beyond ±15°"

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

    def _draw_current_view(self, *, include_curves: bool = True,
                           overlays_only: bool = False):
        tabs = getattr(self, "_tabs", None)
        index = tabs.currentIndex() if tabs is not None else 0
        if index == 0:
            self._draw_bm(overlays_only=overlays_only)
        elif index == 1:
            self._draw_mdc_energy_map()
            if include_curves:
                if hasattr(self, "_mdc_fit_tabs") and self._mdc_fit_tabs.currentIndex() == 1:
                    self._draw_mdc_waterfall()
                else:
                    self._draw_mdc_edc()
        elif index == 3:
            self._draw_fs_tab()

    def _reset_bm_view(self):
        """Recale la BM sur l'étendue des données (kpar/ev courant).

        pcolormesh + autoscale_on(False) → relim/autoscale matplotlib ne
        restaure PAS le cadre data (cause du « dur de revenir au graphe
        initial après zoom »). On fixe les limites explicitement.
        """
        if self._data_disp is None:
            return
        d = self._raw_data
        disp = self._data_disp
        kpar = getattr(self, "_data_disp_kpar", None)
        ev = getattr(self, "_data_disp_ev", None)
        if (kpar is None or ev is None
                or kpar.size != disp.shape[0] or ev.size != disp.shape[1]):
            kpar = d["kpar"]; ev = d["ev_arr"]
        ax = self._bm_canvas.ax
        kp = np.asarray(kpar, dtype=float)
        ee = np.asarray(ev, dtype=float)
        try:
            if np.isfinite(kp).any():
                ax.set_xlim(float(np.nanmin(kp)), float(np.nanmax(kp)))
            if np.isfinite(ee).any():
                ax.set_ylim(float(np.nanmin(ee)), float(np.nanmax(ee)))
        except (ValueError, TypeError):
            return
        self._bm_canvas.redraw()

    def _update_ef_banner(self) -> None:
        # P4.6: "EF not calibrated" banner if neither polynomial fit nor EF
        # offset is set (offset still at historical default = never touched).
        lbl = getattr(self._parent, "_lbl_ef_uncal", None)
        if lbl is None:
            return
        from arpes.core.session import DEFAULT_EF_OFFSET_EV
        entry = self._current_entry()
        uncalibrated = bool(entry is not None) and not getattr(entry, "ef_correction", None) and (
            abs(float(getattr(entry, "ef_offset", DEFAULT_EF_OFFSET_EV)) - DEFAULT_EF_OFFSET_EV) < 1e-9
        )
        lbl.setVisible(uncalibrated)

    def _draw_bm(self, *, overlays_only: bool = False):
        self._update_ef_banner()
        if self._data_disp is None:
            return
        d    = self._raw_data
        disp = self._data_disp
        mode = self._cmb_view.currentText()
        kpar = getattr(self, "_data_disp_kpar", None)
        ev = getattr(self, "_data_disp_ev", None)
        if kpar is None or ev is None or kpar.size != disp.shape[0] or ev.size != disp.shape[1]:
            kpar = d["kpar"]; ev = d["ev_arr"]

        ax = self._bm_canvas.ax
        # FAST PATH (C) : changement purement cosmétique (théorie, distorsion
        # preview, marqueur Γ…). Le mesh + les couleurs sont déjà à l'écran
        # et inchangés → on ne recalcule NI les color kwargs (percentile sur
        # tout le tableau) NI le pcolormesh ; on ne rafraîchit que les
        # overlays. Limites intouchées → zoom préservé.
        state = getattr(self._parent, "_bm_plot_state", None)
        plot_key = getattr(self._parent, "_disp_cache_key", None)
        data_key = getattr(self._parent, "_bm_plot_data_key", None)
        if (
            overlays_only
            and state is not None
            and getattr(state, "mesh", None) is not None
            and plot_key == data_key
        ):
            self._clear_plot_overlays(ax)
            before = self._axis_artist_snapshot(ax)
            self._draw_fit_roi_overlay(ax)
            self._draw_theory_overlay(ax)
            self._draw_kf_overlay(ax)
            self._draw_gamma_preview_axvline(ax)
            self._draw_distortion_preview_overlay(ax)
            self._draw_ef_label(ax, horizontal=True)
            self._tag_new_plot_overlays(ax, before)
            self._bm_canvas.redraw()
            return

        self._clear_plot_overlays(ax)
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=False)
        int_win = self._params.sp_int_win.value()
        fname = Path(d["path"]).name
        state = getattr(self._parent, "_bm_plot_state", None)
        if state is None:
            state = BandmapAxesState()
        plot_key = getattr(self._parent, "_disp_cache_key", None)
        reset_limits = plot_key != getattr(self._parent, "_bm_plot_data_key", None)
        raw_labels, raw_note = self._raw_axes_info(d)
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
            axis_labels=raw_labels, axis_note=raw_note,
        )
        self._parent._bm_plot_data_key = plot_key

        before = self._axis_artist_snapshot(ax)
        self._draw_fit_roi_overlay(ax)
        self._draw_theory_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_gamma_preview_axvline(ax)
        self._draw_distortion_preview_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)
        self._tag_new_plot_overlays(ax, before)
        self._bm_canvas.redraw()

    def _draw_mdc_energy_map(self):
        """Mini BM visible in the MDC Fit tab to choose E,k without returning to BM."""
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
        raw_labels, raw_note = self._raw_axes_info(d)
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
            axis_labels=raw_labels, axis_note=raw_note,
        )
        self._parent._mdc_map_plot_data_key = plot_key
        before = self._axis_artist_snapshot(ax)
        from arpes.ui.controllers.fit_overlay_drawer import apply_mdc_map_view
        # Full map when ≥1 zone exists (every zone visible); else legacy ROI zoom.
        apply_mdc_map_view(self, ax)
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
        legend = ax.get_legend()
        if legend is not None:
            try:
                legend.remove()
            except Exception:
                pass
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
        d_raw = self._raw_data
        if (d_raw.get("metadata", {}) or {}).get("axes_raw_view"):
            # Allowed in browse-only (pure display), but the k/E windows of
            # the spinboxes are now θ°/E kinetic — warn so nobody fits by eye
            # against a window meant for π/a.
            data, kpar, ev = d_raw["data"], d_raw["kpar"], d_raw["ev_arr"]
            self._status("Waterfall on raw θ/E axes (browse-only) — windows are θ° / eV.")
        else:
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

    # Fit-overlay drawing methods live in fit_overlay_drawer.py; thin
    # wrappers keep external callers (other controllers, tests) working.
    def _draw_kf_overlay(self, ax):
        from arpes.ui.controllers.fit_overlay_drawer import draw_kf_overlay
        return draw_kf_overlay(self, ax)

    def _axis_state_mismatch(self, fr: dict) -> bool:
        from arpes.ui.controllers.fit_overlay_drawer import axis_state_mismatch
        return axis_state_mismatch(self, fr)

    def _draw_zone_overlays(self, ax) -> None:
        from arpes.ui.controllers.fit_overlay_drawer import draw_zone_overlays
        return draw_zone_overlays(self, ax)

    def _scatter_kf_with_chi2(self, ax, k_values, ev_f, bad_mask, color, marker,
                              *, kf_std=None) -> None:
        from arpes.ui.controllers.fit_overlay_drawer import scatter_kf_with_chi2
        return scatter_kf_with_chi2(
            ax, k_values, ev_f, bad_mask, color, marker, kf_std=kf_std,
        )

    def _draw_fit_annotations(self, ax, fr: dict) -> None:
        from arpes.ui.controllers.fit_overlay_drawer import draw_fit_annotations
        return draw_fit_annotations(self, ax, fr)

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

    # ---------- A: drag kF init markers + snap-to-peak ----------
    # kF drag handlers extracted to kf_drag_handlers.py — thin wrappers below.
    def _install_kf_drag_handlers(self) -> None:
        from arpes.ui.controllers.kf_drag_handlers import install_kf_drag_handlers
        return install_kf_drag_handlers(self)

    def _on_kf_pick(self, event) -> None:
        from arpes.ui.controllers.kf_drag_handlers import on_kf_pick
        return on_kf_pick(self, event)

    def _on_kf_motion(self, event) -> None:
        from arpes.ui.controllers.kf_drag_handlers import on_kf_motion
        return on_kf_motion(self, event)

    def _on_kf_release(self, event) -> None:
        from arpes.ui.controllers.kf_drag_handlers import on_kf_release
        return on_kf_release(self, event)

    def _snap_to_mdc_peak(self, x_target: float) -> float | None:
        from arpes.ui.controllers.kf_drag_handlers import snap_to_mdc_peak
        return snap_to_mdc_peak(self, x_target)

    def _on_kf_init_drag(self, pair_idx: int, sign: int, kf_new: float) -> None:
        from arpes.ui.controllers.kf_drag_handlers import on_kf_init_drag
        return on_kf_init_drag(self, pair_idx, sign, kf_new)

    def _draw_mdc_edc(self):
        from arpes.ui.controllers.mdc_edc_drawer import draw_mdc_edc
        return draw_mdc_edc(self)

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
