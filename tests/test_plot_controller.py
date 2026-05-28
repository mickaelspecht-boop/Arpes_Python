"""Tests de préparation des données de plot (`arpes_plot_controller`)."""

from __future__ import annotations

import unittest
from collections import OrderedDict
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
import numpy as np

import arpes.ui.controllers.plot_controller as plot_ctrl_mod
from arpes.physics.plot_compute import (
    BandmapAxesState,
    _compute_below_ef_only,
    apply_edcnorm,
    compute_bandmap_display,
    display_grid_config,
    edc_curve,
    fit_roi_bounds,
    fit_roi_data,
    map_color_kwargs,
    mdc_curve,
    prepare_waterfall_data,
    draw_bandmap_axes,
    scroll_zoom_limits,
)
from arpes.ui.controllers.plot_controller import PlotController


class TestPlotController(unittest.TestCase):
    def _raw(self):
        data = np.asarray([
            [2.0, 4.0, 6.0],
            [4.0, 8.0, 12.0],
        ])
        return {
            "data": data,
            "kpar": np.asarray([-0.1, 0.1]),
            "ev_arr": np.asarray([-0.1, 0.0, 0.1]),
        }

    def test_apply_edcnorm(self):
        out = apply_edcnorm(self._raw()["data"])
        np.testing.assert_allclose(np.nanmean(out, axis=0), [1.0, 1.0, 1.0])

    def test_raw_is_unmodified(self):
        raw = self._raw()
        out = compute_bandmap_display(raw, mode="Raw", edc_norm_enabled=True)
        self.assertIs(out.data, raw["data"])

    def test_distortion_applies_to_fs_mean_bandmap(self):
        kpar = np.linspace(-1.0, 1.0, 41)
        ev = np.linspace(-0.4, 0.1, 31)
        kk, ee = np.meshgrid(kpar, ev, indexing="ij")
        data = np.exp(-((kk - 0.2 * ee) ** 2) / 0.04)
        raw = {
            "data": data,
            "kpar": kpar,
            "ev_arr": ev,
            "metadata": {
                "fs_data": np.zeros((3, kpar.size, ev.size), dtype=float),
                "fs_kx": kpar,
                "fs_ky": np.arange(3),
                "fs_energy": ev,
            },
        }
        cfg = {
            "enabled": True,
            "trapezoid": {
                "enabled": True,
                "slope_left": 0.05,
                "slope_right": -0.05,
                "pivot_ev": -0.1,
            },
            "parabola": {"enabled": False, "a": 0.0, "k0": 0.0},
        }

        out = compute_bandmap_display(
            raw,
            mode="Raw",
            edc_norm_enabled=False,
            distortion_correction=cfg,
        )

        self.assertTrue(out.distortion_info.get("applied"))
        self.assertFalse(np.array_equal(out.data, data))

    def test_update_display_data_reuses_cross_file_display_cache(self):
        raw = self._raw()
        calls = []
        old_compute = plot_ctrl_mod.compute_bandmap_display

        def fake_compute(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(
                data=np.asarray(raw["data"]) + 1.0,
                grid_info={"ok": True},
                distortion_info={"distorted": True},
                kpar=np.asarray([-0.2, 0.2]),
                ev=np.asarray([-0.3, 0.0, 0.3]),
            )

        class _Combo:
            def currentText(self):
                return "Raw"

        parent = SimpleNamespace(
            _raw_data=raw,
            _cmb_view=_Combo(),
            _data_disp=None,
            _grid_display_info={},
            _disp_cache_key=None,
            _current_raw_load_cache_key=("raw", "file"),
            _display_cache=OrderedDict(),
            _display_cache_max=4,
            _data_disp_kpar=None,
            _data_disp_ev=None,
            _distortion_display_info={},
        )
        parent._current_entry = lambda: SimpleNamespace(grid_correction={})
        ctrl = PlotController(parent)

        plot_ctrl_mod.compute_bandmap_display = fake_compute
        try:
            ctrl._update_display_data()
            parent._data_disp = None
            parent._disp_cache_key = None
            ctrl._update_display_data()
        finally:
            plot_ctrl_mod.compute_bandmap_display = old_compute

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(parent._data_disp, raw["data"] + 1.0)
        self.assertEqual(parent._grid_display_info, {"ok": True})
        self.assertEqual(parent._distortion_display_info, {"distorted": True})
        np.testing.assert_allclose(parent._data_disp_kpar, [-0.2, 0.2])
        np.testing.assert_allclose(parent._data_disp_ev, [-0.3, 0.0, 0.3])

    def test_axis_cache_signature_detects_internal_axis_change(self):
        a = np.asarray([-1.0, -0.2, 0.0, 1.0])
        b = np.asarray([-1.0, -0.1, 0.0, 1.0])

        self.assertNotEqual(
            plot_ctrl_mod._axis_cache_signature(a),
            plot_ctrl_mod._axis_cache_signature(b),
        )

    def test_mdc_edc_use_cached_edcnorm_display(self):
        raw = self._raw()
        norm = apply_edcnorm(raw["data"])

        class _Combo:
            def currentText(self):
                return "EDCnorm"

        class _Spin:
            def value(self):
                return 0.01

        parent = SimpleNamespace(
            _raw_data=raw,
            _data_disp=norm,
            _sel_ev=0.0,
            _sel_k=0.1,
            _cmb_view=_Combo(),
            _params=SimpleNamespace(sp_int_win=_Spin()),
        )
        parent._update_display_data = lambda: None
        ctrl = PlotController(parent)

        old_apply = plot_ctrl_mod._plot_mdc_curve.__globals__["apply_edcnorm"]
        calls = []

        def forbidden_apply(data):
            calls.append(data)
            raise AssertionError("EDCnorm should come from display cache")

        plot_ctrl_mod._plot_mdc_curve.__globals__["apply_edcnorm"] = forbidden_apply
        try:
            kpar, mdc = ctrl._get_mdc()
            ev, edc = ctrl._get_edc()
        finally:
            plot_ctrl_mod._plot_mdc_curve.__globals__["apply_edcnorm"] = old_apply

        self.assertEqual(calls, [])
        np.testing.assert_allclose(kpar, raw["kpar"])
        np.testing.assert_allclose(ev, raw["ev_arr"])
        np.testing.assert_allclose(mdc, norm[:, 1])
        np.testing.assert_allclose(edc, norm[1, :])

    def test_edcnorm_mode_normalizes(self):
        raw = self._raw()
        out = compute_bandmap_display(raw, mode="EDCnorm", edc_norm_enabled=False)
        np.testing.assert_allclose(np.nanmean(out.data, axis=0), [1.0, 1.0, 1.0])

    def test_derivative_modes_mask_above_ef(self):
        data = np.arange(20, dtype=float).reshape(4, 5)
        ev = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2])

        def fake_compute(d, _k, _e):
            return np.ones_like(d)

        out = _compute_below_ef_only(fake_compute, data, np.arange(4), ev)
        self.assertEqual(out.shape, data.shape)
        np.testing.assert_allclose(out[:, :3], 1.0)
        self.assertTrue(np.isnan(out[:, 3:]).all())

    def test_grid_config_clamps_strength(self):
        self.assertEqual(display_grid_config({"strength": 2.0})["strength"], 1.0)
        self.assertEqual(display_grid_config({"strength": -1.0})["strength"], 0.0)

    def test_grid_artifact_fn_called_and_info_enriched(self):
        raw = self._raw()
        calls = []

        def fake(arr, axis=0, **cfg):
            calls.append((arr.copy(), axis, cfg))
            return arr + 1.0, {"masked_pixels": 3}

        out = compute_bandmap_display(
            raw,
            mode="Raw",
            edc_norm_enabled=False,
            grid_correction={"enabled": True, "strength": 0.5},
            grid_artifact_fn=fake,
        )
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(out.data, raw["data"] + 1.0)
        self.assertEqual(out.grid_info["method"], "display_fft2mask")
        self.assertEqual(out.grid_info["view_mode"], "Raw")
        self.assertEqual(out.grid_info["target"], "display")
        self.assertEqual(out.grid_info["strength"], 0.5)

    def test_grid_artifact_exception_is_reported(self):
        raw = self._raw()

        def bad(*args, **kwargs):
            raise RuntimeError("boom")

        out = compute_bandmap_display(
            raw,
            mode="Raw",
            edc_norm_enabled=False,
            grid_correction={"enabled": True},
            grid_artifact_fn=bad,
        )
        self.assertIn("boom", out.grid_info["error"])

    def test_fit_roi_bounds_clip_to_data(self):
        raw = self._raw()
        bounds = fit_roi_bounds(
            raw["kpar"], raw["ev_arr"],
            k_min=-1.0, k_max=1.0, ev_start=-1.0, ev_end=1.0,
        )
        self.assertEqual(bounds, (-0.1, 0.1, -0.1, 0.1))

    def test_fit_roi_data_extracts_region(self):
        raw = self._raw()
        disp = np.arange(6).reshape(2, 3)
        out = fit_roi_data(disp, raw["kpar"], raw["ev_arr"], (-0.1, -0.1, -0.1, 0.0))
        # k bound degenerate after exact mask still selects first k row; energy two columns.
        np.testing.assert_allclose(out, [[0, 1]])

    def test_map_color_kwargs_raw_and_derivative(self):
        disp = np.asarray([[0.0, 1.0], [2.0, 100.0]])
        cmap, kwargs = map_color_kwargs(disp, mode="Raw")
        self.assertEqual(cmap, "inferno")
        self.assertEqual(kwargs["vmin"], 0)
        self.assertGreater(kwargs["vmax"], 0)
        cmap2, kwargs2 = map_color_kwargs(np.asarray([[-1.0, 0.0], [2.0, 4.0]]), mode="SecDev")
        self.assertEqual(cmap2, "hot_r")
        self.assertEqual(kwargs2["vmin"], 0)

    def test_draw_bandmap_axes_reuses_quadmesh_with_state(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        kpar = np.linspace(-1.0, 1.0, 5)
        ev = np.linspace(-0.3, 0.1, 4)
        disp = np.arange(20, dtype=float).reshape(5, 4)
        state = BandmapAxesState()

        state = draw_bandmap_axes(
            ax, kpar=kpar, ev=ev, disp=disp, cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 20.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="one",
            state=state,
        )
        first_mesh = state.mesh
        state = draw_bandmap_axes(
            ax, kpar=kpar, ev=ev, disp=disp + 1.0, cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 21.0},
            sel_ev=-0.1, sel_k=0.2, int_win=0.02, title="two",
            state=state,
        )

        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        self.assertEqual(len(meshes), 1)
        self.assertIs(state.mesh, first_mesh)
        self.assertEqual(len(state.base_artists), 5)

    def test_draw_bandmap_axes_preserves_limits_when_reusing_mesh(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        kpar = np.linspace(-1.0, 1.0, 5)
        ev = np.linspace(-0.3, 0.1, 4)
        state = BandmapAxesState()
        state = draw_bandmap_axes(
            ax, kpar=kpar, ev=ev, disp=np.ones((5, 4)), cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 1.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="one", state=state,
        )
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.1, 0.0)
        draw_bandmap_axes(
            ax, kpar=kpar, ev=ev, disp=np.ones((5, 4)) * 2.0, cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 2.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="two", state=state,
        )

        self.assertEqual(ax.get_xlim(), (-0.2, 0.2))
        self.assertEqual(ax.get_ylim(), (-0.1, 0.0))

    def test_draw_bandmap_axes_rebuilds_on_shape_change(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        state = BandmapAxesState()
        state = draw_bandmap_axes(
            ax, kpar=np.linspace(-1.0, 1.0, 5), ev=np.linspace(-0.3, 0.1, 4),
            disp=np.ones((5, 4)), cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 1.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="one", state=state,
        )
        first_mesh = state.mesh
        state = draw_bandmap_axes(
            ax, kpar=np.linspace(-1.0, 1.0, 6), ev=np.linspace(-0.3, 0.1, 4),
            disp=np.ones((6, 4)), cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 1.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="two", state=state,
        )

        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        self.assertEqual(len(meshes), 1)
        self.assertIsNot(state.mesh, first_mesh)
        # bornes recalées tight sur l'étendue des données du nouveau fichier,
        # autoscale coupé (les overlays ne doivent plus dilater le cadre)
        self.assertEqual(ax.get_xlim(), (-1.0, 1.0))
        self.assertEqual(ax.get_ylim(), (-0.3, 0.1))
        self.assertFalse(ax.get_autoscale_on())

    def test_draw_bandmap_axes_rebuilds_on_internal_axis_change(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        state = BandmapAxesState()
        ev = np.linspace(-0.3, 0.1, 4)
        state = draw_bandmap_axes(
            ax, kpar=np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]), ev=ev,
            disp=np.ones((5, 4)), cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 1.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="one", state=state,
        )
        first_mesh = state.mesh
        state = draw_bandmap_axes(
            ax, kpar=np.asarray([-1.0, -0.4, 0.0, 0.4, 1.0]), ev=ev,
            disp=np.ones((5, 4)), cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 1.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="two", state=state,
        )

        self.assertIsNot(state.mesh, first_mesh)

    def test_draw_bandmap_axes_without_state_keeps_clear_fallback(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], color="white")
        result = draw_bandmap_axes(
            ax, kpar=np.linspace(-1.0, 1.0, 5), ev=np.linspace(-0.3, 0.1, 4),
            disp=np.ones((5, 4)), cmap="inferno",
            color_kwargs={"vmin": 0.0, "vmax": 1.0},
            sel_ev=0.0, sel_k=0.0, int_win=0.01, title="fallback",
        )

        self.assertIsNone(result)
        self.assertEqual(len([c for c in ax.collections if isinstance(c, QuadMesh)]), 1)
        self.assertEqual(len(ax.lines), 4)

    def test_prepare_waterfall_data_empty_or_valid(self):
        raw = self._raw()
        empty = prepare_waterfall_data(
            raw["data"], raw["kpar"], raw["ev_arr"],
            bounds=(10.0, 11.0, -0.1, 0.1),
        )
        self.assertIsNone(empty)
        wf = prepare_waterfall_data(
            raw["data"], raw["kpar"], raw["ev_arr"],
            bounds=(-0.1, 0.1, -0.1, 0.1),
            n_target=2,
        )
        self.assertIsNotNone(wf)
        self.assertEqual(wf.data_cut.shape, (2, 3))
        self.assertEqual(wf.k_cut.tolist(), [-0.1, 0.1])
        self.assertEqual(wf.ev_sel.tolist(), [-0.1, 0.0, 0.1])
        self.assertTrue(wf.indices)

    def test_mdc_and_edc_curves(self):
        raw = self._raw()
        kpar, mdc = mdc_curve(
            raw,
            selected_ev=0.01,
            int_window=0.02,
            edc_norm_enabled=False,
        )
        np.testing.assert_allclose(kpar, raw["kpar"])
        np.testing.assert_allclose(mdc, [4.0, 8.0])
        ev, edc = edc_curve(raw, selected_k=0.2, edc_norm_enabled=False)
        np.testing.assert_allclose(ev, raw["ev_arr"])
        np.testing.assert_allclose(edc, [4.0, 8.0, 12.0])

    def test_scroll_zoom_limits_zoom_in_and_preserve_reverse_axis(self):
        xlim, ylim = scroll_zoom_limits(
            (0.0, 10.0), (10.0, 0.0),
            xdata=5.0, ydata=5.0, step=1.0,
        )
        self.assertLess(xlim[1] - xlim[0], 10.0)
        self.assertLess(ylim[0] - ylim[1], 10.0)
        self.assertGreater(ylim[0], ylim[1])


if __name__ == "__main__":
    unittest.main()
