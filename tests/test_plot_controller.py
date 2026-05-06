"""Tests de préparation des données de plot (`arpes_plot_controller`)."""

from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.plot_compute import (
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
    scroll_zoom_limits,
)


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
