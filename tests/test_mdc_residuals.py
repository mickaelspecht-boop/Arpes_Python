from __future__ import annotations

import unittest

import numpy as np
from matplotlib.figure import Figure

from arpes.physics.plot_compute import draw_waterfall_axes
from arpes.ui.widgets.plots.mdc_diagnostics import debug_mdc_fit, plot_mdc_waterfall_with_fit
from arpes.ui.widgets.plots.mdc_fit import fit_mdc_peak_pairs
from arpes.ui.widgets.plots.mdc_regions import fit_mdc_free_region_result


def _synthetic_peak_pair_data():
    kpar = np.linspace(-0.5, 0.5, 121)
    ev_arr = np.linspace(-0.2, -0.05, 6)
    data = np.zeros((kpar.size, ev_arr.size), dtype=float)
    for j, energy in enumerate(ev_arr):
        k0 = 0.18 + 0.05 * (energy + 0.2) / 0.15
        gamma = 0.04
        left = gamma**2 / ((kpar + k0) ** 2 + gamma**2)
        right = 0.8 * gamma**2 / ((kpar - k0) ** 2 + gamma**2)
        data[:, j] = 0.05 + left + right
    return data, kpar, ev_arr


def _synthetic_free_peak_data():
    kpar = np.linspace(-0.5, 0.6, 221)
    ev_arr = np.linspace(-0.16, -0.04, 7)
    data = np.zeros((kpar.size, ev_arr.size), dtype=float)
    gamma = 0.035
    left_positions = []
    right_positions = []
    for j, energy in enumerate(ev_arr):
        left = -0.16 + 0.04 * (energy + 0.16) / 0.12
        right = 0.30 + 0.03 * (energy + 0.16) / 0.12
        left_positions.append(left)
        right_positions.append(right)
        data[:, j] = (
            0.04
            + 0.75 * gamma**2 / ((kpar - left) ** 2 + gamma**2)
            + 1.00 * gamma**2 / ((kpar - right) ** 2 + gamma**2)
        )
    return data, kpar, ev_arr, np.asarray(left_positions), np.asarray(right_positions)


class TestMdcResiduals(unittest.TestCase):
    def test_fit_result_contains_residuals_aligned_to_fitted_energies(self):
        data, kpar, ev_arr = _synthetic_peak_pair_data()
        fr = fit_mdc_peak_pairs(
            data, kpar, ev_arr,
            n_pairs=1,
            ev_start=-0.2,
            ev_end=-0.05,
            smooth_fit=0.1,
            smooth_detect=0.1,
            gamma_init=0.04,
            gamma_max=0.12,
            kF_init=[0.18],
            center_init=0.0,
            min_amplitude=0.01,
            max_jump=0.2,
            width_mode="symmetric",
            k_min=-0.4,
            k_max=0.4,
        )

        self.assertEqual(len(fr["residuals"]), len(fr["e_fitted"]))
        self.assertEqual(len(fr["fit_curves"]), len(fr["e_fitted"]))
        self.assertEqual(np.asarray(fr["residuals"][0]).shape, np.asarray(fr["fit_kpar"]).shape)
        self.assertLess(float(np.nanmax(np.abs(fr["residuals"][0]))), 1e-6)
        self.assertEqual(fr["width_convention"], "HWHM")
        self.assertAlmostEqual(float(np.nanmedian(fr["gamma_brut"][0])), 0.04, delta=0.003)

    def test_hold_center_and_gamma_constraints_are_recorded(self):
        data, kpar, ev_arr = _synthetic_peak_pair_data()
        fr = fit_mdc_peak_pairs(
            data, kpar, ev_arr,
            n_pairs=1,
            ev_start=-0.2,
            ev_end=-0.05,
            smooth_fit=0.1,
            smooth_detect=0.1,
            gamma_init=0.04,
            gamma_max=0.12,
            kF_init=[0.18],
            center_init=0.0,
            min_amplitude=0.01,
            max_jump=0.2,
            width_mode="symmetric",
            k_min=-0.4,
            k_max=0.4,
            hold_center=True,
            hold_gamma=True,
        )

        self.assertTrue(fr["fit_constraints"]["hold_center"])
        self.assertTrue(fr["fit_constraints"]["hold_gamma"])
        self.assertLess(float(np.nanmax(np.abs(fr["xg"]))), 1e-6)
        self.assertAlmostEqual(float(np.nanmedian(fr["gamma_brut"][0])), 0.04, delta=1e-5)

    def test_free_peaks_recovers_non_symmetric_mdc(self):
        data, kpar, ev_arr, left, right = _synthetic_free_peak_data()
        fr = fit_mdc_peak_pairs(
            data, kpar, ev_arr,
            n_pairs=1,
            ev_start=-0.16,
            ev_end=-0.04,
            smooth_fit=0.1,
            smooth_detect=0.1,
            gamma_init=0.035,
            gamma_max=0.10,
            kF_init=[0.24],
            center_init=0.0,
            min_amplitude=0.01,
            max_jump=0.2,
            width_mode="free",
            k_min=-0.35,
            k_max=0.45,
        )

        self.assertEqual(fr["width_mode"], "free")
        np.testing.assert_allclose(fr["kF_minus"][0], left, atol=0.01)
        np.testing.assert_allclose(fr["kF_plus"][0], right, atol=0.01)
        self.assertAlmostEqual(float(np.nanmedian(fr["gamma_brut"][0])), 0.035, delta=0.004)
        self.assertEqual(len(fr["residuals"]), len(fr["e_fitted"]))

    def test_debug_mdc_fit_accepts_free_peaks_mode(self):
        data, kpar, ev_arr, left, right = _synthetic_free_peak_data()
        r = debug_mdc_fit(
            data, kpar, ev_arr,
            energy=float(ev_arr[0]),
            n_pairs=1,
            smooth_fit=0.1,
            smooth_detect=0.1,
            gamma_init=0.035,
            gamma_max=0.10,
            kF_init=[0.24],
            center_init=0.0,
            width_mode="free",
            k_min=-0.35,
            k_max=0.45,
            verbose=False,
        )
        try:
            self.assertTrue(r["success"])
            self.assertAlmostEqual(float(r["k0"][0]), 0.5 * abs(right[0] - left[0]), delta=0.01)
            self.assertEqual(r["fit_result"]["width_mode"], "free")
        finally:
            r["ax"].figure.clf()

    def test_free_region_result_is_results_compatible_without_mirrored_branch(self):
        kpar = np.linspace(-0.4, 0.6, 201)
        ev_arr = np.linspace(-0.16, -0.04, 7)
        data = np.zeros((kpar.size, ev_arr.size), dtype=float)
        gamma = 0.032
        right = []
        for j, energy in enumerate(ev_arr):
            pos = 0.22 + 0.05 * (energy + 0.16) / 0.12
            right.append(pos)
            data[:, j] = 0.03 + 0.9 * gamma**2 / ((kpar - pos) ** 2 + gamma**2)
        fr = fit_mdc_free_region_result(
            data, kpar, ev_arr,
            k_min=0.10,
            k_max=0.35,
            ev_start=-0.16,
            ev_end=-0.04,
            n_lor=1,
            smooth_fit=0.1,
            smooth_detect=0.1,
            gamma_init=gamma,
            gamma_max=0.09,
            min_amplitude=0.01,
            center_init=0.0,
        )

        self.assertEqual(fr["fit_model"], "free_region")
        self.assertEqual(fr["width_convention"], "HWHM")
        self.assertEqual(len(fr["kF_plus"]), 1)
        np.testing.assert_allclose(fr["kF_plus"][0], np.asarray(right), atol=0.01)
        self.assertTrue(np.isnan(fr["kF_minus"][0]).all())
        self.assertAlmostEqual(float(np.nanmedian(fr["gamma_brut"][0])), gamma, delta=0.004)
        self.assertEqual(len(fr["residuals"]), len(fr["e_fitted"]))
        self.assertEqual(len(fr["chi2_red"]), len(fr["e_fitted"]))

    def test_diagnostic_waterfall_adds_residual_axis(self):
        data, kpar, ev_arr = _synthetic_peak_pair_data()
        fr = fit_mdc_peak_pairs(
            data, kpar, ev_arr,
            n_pairs=1,
            ev_start=-0.2,
            ev_end=-0.05,
            smooth_fit=0.1,
            smooth_detect=0.1,
            gamma_init=0.04,
            gamma_max=0.12,
            kF_init=[0.18],
            center_init=0.0,
            min_amplitude=0.01,
            max_jump=0.2,
            width_mode="symmetric",
            k_min=-0.4,
            k_max=0.4,
        )

        fig, axes = plot_mdc_waterfall_with_fit(fr, data_cut=data)
        try:
            self.assertEqual(len(axes), 3)
            self.assertIn("Residus MDC", axes[2].get_title())
            self.assertGreater(len(axes[2].lines), 0)
        finally:
            fig.clf()

    def test_ui_waterfall_draws_residual_axis_when_available(self):
        data, kpar, ev_arr = _synthetic_peak_pair_data()
        fr = fit_mdc_peak_pairs(
            data, kpar, ev_arr,
            n_pairs=1,
            ev_start=-0.2,
            ev_end=-0.05,
            smooth_fit=0.1,
            smooth_detect=0.1,
            gamma_init=0.04,
            gamma_max=0.12,
            kF_init=[0.18],
            center_init=0.0,
            min_amplitude=0.01,
            max_jump=0.2,
            width_mode="symmetric",
            k_min=-0.4,
            k_max=0.4,
        )
        fig = Figure()
        ax = fig.add_subplot(211)
        residual_ax = fig.add_subplot(212)

        ok = draw_waterfall_axes(
            ax, data, kpar, ev_arr,
            bounds=(-0.4, 0.4, -0.2, -0.05),
            n_target=6,
            fit_result=fr,
            n_pairs=1,
            residual_ax=residual_ax,
        )

        self.assertTrue(ok)
        self.assertIn("MDC residuals", residual_ax.get_title())
        self.assertGreater(len(residual_ax.lines), 0)


if __name__ == "__main__":
    unittest.main()
