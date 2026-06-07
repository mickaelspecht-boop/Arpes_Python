from __future__ import annotations

import unittest

import numpy as np
from matplotlib.figure import Figure

from arpes.physics.plot_compute import draw_waterfall_axes
from arpes.ui.widgets.plots.mdc_diagnostics import plot_mdc_waterfall_with_fit
from arpes.ui.widgets.plots.mdc_fit import fit_mdc_peak_pairs


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
