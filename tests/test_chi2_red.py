from __future__ import annotations

import unittest

import numpy as np

from arpes.ui.widgets.plots.mdc_fit import fit_mdc_peak_pairs


def _synthetic_data(noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    kpar = np.linspace(-0.5, 0.5, 121)
    ev_arr = np.linspace(-0.2, -0.05, 6)
    data = np.zeros((kpar.size, ev_arr.size), dtype=float)
    for j, energy in enumerate(ev_arr):
        k0 = 0.18 + 0.05 * (energy + 0.2) / 0.15
        gamma = 0.04
        left = gamma**2 / ((kpar + k0) ** 2 + gamma**2)
        right = 0.8 * gamma**2 / ((kpar - k0) ** 2 + gamma**2)
        data[:, j] = 0.05 + left + right
    if noise:
        data += noise * rng.standard_normal(data.shape)
    return data, kpar, ev_arr


def _fit(data, kpar, ev_arr):
    return fit_mdc_peak_pairs(
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


class TestChi2Red(unittest.TestCase):
    def test_chi2_red_shape_matches_fitted_energies(self):
        data, kpar, ev_arr = _synthetic_data()
        fr = _fit(data, kpar, ev_arr)

        self.assertIn("chi2_red", fr)
        self.assertEqual(np.asarray(fr["chi2_red"]).shape, np.asarray(fr["e_fitted"]).shape)
        self.assertTrue(np.isfinite(fr["chi2_red"]).all())

    def test_noisy_fit_has_larger_chi2_red_than_clean_fit(self):
        clean = _fit(*_synthetic_data(noise=0.0))
        noisy = _fit(*_synthetic_data(noise=0.05, seed=4))

        clean_med = float(np.nanmedian(clean["chi2_red"]))
        noisy_med = float(np.nanmedian(noisy["chi2_red"]))
        self.assertLess(clean_med, 1e-10)
        self.assertGreater(noisy_med, clean_med + 1e-5)


if __name__ == "__main__":
    unittest.main()
