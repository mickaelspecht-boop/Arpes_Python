from __future__ import annotations

import unittest

import numpy as np

from arpes.ui.widgets.plots.mdc_fit import fit_mdc_peak_pairs


def _flat_band(noise: float, seed: int):
    """Flat (non-dispersing) band so kF jitter is pure noise, not dispersion."""
    rng = np.random.default_rng(seed)
    kpar = np.linspace(-0.5, 0.5, 121)
    ev_arr = np.linspace(-0.20, -0.02, 41)
    k0, gamma = 0.18, 0.04
    data = np.zeros((kpar.size, ev_arr.size), dtype=float)
    for j in range(ev_arr.size):
        left = gamma**2 / ((kpar + k0) ** 2 + gamma**2)
        right = gamma**2 / ((kpar - k0) ** 2 + gamma**2)
        data[:, j] = 0.05 + left + right
    data += noise * rng.standard_normal(data.shape)
    return data, kpar, ev_arr


def _fit(data, kpar, ev_arr, window):
    return fit_mdc_peak_pairs(
        data, kpar, ev_arr, n_pairs=1, ev_start=-0.20, ev_end=-0.02,
        smooth_fit=0.1, smooth_detect=0.1, gamma_init=0.04, gamma_max=0.12,
        kF_init=[0.18], center_init=0.0, min_amplitude=0.01, max_jump=0.2,
        width_mode="symmetric", k_min=-0.4, k_max=0.4,
        mdc_energy_window=window,
    )


class TestMdcEnergyWindow(unittest.TestCase):
    def test_window_reduces_kf_jitter_on_flat_band(self):
        data, kpar, ev_arr = _flat_band(noise=0.06, seed=3)
        kf0 = np.asarray(_fit(data, kpar, ev_arr, 0.0)["kF_plus"][0], dtype=float)
        kfw = np.asarray(_fit(data, kpar, ev_arr, 0.03)["kF_plus"][0], dtype=float)
        std0 = float(np.nanstd(kf0))
        stdw = float(np.nanstd(kfw))
        # Integrating ±15 meV must make the (flat) kF noticeably smoother.
        self.assertLess(stdw, std0)

    def test_window_zero_is_unchanged_behaviour(self):
        data, kpar, ev_arr = _flat_band(noise=0.0, seed=0)
        kf = np.asarray(_fit(data, kpar, ev_arr, 0.0)["kF_plus"][0], dtype=float)
        # Clean flat band, no window: kF sits at ~0.18 everywhere.
        self.assertTrue(np.all(np.abs(kf[np.isfinite(kf)] - 0.18) < 0.02))


if __name__ == "__main__":
    unittest.main()
