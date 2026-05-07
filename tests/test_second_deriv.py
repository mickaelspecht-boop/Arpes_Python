from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.plot_compute import compute_bandmap_display, compute_second_deriv_e


class TestSecondDerivativeE(unittest.TestCase):
    def test_gaussian_peak_has_positive_second_deriv_at_center(self):
        kpar = np.linspace(-0.2, 0.2, 5)
        ev = np.linspace(-0.2, 0.0, 101)
        center = -0.08
        sigma = 0.02
        profile = np.exp(-0.5 * ((ev - center) / sigma) ** 2)
        data = np.tile(profile, (kpar.size, 1))

        out = compute_second_deriv_e(data, kpar, ev, sigma_smooth=0.0)
        idx = int(np.argmin(np.abs(ev - center)))

        self.assertGreater(out[2, idx], 0.0)
        self.assertEqual(int(np.nanargmax(out[2])), idx)

    def test_display_mode_masks_above_ef(self):
        kpar = np.linspace(-0.2, 0.2, 5)
        ev = np.linspace(-0.1, 0.1, 21)
        data = np.tile(np.exp(-0.5 * ((ev + 0.04) / 0.02) ** 2), (kpar.size, 1))
        result = compute_bandmap_display(
            {"data": data, "kpar": kpar, "ev_arr": ev},
            mode="2nd deriv E",
            edc_norm_enabled=False,
        )

        self.assertTrue(np.isnan(result.data[:, ev > 0]).all())
        self.assertTrue(np.isfinite(result.data[:, ev <= 0]).any())


if __name__ == "__main__":
    unittest.main()
