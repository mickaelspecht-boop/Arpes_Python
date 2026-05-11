from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from arpes.physics.plot_compute import compute_bandmap_display, compute_curvature


class TestDerivativeDisplays(unittest.TestCase):
    def test_secdev_display_mode_masks_above_ef(self):
        kpar = np.linspace(-0.2, 0.2, 5)
        ev = np.linspace(-0.1, 0.1, 21)
        data = np.tile(np.exp(-0.5 * ((ev + 0.04) / 0.02) ** 2), (kpar.size, 1))
        result = compute_bandmap_display(
            {"data": data, "kpar": kpar, "ev_arr": ev},
            mode="SecDev",
            edc_norm_enabled=False,
        )

        self.assertTrue(np.isnan(result.data[:, ev > 0]).all())
        self.assertTrue(np.isfinite(result.data[:, ev <= 0]).any())

    def test_curvature_uses_zhang_2d_formula(self):
        kpar = np.linspace(-0.4, 0.4, 21)
        ev = np.linspace(-0.25, 0.0, 31)
        kk, ee = np.meshgrid(kpar, ev, indexing="ij")
        data = np.exp(-0.5 * ((ee + 0.08 - 0.25 * kk) / 0.025) ** 2)

        out = compute_curvature(data, kpar, ev, sigma_k=1.0, sigma_e=1.0)

        smooth = gaussian_filter(data.astype(float), sigma=[1.0, 1.0])
        dI_dE = np.gradient(smooth, ev, axis=1)
        dI_dk = np.gradient(smooth, kpar, axis=0)
        d2I_dk2 = np.gradient(dI_dk, kpar, axis=0)
        d2I_dkdE = np.gradient(dI_dk, ev, axis=1)
        bc = 3
        interior = (slice(bc, -bc), slice(bc, -bc))
        C0 = 0.05 * (
            np.abs(dI_dk[interior]).max() ** 2
            + np.abs(dI_dE[interior]).max() ** 2
        )
        expected = -(
            (C0 + dI_dE**2) * d2I_dk2 - dI_dk * dI_dE * d2I_dkdE
        ) / ((C0 + dI_dk**2 + dI_dE**2) ** 1.5 + 1e-30)
        expected[:bc, :] = np.nan
        expected[-bc:, :] = np.nan
        expected[:, :bc] = np.nan
        expected[:, -bc:] = np.nan

        np.testing.assert_allclose(out, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
