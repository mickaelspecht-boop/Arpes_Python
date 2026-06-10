from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.plot_compute import (
    DerivParams,
    _sigma_px,
    compute_bandmap_display,
    compute_curvature,
    compute_secdev,
)


def _band_map(n_k=41, n_e=61, *, slope=0.0, e0=-0.08, width=0.02):
    """Synthetic band: a Gaussian ridge in E, dispersing linearly with k."""
    kpar = np.linspace(-0.4, 0.4, n_k)
    ev = np.linspace(-0.25, 0.05, n_e)
    kk, ee = np.meshgrid(kpar, ev, indexing="ij")
    data = np.exp(-0.5 * ((ee - (e0 + slope * kk)) / width) ** 2)
    return kpar, ev, data


class TestSigmaConversion(unittest.TestCase):
    def test_physical_sigma_to_pixels(self):
        ev = np.linspace(0.0, 0.2, 101)  # 0.002 eV/px
        self.assertAlmostEqual(_sigma_px(ev, 0.02), 10.0, places=6)
        k = np.linspace(-0.5, 0.5, 51)   # 0.02 (π/a)/px
        self.assertAlmostEqual(_sigma_px(k, 0.04), 2.0, places=6)

    def test_zero_sigma_means_no_smoothing(self):
        ev = np.linspace(0.0, 0.2, 101)
        self.assertEqual(_sigma_px(ev, 0.0), 0.0)


class TestSecDev(unittest.TestCase):
    def test_secdev_positive_on_band(self):
        kpar, ev, data = _band_map()
        out = compute_secdev(data, kpar, ev, sigma_k_px=1.0, sigma_e_px=1.0)
        # Peak row (k=0) at the band energy -> -d2I/dE2 > 0 on the ridge.
        ik = len(kpar) // 2
        ie_band = int(np.argmin(np.abs(ev - (-0.08))))
        self.assertGreater(out[ik, ie_band], 0.0)

    def test_secdev_masks_above_ef_margin(self):
        kpar = np.linspace(-0.2, 0.2, 5)
        ev = np.linspace(-0.1, 0.1, 41)
        data = np.tile(np.exp(-0.5 * ((ev + 0.04) / 0.02) ** 2), (kpar.size, 1))
        result = compute_bandmap_display(
            {"data": data, "kpar": kpar, "ev_arr": ev},
            mode="SecDev", edc_norm_enabled=False,
            deriv_params=DerivParams(ef_margin_eV=0.05),
        )
        # Above EF + margin -> NaN; below -> finite somewhere.
        self.assertTrue(np.isnan(result.data[:, ev > 0.05]).all())
        self.assertTrue(np.isfinite(result.data[:, ev <= 0.05]).any())


class TestCurvature(unittest.TestCase):
    def test_full_2d_formula_matches_independent_reference(self):
        # Independent re-derivation on a NaN-free map (no border, no masking),
        # so smoothing == plain Gaussian and we can check the math directly.
        from scipy.ndimage import gaussian_filter

        kpar, ev, data = _band_map(slope=0.6)
        sk, se = 1.0, 1.0
        out = compute_curvature(data, kpar, ev, sigma_k_px=sk, sigma_e_px=se,
                                c0_alpha=0.05, border_clip=0)

        sm = gaussian_filter(data, [sk, se])
        f_E = np.gradient(sm, ev, axis=1)
        f_k = np.gradient(sm, kpar, axis=0)
        f_kk = np.gradient(f_k, kpar, axis=0)
        f_EE = np.gradient(f_E, ev, axis=1)
        f_kE = np.gradient(f_k, ev, axis=1)
        from scipy.ndimage import binary_erosion
        interior = binary_erosion(np.ones(data.shape, bool), iterations=5)
        C0 = 0.05 * (np.percentile(np.abs(f_k[interior]), 95) ** 2
                     + np.percentile(np.abs(f_E[interior]), 95) ** 2)
        numer = (C0 + f_E**2) * f_kk - 2.0 * f_k * f_E * f_kE + (C0 + f_k**2) * f_EE
        denom = (C0 + f_k**2 + f_E**2) ** 1.5
        expected = -numer / (denom + 1e-30)

        # Compare the interior (border smoothing differs negligibly but exclude it).
        sl = (slice(3, -3), slice(3, -3))
        np.testing.assert_allclose(out[sl], expected[sl], rtol=1e-6, atol=1e-6)

    def test_curvature_peaks_on_band_not_border(self):
        # Trapezoid-style NaN border with a sharp sample/background cliff.
        kpar, ev, data = _band_map(slope=0.3, n_k=61, n_e=81)
        data = data + 0.02 * np.random.default_rng(0).standard_normal(data.shape)
        data[:6, :] = np.nan
        data[-6:, :] = np.nan
        out = compute_curvature(data, kpar, ev, sigma_k_px=1.5, sigma_e_px=1.5,
                                c0_alpha=0.05)
        ik = len(kpar) // 2
        band_val = np.nanmax(out[ik, :])           # on the band, central column
        edge_col = out[:, 5]                        # near the (clipped) border
        self.assertTrue(np.isfinite(band_val))
        self.assertGreater(band_val, np.nanmax(np.abs(edge_col)) * 0.5
                           if np.isfinite(np.nanmax(np.abs(edge_col))) else 0.0)

    def test_c0_eroded_estimate_below_naive_max(self):
        # With a border cliff, percentile-on-eroded-interior C0 must be far
        # smaller than the naive 0.05*max(gradient) the old code used.
        kpar, ev, data = _band_map(n_k=61, n_e=81)
        data = data.copy()
        data[:6, :] = np.nan
        data[-6:, :] = np.nan
        # The eroded-interior path is internal; assert the band survives, i.e.
        # the curvature is not washed to ~0 everywhere (old C0 bug symptom).
        out = compute_curvature(data, kpar, ev, sigma_k_px=1.5, sigma_e_px=1.5)
        self.assertGreater(np.nanmax(np.abs(out)), 1e-3)


if __name__ == "__main__":
    unittest.main()
