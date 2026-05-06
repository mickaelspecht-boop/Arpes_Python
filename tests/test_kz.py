from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.kz import (
    KzParams,
    KzScanInput,
    compute_hv_k_map,
    compute_kz_map,
    compute_mdc_waterfall,
    energy_slice,
    kz_from_hv_kpar,
    standardize_scan,
)


class TestKzPhysics(unittest.TestCase):
    def _scan(self, hv: float, scale: float = 1.0) -> KzScanInput:
        k = np.linspace(-0.4, 0.4, 21)
        e = np.linspace(-0.1, 0.1, 11)
        kk, ee = np.meshgrid(k, e, indexing="ij")
        data = scale * np.exp(-(kk / 0.18) ** 2) * np.exp(-(ee / 0.04) ** 2)
        return KzScanInput(data=data, kpar=k, energy=e, hv=hv)

    def test_kz_monotonic_with_hv(self):
        k = np.asarray([0.0])
        low = kz_from_hv_kpar(60.0, k, work_func=4.0, inner_potential=12.0, a_lattice=4.0)[0]
        high = kz_from_hv_kpar(100.0, k, work_func=4.0, inner_potential=12.0, a_lattice=4.0)[0]
        self.assertGreater(high, low)

    def test_invalid_v0_rejected(self):
        with self.assertRaises(ValueError):
            kz_from_hv_kpar(60.0, [0.0], work_func=4.0, inner_potential=0.0, a_lattice=4.0)

    def test_energy_slice_uses_requested_window(self):
        scan = self._scan(80.0)
        out = energy_slice(scan, 0.0, 0.02)
        self.assertEqual(out.shape, scan.kpar.shape)
        self.assertGreater(float(np.nanmax(out)), 0.5)

    def test_axes_descending_are_standardized(self):
        scan = self._scan(80.0)
        rev = KzScanInput(
            data=scan.data[::-1, ::-1],
            kpar=scan.kpar[::-1],
            energy=scan.energy[::-1],
            hv=scan.hv,
        )
        out = standardize_scan(rev)
        np.testing.assert_allclose(out.kpar, scan.kpar)
        np.testing.assert_allclose(out.energy, scan.energy)
        np.testing.assert_allclose(out.data, scan.data)

    def test_compute_kz_map_shape_and_nan_outside_points(self):
        params = KzParams(
            work_func=4.0,
            inner_potential=12.0,
            a_lattice=4.0,
            c_lattice=12.0,
            k_bins=32,
            kz_bins=24,
            energy_center=0.0,
            energy_window=0.02,
        )
        out = compute_kz_map([self._scan(60.0), self._scan(80.0, scale=2.0)], params)
        self.assertEqual(out.image.shape, (24, 32))
        self.assertGreater(out.diagnostics["n_points"], 0)
        self.assertTrue(np.isfinite(out.image).any())
        self.assertIn("n_bins_filled", out.diagnostics)
        self.assertEqual(out.diagnostics["display_mode"], "interpolated")
        self.assertEqual(out.diagnostics["point_k"].shape, out.diagnostics["point_i"].shape)

    def test_compute_kz_map_points_mode_keeps_raw_cloud(self):
        params = KzParams(
            work_func=4.0,
            inner_potential=12.0,
            a_lattice=4.0,
            c_lattice=12.0,
            k_bins=16,
            kz_bins=12,
            display_mode="points",
        )
        out = compute_kz_map([self._scan(60.0), self._scan(80.0)], params)
        self.assertEqual(out.image.shape, (12, 16))
        self.assertEqual(out.diagnostics["display_mode"], "points")
        self.assertGreater(out.diagnostics["point_k"].size, 0)

    def test_unknown_display_mode_rejected(self):
        params = KzParams(display_mode="bad")
        with self.assertRaises(ValueError):
            compute_kz_map([self._scan(60.0), self._scan(80.0)], params)

    def test_compute_hv_k_map_returns_raw_hv_axis(self):
        params = KzParams(k_bins=17, energy_center=0.0, energy_window=0.02)
        out = compute_hv_k_map([self._scan(80.0, scale=2.0), self._scan(60.0)], params)
        self.assertEqual(out.image.shape, (2, 17))
        np.testing.assert_allclose(out.hv_grid, [60.0, 80.0])
        self.assertEqual(out.k_grid.size, 17)
        self.assertEqual(out.diagnostics["display_mode"], "hv map")
        self.assertGreater(out.diagnostics["n_points"], 0)

    def test_compute_hv_k_map_requires_varying_hv(self):
        params = KzParams()
        with self.assertRaises(ValueError):
            compute_hv_k_map([self._scan(60.0), self._scan(60.0)], params)

    def test_compute_mdc_waterfall_stacks_curves_by_hv(self):
        params = KzParams(k_bins=19, energy_center=0.0, energy_window=0.02)
        out = compute_mdc_waterfall([self._scan(80.0), self._scan(60.0)], params)
        self.assertEqual(out.curves.shape, (2, 19))
        self.assertEqual(out.offsets.shape, (2,))
        np.testing.assert_allclose(out.hv_grid, [60.0, 80.0])
        self.assertEqual(out.diagnostics["display_mode"], "MDC waterfall")
        self.assertGreater(out.diagnostics["offset_step"], 0)


if __name__ == "__main__":
    unittest.main()
