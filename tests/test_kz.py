from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.kz import (
    KzParams,
    KzScanInput,
    compute_hv_k_map,
    compute_kz_map,
    convert_kz_unit,
    energy_slice,
    fit_inner_potential,
    hv_for_kz,
    kz_coverage_summary,
    kz_from_hv_kpar,
    kz_high_symmetry_planes,
    kz_profile_at_normal_emission,
    kz_unit_to_inv_a,
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
        self.assertEqual(out.diagnostics["point_k"].shape, out.diagnostics["point_i"].shape)
        # Raw sample cloud is always returned for the optional overlay.
        self.assertGreater(out.diagnostics["point_k"].size, 0)

    def test_compute_kz_map_degenerate_kpar_falls_back_to_binned(self):
        # Normal-emission scans with k//≡0 (e.g. lattice a missing at load):
        # the (k//, kz) cloud is collinear -> must not raise (was a QhullError).
        flat = [
            KzScanInput(
                data=self._scan(hv).data,
                kpar=np.zeros_like(self._scan(hv).kpar),
                energy=self._scan(hv).energy,
                hv=hv,
            )
            for hv in (60.0, 80.0, 100.0)
        ]
        params = KzParams(work_func=4.0, inner_potential=12.0, a_lattice=4.0,
                          c_lattice=12.0, k_bins=24, kz_bins=24)
        out = compute_kz_map(flat, params)
        self.assertTrue(out.diagnostics["degenerate_kpar"])
        self.assertTrue(np.isfinite(out.image).any())

    def test_hv_for_kz_inverts_kz_from_hv_at_normal_emission(self):
        hv = 72.0
        kz = kz_from_hv_kpar(hv, [0.0], work_func=4.5, inner_potential=12.0, a_lattice=4.0)[0]
        back = hv_for_kz(kz, work_func=4.5, inner_potential=12.0)
        self.assertAlmostEqual(back, hv, places=4)

    def test_kz_unit_roundtrip(self):
        kz = np.array([0.3, 1.7, 2.9])
        disp = convert_kz_unit(kz, unit="pi/c", c_lattice=11.6)
        back = kz_unit_to_inv_a(disp, unit="pi/c", c_lattice=11.6)
        np.testing.assert_allclose(back, kz)

    def test_kz_high_symmetry_planes_spacing_and_labels(self):
        c = 12.0
        planes = kz_high_symmetry_planes(0.0, 3.0 * np.pi / c, c, unit="A^-1")
        kz_vals = [p["kz"] for p in planes]
        # Planes at n*pi/c for n=0..3 inside the range.
        np.testing.assert_allclose(kz_vals, [n * np.pi / c for n in range(4)], atol=1e-9)
        self.assertEqual([p["label"] for p in planes], ["Γ", "Z", "Γ", "Z"])

    def test_kz_high_symmetry_planes_in_pi_c_unit(self):
        c = 12.0
        planes = kz_high_symmetry_planes(0.0, 2.0 * np.pi / c, c, unit="pi/c")
        np.testing.assert_allclose([p["kz"] for p in planes], [0.0, 1.0, 2.0], atol=1e-9)

    def test_kz_coverage_summary_counts_zones(self):
        c = 12.0
        cov = kz_coverage_summary(
            0.0, 2.0 * np.pi / c, c,
            work_func=4.5, inner_potential=12.0,
        )
        self.assertAlmostEqual(cov["n_zones"], 2.0, places=6)
        self.assertTrue(all(np.isfinite(cov["gamma_hv"])))

    def test_compute_hv_k_map_returns_raw_hv_axis(self):
        params = KzParams(k_bins=17, energy_center=0.0, energy_window=0.02)
        out = compute_hv_k_map([self._scan(80.0, scale=2.0), self._scan(60.0)], params)
        self.assertEqual(out.image.shape, (2, 17))
        np.testing.assert_allclose(out.hv_grid, [60.0, 80.0])
        self.assertEqual(out.k_grid.size, 17)
        self.assertGreater(out.diagnostics["n_points"], 0)

    def test_compute_hv_k_map_requires_varying_hv(self):
        params = KzParams()
        with self.assertRaises(ValueError):
            compute_hv_k_map([self._scan(60.0), self._scan(60.0)], params)

    # --- A+C: inner-potential fit + normal-emission profile ----------------

    def _periodic_scan(self, hv, *, v0_true, c, phi=4.5):
        """Scan whose E_F intensity at k//=0 modulates with period 2π/c in kz."""
        k = np.linspace(-0.4, 0.4, 41)
        e = np.linspace(-0.05, 0.05, 11)
        kz_true = 0.5123167 * np.sqrt(hv - phi + v0_true)
        amp = 1.0 + 0.8 * np.cos(c * kz_true)  # peaks at kz = n·2π/c (Γ)
        profile = np.where(np.abs(k) <= 0.05, amp, 1.0)
        data = np.repeat(profile[:, None], e.size, axis=1)
        return KzScanInput(data=data, kpar=k, energy=e, hv=hv)

    def test_fit_inner_potential_recovers_known_v0(self):
        v0_true, c = 14.0, 12.0
        scans = [self._periodic_scan(hv, v0_true=v0_true, c=c)
                 for hv in np.linspace(50.0, 150.0, 40)]
        params = KzParams(work_func=4.5, c_lattice=c, energy_center=0.0,
                          energy_window=0.03, inner_potential=10.0)
        out = fit_inner_potential(scans, params, v0_min=8.0, v0_max=22.0)
        self.assertAlmostEqual(out["v0_best"], v0_true, delta=1.5)
        self.assertGreater(out["power"], 0.5)
        self.assertEqual(out["confidence"], "ok")

    def test_fit_inner_potential_flags_no_periodicity(self):
        # Flat E_F intensity (no kz modulation) → V0 must NOT be claimed.
        scans = [self._scan(hv) for hv in np.linspace(55.0, 105.0, 20)]
        params = KzParams(work_func=4.5, c_lattice=11.6, energy_center=0.0,
                          energy_window=0.03)
        out = fit_inner_potential(scans, params)
        self.assertEqual(out["confidence"], "low")
        self.assertLess(out["power"], 0.5)

    def test_kz_profile_sorted_and_implies_c(self):
        v0_true, c = 14.0, 12.0
        scans = [self._periodic_scan(hv, v0_true=v0_true, c=c)
                 for hv in np.linspace(50.0, 150.0, 40)]
        params = KzParams(work_func=4.5, c_lattice=c, energy_center=0.0,
                          energy_window=0.03, inner_potential=v0_true)
        prof = kz_profile_at_normal_emission(scans, params)
        self.assertTrue(np.all(np.diff(prof["kz"]) >= 0))
        self.assertEqual(prof["kz"].size, prof["intensity"].size)
        self.assertGreater(prof["c_implied"], 8.0)
        self.assertLess(prof["c_implied"], 18.0)

    def test_fit_inner_potential_requires_two_scans(self):
        params = KzParams(c_lattice=12.0)
        with self.assertRaises(ValueError):
            fit_inner_potential([self._scan(60.0)], params)


if __name__ == "__main__":
    unittest.main()
