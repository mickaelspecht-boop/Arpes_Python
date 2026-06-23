from __future__ import annotations

import unittest

from arpes.physics.resolution import estimate_resolutions
try:
    from arpes_plots import _resolution_correct_gamma
except ModuleNotFoundError:
    _resolution_correct_gamma = None


class TestResolutionEstimation(unittest.TestCase):
    def test_da30_pass_energy_estimates_energy_resolution(self):
        res = estimate_resolutions({"pass_energy_eV": 50, "lens_mode": "DA30L_01"})
        self.assertAlmostEqual(res["dE_meV"], 25.0, places=6)
        self.assertAlmostEqual(res["dk_inv_a"], 0.005, places=6)
        self.assertIn("PE=50", res["source"])

    def test_r8000_pass_energy_estimates_energy_resolution(self):
        res = estimate_resolutions({"pass_energy_eV": 20, "lens_mode": "Angular30"})
        self.assertAlmostEqual(res["dE_meV"], 12.0, places=6)

    def test_empty_metadata_uses_default(self):
        res = estimate_resolutions({})
        self.assertAlmostEqual(res["dE_meV"], 15.0, places=6)
        self.assertAlmostEqual(res["dk_inv_a"], 0.005, places=6)
        self.assertIn("defaut", res["source"])

    def test_pass_energy_unknown_lens_falls_back(self):
        res = estimate_resolutions({"pass_energy_eV": 50, "lens_mode": "Transmission"})
        self.assertAlmostEqual(res["dE_meV"], 15.0, places=6)
        self.assertIn("lens inconnu", res["source"])

    def test_angle_step_gives_physical_dk(self):
        # Independent reference: dk = 0.51233*sqrt(Ek)*step_rad*a/pi.
        import math
        meta = {"angle_step_deg": 0.1, "ef_kinetic_from_hv": 16.8, "a_lattice": 3.96}
        res = estimate_resolutions(meta)
        expected = 0.51233 * math.sqrt(16.8) * math.radians(0.1) * 3.96 / math.pi
        self.assertAlmostEqual(res["dk_inv_a"], expected, places=9)
        self.assertIn("angle_step", res["source"])

    def test_angle_step_ef_kin_from_hv_minus_phi(self):
        import math
        meta = {"angle_step_deg": 0.1, "hv": 21.2, "work_function_eV": 4.4,
                "a_lattice": 3.96}
        res = estimate_resolutions(meta)
        expected = 0.51233 * math.sqrt(21.2 - 4.4) * math.radians(0.1) * 3.96 / math.pi
        self.assertAlmostEqual(res["dk_inv_a"], expected, places=9)

    def test_angle_step_without_hv_keeps_default(self):
        res = estimate_resolutions({"angle_step_deg": 0.1, "a_lattice": 3.96})
        self.assertAlmostEqual(res["dk_inv_a"], 0.005, places=9)
        self.assertIn("hv absent", res["source"])

    def test_angle_step_without_lattice_keeps_default(self):
        res = estimate_resolutions({"angle_step_deg": 0.1, "hv": 21.2})
        self.assertAlmostEqual(res["dk_inv_a"], 0.005, places=9)
        self.assertIn("a_lattice absent", res["source"])

    def test_non_numeric_metadata_ignored(self):
        res = estimate_resolutions({"pass_energy_eV": "n/a", "angle_step_deg": "x"})
        self.assertAlmostEqual(res["dE_meV"], 15.0, places=6)
        self.assertAlmostEqual(res["dk_inv_a"], 0.005, places=6)

    def test_zero_resolution_keeps_raw_gamma(self):
        if _resolution_correct_gamma is None:
            self.skipTest("scipy unavailable")
        gamma = [0.04, 0.05, 0.06]
        gmin, gcorr = _resolution_correct_gamma(
            [-0.2, -0.1, 0.0], [0.3, 0.25, 0.2], gamma,
            dE_eV=0.0, dk_inv_a=0.0,
        )
        self.assertTrue((gmin == 0).all())
        for got, exp in zip(gcorr, gamma):
            self.assertAlmostEqual(float(got), exp, places=8)

    def test_large_dk_marks_resolution_limited(self):
        if _resolution_correct_gamma is None:
            self.skipTest("scipy unavailable")
        _, gcorr = _resolution_correct_gamma(
            [-0.2, -0.1, 0.0], [0.3, 0.25, 0.2], [0.04, 0.04, 0.04],
            dE_eV=0.0, dk_inv_a=0.08,
        )
        self.assertTrue((gcorr == 0).all())

    def test_resolution_is_hwhm_converted(self):
        # gamma is HWHM; dk_inv_a is FWHM, so the subtracted floor is dk/2.
        # dE=0 -> gamma_min = HWHM(dk) = 0.05 for dk=0.10.
        # sqrt(0.13^2 - 0.05^2) = 0.12.
        if _resolution_correct_gamma is None:
            self.skipTest("scipy unavailable")
        gmin, gcorr = _resolution_correct_gamma(
            [-0.2, -0.1, 0.0], [0.3, 0.25, 0.2], [0.13, 0.13, 0.13],
            dE_eV=0.0, dk_inv_a=0.10,
        )
        for gm in gmin:
            self.assertAlmostEqual(float(gm), 0.05, places=6)
        for g in gcorr:
            self.assertAlmostEqual(float(g), 0.12, places=6)

    def test_smooth_fit_broadening_is_deconvolved(self):
        # P2: the Gaussian k-smoothing (smooth_fit) folds into Δk_eff.
        # σ=3 px, dk_pixel=0.01 -> Δk_smooth FWHM = 2√(2ln2)·0.03 = 0.07064,
        # HWHM floor = 0.03532 (dE=0, no instrument dk). gamma_min must reflect it
        # and gamma_corrige must drop below the raw HWHM.
        if _resolution_correct_gamma is None:
            self.skipTest("scipy unavailable")
        gmin, gcorr = _resolution_correct_gamma(
            [-0.2, -0.1, 0.0], [0.3, 0.25, 0.2], [0.13, 0.13, 0.13],
            dE_eV=0.0, dk_inv_a=0.0,
            smooth_fit_sigma_px=3.0, dk_pixel=0.01,
        )
        for gm in gmin:
            self.assertAlmostEqual(float(gm), 0.5 * 2.3548200450309493 * 0.03, places=6)
        for g in gcorr:
            self.assertLess(float(g), 0.13)
        # Backward compat: zero smoothing reproduces the un-deconvolved width.
        _, gcorr0 = _resolution_correct_gamma(
            [-0.2, -0.1, 0.0], [0.3, 0.25, 0.2], [0.13, 0.13, 0.13],
            dE_eV=0.0, dk_inv_a=0.0,
        )
        for g in gcorr0:
            self.assertAlmostEqual(float(g), 0.13, places=8)


if __name__ == "__main__":
    unittest.main()
