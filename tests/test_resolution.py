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

    def test_zero_resolution_keeps_raw_gamma(self):
        if _resolution_correct_gamma is None:
            self.skipTest("scipy indisponible")
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
            self.skipTest("scipy indisponible")
        _, gcorr = _resolution_correct_gamma(
            [-0.2, -0.1, 0.0], [0.3, 0.25, 0.2], [0.04, 0.04, 0.04],
            dE_eV=0.0, dk_inv_a=0.08,
        )
        self.assertTrue((gcorr == 0).all())


if __name__ == "__main__":
    unittest.main()
