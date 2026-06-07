"""Tests P2.1a — k∥ geometry: single C_ARPES source + tilt guard."""
from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.kpar_geometry import (
    C_ARPES,
    TILT_GUARD_DEG,
    kpar_scale,
    tilt_within_guard,
    ky_residual_pi_a,
)


class TestConstantSingleSource(unittest.TestCase):
    def test_constant_value(self):
        self.assertAlmostEqual(C_ARPES, 0.51233)

    def test_gamma_and_overlay_share_constant(self):
        from arpes.physics import gamma, bm_cut_overlay
        from arpes.io.loaders import common
        self.assertIs(gamma.C_ARPES, C_ARPES)
        self.assertIs(bm_cut_overlay.C_ARPES, C_ARPES)
        self.assertIs(common._C_ARPES, C_ARPES)


class TestKparScale(unittest.TestCase):
    def test_positive_for_valid_ek(self):
        self.assertGreater(kpar_scale(60.0, 4.5, 3.96), 0)

    def test_none_for_invalid_ek(self):
        self.assertIsNone(kpar_scale(3.0, 4.5, 3.96))

    def test_none_for_unknown_lattice(self):
        self.assertIsNone(kpar_scale(60.0, 4.5, 0.0))

    def test_matches_historical_formula(self):
        ek = 60.0 - 4.5
        expected = C_ARPES * np.sqrt(ek) * 3.96 / np.pi
        self.assertAlmostEqual(kpar_scale(60.0, 4.5, 3.96), expected, places=10)


class TestTiltGuard(unittest.TestCase):
    def test_none_tilt_within_guard(self):
        # Missing tilt = standard scan without tilt → allowed (regression).
        self.assertTrue(tilt_within_guard(None))

    def test_zero_within_guard(self):
        self.assertTrue(tilt_within_guard(0.0))

    def test_boundary_inclusive(self):
        self.assertTrue(tilt_within_guard(TILT_GUARD_DEG))

    def test_above_guard_rejected(self):
        self.assertFalse(tilt_within_guard(3.0))
        self.assertFalse(tilt_within_guard(-3.0))

    def test_nan_treated_as_zero(self):
        self.assertTrue(tilt_within_guard(float("nan")))


class TestKyResidual(unittest.TestCase):
    def test_zero_tilt_zero_residual(self):
        self.assertEqual(
            ky_residual_pi_a(0.0, hv=60.0, work_func=4.5, a_lattice=3.96), 0.0
        )

    def test_positive_residual_grows_with_tilt(self):
        r1 = ky_residual_pi_a(1.0, hv=60.0, work_func=4.5, a_lattice=3.96)
        r2 = ky_residual_pi_a(2.0, hv=60.0, work_func=4.5, a_lattice=3.96)
        self.assertGreater(r1, 0.0)
        self.assertGreater(r2, r1)

    def test_matches_scale_sin(self):
        scale = kpar_scale(60.0, 4.5, 3.96)
        expected = scale * np.sin(np.radians(1.5))
        self.assertAlmostEqual(
            ky_residual_pi_a(1.5, hv=60.0, work_func=4.5, a_lattice=3.96),
            expected, places=10,
        )

    def test_zero_when_scale_unknown(self):
        self.assertEqual(
            ky_residual_pi_a(1.5, hv=60.0, work_func=4.5, a_lattice=0.0), 0.0
        )


if __name__ == "__main__":
    unittest.main()
