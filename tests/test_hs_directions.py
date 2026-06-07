from __future__ import annotations

import unittest

from arpes.physics.hs_directions import (
    bz_directions,
    direction_from_azimuth,
    normalize_direction_label,
)


class TestNormalizeDirection(unittest.TestCase):
    def test_shortcuts_from_gamma(self):
        self.assertEqual(normalize_direction_label("GX"), "Γ-X")
        self.assertEqual(normalize_direction_label("GM"), "Γ-M")
        self.assertEqual(normalize_direction_label("GK"), "Γ-K")
        self.assertEqual(normalize_direction_label("GY"), "Γ-Y")

    def test_gs_is_gamma_sigma(self):
        self.assertEqual(normalize_direction_label("GS"), "Γ-Σ")
        self.assertEqual(normalize_direction_label("gamma-sigma"), "Γ-Σ")
        self.assertEqual(normalize_direction_label("Γ→Σ"), "Γ-Σ")

    def test_edge_to_edge_segments(self):
        self.assertEqual(normalize_direction_label("XM"), "X-M")
        self.assertEqual(normalize_direction_label("MK"), "M-K")
        self.assertEqual(normalize_direction_label("KM"), "K-M")

    def test_lab_variants(self):
        for raw in ("G-X", "GtoX", "Gamma-X", "ΓX", "gamma to x", "g_x", "G/X", "Γ→X"):
            self.assertEqual(normalize_direction_label(raw), "Γ-X", raw)

    def test_multipoint_path(self):
        self.assertEqual(normalize_direction_label("Γ-X-M"), "Γ-X-M")
        self.assertEqual(normalize_direction_label("gxm"), "Γ-X-M")

    def test_empty_and_garbage(self):
        self.assertEqual(normalize_direction_label(""), "")
        self.assertEqual(normalize_direction_label(None), "")
        self.assertEqual(normalize_direction_label("   "), "")

    def test_single_point(self):
        self.assertEqual(normalize_direction_label("G"), "Γ")


class TestBzDirections(unittest.TestCase):
    def test_per_shape(self):
        self.assertIn("Γ-X", bz_directions("square"))
        self.assertIn("Γ-M", bz_directions("square"))
        self.assertIn("Γ-K", bz_directions("hexagon"))
        self.assertIn("Γ-Y", bz_directions("rectangle"))

    def test_unknown_shape_empty(self):
        self.assertEqual(bz_directions("oblique"), [])
        self.assertEqual(bz_directions(""), [])


class TestDirectionFromAzimuth(unittest.TestCase):
    def test_uncalibrated_never_invents(self):
        label, note = direction_from_azimuth(37.0, None, "square")
        self.assertEqual(label, "")
        self.assertIn("UNCALIBRATED", note)

    def test_square_gamma_x_and_m(self):
        # Square: X at edge center (0°), M at corner (45°).
        lbl_x, _ = direction_from_azimuth(0.0, 0.0, "square")
        self.assertEqual(lbl_x, "Γ-X")
        lbl_m, _ = direction_from_azimuth(45.0, 0.0, "square")
        self.assertEqual(lbl_m, "Γ-M")

    def test_reference_offset_applied(self):
        # azi 30 with ref 30 -> crystal 0 -> Γ-X.
        lbl, _ = direction_from_azimuth(30.0, 30.0, "square")
        self.assertEqual(lbl, "Γ-X")

    def test_no_direction_within_tolerance(self):
        lbl, note = direction_from_azimuth(20.0, 0.0, "square", tol_deg=5.0)
        self.assertEqual(lbl, "")
        self.assertIn("no Γ-dir", note)


if __name__ == "__main__":
    unittest.main()
