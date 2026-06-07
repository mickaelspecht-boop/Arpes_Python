from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.bz import BZ_PRESETS, bz_high_symmetry_points, bz_polygon, resolve_bz_preset


class TestBZPresets(unittest.TestCase):
    def test_rectangle_polygon_is_closed(self):
        poly = bz_polygon("rectangle", 1.0, 0.75)
        self.assertEqual(poly.shape, (5, 2))
        np.testing.assert_allclose(poly[0], poly[-1])
        self.assertEqual(len(BZ_PRESETS), 5)
        self.assertEqual(resolve_bz_preset("orthorhombic").shape, "rectangle")

    def test_hexagon_polygon_and_labels(self):
        poly = bz_polygon("hexagon", 1.0, 0.866)
        self.assertEqual(poly.shape, (7, 2))
        np.testing.assert_allclose(poly[0], poly[-1])
        labels = [p[2] for p in bz_high_symmetry_points("hexagon", 1.0, 0.866)]
        self.assertIn("Γ", labels)
        self.assertIn("K", labels)
        self.assertIn("M", labels)

    def test_all_2d_bravais_shapes_are_closed(self):
        shapes = ["square", "rectangle", "hexagon", "centered_rect", "oblique"]
        for shape in shapes:
            with self.subTest(shape=shape):
                poly = bz_polygon(shape, 1.0, 0.75, angle_deg=75.0)
                self.assertGreaterEqual(poly.shape[0], 5)
                self.assertEqual(poly.shape[1], 2)
                np.testing.assert_allclose(poly[0], poly[-1])

    def test_rectangle_labels_xys(self):
        pts = bz_high_symmetry_points("rectangle", 1.0, 0.75)
        by_lbl = {}
        for x, y, lbl, _c in pts:
            by_lbl.setdefault(lbl, []).append((round(x, 6), round(y, 6)))
        self.assertIn((1.0, 0.0), by_lbl["X"])
        self.assertIn((0.0, 0.75), by_lbl["Y"])
        self.assertIn((1.0, 0.75), by_lbl["S"])
        self.assertNotIn("M", by_lbl)  # M reserved for square/hexagonal

    def test_centered_rect_no_m(self):
        labels = {p[2] for p in bz_high_symmetry_points("centered_rect", 1.0, 0.75)}
        self.assertEqual(labels, {"Γ", "X", "S"})

    def test_oblique_only_gamma_named(self):
        named = [p[2] for p in bz_high_symmetry_points("oblique", 1.0, 0.8, 75.0)
                 if p[2]]
        self.assertEqual(named, ["Γ"])

    def test_square_m_corner_preserved(self):
        pts = bz_high_symmetry_points("square", 1.0, 1.0)
        m = [(round(x, 6), round(y, 6)) for x, y, lbl, _c in pts if lbl == "M"]
        self.assertIn((1.0, 1.0), m)

    def test_oblique_angle_changes_polygon(self):
        poly_60 = bz_polygon("oblique", 1.0, 0.8, angle_deg=60.0)
        poly_110 = bz_polygon("oblique", 1.0, 0.8, angle_deg=110.0)
        self.assertEqual(poly_60.shape, poly_110.shape)
        self.assertGreater(np.max(np.abs(poly_60 - poly_110)), 1e-3)


if __name__ == "__main__":
    unittest.main()
