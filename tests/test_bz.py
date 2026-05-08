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

    def test_oblique_angle_changes_polygon(self):
        poly_60 = bz_polygon("oblique", 1.0, 0.8, angle_deg=60.0)
        poly_110 = bz_polygon("oblique", 1.0, 0.8, angle_deg=110.0)
        self.assertEqual(poly_60.shape, poly_110.shape)
        self.assertGreater(np.max(np.abs(poly_60 - poly_110)), 1e-3)


if __name__ == "__main__":
    unittest.main()
