from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.bz import BZ_PRESETS, bz_high_symmetry_points, bz_polygon


class TestBZPresets(unittest.TestCase):
    def test_rectangle_polygon_is_closed(self):
        poly = bz_polygon("rectangle", 1.0, 0.75)
        self.assertEqual(poly.shape, (5, 2))
        np.testing.assert_allclose(poly[0], poly[-1])
        self.assertIn("orthorhombic", BZ_PRESETS)

    def test_hexagon_polygon_and_labels(self):
        poly = bz_polygon("hexagon", 1.0, 0.866)
        self.assertEqual(poly.shape, (7, 2))
        np.testing.assert_allclose(poly[0], poly[-1])
        labels = [p[2] for p in bz_high_symmetry_points("hexagon", 1.0, 0.866)]
        self.assertIn("Γ", labels)
        self.assertIn("K", labels)
        self.assertIn("M", labels)


if __name__ == "__main__":
    unittest.main()
