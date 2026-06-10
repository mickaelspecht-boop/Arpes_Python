"""Headless tests for the pocket lasso seeding (human-in-the-loop entry)."""
from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.pocket_lasso import (
    LassoSeed,
    contour_convexity_ratio,
    contour_touches_boundary,
    lasso_to_seed,
)


def _ring_map(n=101, r=0.3, cx=0.0, cy=0.0, width=0.05):
    """FS map with one ring-shaped pocket (wall intensity ~1, inside/out ~0)."""
    kx = np.linspace(-1.0, 1.0, n)
    ky = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(kx, ky)  # (ny, nx)
    rr = np.hypot(xx - cx, yy - cy)
    fs = np.exp(-0.5 * ((rr - r) / width) ** 2)
    return kx, ky, fs


class TestLassoToSeed(unittest.TestCase):
    def test_box_around_pocket_gives_center_seed_and_sane_level(self):
        kx, ky, fs = _ring_map(cx=0.2, cy=-0.1)
        out = lasso_to_seed(kx, ky, fs, (-0.25, 0.65, -0.55, 0.35))
        self.assertIsInstance(out, LassoSeed)
        self.assertAlmostEqual(out.seed_kx, 0.2, delta=0.01)
        self.assertAlmostEqual(out.seed_ky, -0.1, delta=0.01)
        self.assertGreater(out.level, 0.0)
        self.assertLess(out.level, 1.0)

    def test_degenerate_selection_raises_loudly(self):
        kx, ky, fs = _ring_map()
        with self.assertRaisesRegex(ValueError, "too small"):
            lasso_to_seed(kx, ky, fs, (0.0, 0.001, 0.0, 0.001))

    def test_nan_region_raises_loudly(self):
        kx, ky, fs = _ring_map()
        fs = fs.copy()
        fs[:, :] = np.nan
        with self.assertRaisesRegex(ValueError, "no data"):
            lasso_to_seed(kx, ky, fs, (-0.5, 0.5, -0.5, 0.5))

    def test_flat_region_raises_loudly(self):
        kx, ky, fs = _ring_map()
        flat = np.full_like(fs, 0.3)
        with self.assertRaisesRegex(ValueError, "no intensity contrast"):
            lasso_to_seed(kx, ky, flat, (-0.5, 0.5, -0.5, 0.5))


class TestContourGuards(unittest.TestCase):
    def test_convexity_near_one_for_circle(self):
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        self.assertAlmostEqual(contour_convexity_ratio(circle), 1.0, delta=0.02)

    def test_convexity_large_for_figure_eight(self):
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        # Lemniscate of Bernoulli (figure eight).
        denom = 1 + np.sin(t) ** 2
        eight = np.column_stack([np.cos(t) / denom, np.sin(t) * np.cos(t) / denom])
        self.assertGreater(contour_convexity_ratio(eight), 1.4)

    def test_boundary_touch_detection(self):
        kx = np.linspace(-1, 1, 50)
        ky = np.linspace(-1, 1, 50)
        inside = np.column_stack([np.zeros(20), np.linspace(-0.3, 0.3, 20)])
        self.assertFalse(contour_touches_boundary(inside, kx, ky))
        edge = np.column_stack([np.full(20, 0.999), np.linspace(-0.3, 0.3, 20)])
        self.assertTrue(contour_touches_boundary(edge, kx, ky))


if __name__ == "__main__":
    unittest.main()
