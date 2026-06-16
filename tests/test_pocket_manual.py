from __future__ import annotations

import math
import unittest

import numpy as np

from arpes.physics.bz import bz_polygon, bz_high_symmetry_points
from arpes.physics.pocket_manual import (
    characterize_manual_contour,
    snap_manual_contour_points,
)


def _filled_disk(n=121, radius=0.45):
    kx = np.linspace(-1.0, 1.0, n)
    ky = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(kx, ky, indexing="xy")
    r = np.sqrt(x * x + y * y)
    img = 0.1 + 0.8 / (1.0 + np.exp((r - radius) / 0.025))
    return img, kx, ky


def _circle_points(radius=0.45, n=24):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


class TestManualPocketPhysics(unittest.TestCase):
    def test_snap_moves_points_to_local_edge(self):
        img, kx, ky = _filled_disk(radius=0.45)
        pts = _circle_points(radius=0.40, n=16)

        snapped = snap_manual_contour_points(img, kx, ky, pts, radius_px=5)
        radii = np.linalg.norm(snapped, axis=1)

        self.assertAlmostEqual(float(np.median(radii)), 0.45, delta=0.06)

    def test_characterize_manual_contour_returns_geometry(self):
        img, kx, ky = _filled_disk(radius=0.45)
        pts = _circle_points(radius=0.45, n=24)
        hs = {
            name: [(float(x), float(y))]
            for x, y, name, _color in bz_high_symmetry_points("rectangle", 1.0, 1.0, 90.0)
        }

        props, contour = characterize_manual_contour(
            img,
            kx,
            ky,
            pts,
            bz_polygon=bz_polygon("rectangle", 1.0, 1.0, 90.0),
            hs_points=hs,
            contour_window=5,
        )

        self.assertEqual(props.analysis_mode, "manual_contour")
        self.assertEqual(props.topology, "electron")
        self.assertGreater(props.area_pct_bz, 10.0)
        self.assertEqual(contour.shape[1], 2)

    def test_rejects_too_few_points(self):
        img, kx, ky = _filled_disk()
        with self.assertRaisesRegex(ValueError, "at least 5"):
            characterize_manual_contour(
                img,
                kx,
                ky,
                _circle_points(n=4),
                bz_polygon=bz_polygon("rectangle", 1.0, 1.0, 90.0),
                hs_points={},
            )


if __name__ == "__main__":
    unittest.main()
