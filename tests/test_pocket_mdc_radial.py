from __future__ import annotations

import math
import unittest

import numpy as np

from arpes.physics.pocket_mdc_radial import (
    characterize_pocket_mdc_radial,
    fit_electron_edge_mdc,
    fit_lorentzian_mdc,
    kf_radial_mdc,
    sample_radial_mdc,
)


def _lorentzian_ring(n=121, r0=0.4, gamma=0.04, amp=1.0):
    kx = np.linspace(-1.0, 1.0, n)
    ky = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(kx, ky, indexing="xy")
    r = np.sqrt(x * x + y * y)
    img = amp * gamma ** 2 / ((r - r0) ** 2 + gamma ** 2)
    return img, kx, ky


def _elliptic_lorentzian_ring(n=141, a=0.5, b=0.3, gamma=0.04, amp=1.0):
    kx = np.linspace(-1.0, 1.0, n)
    ky = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(kx, ky, indexing="xy")
    # iso-contour ellipse (x/a)² + (y/b)² = 1, distance signed approximated:
    norm = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    img = amp * gamma ** 2 / ((norm - 1.0) ** 2 * (a * b) + gamma ** 2)
    return img, kx, ky


class TestSampleMDC(unittest.TestCase):
    def test_sample_radial_returns_intensity_along_radius(self):
        img, kx, ky = _lorentzian_ring()
        radii, ints = sample_radial_mdc(img, kx, ky, (0.0, 0.0), 0.0, r_max=0.9, n_points=64)
        self.assertEqual(radii.size, 64)
        self.assertAlmostEqual(float(radii[np.nanargmax(ints)]), 0.4, delta=0.04)


class TestFitLorentzian(unittest.TestCase):
    def test_fit_recovers_known_peak(self):
        radii = np.linspace(0.0, 1.0, 80)
        gamma = 0.05
        ints = 0.7 * gamma ** 2 / ((radii - 0.42) ** 2 + gamma ** 2) + 0.05
        kF, std, r2, n_used = fit_lorentzian_mdc(radii, ints)
        self.assertAlmostEqual(kF, 0.42, delta=0.01)
        self.assertGreater(r2, 0.95)
        self.assertEqual(n_used, 80)

    def test_fit_returns_nan_on_flat_input(self):
        radii = np.linspace(0.0, 1.0, 50)
        ints = np.ones_like(radii)
        kF, std, r2, n_used = fit_lorentzian_mdc(radii, ints)
        self.assertTrue(math.isnan(kF))

    def test_electron_edge_recovers_falling_half_height(self):
        radii = np.linspace(0.0, 1.0, 90)
        ints = 0.15 + 0.8 / (1.0 + np.exp((radii - 0.43) / 0.025))
        kF, std, conf, n_used = fit_electron_edge_mdc(radii, ints)
        self.assertAlmostEqual(kF, 0.43, delta=0.03)
        self.assertGreater(conf, 0.6)
        self.assertEqual(n_used, 90)


class TestKfRadialMDC(unittest.TestCase):
    def test_circle_kf_is_uniform(self):
        img, kx, ky = _lorentzian_ring(r0=0.35, gamma=0.05)
        for theta in (0.0, 45.0, 90.0, 180.0):
            res = kf_radial_mdc(img, kx, ky, (0.0, 0.0), theta, r_max=0.9)
            self.assertTrue(res.ok)
            self.assertAlmostEqual(res.kF, 0.35, delta=0.03)

    def test_filled_electron_pocket_uses_edge_fallback(self):
        kx = np.linspace(-1.0, 1.0, 121)
        ky = np.linspace(-1.0, 1.0, 121)
        x, y = np.meshgrid(kx, ky, indexing="xy")
        r = np.sqrt(x * x + y * y)
        img = 0.10 + 0.85 / (1.0 + np.exp((r - 0.38) / 0.025))

        res = kf_radial_mdc(img, kx, ky, (0.0, 0.0), 0.0, r_max=0.9, r2_min=0.5)

        self.assertTrue(res.ok)
        self.assertIn(res.method, {"lorentzian", "edge_half_height"})
        self.assertAlmostEqual(res.kF, 0.38, delta=0.04)


class TestCharacterizePocketMDC(unittest.TestCase):
    def test_circular_pocket_full_pipeline(self):
        img, kx, ky = _lorentzian_ring(r0=0.4)
        contour, results, center = characterize_pocket_mdc_radial(
            img, kx, ky,
            seed_point=(0.02, 0.0),
            n_directions=24, r_max=0.9, r2_min=0.6, refine_center=True,
        )
        self.assertEqual(contour.shape[1], 2)
        self.assertGreater(contour.shape[0], 16)
        radii = np.linalg.norm(contour[:-1] - np.array(center), axis=1)
        self.assertAlmostEqual(float(np.median(radii)), 0.4, delta=0.04)
        self.assertGreater(sum(1 for r in results if r.ok), 12)
        self.assertAlmostEqual(center[0], 0.0, delta=0.05)
        self.assertAlmostEqual(center[1], 0.0, delta=0.05)

    def test_elliptic_pocket_recovers_anisotropy(self):
        img, kx, ky = _elliptic_lorentzian_ring(a=0.5, b=0.3)
        contour, _results, center = characterize_pocket_mdc_radial(
            img, kx, ky,
            seed_point=(0.0, 0.0),
            n_directions=36, r_max=0.85, r2_min=0.4,
        )
        vec = contour[:-1] - np.array(center)
        rx = float(np.max(np.abs(vec[:, 0])))
        ry = float(np.max(np.abs(vec[:, 1])))
        self.assertGreater(rx, ry)
        self.assertAlmostEqual(rx, 0.5, delta=0.08)
        self.assertAlmostEqual(ry, 0.3, delta=0.08)


if __name__ == "__main__":
    unittest.main()
