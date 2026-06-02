from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.pocket_quality import (
    contour_touches_border,
    local_snr,
    run_pocket_guards,
    smoothing_warning,
)


def _disk_image(noise=0.0, n=121, seed=0):
    kx = np.linspace(-1.0, 1.0, n)
    ky = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(kx, ky, indexing="xy")
    img = np.clip(1.0 - np.sqrt(x * x + y * y) / 0.5, 0.0, 1.0)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise, img.shape)
    return img, kx, ky


def _circle_contour(r=0.3, n=180):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


class TestLocalSNR(unittest.TestCase):
    def test_clean_disk_high_snr(self):
        img, kx, ky = _disk_image(noise=0.0)
        snr = local_snr(img, kx, ky, (0.0, 0.0), radius=0.05)
        self.assertTrue(snr > 10 or snr == float("inf"))

    def test_noisy_low_snr(self):
        img, kx, ky = _disk_image(noise=0.5)
        snr = local_snr(img, kx, ky, (0.0, 0.0), radius=0.1)
        self.assertLess(snr, 5.0)

    def test_empty_region_returns_nan(self):
        img, kx, ky = _disk_image()
        snr = local_snr(img, kx, ky, (10.0, 10.0), radius=0.001)
        self.assertTrue(np.isnan(snr))


class TestBorderCheck(unittest.TestCase):
    def test_interior_contour_not_border(self):
        _, kx, ky = _disk_image()
        c = _circle_contour(r=0.3)
        self.assertFalse(contour_touches_border(c, kx, ky))

    def test_edge_contour_is_border(self):
        _, kx, ky = _disk_image()
        c = _circle_contour(r=0.99)
        self.assertTrue(contour_touches_border(c, kx, ky))


class TestSmoothingWarning(unittest.TestCase):
    def test_small_sigma_ok(self):
        w = smoothing_warning((1.0, 1.0), (0.01, 0.01), kf_mean=0.5)
        self.assertTrue(w.ok)

    def test_huge_sigma_blocks(self):
        w = smoothing_warning((20.0, 20.0), (0.02, 0.02), kf_mean=0.3)
        self.assertFalse(w.ok)
        self.assertEqual(w.code, "smooth_excess")


class TestRunPocketGuards(unittest.TestCase):
    def test_clean_disk_passes(self):
        img, kx, ky = _disk_image()
        c = _circle_contour(r=0.3)
        results = run_pocket_guards(
            image=img, kx=kx, ky=ky,
            seed_point=(0.0, 0.0), contour=c,
            sigma_pixels=(1.0, 1.0), kf_mean=0.3,
        )
        self.assertTrue(all(r.ok for r in results))

    def test_noisy_image_blocks(self):
        img, kx, ky = _disk_image(noise=0.5)
        c = _circle_contour(r=0.3)
        results = run_pocket_guards(
            image=img, kx=kx, ky=ky,
            seed_point=(0.0, 0.0), contour=c,
            sigma_pixels=(1.0, 1.0), kf_mean=0.3,
        )
        self.assertFalse(all(r.ok for r in results))
        codes = [r.code for r in results if not r.ok]
        self.assertIn("snr_low", codes)


if __name__ == "__main__":
    unittest.main()
