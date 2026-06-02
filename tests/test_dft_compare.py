from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arpes.io.dft_grid import load_dft_grid_npz
from arpes.physics.dft_slice import (
    isocontour_at_energy,
    kz_from_hv,
    slice_grid_at_kz,
)
from arpes.physics.pocket_compare import (
    compare_pocket_contours,
    hausdorff_distance,
)


def _ellipse(a, b, n=361):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([a * np.cos(t), b * np.sin(t)])


class TestKzFromHv(unittest.TestCase):
    def test_normal_value(self):
        kz = kz_from_hv(80.0, 12.0, work_function_eV=4.5)
        self.assertAlmostEqual(kz, 0.5123 * math.sqrt(87.5), delta=1e-6)

    def test_negative_inner_raises(self):
        with self.assertRaises(ValueError):
            kz_from_hv(2.0, 0.0, work_function_eV=10.0)


class TestSliceGrid(unittest.TestCase):
    def _grid(self):
        kx = np.linspace(-0.5, 0.5, 21)
        ky = np.linspace(-0.5, 0.5, 21)
        kz = np.linspace(0.0, 2.0, 11)
        kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing="xy")
        # paraboloid: E = (kx² + ky²) - 0.1·(kz-1)² - 0.05
        e = (kxg ** 2 + kyg ** 2 - 0.1 * (kzg - 1.0) ** 2 - 0.05)
        e = np.transpose(e, (2, 0, 1))  # (n_kz, n_ky, n_kx)
        return kx, ky, kz, e

    def test_slice_at_z_returns_2d(self):
        kx, ky, kz, e = self._grid()
        s = slice_grid_at_kz(kx, ky, kz, e, kz_target=1.0)
        self.assertEqual(s.energy_2d.shape, (ky.size, kx.size))
        self.assertAlmostEqual(s.kz_used, 1.0, delta=1e-6)

    def test_isocontour_circle(self):
        kx, ky, kz, e = self._grid()
        s = slice_grid_at_kz(kx, ky, kz, e, kz_target=1.0)
        contour = isocontour_at_energy(s, energy_eV=0.0, seed_point_1_per_ang=(0.0, 0.0))
        radii = np.linalg.norm(contour, axis=1)
        self.assertAlmostEqual(float(np.nanmean(radii)), math.sqrt(0.05), delta=0.02)


class TestPocketCompare(unittest.TestCase):
    def test_identical_contours_zero_metrics(self):
        c = _ellipse(0.5, 0.3)
        res = compare_pocket_contours(c, c)
        self.assertAlmostEqual(res.delta_area_pct, 0.0, delta=1e-9)
        self.assertAlmostEqual(res.delta_kF_mean_pct, 0.0, delta=1e-9)
        self.assertAlmostEqual(res.hausdorff, 0.0, delta=1e-9)

    def test_scaled_contour_area_delta(self):
        c1 = _ellipse(0.5, 0.5)
        c2 = _ellipse(0.55, 0.55)
        res = compare_pocket_contours(c2, c1)
        expected = 100.0 * ((0.55 / 0.5) ** 2 - 1.0)
        self.assertAlmostEqual(res.delta_area_pct, expected, delta=0.1)
        self.assertAlmostEqual(res.delta_kF_mean_pct, 10.0, delta=0.1)

    def test_hausdorff_translation(self):
        c1 = _ellipse(0.4, 0.4, n=181)
        c2 = c1 + np.array([0.1, 0.0])
        d = hausdorff_distance(c1, c2)
        self.assertAlmostEqual(d, 0.1, delta=0.02)


class TestDFTGridIO(unittest.TestCase):
    def test_load_npz_roundtrip(self):
        kx = np.linspace(-0.5, 0.5, 11)
        ky = np.linspace(-0.5, 0.5, 11)
        kz = np.linspace(0.0, 1.0, 5)
        e = np.random.default_rng(0).standard_normal((kz.size, ky.size, kx.size))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "grid.npz"
            np.savez(p, kx=kx, ky=ky, kz=kz, energies=e, a_lattice=4.0)
            grid = load_dft_grid_npz(p)
        self.assertEqual(grid.kx.shape, (11,))
        self.assertEqual(grid.energies.shape, (5, 11, 11))
        self.assertAlmostEqual(grid.a_lattice, 4.0, delta=1e-9)

    def test_load_npz_missing_a_lattice_uses_fallback(self):
        kx = np.linspace(-0.5, 0.5, 5)
        ky = np.linspace(-0.5, 0.5, 5)
        kz = np.linspace(0.0, 1.0, 3)
        e = np.zeros((kz.size, ky.size, kx.size))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "grid.npz"
            np.savez(p, kx=kx, ky=ky, kz=kz, energies=e)
            grid = load_dft_grid_npz(p, a_lattice_fallback=3.5)
        self.assertAlmostEqual(grid.a_lattice, 3.5, delta=1e-9)

    def test_load_npz_bad_shape_raises(self):
        kx = np.linspace(-0.5, 0.5, 5)
        ky = np.linspace(-0.5, 0.5, 4)
        kz = np.linspace(0.0, 1.0, 3)
        e = np.zeros((3, 5, 5))  # wrong: should be (3, 4, 5)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "grid.npz"
            np.savez(p, kx=kx, ky=ky, kz=kz, energies=e, a_lattice=3.0)
            with self.assertRaises(ValueError):
                load_dft_grid_npz(p)


if __name__ == "__main__":
    unittest.main()
