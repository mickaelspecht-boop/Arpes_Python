from __future__ import annotations

import math
import unittest

import numpy as np

from arpes.physics.pocket import (
    assign_hs_label,
    characterize_pocket,
    characterize_pocket_bootstrap,
    extract_fs_contour,
    fit_pocket_ellipse,
    kf_along_direction,
    luttinger_count,
    pocket_area,
    pocket_curvature,
    pocket_topology,
    simplify_closed_contour,
    smooth_closed_contour,
    smooth_fs_image,
)


def _axes(n: int = 161):
    kx = np.linspace(-1.2, 1.2, n)
    ky = np.linspace(-1.2, 1.2, n)
    x, y = np.meshgrid(kx, ky)
    return kx, ky, x, y


def _electron_disk(radius: float = 0.6):
    kx, ky, x, y = _axes()
    r = np.sqrt(x * x + y * y)
    img = np.clip(1.0 - r / radius, 0.0, 1.0)
    return img, kx, ky


def _hole_disk(radius: float = 0.6):
    img, kx, ky = _electron_disk(radius)
    return 1.0 - img, kx, ky


def _rotated_ellipse(a: float, b: float, angle_deg: float, n: int = 361):
    t = np.linspace(0, 2 * np.pi, n)
    ca = math.cos(math.radians(angle_deg))
    sa = math.sin(math.radians(angle_deg))
    x0 = a * np.cos(t)
    y0 = b * np.sin(t)
    return np.column_stack([ca * x0 - sa * y0, sa * x0 + ca * y0])


class TestPocketContours(unittest.TestCase):
    def test_extract_contour_contains_seed(self):
        img, kx, ky = _electron_disk()
        contour = extract_fs_contour(img, kx, ky, 0.5, seed_point=(0.0, 0.0))

        self.assertEqual(contour.shape[1], 2)
        self.assertGreater(contour.shape[0], 20)
        self.assertLess(abs(pocket_area(contour)), math.pi * 0.6**2)

    def test_seed_outside_contour_raises(self):
        img, kx, ky = _electron_disk()
        with self.assertRaises(ValueError):
            extract_fs_contour(img, kx, ky, 0.5, seed_point=(1.0, 1.0))

    def test_level_outside_range_raises(self):
        img, kx, ky = _electron_disk()
        with self.assertRaises(ValueError):
            extract_fs_contour(img, kx, ky, 2.0, seed_point=(0.0, 0.0))

    def test_without_seed_returns_largest_closed_contour(self):
        kx, ky, x, y = _axes()
        r_big = np.sqrt((x + 0.25) ** 2 + y * y)
        r_small = np.sqrt((x - 0.75) ** 2 + y * y)
        img = np.maximum(
            np.clip(1.0 - r_big / 0.5, 0.0, 1.0),
            np.clip(1.0 - r_small / 0.25, 0.0, 1.0),
        )

        contour = extract_fs_contour(img, kx, ky, 0.5)
        area = abs(pocket_area(contour))

        self.assertGreater(area, 0.15)
        self.assertLess(area, 0.25)

    def test_invalid_axes_raise(self):
        img, kx, ky = _electron_disk()
        with self.assertRaises(ValueError):
            extract_fs_contour(img, kx[::-1], ky, 0.5, seed_point=(0.0, 0.0))


class TestPocketTopology(unittest.TestCase):
    def test_electron_topology_for_bright_inside(self):
        img, kx, ky = _electron_disk()
        contour = extract_fs_contour(img, kx, ky, 0.5, seed_point=(0.0, 0.0))
        topology, confidence, rays = pocket_topology(img, kx, ky, contour)

        self.assertEqual(topology, "electron")
        self.assertGreater(confidence, 0.9)
        self.assertGreater(rays, 0)

    def test_hole_topology_for_dark_inside(self):
        img, kx, ky = _hole_disk()
        contour = extract_fs_contour(img, kx, ky, 0.5, seed_point=(0.0, 0.0))
        topology, confidence, rays = pocket_topology(img, kx, ky, contour)

        self.assertEqual(topology, "hole")
        self.assertGreater(confidence, 0.9)
        self.assertGreater(rays, 0)

    def test_flat_image_topology_is_unclear(self):
        img, kx, ky = _electron_disk()
        contour = extract_fs_contour(img, kx, ky, 0.5, seed_point=(0.0, 0.0))
        topology, confidence, _rays = pocket_topology(np.ones_like(img), kx, ky, contour)

        self.assertEqual(topology, "unclear")
        self.assertEqual(confidence, 0.0)


class TestPocketGeometry(unittest.TestCase):
    def test_ellipse_fit_recovers_axes_order(self):
        contour = _rotated_ellipse(0.8, 0.3, 0.0)
        a, b, angle = fit_pocket_ellipse(contour)

        self.assertAlmostEqual(a, 0.8, delta=0.02)
        self.assertAlmostEqual(b, 0.3, delta=0.02)
        self.assertLess(abs(angle), 2.0)

    def test_ellipse_fit_recovers_rotation(self):
        contour = _rotated_ellipse(0.7, 0.25, 35.0)
        a, b, angle = fit_pocket_ellipse(contour)

        self.assertAlmostEqual(a, 0.7, delta=0.02)
        self.assertAlmostEqual(b, 0.25, delta=0.02)
        self.assertAlmostEqual(angle, 35.0, delta=2.0)

    def test_assign_nearest_high_symmetry_label(self):
        label, distance = assign_hs_label(
            (0.05, 0.02),
            {"Gamma": (0.0, 0.0), "X": (1.0, 0.0), "M": (1.0, 1.0)},
        )

        self.assertEqual(label, "Gamma")
        self.assertLess(distance, 0.06)

    def test_assign_label_empty_returns_nan_distance(self):
        label, distance = assign_hs_label((0.0, 0.0), {})

        self.assertEqual(label, "")
        self.assertTrue(math.isnan(distance))

    def test_assign_label_with_grouped_dict_picks_nearest_copy(self):
        # Square BZ : 4 X copies, 1 Γ. Pocket à (1.05, 0.02) → nearest X (1, 0), pas Γ.
        hs = {
            "Γ": [(0.0, 0.0)],
            "X": [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],
            "M": [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)],
        }
        label, dist = assign_hs_label((1.05, 0.02), hs)
        self.assertEqual(label, "X")
        self.assertLess(dist, 0.1)
        # Pocket à (-1.0, 1.05) → nearest M(-1, 1)
        label, dist = assign_hs_label((-1.0, 1.05), hs)
        self.assertEqual(label, "M")
        self.assertLess(dist, 0.1)

    def test_assign_label_with_iterable_tuples(self):
        hs = [("Γ", 0.0, 0.0), ("X", 1.0, 0.0), ("M", 1.0, 1.0)]
        label, _ = assign_hs_label((0.9, 0.95), hs)
        self.assertEqual(label, "M")

    def test_assign_label_accepts_numpy_arrays(self):
        hs = {
            "Γ": np.array([0.0, 0.0]),
            "X": np.array([[1.0, 0.0], [-1.0, 0.0]]),
        }
        label, dist = assign_hs_label((-0.95, 0.02), hs)

        self.assertEqual(label, "X")
        self.assertLess(dist, 0.1)

    def test_smooth_and_simplify_closed_contour_preserves_area_scale(self):
        contour = _rotated_ellipse(0.6, 0.25, 20.0, n=721)
        noisy = contour.copy()
        noisy[:, 0] += 0.015 * np.sin(np.linspace(0, 80, noisy.shape[0]))

        smooth = smooth_closed_contour(noisy, window=11)
        simple = simplify_closed_contour(smooth, min_step=0.025)

        self.assertLess(simple.shape[0], smooth.shape[0])
        self.assertAlmostEqual(abs(pocket_area(simple)), abs(pocket_area(contour)), delta=0.08)

    def test_smooth_fs_image_reduces_stripe_noise(self):
        img, _kx, _ky = _electron_disk()
        stripes = img.copy()
        stripes[:, ::2] += 0.2

        smooth = smooth_fs_image(stripes, sigma=(1.0, 2.0))

        self.assertLess(np.nanstd(smooth - img), np.nanstd(stripes - img))


class TestCharacterizePocket(unittest.TestCase):
    def test_characterize_electron_disk(self):
        img, kx, ky = _electron_disk()
        bz = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        props = characterize_pocket(
            img, kx, ky,
            seed_point=(0.0, 0.0),
            level=0.5,
            bz_polygon=bz,
            hs_points={"Gamma": (0.0, 0.0), "X": (1.0, 0.0)},
        )

        self.assertEqual(props.topology, "electron")
        self.assertEqual(props.hs_label_nearest, "Gamma")
        self.assertAlmostEqual(props.centroid_kx, 0.0, delta=0.02)
        self.assertAlmostEqual(props.centroid_ky, 0.0, delta=0.02)
        self.assertGreater(props.area_pct_bz, 5.0)
        self.assertLess(props.area_pct_bz, 10.0)
        self.assertEqual(props.asdict()["topology"], "electron")

    def test_characterize_rejects_degenerate_bz_polygon(self):
        img, kx, ky = _electron_disk()
        with self.assertRaises(ValueError):
            characterize_pocket(
                img, kx, ky,
                seed_point=(0.0, 0.0),
                level=0.5,
                bz_polygon=np.array([[0.0, 0.0], [1.0, 0.0]]),
                hs_points={},
            )


class TestPocketEnrichment(unittest.TestCase):
    def test_kf_along_direction_circle(self):
        contour = _rotated_ellipse(0.5, 0.5, 0.0)
        r_x = kf_along_direction(contour, (0.0, 0.0), 0.0, 10.0)
        r_m = kf_along_direction(contour, (0.0, 0.0), 45.0, 10.0)
        self.assertAlmostEqual(r_x, 0.5, delta=0.02)
        self.assertAlmostEqual(r_m, 0.5, delta=0.02)

    def test_kf_along_direction_ellipse_anisotropy(self):
        contour = _rotated_ellipse(0.8, 0.3, 0.0)
        r_x = kf_along_direction(contour, (0.0, 0.0), 0.0, 8.0)
        r_y = kf_along_direction(contour, (0.0, 0.0), 90.0, 8.0)
        self.assertGreater(r_x, r_y)
        self.assertAlmostEqual(r_x, 0.8, delta=0.03)
        self.assertAlmostEqual(r_y, 0.3, delta=0.03)

    def test_kf_along_direction_no_points_returns_nan(self):
        contour = _rotated_ellipse(0.5, 0.5, 0.0, n=12)
        r = kf_along_direction(contour, (10.0, 10.0), 0.0, 1.0)
        self.assertTrue(math.isnan(r))

    def test_pocket_curvature_circle(self):
        contour = _rotated_ellipse(0.5, 0.5, 0.0, n=361)
        mean_k, var_k = pocket_curvature(contour)
        self.assertAlmostEqual(mean_k, 2.0, delta=0.1)
        self.assertLess(var_k, 0.05)

    def test_pocket_curvature_ellipse_has_variance(self):
        contour = _rotated_ellipse(0.8, 0.3, 0.0, n=361)
        mean_k, var_k = pocket_curvature(contour)
        self.assertGreater(var_k, 1.0)
        self.assertGreater(mean_k, 0.5)

    def test_luttinger_count_half_filled_band(self):
        n = luttinger_count(0.5, 1.0, n_bands=1, spin=2)
        self.assertAlmostEqual(n, 1.0, delta=1e-9)
        n2 = luttinger_count(0.5, 1.0, n_bands=2, spin=2)
        self.assertAlmostEqual(n2, 2.0, delta=1e-9)

    def test_luttinger_count_zero_bz_returns_nan(self):
        self.assertTrue(math.isnan(luttinger_count(0.5, 0.0)))

    def test_characterize_enrichment_fields_populated(self):
        img, kx, ky = _electron_disk()
        bz = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        props = characterize_pocket(
            img, kx, ky,
            seed_point=(0.0, 0.0),
            level=0.5,
            bz_polygon=bz,
            hs_points={"Gamma": (0.0, 0.0)},
            n_bands=1, spin=2,
        )
        self.assertTrue(math.isfinite(props.kF_gamma_x))
        self.assertTrue(math.isfinite(props.kF_gamma_m))
        self.assertAlmostEqual(props.kF_gamma_x, props.kF_gamma_m, delta=0.02)
        self.assertAlmostEqual(props.aspect_ratio, 1.0, delta=0.05)
        self.assertLess(props.eccentricity, 0.3)
        self.assertGreater(props.curvature_mean, 0.5)
        self.assertGreater(props.n_carriers_2D, 0.0)
        self.assertGreater(props.topology_rays_used, 0)

    def test_characterize_neighbor_pocket_filtered(self):
        kx, ky, x, y = _axes()
        r_main = np.sqrt(x * x + y * y)
        r_neighbor = np.sqrt((x - 1.0) ** 2 + y * y)
        img = np.maximum(
            np.clip(1.0 - r_main / 0.35, 0.0, 1.0),
            np.clip(1.0 - r_neighbor / 0.25, 0.0, 1.0),
        )
        bz = np.array([[-1.2, -1.2], [1.2, -1.2], [1.2, 1.2], [-1.2, 1.2]])
        props = characterize_pocket(
            img, kx, ky,
            seed_point=(0.0, 0.0),
            level=0.5,
            bz_polygon=bz,
            hs_points={"Gamma": (0.0, 0.0)},
        )
        self.assertEqual(props.topology, "electron")
        self.assertGreater(props.topology_rays_used, 2)


class TestPocketBootstrap(unittest.TestCase):
    def test_bootstrap_produces_finite_central_and_stds(self):
        img, kx, ky = _electron_disk()
        bz = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        rng = np.random.default_rng(123)
        bs = characterize_pocket_bootstrap(
            img, kx, ky,
            seed_point=(0.0, 0.0),
            level=0.5,
            bz_polygon=bz,
            hs_points={"Gamma": (0.0, 0.0)},
            smooth_sigma=(1.0, 2.0),
            n_bootstrap=12,
            level_rel_jitter=0.10,
            smooth_rel_jitter=0.20,
            rng=rng,
        )
        self.assertGreater(bs.n_valid, 6)
        self.assertEqual(bs.n_total, 12)
        self.assertEqual(bs.central.topology, "electron")
        self.assertTrue(math.isfinite(bs.central.kF_mean))
        self.assertGreaterEqual(bs.std["kF_mean"], 0.0)
        self.assertGreater(bs.std["area_pct_bz"], 0.0)
        d = bs.asdict()
        self.assertIn("uncertainty", d)
        self.assertIn("kF_mean", d["uncertainty"])
        self.assertEqual(d["n_bootstrap_valid"], bs.n_valid)

    def test_bootstrap_zero_jitter_has_zero_std(self):
        img, kx, ky = _electron_disk()
        bz = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        bs = characterize_pocket_bootstrap(
            img, kx, ky,
            seed_point=(0.0, 0.0),
            level=0.5,
            bz_polygon=bz,
            hs_points={"Gamma": (0.0, 0.0)},
            smooth_sigma=(1.0, 2.0),
            n_bootstrap=5,
            level_rel_jitter=0.0,
            smooth_rel_jitter=0.0,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(bs.n_valid, 5)
        self.assertEqual(bs.std["kF_mean"], 0.0)
        self.assertEqual(bs.std["area_inv_a2"], 0.0)


if __name__ == "__main__":
    unittest.main()
