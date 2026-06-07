"""Tests fs_isocontour: cylinder, ellipsoid, multi-band, input validation."""
from __future__ import annotations

import numpy as np
import pytest

from arpes.theory.fs_isocontour import (
    FsContour,
    extract_fs_isocontour,
    isocontour_at_planes,
)


# ---- helpers: synthetic grid generation ------------------------------------


def _make_axes(n=51, half=2.0):
    ax = np.linspace(-half, half, n)
    return ax, ax.copy(), np.linspace(-1.0, 1.0, 21)


def _cylinder_band(kx, ky, kz, radius=1.0):
    """E(k) = kx² + ky² − r². Iso=0 → circle of radius r, independent of kz."""
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return KX**2 + KY**2 - radius**2


def _ellipsoid_band(kx, ky, kz, a=1.5, b=1.0, c=0.8):
    """E = (kx/a)² + (ky/b)² + (kz/c)² − 1. Iso=0 → ellipsoid."""
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return (KX/a)**2 + (KY/b)**2 + (KZ/c)**2 - 1.0


# ---- cylinder --------------------------------------------------------------


class TestCylinder:
    def test_one_contour_at_kz_zero(self):
        kx, ky, kz = _make_axes()
        band = _cylinder_band(kx, ky, kz, radius=1.0)[None, ...]  # n_b=1
        contours = extract_fs_isocontour(band, kx, ky, kz, kz_value=0.0, ef=0.0)
        assert len(contours) == 1
        c = contours[0]
        assert isinstance(c, FsContour)
        assert c.band_index == 0
        # Check mean radius ≈ 1.0 (grid tolerance).
        r = np.linalg.norm(c.points, axis=1)
        assert abs(r.mean() - 1.0) < 0.05

    def test_invariant_along_kz_for_cylinder(self):
        kx, ky, kz = _make_axes()
        band = _cylinder_band(kx, ky, kz, radius=0.8)[None, ...]
        c0 = extract_fs_isocontour(band, kx, ky, kz, kz_value=0.0, ef=0.0)
        c1 = extract_fs_isocontour(band, kx, ky, kz, kz_value=0.5, ef=0.0)
        assert len(c0) == len(c1) == 1
        r0 = float(np.linalg.norm(c0[0].points, axis=1).mean())
        r1 = float(np.linalg.norm(c1[0].points, axis=1).mean())
        assert abs(r0 - r1) < 1e-3

    def test_contour_is_closed_cylinder(self):
        kx, ky, kz = _make_axes()
        band = _cylinder_band(kx, ky, kz, radius=1.0)[None, ...]
        c = extract_fs_isocontour(band, kx, ky, kz, kz_value=0.0, ef=0.0)[0]
        assert c.closed is True


# ---- ellipsoid -------------------------------------------------------------


class TestEllipsoid:
    def test_axes_ratio_at_gamma_plane(self):
        kx, ky, kz = _make_axes(n=81, half=2.0)
        a, b = 1.5, 1.0
        band = _ellipsoid_band(kx, ky, kz, a=a, b=b, c=0.8)[None, ...]
        c = extract_fs_isocontour(band, kx, ky, kz, kz_value=0.0, ef=0.0)
        assert len(c) == 1
        pts = c[0].points
        # max kx ≈ a, max ky ≈ b.
        kx_max = float(np.max(np.abs(pts[:, 0])))
        ky_max = float(np.max(np.abs(pts[:, 1])))
        assert abs(kx_max - a) < 0.06
        assert abs(ky_max - b) < 0.06

    def test_shrinks_with_kz(self):
        kx, ky, kz = _make_axes(n=81, half=2.0)
        a, b, c_ax = 1.5, 1.0, 0.8
        band = _ellipsoid_band(kx, ky, kz, a=a, b=b, c=c_ax)[None, ...]
        c0 = extract_fs_isocontour(band, kx, ky, kz, kz_value=0.0, ef=0.0)[0]
        c1 = extract_fs_isocontour(band, kx, ky, kz, kz_value=0.5, ef=0.0)[0]
        # At kz=0.5, expected axes a*sqrt(1-(0.5/0.8)²), b*same.
        shrink = np.sqrt(1.0 - (0.5/c_ax)**2)
        kx_max_0 = float(np.max(np.abs(c0.points[:, 0])))
        kx_max_1 = float(np.max(np.abs(c1.points[:, 0])))
        expected = a * shrink
        assert kx_max_1 < kx_max_0
        assert abs(kx_max_1 - expected) < 0.08

    def test_no_contour_above_kz_max(self):
        kx, ky, kz = _make_axes(n=81, half=2.0)
        band = _ellipsoid_band(kx, ky, kz, a=1.5, b=1.0, c=0.8)[None, ...]
        # kz beyond c (clamped to kz_axis[-1]=1.0) → empty ellipsoid if c<1.
        c = extract_fs_isocontour(band, kx, ky, kz, kz_value=1.0, ef=0.0)
        assert c == []


# ---- multi-band ------------------------------------------------------------


class TestMultiBands:
    def test_two_bands_two_contours(self):
        kx, ky, kz = _make_axes(n=51, half=3.0)
        b1 = _cylinder_band(kx, ky, kz, radius=1.0)[None, ...]
        b2 = _cylinder_band(kx, ky, kz, radius=2.0)[None, ...]
        bands = np.concatenate([b1, b2], axis=0)
        c = extract_fs_isocontour(bands, kx, ky, kz, kz_value=0.0, ef=0.0)
        assert len(c) == 2
        idx = sorted(co.band_index for co in c)
        assert idx == [0, 1]

    def test_band_indices_filter(self):
        kx, ky, kz = _make_axes(n=51, half=3.0)
        b1 = _cylinder_band(kx, ky, kz, radius=1.0)[None, ...]
        b2 = _cylinder_band(kx, ky, kz, radius=2.0)[None, ...]
        bands = np.concatenate([b1, b2], axis=0)
        c = extract_fs_isocontour(
            bands, kx, ky, kz, kz_value=0.0, ef=0.0, band_indices=[1]
        )
        assert len(c) == 1
        assert c[0].band_index == 1


# ---- isocontour_at_planes --------------------------------------------------


class TestPlanesHelper:
    def test_two_planes(self):
        kx, ky, kz = _make_axes()
        band = _cylinder_band(kx, ky, kz, radius=1.0)[None, ...]
        result = isocontour_at_planes(
            band, kx, ky, kz, kz_values=[0.0, 0.5], ef=0.0
        )
        assert set(result.keys()) == {0.0, 0.5}
        assert all(len(v) == 1 for v in result.values())


# ---- input validation ------------------------------------------------------


class TestInputValidation:
    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            extract_fs_isocontour(
                np.zeros((10, 10)),  # 2D instead of 4D
                np.arange(10), np.arange(10), np.arange(5),
                kz_value=0.0,
            )

    def test_axes_mismatch_raises(self):
        bands = np.zeros((1, 10, 10, 5))
        with pytest.raises(ValueError):
            extract_fs_isocontour(
                bands, np.arange(8), np.arange(10), np.arange(5), kz_value=0.0
            )

    def test_kz_clamp_below(self):
        kx, ky, kz = _make_axes()
        band = _cylinder_band(kx, ky, kz, radius=1.0)[None, ...]
        # Very low kz_value → clamp to the first kz.
        c_lo = extract_fs_isocontour(band, kx, ky, kz, kz_value=-100.0)
        c_ref = extract_fs_isocontour(band, kx, ky, kz, kz_value=float(kz[0]))
        assert len(c_lo) == len(c_ref) == 1

    def test_min_points_filter(self):
        kx, ky, kz = _make_axes(n=21, half=1.5)
        band = _cylinder_band(kx, ky, kz, radius=0.05)[None, ...]
        # Tiny circle → very short contour, filtered by high min_points.
        c = extract_fs_isocontour(
            band, kx, ky, kz, kz_value=0.0, ef=0.0, min_points=1000
        )
        assert c == []
