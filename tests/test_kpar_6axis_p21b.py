"""Tests P2.1b — 6-axis angle→k conversion (Ishida & Shin 2018).

Validation WITHOUT real tilt data: analytical invariants (one-angle reduction,
norm bound, azimuthal covariance, normal emission).
"""
from __future__ import annotations

import numpy as np
import pytest

from arpes.physics.kpar_geometry import C_ARPES, kpar_from_angles


EK = 90.0
ALPHA = np.linspace(-15.0, 15.0, 31)


class TestReduction:
    def test_reduces_to_one_angle_formula(self):
        # (a) tilt=azi=polar=0 → kx = C·√Ek·sinα, ky = 0.
        kx, ky = kpar_from_angles(ALPHA, 0.0, 0.0, 0.0, ek=EK)
        expected = C_ARPES * np.sqrt(EK) * np.sin(np.radians(ALPHA))
        np.testing.assert_allclose(kx, expected, rtol=1e-12)
        np.testing.assert_allclose(ky, 0.0, atol=1e-14)

    def test_slit_y_swaps_axes(self):
        kx, ky = kpar_from_angles(ALPHA, 0.0, 0.0, 0.0, ek=EK, slit_axis="y")
        expected = C_ARPES * np.sqrt(EK) * np.sin(np.radians(ALPHA))
        np.testing.assert_allclose(ky, expected, rtol=1e-12)
        np.testing.assert_allclose(kx, 0.0, atol=1e-14)


class TestNormBound:
    def test_kpar_norm_le_total(self):
        # (b) kx²+ky² ≤ C²·Ek for any geometry.
        kx, ky = kpar_from_angles(ALPHA, 12.0, 8.0, 33.0, ek=EK)
        kmax2 = C_ARPES ** 2 * EK
        assert np.all(kx ** 2 + ky ** 2 <= kmax2 + 1e-9)


class TestAzimuthCovariance:
    def test_azimuth_rotates_in_plane(self):
        # (c) φ → rotation of (kx, ky) by φ.
        kx1, ky1 = kpar_from_angles(ALPHA, 10.0, 6.0, 30.0, ek=EK)
        kx2, ky2 = kpar_from_angles(ALPHA, 10.0, 6.0, 75.0, ek=EK)
        d = np.radians(45.0)
        np.testing.assert_allclose(kx2, np.cos(d) * kx1 - np.sin(d) * ky1, rtol=1e-12)
        np.testing.assert_allclose(ky2, np.sin(d) * kx1 + np.cos(d) * ky1, rtol=1e-12)


class TestNormalEmission:
    def test_normal_emission_zero(self):
        # (d) α=θ=β=0 → kx=ky=0 for any φ.
        for phi in (0.0, 37.0, 180.0):
            kx, ky = kpar_from_angles(0.0, 0.0, 0.0, phi, ek=EK)
            assert abs(float(kx)) < 1e-14 and abs(float(ky)) < 1e-14


class TestTiltShiftsKy:
    def test_tilt_introduces_ky_offset(self):
        # Pure tilt (polar=azi=0) shifts ky by −C·√Ek·sinβ at α=0.
        kx, ky = kpar_from_angles(0.0, 0.0, 5.0, 0.0, ek=EK)
        expected_ky = -C_ARPES * np.sqrt(EK) * np.sin(np.radians(5.0))
        assert float(ky) == pytest.approx(expected_ky, rel=1e-10)

    def test_invalid_ek_raises(self):
        with pytest.raises(ValueError):
            kpar_from_angles(ALPHA, ek=-1.0)
