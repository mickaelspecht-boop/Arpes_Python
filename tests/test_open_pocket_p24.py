"""Tests P2.4 — open pockets: conic fit on arc, extrapolation, rejection.

Covers arpes.physics.ellipse_conic and wiring in
arpes.physics.pocket._properties_from_contour (extrapolated flag, σ axes,
rejection if arc is too short, Luttinger NaN if extrapolated, topology threshold
0.50).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arpes.physics.ellipse_conic import (
    ARC_FULL_DEG,
    ARC_REFUSE_DEG,
    PocketFitRefusedError,
    contiguous_coverage_deg,
    fit_ellipse_conic,
)
from arpes.physics.pocket import _properties_from_contour, _TOPOLOGY_CONFIDENCE_MIN


def _ellipse_points(a, b, ang_deg, cx, cy, theta):
    ang = np.radians(ang_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    base = np.vstack([a * np.cos(theta), b * np.sin(theta)])
    return (R @ base).T + [cx, cy]


# -------------------------------------------------------------- conic module

class TestFitEllipseConic:
    def test_full_ellipse_recovered(self):
        t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        pts = _ellipse_points(0.4, 0.2, 25.0, 0.1, -0.05, t)
        g = fit_ellipse_conic(pts)
        assert g["ok"]
        assert g["a"] == pytest.approx(0.4, abs=0.01)
        assert g["b"] == pytest.approx(0.2, abs=0.01)
        assert g["angle_deg"] == pytest.approx(25.0, abs=2.0)

    def test_half_arc_recovered(self):
        t = np.linspace(0, np.pi, 40)            # 180°
        pts = _ellipse_points(0.4, 0.2, 25.0, 0.1, -0.05, t)
        g = fit_ellipse_conic(pts)
        assert g["ok"]
        assert g["a"] == pytest.approx(0.4, abs=0.03)
        assert g["b"] == pytest.approx(0.2, abs=0.03)

    def test_collinear_refused(self):
        pts = np.column_stack([np.linspace(0, 1, 10), np.linspace(0, 1, 10)])
        assert fit_ellipse_conic(pts)["ok"] is False

    def test_too_few_points(self):
        assert fit_ellipse_conic(np.zeros((4, 2)))["ok"] is False


class TestCoverage:
    def test_full_circle_near_360(self):
        t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        pts = _ellipse_points(0.4, 0.4, 0.0, 0.0, 0.0, t)
        assert contiguous_coverage_deg(pts, (0.0, 0.0)) > 350

    def test_half_arc_near_180(self):
        t = np.linspace(0, np.pi, 40)
        pts = _ellipse_points(0.4, 0.4, 0.0, 0.0, 0.0, t)
        assert 170 < contiguous_coverage_deg(pts, (0.0, 0.0)) < 190

    def test_two_separated_arcs_low_coverage(self):
        # Two separated 30° arcs → large gap → low contiguous coverage.
        t = np.concatenate([np.linspace(0, np.pi / 6, 8),
                            np.linspace(np.pi, np.pi + np.pi / 6, 8)])
        pts = _ellipse_points(0.4, 0.4, 0.0, 0.0, 0.0, t)
        assert contiguous_coverage_deg(pts, (0.0, 0.0)) < 220


# ---------------------------------------------- _properties_from_contour wiring

def _disk_image(n=161, span=1.2):
    kx = np.linspace(-span, span, n)
    ky = np.linspace(-span, span, n)
    x, y = np.meshgrid(kx, ky)
    img = np.exp(-((x ** 2 + y ** 2) / (2 * 0.3 ** 2)))  # bright center → electron
    return img, kx, ky


_BZ = np.array([[-1.2, -1.2], [1.2, -1.2], [1.2, 1.2], [-1.2, 1.2]])
_HS = {"Gamma": (0.0, 0.0)}


def _props(contour):
    img, kx, ky = _disk_image()
    return _properties_from_contour(
        img, kx, ky, contour,
        bz_polygon=_BZ, hs_points=_HS, n_bands=1, spin=2,
        hs_dir_x_deg=0.0, hs_dir_m_deg=45.0, hs_dir_tol_deg=15.0,
        analysis_mode="test",
    )


class TestPropertiesOpenPocket:
    def test_closed_pocket_not_extrapolated(self):
        t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        c = _ellipse_points(0.45, 0.30, 0.0, 0.0, 0.0, t)
        p = _props(np.vstack([c, c[0]]))
        assert p.is_extrapolated is False
        assert p.arc_coverage_deg > ARC_FULL_DEG
        assert math.isfinite(p.n_carriers_2D)        # Luttinger defined if closed

    def test_partial_arc_extrapolated_flagged(self):
        t = np.linspace(0, np.radians(220), 44)      # arc 220°
        c = _ellipse_points(0.45, 0.30, 0.0, 0.0, 0.0, t)
        p = _props(c)
        assert p.is_extrapolated is True
        assert p.fit_method == "conic"
        assert ARC_REFUSE_DEG <= p.arc_coverage_deg < ARC_FULL_DEG
        assert math.isnan(p.n_carriers_2D)           # no Luttinger if extrapolated
        assert p.kF_a_sigma >= 0 and math.isfinite(p.kF_a_sigma)
        # area = π·a·b of the fitted ellipse.
        assert p.area_inv_a2 == pytest.approx(math.pi * p.kF_a * p.kF_b, rel=0.05)

    def test_short_arc_refused(self):
        t = np.linspace(0, np.radians(80), 20)       # arc 80° < 120°
        c = _ellipse_points(0.45, 0.30, 0.0, 0.0, 0.0, t)
        with pytest.raises(PocketFitRefusedError):
            _props(c)


def test_topology_confidence_threshold_bumped():
    assert _TOPOLOGY_CONFIDENCE_MIN == 0.50
