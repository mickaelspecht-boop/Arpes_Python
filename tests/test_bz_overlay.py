"""Tests : projection points HS cristal → repère détecteur + fold kz."""
from __future__ import annotations

import numpy as np
import pytest

from arpes.physics.bz import Lattice3D, bz_points_for_lattice_plane
from arpes.physics.bz_overlay import (
    HSProjection,
    fit_phi_c_from_clicks,
    project_hs_points,
)
from arpes.physics.kz import fold_kz_to_1bz


# ---- fold_kz_to_1bz --------------------------------------------------------


class TestFoldKz:
    def test_zero_kz_is_gamma(self):
        c = 11.6
        res = fold_kz_to_1bz(0.0, c)
        assert res["plane"] == "Gamma"
        assert res["n_zone"] == 0
        assert res["near_boundary"] is True

    def test_pi_over_c_is_z(self):
        c = 11.6
        res = fold_kz_to_1bz(np.pi / c, c)
        assert res["plane"] == "Z"
        assert abs(res["kz_reduced_pi_over_c"] - 1.0) < 1e-6

    def test_periodicity_2pi_over_c(self):
        c = 11.6
        kz1 = 0.0
        kz2 = 2.0 * np.pi / c
        r1 = fold_kz_to_1bz(kz1, c)
        r2 = fold_kz_to_1bz(kz2, c)
        assert r1["plane"] == r2["plane"] == "Gamma"
        assert r2["n_zone"] == 1

    def test_large_kz_zone_index(self):
        # Cas Bi2Se3-like : c=28.6, hv=120 → kz~5.8
        res = fold_kz_to_1bz(5.8, 28.6)
        assert res["n_zone"] >= 1
        assert 0.0 <= res["kz_reduced_pi_over_c"] <= 1.0

    def test_invalid_c_raises(self):
        with pytest.raises(ValueError):
            fold_kz_to_1bz(1.0, 0.0)

    def test_nan_kz_raises(self):
        with pytest.raises(ValueError):
            fold_kz_to_1bz(float("nan"), 10.0)


# ---- bz_points_for_lattice_plane -------------------------------------------


class TestBzLatticePlane:
    def test_tetragonal_gamma_plane_labels(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        _, pts, key = bz_points_for_lattice_plane(lat, plane="Gamma")
        labels = {lab for _, _, lab, _ in pts}
        assert key == "square"
        assert "Γ" in labels
        assert "X" in labels
        assert "M" in labels
        # Le label Z ne doit PAS apparaître dans le plan Γ.
        assert "Z" not in labels

    def test_tetragonal_z_plane_labels(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        _, pts, _ = bz_points_for_lattice_plane(lat, plane="Z")
        labels = {lab for _, _, lab, _ in pts}
        # Plan Z : centre=Z, milieux d'arête=R, coins=A.
        assert "Z" in labels
        assert "R" in labels
        assert "A" in labels
        assert "Γ" not in labels
        assert "X" not in labels

    def test_orthorhombic_distinct_x_y(self):
        lat = Lattice3D(a=3.5, b=6.0, c=12.0, bravais="orthorhombic")
        _, pts, key = bz_points_for_lattice_plane(lat, plane="Gamma")
        labels = {lab for _, _, lab, _ in pts}
        assert key == "rectangle"
        assert {"Γ", "X", "Y", "S"}.issubset(labels)

    def test_hexagonal_k_m(self):
        lat = Lattice3D(a=3.0, b=3.0, c=10.0, gamma_deg=120.0, bravais="hexagonal")
        _, pts, key = bz_points_for_lattice_plane(lat, plane="Gamma")
        labels = {lab for _, _, lab, _ in pts}
        assert key == "hexagonal"
        assert {"Γ", "K", "M"}.issubset(labels)

    def test_unknown_bravais_fallback_square(self):
        lat = Lattice3D(a=3.0, b=3.0, c=10.0, bravais="triclinic")
        _, _, key = bz_points_for_lattice_plane(lat)
        assert key == "square"


# ---- project_hs_points -----------------------------------------------------


class TestProjectHsPoints:
    def test_identity_no_rotation_no_translation(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        proj, _ = project_hs_points(lat, plane="Gamma", phi_c_deg=0.0)
        gammas = [p for p in proj if p.label == "Γ"]
        assert gammas, "Γ doit être projeté"
        g = gammas[0]
        assert abs(g.kx) < 1e-9 and abs(g.ky) < 1e-9

    def test_translation_gamma_shifts_all(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        proj0, _ = project_hs_points(lat, plane="Gamma")
        proj1, _ = project_hs_points(lat, plane="Gamma", gamma_kx=0.3, gamma_ky=-0.2)
        for p0, p1 in zip(proj0, proj1):
            assert abs((p1.kx - p0.kx) - 0.3) < 1e-9
            assert abs((p1.ky - p0.ky) - (-0.2)) < 1e-9

    def test_rotation_90_swaps_x_axes_square(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        proj, _ = project_hs_points(lat, plane="Gamma", phi_c_deg=90.0)
        # Convention R(+90°) cohérente avec project_gamma_by_azi :
        # R = [[cos, sin],[-sin, cos]] ; un point initial (1, 0) → (0, -1).
        xs = [p for p in proj if p.label == "X"]
        # Au moins un X devrait tomber sur axe ky désormais.
        on_ky_axis = [p for p in xs if abs(p.kx) < 1e-9 and abs(abs(p.ky) - 1.0) < 1e-9]
        assert on_ky_axis, f"X attendu sur axe ky après 90°, got {[(p.kx, p.ky) for p in xs]}"

    def test_azi_diff_equivalent_to_phi_c(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        a, _ = project_hs_points(lat, plane="Gamma", phi_c_deg=30.0)
        b, _ = project_hs_points(
            lat, plane="Gamma", phi_c_deg=0.0, azi_ref_deg=10.0, azi_target_deg=40.0
        )
        for pa, pb in zip(a, b):
            assert abs(pa.kx - pb.kx) < 1e-9
            assert abs(pa.ky - pb.ky) < 1e-9

    def test_polygon_closed_after_rotation(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        _, poly = project_hs_points(lat, plane="Gamma", phi_c_deg=37.0)
        # Premier == dernier (polygone fermé).
        assert np.allclose(poly[0], poly[-1])


# ---- fit_phi_c_from_clicks -------------------------------------------------


class TestFitPhiC:
    def test_recovers_zero_rotation(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        # X attendu à (1, 0), M à (1, 1) avec phi_c=0, Γ=0.
        res = fit_phi_c_from_clicks(
            lat, plane="Gamma",
            clicks_kx_ky=[(1.0, 0.0), (1.0, 1.0)],
            expected_labels=["X", "M"],
        )
        assert abs(res["phi_c_deg"]) < 0.2 or abs(res["phi_c_deg"] - 360.0) < 0.2
        assert abs(res["gamma_kx"]) < 1e-3
        assert abs(res["gamma_ky"]) < 1e-3
        assert res["residual"] < 1e-4

    def test_recovers_45_rotation(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        # Après rotation 45° : X initial (1,0) → R·(1,0) avec
        # R=[[cos,sin],[-sin,cos]] = (cos45, -sin45) = (0.707, -0.707).
        c45 = float(np.cos(np.radians(45.0)))
        s45 = float(np.sin(np.radians(45.0)))
        clicks = [(c45, -s45), (c45 + s45, -s45 + c45)]
        res = fit_phi_c_from_clicks(
            lat, plane="Gamma",
            clicks_kx_ky=clicks,
            expected_labels=["X", "M"],
        )
        # Tetragonal : ambiguïté mod 90°. Vérifier au moins un candidat ≈ 45°.
        modulo = sorted(res["phi_c_deg"] % 90.0 for _ in range(1))
        assert any(abs((res["phi_c_deg"] - 45.0) % 90.0) < 0.3
                   or abs((res["phi_c_deg"] - 45.0) % 90.0 - 90.0) < 0.3
                   for _ in [0])
        assert res["residual"] < 1e-3
        del modulo

    def test_recovers_translation(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        res = fit_phi_c_from_clicks(
            lat, plane="Gamma",
            clicks_kx_ky=[(1.3, 0.2), (1.3, 1.2)],  # X et M décalés Γ=(0.3, 0.2)
            expected_labels=["X", "M"],
        )
        assert abs(res["gamma_kx"] - 0.3) < 0.02
        assert abs(res["gamma_ky"] - 0.2) < 0.02

    def test_candidates_have_4_for_tetragonal(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        res = fit_phi_c_from_clicks(
            lat, plane="Gamma",
            clicks_kx_ky=[(1.0, 0.0)],
            expected_labels=["X"],
        )
        assert len(res["candidates"]) == 4  # tétragonal : phi mod 90°

    def test_label_mismatch_raises(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        with pytest.raises(ValueError):
            fit_phi_c_from_clicks(
                lat, plane="Gamma",
                clicks_kx_ky=[(1.0, 0.0)],
                expected_labels=["X", "M"],
            )

    def test_empty_raises(self):
        lat = Lattice3D(a=3.96, b=3.96, c=11.6, bravais="tetragonal")
        with pytest.raises(ValueError):
            fit_phi_c_from_clicks(
                lat, plane="Gamma", clicks_kx_ky=[], expected_labels=[]
            )
