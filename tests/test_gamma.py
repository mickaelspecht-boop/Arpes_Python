"""Tests des fonctions pures Γ / angles (`arpes_gamma`).

But : verrouiller signes et conventions de projection azimutale, ainsi que
la formule k → angle. Toute régression sur ces invariants casse les fits CLS.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from arpes.physics.gamma import (
    A_LATTICE_DEFAULT,
    POLAR_TOLERANCE_DEG,
    angle_offset_candidates_for_load,
    angle_offsets_from_k_center,
    apply_bm_gamma_axis_shift,
    build_gamma_reference,
    gamma_reference_to_bm_center,
    k_to_angle_offset_deg,
    project_gamma_by_azi,
    score_bm_gamma_residual,
    stored_gamma_reference,
)
import numpy as np


class TestKToAngle(unittest.TestCase):
    def test_zero_k_gives_zero_angle(self):
        ang = k_to_angle_offset_deg(0.0, hv=80.0, work_func=4.5)
        self.assertIsNotNone(ang)
        self.assertAlmostEqual(ang, 0.0, places=6)

    def test_invalid_ek_returns_none(self):
        # hv < φ → ek <= 0
        self.assertIsNone(k_to_angle_offset_deg(0.1, hv=4.0, work_func=4.5))

    def test_clip_when_arg_above_one(self):
        # k arbitrairement grand → arcsin clippé à ±90°
        ang = k_to_angle_offset_deg(1e6, hv=80.0, work_func=4.5)
        self.assertAlmostEqual(ang, 90.0, places=4)
        ang_neg = k_to_angle_offset_deg(-1e6, hv=80.0, work_func=4.5)
        self.assertAlmostEqual(ang_neg, -90.0, places=4)

    def test_round_trip_k_angle_k(self):
        # k → angle → k doit redonner le même k (formule inverse)
        hv, phi = 80.0, 4.5
        k0 = 0.05
        ang = k_to_angle_offset_deg(k0, hv=hv, work_func=phi)
        self.assertIsNotNone(ang)
        ek = hv - phi
        scale = 0.51233 * math.sqrt(ek) * A_LATTICE_DEFAULT / math.pi
        k_back = scale * math.sin(math.radians(ang))
        self.assertAlmostEqual(k_back, k0, places=10)


class TestAngleOffsetsDict(unittest.TestCase):
    def test_returns_empty_when_hv_none(self):
        self.assertEqual(angle_offsets_from_k_center(0.1, 0.0, hv=None, work_func=4.5), {})

    def test_returns_empty_when_invalid_ek(self):
        self.assertEqual(angle_offsets_from_k_center(0.1, 0.0, hv=2.0, work_func=4.5), {})

    def test_full_dict_shape(self):
        out = angle_offsets_from_k_center(
            0.05, 0.0, hv=80.0, work_func=4.5, source="test", azi=12.0,
        )
        self.assertEqual(out["mode"], "cls_angle_offsets")
        self.assertIn("theta0_deg", out)
        self.assertIn("tilt0_deg", out)
        self.assertEqual(out["source"], "test")
        self.assertEqual(out["azi"], 12.0)
        self.assertEqual(out["work_func"], 4.5)
        self.assertEqual(out["a_lattice"], A_LATTICE_DEFAULT)


class TestProjectGammaByAzi(unittest.TestCase):
    def test_no_azi_returns_ref_kx_ky(self):
        ref = {"kx": 0.10, "ky": 0.0}
        kx, ky = project_gamma_by_azi(ref, azi_target=None)
        self.assertAlmostEqual(kx, 0.10)
        self.assertAlmostEqual(ky, 0.0)

    def test_warn_called_when_ky_significant_and_azi_missing(self):
        ref = {"kx": 0.10, "ky": 0.05}  # ky non négligeable
        warns = []
        kx, ky = project_gamma_by_azi(
            ref, azi_target=None, on_warn=warns.append, warn_label="X",
        )
        self.assertEqual(len(warns), 1)
        self.assertIn("X", warns[0])
        self.assertAlmostEqual(kx, 0.10)
        self.assertAlmostEqual(ky, 0.05)

    def test_zero_azi_difference_no_rotation(self):
        ref = {"kx": 0.10, "ky": 0.05, "azi": 30.0}
        kx, ky = project_gamma_by_azi(ref, azi_target=30.0)
        self.assertAlmostEqual(kx, 0.10, places=10)
        self.assertAlmostEqual(ky, 0.05, places=10)

    def test_90deg_rotation(self):
        # azi cible = azi ref + 90 → (kx, ky) → (ky, -kx)
        ref = {"kx": 0.10, "ky": 0.0, "azi": 0.0}
        kx, ky = project_gamma_by_azi(ref, azi_target=90.0)
        self.assertAlmostEqual(kx, 0.0, places=10)
        self.assertAlmostEqual(ky, -0.10, places=10)

    def test_180deg_rotation(self):
        ref = {"kx": 0.10, "ky": 0.05, "azi": 0.0}
        kx, ky = project_gamma_by_azi(ref, azi_target=180.0)
        self.assertAlmostEqual(kx, -0.10, places=10)
        self.assertAlmostEqual(ky, -0.05, places=10)

    def test_invalid_ref_returns_nan(self):
        kx, ky = project_gamma_by_azi({"kx": float("nan"), "ky": 0.0}, azi_target=10.0)
        self.assertTrue(math.isnan(kx))
        self.assertTrue(math.isnan(ky))


class TestStoredGammaReference(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(stored_gamma_reference(None), {})

    def test_invalid_returns_empty(self):
        self.assertEqual(stored_gamma_reference({"kx": float("nan"), "ky": 0.0}), {})
        self.assertEqual(stored_gamma_reference({}), {})

    def test_valid_returns_ref(self):
        ref = {"kx": 0.1, "ky": 0.0, "extra": 1}
        self.assertEqual(stored_gamma_reference(ref), ref)


class TestBuildGammaReference(unittest.TestCase):
    def test_minimal(self):
        ref = build_gamma_reference(
            kx=0.10, ky=0.05,
            metadata={"polar": 12.0, "tilt_ref": 1.0, "polar_already_applied_to_kx": True},
            hv=80.0, path="/x/y.zip", azi=30.0, source="fs",
        )
        self.assertAlmostEqual(ref["kx"], 0.10)
        self.assertAlmostEqual(ref["ky"], 0.05)
        self.assertAlmostEqual(ref["polar"], 12.0)
        self.assertAlmostEqual(ref["tilt"], 1.0)
        self.assertAlmostEqual(ref["azi"], 30.0)
        self.assertEqual(ref["hv"], 80.0)
        self.assertEqual(ref["path"], "/x/y.zip")
        self.assertTrue(ref["polar_already_applied_to_kx"])
        self.assertEqual(ref["source"], "fs")
        self.assertNotIn("direction", ref)

    def test_direction_kept(self):
        ref = build_gamma_reference(
            kx=0.0, ky=0.0, metadata={}, hv=None, path=None, azi=None,
            source="bm", direction="GX",
        )
        self.assertEqual(ref["direction"], "GX")
        self.assertIsNone(ref["azi"])
        self.assertIsNone(ref["hv"])


class TestGammaReferenceToBmCenter(unittest.TestCase):
    def test_no_metadata_returns_nan(self):
        gamma, corr = gamma_reference_to_bm_center(
            {"kx": 0.1, "ky": 0.0, "polar": 0.0},
            bm_metadata=None, bm_hv=80.0, work_func=4.5, bm_azi=0.0,
        )
        self.assertTrue(math.isnan(gamma))

    def test_polar_above_tolerance_refused_when_polar_not_baked(self):
        warns = []
        ref = {"kx": 0.10, "ky": 0.0, "polar": 0.0, "polar_already_applied_to_kx": False}
        gamma, _ = gamma_reference_to_bm_center(
            ref,
            bm_metadata={"polar": 5.0, "polar_already_applied_to_kx": False},
            bm_hv=80.0, work_func=4.5, bm_azi=0.0, on_warn=warns.append,
        )
        self.assertTrue(math.isnan(gamma))
        self.assertEqual(len(warns), 1)
        self.assertIn("Γ FS→BM", warns[0])

    def test_polar_above_tolerance_accepted_when_polar_baked_both_sides(self):
        # CLS case : polar absorbed into kpar conversion on both sides → polar
        # diff is irrelevant, tolerance must NOT block the transfer.
        ref = {
            "kx": -1.81, "ky": 0.0, "polar": 0.0,
            "azi": 0.0, "polar_already_applied_to_kx": True,
        }
        gamma, corr = gamma_reference_to_bm_center(
            ref,
            bm_metadata={"polar": 40.0, "polar_already_applied_to_kx": True},
            bm_hv=78.0, work_func=4.5, bm_azi=0.0,
        )
        self.assertAlmostEqual(corr, 0.0)
        self.assertAlmostEqual(gamma, -1.81, places=10)

    def test_within_tolerance_no_polar_correction_when_already_applied(self):
        ref = {
            "kx": 0.10, "ky": 0.0, "polar": 1.0,
            "azi": 0.0, "polar_already_applied_to_kx": True,
        }
        gamma, corr = gamma_reference_to_bm_center(
            ref,
            bm_metadata={"polar": 1.5, "polar_already_applied_to_kx": True},
            bm_hv=80.0, work_func=4.5, bm_azi=0.0,
        )
        self.assertAlmostEqual(corr, 0.0)
        self.assertAlmostEqual(gamma, 0.10, places=10)

    def test_polar_correction_applied_when_not_already(self):
        ref = {
            "kx": 0.10, "ky": 0.0, "polar": 0.0,
            "azi": 0.0, "polar_already_applied_to_kx": False,
        }
        # polar BM = 1° → correction non nulle
        gamma, corr = gamma_reference_to_bm_center(
            ref,
            bm_metadata={"polar": 1.0, "polar_already_applied_to_kx": False},
            bm_hv=80.0, work_func=4.5, bm_azi=0.0,
        )
        self.assertNotAlmostEqual(corr, 0.0)
        self.assertAlmostEqual(gamma - corr, 0.10, places=10)

    def test_invalid_ref_returns_nan(self):
        gamma, _ = gamma_reference_to_bm_center(
            {"kx": float("nan"), "ky": 0.0, "polar": 0.0},
            bm_metadata={"polar": 0.0}, bm_hv=80.0, work_func=4.5, bm_azi=0.0,
        )
        self.assertTrue(math.isnan(gamma))


class TestApplyBmGammaAxisShift(unittest.TestCase):
    def _make_raw(self, kpar, **meta_extra):
        return {
            "kpar": np.asarray(kpar, dtype=float),
            "metadata": dict(meta_extra),
        }

    def test_basic_shift(self):
        raw = self._make_raw([-0.1, 0.0, 0.1])
        ok = apply_bm_gamma_axis_shift(raw, 0.05, ref={"source": "fs", "path": "/x", "azi": 30.0})
        self.assertTrue(ok)
        np.testing.assert_allclose(raw["kpar"], [-0.15, -0.05, 0.05])
        m = raw["metadata"]
        self.assertTrue(m["bm_gamma_axis_centered"])
        self.assertAlmostEqual(m["bm_gamma_axis_shift"], 0.05)
        self.assertEqual(m["bm_gamma_reference_source"], "fs")
        self.assertEqual(m["bm_gamma_reference_path"], "/x")
        self.assertAlmostEqual(m["bm_gamma_reference_azi"], 30.0)

    def test_refuses_fs(self):
        raw = self._make_raw([-0.1, 0.1], fs_data=np.zeros((2, 2, 2)))
        self.assertFalse(apply_bm_gamma_axis_shift(raw, 0.05))

    def test_refuses_when_offsets_already_applied(self):
        raw = self._make_raw([-0.1, 0.1], angle_offsets_applied=True)
        self.assertFalse(apply_bm_gamma_axis_shift(raw, 0.05))

    def test_refuses_when_already_centered(self):
        raw = self._make_raw([-0.1, 0.1], bm_gamma_axis_centered=True)
        self.assertFalse(apply_bm_gamma_axis_shift(raw, 0.05))

    def test_refuses_nan_gamma(self):
        raw = self._make_raw([-0.1, 0.1])
        self.assertFalse(apply_bm_gamma_axis_shift(raw, float("nan")))

    def test_refuses_empty_kpar(self):
        raw = self._make_raw([])
        self.assertFalse(apply_bm_gamma_axis_shift(raw, 0.05))


class TestAngleOffsetCandidates(unittest.TestCase):
    def test_no_primary_returns_empty(self):
        self.assertEqual(
            angle_offset_candidates_for_load(
                primary=None, is_file=True, ref=None, target_geom=None,
                target_azi_fallback=None, hv=80.0, work_func=4.5,
            ),
            [],
        )

    def test_fs_returns_only_primary(self):
        primary = {"theta0_deg": 1.0, "tilt0_deg": 0.0}
        out = angle_offset_candidates_for_load(
            primary=primary, is_file=False, ref=None, target_geom=None,
            target_azi_fallback=None, hv=80.0, work_func=4.5,
        )
        self.assertEqual(out, [primary])

    def test_bm_without_ref_yields_two_signs(self):
        primary = {"theta0_deg": 1.5, "tilt0_deg": 0.0, "gamma_bm_pi_over_a": 0.05}
        out = angle_offset_candidates_for_load(
            primary=primary, is_file=True, ref=None, target_geom=None,
            target_azi_fallback=None, hv=80.0, work_func=4.5,
        )
        self.assertEqual(len(out), 2)
        labels = sorted(c["candidate"] for c in out)
        self.assertEqual(labels, ["-theta0", "theta0"])
        # signs flipped on theta0_deg and gamma_bm
        neg = [c for c in out if c["candidate"] == "-theta0"][0]
        self.assertAlmostEqual(neg["theta0_deg"], -1.5)
        self.assertAlmostEqual(neg["gamma_bm_pi_over_a"], -0.05)

    def test_bm_with_ref_and_geom_yields_more_candidates(self):
        primary = {"theta0_deg": 1.5, "tilt0_deg": 0.0, "gamma_bm_pi_over_a": 0.05}
        ref = {
            "kx": 0.10, "ky": 0.0, "azi": 0.0, "polar": 0.0,
            "path": "/x/y", "source": "fs",
        }
        geom = {"polar": 1.0, "azi": 30.0}
        out = angle_offset_candidates_for_load(
            primary=primary, is_file=True, ref=ref, target_geom=geom,
            target_azi_fallback=None, hv=80.0, work_func=4.5,
        )
        labels = {c["candidate"] for c in out}
        # base ±theta0, raw_polar/raw_polar_neg, azi_plus/azi_minus + variantes
        self.assertIn("theta0", labels)
        self.assertIn("-theta0", labels)
        self.assertIn("raw_polar", labels)
        self.assertIn("raw_polar_neg", labels)
        self.assertIn("azi_plus", labels)
        self.assertIn("azi_minus", labels)
        # pas de doublons
        keys = [(round(c["theta0_deg"], 8), round(c["tilt0_deg"], 8), c["candidate"]) for c in out]
        self.assertEqual(len(keys), len(set(keys)))


class TestScoreBmGammaResidual(unittest.TestCase):
    def _make_loaded(self, kpar):
        return {
            "data": np.zeros((2, 2)),
            "kpar": np.asarray(kpar, dtype=float),
            "ev_arr": np.asarray([0.0, -0.1]),
        }

    def test_inf_when_estimator_none(self):
        s = score_bm_gamma_residual(
            self._make_loaded([-0.1, 0.1]),
            ev_range=(-0.05, 0.0), k_range=(-0.5, 0.5),
            center_window=0.5, smooth_sigma=0.0, estimate_fn=None,
        )
        self.assertEqual(s, float("inf"))

    def test_inf_when_estimator_returns_too_few_points(self):
        def fake(*a, **kw):
            return {"gamma": 0.0, "mad": 0.0, "n": 1}
        s = score_bm_gamma_residual(
            self._make_loaded([-0.1, 0.1]),
            ev_range=(-0.05, 0.0), k_range=(-0.5, 0.5),
            center_window=0.5, smooth_sigma=0.0, estimate_fn=fake,
        )
        self.assertEqual(s, float("inf"))

    def test_score_components(self):
        # gamma=0.02, mad=0.04, n=10, k_mid=0 → 0.02 + 0.25*0.04 + 0 = 0.03
        def fake(*a, **kw):
            return {"gamma": 0.02, "mad": 0.04, "n": 10}
        s = score_bm_gamma_residual(
            self._make_loaded([-0.1, 0.1]),
            ev_range=(-0.05, 0.0), k_range=(-0.5, 0.5),
            center_window=0.5, smooth_sigma=0.0, estimate_fn=fake,
        )
        self.assertAlmostEqual(s, 0.03, places=10)

    def test_inf_on_estimator_exception(self):
        def fake(*a, **kw):
            raise RuntimeError("nope")
        s = score_bm_gamma_residual(
            self._make_loaded([-0.1, 0.1]),
            ev_range=(-0.05, 0.0), k_range=(-0.5, 0.5),
            center_window=0.5, smooth_sigma=0.0, estimate_fn=fake,
        )
        self.assertEqual(s, float("inf"))


if __name__ == "__main__":
    unittest.main()
