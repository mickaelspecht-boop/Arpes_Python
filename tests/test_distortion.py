"""Tests for arpes.physics.distortion (pure module + guards)."""
from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.distortion import (
    angle_offsets_hash,
    apply_distortion,
    auto_detect_parabola,
    auto_detect_trapezoid,
    cache_signature,
    calib_key_for_meta,
    clamp_params,
    gamma_shift_signature,
    is_distortion_active,
    is_fs_data,
    signal_bbox,
)


def _synthetic_bm(n_kpar: int = 100, n_e: int = 80,
                  k_min: float = -1.0, k_max: float = 1.0,
                  ev_min: float = -0.4, ev_max: float = 0.1) -> tuple:
    kpar = np.linspace(k_min, k_max, n_kpar)
    ev = np.linspace(ev_min, ev_max, n_e)
    K, E = np.meshgrid(kpar, ev, indexing="ij")
    # Bande parabolique simple : E = -k² → intensité gaussienne autour
    band_e = -0.5 * K ** 2
    data = np.exp(-((E - band_e) ** 2) / (2 * 0.05 ** 2))
    return data.astype(np.float32), kpar, ev


class TestIsActive(unittest.TestCase):
    def test_empty_disabled(self):
        self.assertFalse(is_distortion_active({}))
        self.assertFalse(is_distortion_active(None))

    def test_disabled_top_flag(self):
        self.assertFalse(is_distortion_active({
            "enabled": False,
            "trapezoid": {"enabled": True, "slope_left": 0.1},
        }))

    def test_zero_slopes_zero_a(self):
        self.assertFalse(is_distortion_active({
            "enabled": True,
            "trapezoid": {"enabled": True, "slope_left": 0.0, "slope_right": 0.0},
            "parabola": {"enabled": True, "a": 0.0},
        }))

    def test_active_trapezoid(self):
        self.assertTrue(is_distortion_active({
            "enabled": True,
            "trapezoid": {"enabled": True, "slope_left": 0.05, "slope_right": -0.05},
        }))

    def test_active_parabola(self):
        self.assertTrue(is_distortion_active({
            "enabled": True,
            "parabola": {"enabled": True, "a": -0.3, "k0": 0.0},
        }))


class TestApplyDistortion(unittest.TestCase):
    def test_identity_when_disabled(self):
        data, kpar, ev = _synthetic_bm()
        out, info = apply_distortion(data, kpar, ev, {})
        self.assertIs(out, data)  # bit-exact identity, no copy
        self.assertFalse(info["applied"])

    def test_identity_when_no_effective_change(self):
        data, kpar, ev = _synthetic_bm()
        cfg = {"enabled": True,
               "trapezoid": {"enabled": True, "slope_left": 0.0, "slope_right": 0.0}}
        out, info = apply_distortion(data, kpar, ev, cfg)
        # is_distortion_active=False (zero slopes) → identity short-circuit
        self.assertFalse(info["applied"])

    def test_trapezoid_changes_data(self):
        data, kpar, ev = _synthetic_bm()
        cfg = {"enabled": True,
               "trapezoid": {"enabled": True,
                             "slope_left": 0.5, "slope_right": -0.5,
                             "pivot_ev": 0.0}}
        out, info = apply_distortion(data, kpar, ev, cfg)
        self.assertTrue(info["applied"])
        self.assertTrue(info["trapezoid_applied"])
        self.assertEqual(out.shape, data.shape)
        # certains pixels diffèrent (warp non-trivial)
        diff = np.nansum(np.abs(out - data))
        self.assertGreater(diff, 0)

    def test_parabola_changes_data(self):
        data, kpar, ev = _synthetic_bm()
        cfg = {"enabled": True,
               "parabola": {"enabled": True, "a": -0.5, "k0": 0.0}}
        out, info = apply_distortion(data, kpar, ev, cfg)
        self.assertTrue(info["applied"])
        self.assertTrue(info["parabola_applied"])
        self.assertEqual(out.shape, data.shape)

    def test_shape_mismatch_raises(self):
        data, kpar, ev = _synthetic_bm()
        with self.assertRaises(ValueError):
            apply_distortion(data, kpar[:-2], ev, {
                "enabled": True,
                "parabola": {"enabled": True, "a": 0.1, "k0": 0.0},
            })

    def test_3d_data_raises(self):
        data = np.zeros((10, 20, 30), dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_distortion(data, np.linspace(-1, 1, 10), np.linspace(0, 1, 20), {
                "enabled": True,
                "parabola": {"enabled": True, "a": 0.1, "k0": 0.0},
            })

    def test_nan_at_boundaries_after_warp(self):
        data, kpar, ev = _synthetic_bm()
        cfg = {"enabled": True,
               "trapezoid": {"enabled": True, "slope_left": 0.5, "slope_right": -0.5,
                             "pivot_ev": ev[0]}}
        out, _ = apply_distortion(data, kpar, ev, cfg)
        # Au pivot rien ne bouge ; loin du pivot, certains bords sortent → NaN.
        self.assertTrue(np.isnan(out).any())


class TestClampParams(unittest.TestCase):
    def test_clamp_excessive_slope(self):
        kpar = np.linspace(-1.0, 1.0, 50)
        ev = np.linspace(-0.5, 0.0, 30)
        cfg = {
            "enabled": True,
            "trapezoid": {"enabled": True, "slope_left": 100.0, "slope_right": -100.0},
        }
        out = clamp_params(cfg, kpar, ev)
        max_slope = 0.5 * 2.0 / 0.5  # = 2.0
        self.assertLessEqual(abs(out["trapezoid"]["slope_left"]), max_slope + 1e-6)
        self.assertIn("slope_left", out["trapezoid"]["clamped"])

    def test_clamp_excessive_a(self):
        kpar = np.linspace(-1.0, 1.0, 50)
        ev = np.linspace(-0.5, 0.0, 30)
        cfg = {"enabled": True, "parabola": {"enabled": True, "a": 100.0, "k0": 0.0}}
        out = clamp_params(cfg, kpar, ev)
        max_a = 0.5 * 0.5 / (1.0 ** 2)  # = 0.25
        self.assertLessEqual(abs(out["parabola"]["a"]), max_a + 1e-6)
        self.assertIn("a", out["parabola"]["clamped"])

    def test_no_mutation_of_input(self):
        cfg = {"enabled": True,
               "trapezoid": {"enabled": True, "slope_left": 100.0, "slope_right": -100.0}}
        original = dict(cfg["trapezoid"])
        clamp_params(cfg, np.linspace(-1, 1, 10), np.linspace(0, 1, 10))
        self.assertEqual(cfg["trapezoid"], original)


class TestCacheSignature(unittest.TestCase):
    def test_inactive_signature(self):
        sig = cache_signature({})
        self.assertEqual(sig, ("distortion", False))

    def test_active_signature_hashable(self):
        cfg = {"enabled": True,
               "trapezoid": {"enabled": True, "slope_left": 0.1, "slope_right": -0.1}}
        sig = cache_signature(cfg)
        self.assertNotEqual(sig, ("distortion", False))
        # Hashable (use as dict key)
        d = {sig: True}
        self.assertTrue(d[sig])

    def test_signature_changes_with_params(self):
        cfg1 = {"enabled": True,
                "trapezoid": {"enabled": True, "slope_left": 0.10, "slope_right": -0.10}}
        cfg2 = {"enabled": True,
                "trapezoid": {"enabled": True, "slope_left": 0.11, "slope_right": -0.10}}
        self.assertNotEqual(cache_signature(cfg1), cache_signature(cfg2))


class TestAutoDetect(unittest.TestCase):
    def test_trapezoid_refuses_narrow_bm(self):
        data, kpar, ev = _synthetic_bm(n_kpar=8)
        self.assertIsNone(auto_detect_trapezoid(data, kpar, ev))

    def test_parabola_refuses_narrow_bm(self):
        data, kpar, ev = _synthetic_bm(n_kpar=8)
        self.assertIsNone(auto_detect_parabola(data, kpar, ev))

    def test_parabola_recovers_known_curvature(self):
        # Bande E = -0.5*k^2 (a=-0.5). argmax_k par ligne E doit donner k tel
        # que E = -0.5*k² → fit polyfit deg 2 retrouve a≈-0.5, k0≈0.
        data, kpar, ev = _synthetic_bm(n_kpar=120, n_e=100,
                                        ev_min=-0.4, ev_max=-0.01)
        res = auto_detect_parabola(data, kpar, ev)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["a"], -0.5, delta=0.1)
        self.assertAlmostEqual(res["k0"], 0.0, delta=0.1)

    def test_parabola_refuses_flat_bm(self):
        # Carte uniforme : pas de dispersion → refus
        data = np.ones((50, 40), dtype=np.float32)
        kpar = np.linspace(-1, 1, 50)
        ev = np.linspace(-0.4, 0.1, 40)
        self.assertIsNone(auto_detect_parabola(data, kpar, ev))


class TestGuards(unittest.TestCase):
    def test_is_fs_data_true(self):
        self.assertTrue(is_fs_data({"fs_data": np.zeros((2, 2, 2))}))

    def test_is_fs_data_false(self):
        self.assertFalse(is_fs_data({}))
        self.assertFalse(is_fs_data(None))
        self.assertFalse(is_fs_data({"fs_data": None}))

    def test_angle_offsets_hash_stable(self):
        h1 = angle_offsets_hash({"theta0_deg": 0.5, "tilt0_deg": 0.0})
        h2 = angle_offsets_hash({"tilt0_deg": 0.0, "theta0_deg": 0.5})  # ordre différent
        self.assertEqual(h1, h2)

    def test_angle_offsets_hash_changes_with_value(self):
        h1 = angle_offsets_hash({"theta0_deg": 0.5})
        h2 = angle_offsets_hash({"theta0_deg": 0.6})
        self.assertNotEqual(h1, h2)

    def test_gamma_shift_signature(self):
        sig = gamma_shift_signature({"bm_gamma_axis_centered": True,
                                      "bm_gamma_axis_shift": -1.81})
        self.assertEqual(sig["bm_gamma_axis_centered"], True)
        self.assertAlmostEqual(sig["bm_gamma_axis_shift"], -1.81)

    def test_calib_key_for_meta(self):
        k = calib_key_for_meta({"lens_mode": "MAM", "pass_energy": 20, "hv": 21.2})
        self.assertEqual(k, ("MAM", "20.0", "21.2"))

    def test_calib_key_missing_fields(self):
        k = calib_key_for_meta({"hv": 100})
        self.assertEqual(k[1], "?")  # pass_energy absent


class TestRoundtripReversibility(unittest.TestCase):
    def test_disabled_after_apply_returns_original(self):
        data, kpar, ev = _synthetic_bm()
        # Apply non-trivial correction
        cfg_on = {"enabled": True,
                  "trapezoid": {"enabled": True, "slope_left": 0.2, "slope_right": -0.2,
                                "pivot_ev": 0.0}}
        warped, _ = apply_distortion(data, kpar, ev, cfg_on)
        self.assertFalse(np.array_equal(warped, data))
        # Reset → identity must restore original from raw
        restored, _ = apply_distortion(data, kpar, ev, {})
        self.assertTrue(np.array_equal(restored, data))


class TestSignalBbox(unittest.TestCase):
    def test_bbox_finds_centered_blob(self):
        kpar = np.linspace(-1.0, 1.0, 100)
        ev = np.linspace(-0.5, 0.0, 80)
        K, E = np.meshgrid(kpar, ev, indexing="ij")
        # blob signal entre k∈[-0.3, 0.3], E∈[-0.3, -0.1]
        data = np.exp(-((K / 0.15) ** 2) - (((E + 0.2) / 0.05) ** 2))
        # ajoute zone vide (basse intensité) ailleurs
        bbox = signal_bbox(data, kpar, ev, intensity_percentile=80.0)
        self.assertTrue(bbox["valid"])
        self.assertGreater(bbox["k_min"], -0.6)
        self.assertLess(bbox["k_max"], 0.6)
        self.assertGreater(bbox["ev_min"], -0.4)
        self.assertLess(bbox["ev_max"], -0.05)

    def test_bbox_fallback_on_all_nan(self):
        kpar = np.linspace(-1.0, 1.0, 50)
        ev = np.linspace(-0.5, 0.0, 40)
        data = np.full((50, 40), np.nan)
        bbox = signal_bbox(data, kpar, ev)
        self.assertFalse(bbox["valid"])
        self.assertAlmostEqual(bbox["k_min"], -1.0)
        self.assertAlmostEqual(bbox["k_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
