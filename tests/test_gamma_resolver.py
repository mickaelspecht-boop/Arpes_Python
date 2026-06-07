"""Headless tests for the Γ resolver (P2)."""
from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.gamma import apply_bm_gamma_axis_shift
from arpes.physics.gamma_resolver import ResolvedGamma, resolve


def _raw_bm(*, kpar=None, shift=0.0, centered=False, loader=False, polar=0.0):
    meta = {"scan_kind": "BM", "polar": polar, "polar_already_applied_to_kx": True}
    if centered:
        meta["bm_gamma_axis_centered"] = True
        meta["bm_gamma_axis_shift"] = shift
    if loader:
        meta["angle_offsets_applied"] = {"theta0_deg": 0.3, "candidate": "loader_auto"}
    return {
        "path": "/tmp/bm04",
        "hv": 60.0,
        "kpar": np.array(kpar if kpar is not None else [-1.0, 0.0, 1.0], dtype=float),
        "metadata": meta,
    }


def _raw_fs(*, kpar=None, shift_kx=0.0, centered=False, loader=False):
    meta = {
        "fs_data": object(),
        "fs_kx": np.array([-1.0, 0.0, 1.0]),
        "fs_ky": np.array([-1.0, 0.0, 1.0]),
        "fs_kind": "kxky",
    }
    if centered:
        meta["bm_gamma_axis_centered"] = True
        meta["fs_gamma_axis_centered"] = True
        meta["bm_gamma_axis_shift"] = shift_kx
        meta["fs_gamma_axis_shift_kx"] = shift_kx
        meta["fs_gamma_axis_shift_ky"] = 0.0
    if loader:
        meta["angle_offsets_applied"] = {"theta0_deg": 0.3}
    return {
        "path": "/tmp/fs1",
        "hv": 60.0,
        "kpar": np.array(kpar if kpar is not None else [-1.0, 0.0, 1.0], dtype=float),
        "metadata": meta,
    }


class TestResolveBasics(unittest.TestCase):
    def test_no_raw_returns_none_mode(self):
        r = resolve(None, {"kx": 0.1, "ky": 0.0}, work_func=4.5)
        self.assertEqual(r.mode, "none")
        self.assertEqual(r.axis_shift_delta, 0.0)

    def test_no_ref_returns_none_mode(self):
        r = resolve(_raw_bm(), None, work_func=4.5)
        self.assertEqual(r.mode, "none")

    def test_loader_baked_overrides_everything(self):
        r = resolve(
            _raw_bm(loader=True),
            {"kx": 0.3, "ky": 0.0, "path": "/elsewhere"},
            work_func=4.5,
        )
        self.assertEqual(r.mode, "loader_baked")
        self.assertEqual(r.axis_shift_delta, 0.0)
        self.assertEqual(r.display_center, 0.0)
        self.assertEqual(r.fit_center_init, 0.0)


class TestResolveBm(unittest.TestCase):
    def test_bm_same_path_full_delta_when_axis_not_centered(self):
        raw = _raw_bm()
        ref = {
            "kx": 0.07, "ky": 0.0, "path": "/tmp/bm04", "source": "bm",
            "polar": 0.0, "polar_already_applied_to_kx": True, "azi": 0.0,
        }
        r = resolve(raw, ref, work_func=4.5, entry_azi=0.0)
        self.assertEqual(r.mode, "axis_shifted")
        self.assertAlmostEqual(r.axis_shift_target, 0.07)
        self.assertAlmostEqual(r.axis_shift_delta, 0.07)
        self.assertFalse(r.is_fs)

    def test_bm_already_centered_delta_zero(self):
        raw = _raw_bm(centered=True, shift=0.07)
        ref = {
            "kx": 0.07, "ky": 0.0, "path": "/tmp/bm04", "source": "bm",
            "polar": 0.0, "polar_already_applied_to_kx": True, "azi": 0.0,
        }
        r = resolve(raw, ref, work_func=4.5, entry_azi=0.0)
        self.assertEqual(r.mode, "axis_shifted")
        self.assertAlmostEqual(r.axis_shift_delta, 0.0,
                               msg="delta != 0 even though the axis is already up to date")


class TestResolveFs(unittest.TestCase):
    def test_fs_same_path_marker_kx_ky(self):
        raw = _raw_fs()
        ref = {
            "kx": 0.30, "ky": 0.10, "path": "/tmp/fs1", "source": "fs_auto",
            "polar": 0.0, "polar_already_applied_to_kx": False, "azi": 0.0,
        }
        r = resolve(raw, ref, work_func=4.5, entry_azi=0.0)
        self.assertEqual(r.mode, "axis_shifted")
        self.assertTrue(r.is_fs)
        self.assertTrue(r.same_ref_path)
        self.assertAlmostEqual(r.fs_marker_kx, 0.30)
        self.assertAlmostEqual(r.fs_marker_ky, 0.10)
        self.assertAlmostEqual(r.axis_shift_target, 0.30)

    def test_fs_different_path_projected_by_azi(self):
        raw = _raw_fs()
        ref = {
            "kx": 0.30, "ky": 0.0, "path": "/tmp/fs_ref", "source": "fs_auto",
            "polar": 0.0, "polar_already_applied_to_kx": False, "azi": 0.0,
        }
        r = resolve(raw, ref, work_func=4.5, entry_azi=90.0)
        self.assertEqual(r.mode, "axis_shifted")
        # 90° rotation: kx_ref → ky, ky_ref(0) → -kx (near 0).
        self.assertAlmostEqual(r.fs_marker_kx, 0.0, places=10)
        self.assertAlmostEqual(r.fs_marker_ky, -0.30, places=10)


class TestIdempotence(unittest.TestCase):
    """Key P2 invariant: resolve∘apply∘resolve == resolve (delta=0 on second pass)."""

    def test_bm_apply_then_resolve_gives_delta_zero(self):
        raw = _raw_bm()
        ref = {
            "kx": 0.12, "ky": 0.0, "path": "/tmp/bm04", "source": "bm",
            "polar": 0.0, "polar_already_applied_to_kx": True, "azi": 0.0,
        }
        r1 = resolve(raw, ref, work_func=4.5, entry_azi=0.0)
        self.assertNotEqual(r1.axis_shift_delta, 0.0)
        # Apply: shift the axis with the delta.
        ok = apply_bm_gamma_axis_shift(raw, r1.axis_shift_target, ref=ref)
        self.assertTrue(ok)
        # Resolve again: delta must be 0.
        r2 = resolve(raw, ref, work_func=4.5, entry_azi=0.0)
        self.assertEqual(r2.mode, r1.mode)
        self.assertAlmostEqual(r2.axis_shift_delta, 0.0,
                               msg="resolve∘apply∘resolve non idempotent")
        self.assertAlmostEqual(r2.axis_shift_target, r1.axis_shift_target)


if __name__ == "__main__":
    unittest.main()
