"""Tests B.1 — BM projection in the FS frame."""
from __future__ import annotations

import unittest

import numpy as np

from arpes.core.session import FileEntry, FileMeta
from arpes.physics.bm_cut_overlay import (
    BMCutLine,
    _scale_factor,
    compute_bm_cut_in_fs_frame,
)

TEST_A = 3.96


def _bm_entry(
    *, polar=0.0, azi=0.0, hv=60.0, pol="LH", tilt=None, direction=""
) -> FileEntry:
    return FileEntry(meta=FileMeta(
        hv=hv, polar=polar, azi=azi, tilt=tilt, direction=direction,
        polarization=pol, scan_kind="BM",
    ))


def _fs_entry(
    *, polar=0.0, azi=0.0, hv=60.0, pol="LH", tilt=None, direction=""
) -> FileEntry:
    return FileEntry(meta=FileMeta(
        hv=hv, polar=polar, azi=azi, tilt=tilt, direction=direction,
        polarization=pol, scan_kind="FS",
    ))


def _fs_metadata(scan_center: float | None = 0.0) -> dict:
    md: dict = {"fs_data": object()}
    if scan_center is not None:
        md["fs_scan_axis_deg"] = {
            "min": scan_center - 5.0, "max": scan_center + 5.0,
            "center": scan_center, "step": 0.1, "n": 100,
        }
    return md


class TestScaleFactor(unittest.TestCase):
    def test_positive_for_valid_hv(self):
        s = _scale_factor(60.0, 4.5, 3.96)
        self.assertGreater(s, 0)

    def test_none_for_negative_ek(self):
        self.assertIsNone(_scale_factor(3.0, 4.5, 3.96))


class TestComputeBmCut(unittest.TestCase):
    WF = 4.5

    def test_returns_none_for_non_bm(self):
        fs = _fs_entry()
        not_bm = FileEntry(meta=FileMeta(scan_kind="KZ"))
        out = compute_bm_cut_in_fs_frame(
            not_bm, "/d/kz.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertIsNone(out)

    def test_polar_zero_gives_ky_zero_in_fs_center(self):
        bm = _bm_entry(polar=0.0)
        fs = _fs_entry(polar=0.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out.ky_points, 0.0, atol=1e-10)
        self.assertEqual(out.quality, "exact")

    def test_polar_nonzero_shifts_ky_in_fs(self):
        bm = _bm_entry(polar=2.0)
        fs = _fs_entry(polar=0.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertIsNotNone(out)
        # All ky points = same value ≠ 0.
        ky0 = out.ky_points[0]
        self.assertGreater(abs(ky0), 1e-3)
        np.testing.assert_allclose(out.ky_points, ky0, atol=1e-12)

    def test_quality_rotated_for_azi_diff(self):
        bm = _bm_entry(azi=0.0)
        fs = _fs_entry(azi=30.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertEqual(out.quality, "rotated")
        self.assertIn("Δazi", out.warning)

    def test_quality_exact_for_azi_wraparound(self):
        bm = _bm_entry(azi=359.0)
        fs = _fs_entry(azi=1.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF, a_lattice=TEST_A, azi_tolerance_deg=2.1,
        )
        self.assertEqual(out.quality, "exact")

    def test_quality_scaled_for_hv_diff(self):
        bm = _bm_entry(hv=60.0)
        fs = _fs_entry(hv=80.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF, a_lattice=TEST_A,
            overlay_max_hv_rel=1.0,  # disables strict guard for the scaled test
        )
        self.assertEqual(out.quality, "scaled")
        self.assertIn("Δhv", out.warning)

    def test_overlay_masked_above_strict_hv_threshold(self):
        bm = _bm_entry(hv=60.0)
        fs = _fs_entry(hv=80.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF, a_lattice=TEST_A, overlay_max_hv_rel=0.05,
        )
        self.assertEqual(out.quality, "incompatible")
        self.assertEqual(out.kx_points.size, 0)
        self.assertIn("overlay hidden", out.warning)

    def test_rotation_90deg_swaps_axes(self):
        bm = _bm_entry(polar=1.0, azi=0.0)
        fs = _fs_entry(polar=0.0, azi=90.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF, a_lattice=TEST_A,
        )
        # Original line ky=ky0 (constant), kx varies along t.
        # After 90° rotation: x_new = -ky0, y_new = t.
        # → kx must be constant, ky must vary.
        kx_unique = np.unique(np.round(out.kx_points, 6))
        self.assertEqual(len(kx_unique), 1)
        self.assertGreater(out.ky_points.max() - out.ky_points.min(), 0.1)

    def test_direction_fallback_rotates_when_azi_missing(self):
        bm = _bm_entry(polar=0.0, azi=None, direction="Γ-M")
        fs = _fs_entry(polar=0.0, azi=None, direction="Γ-X")
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertEqual(out.quality, "rotated")
        self.assertIn("direction", out.warning)
        self.assertGreater(out.ky_points.max() - out.ky_points.min(), 0.1)

    def test_returns_incompatible_for_invalid_fs_hv(self):
        bm = _bm_entry(hv=60.0)
        fs = _fs_entry(hv=0.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertEqual(out.quality, "incompatible")

    def _ky_mean(self, bm_tilt, fs_tilt):
        bm = _bm_entry(tilt=bm_tilt)
        fs = _fs_entry(tilt=fs_tilt)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF, a_lattice=TEST_A,
        )
        return out

    def test_tilt_corrected_not_disabled(self):
        # P2.1b — tilt > 2° is no longer rejected: overlay drawn, ky shifted.
        out = self._ky_mean(3.0, 0.0)
        self.assertNotEqual(out.quality, "incompatible")
        self.assertGreater(out.kx_points.size, 0)
        base = self._ky_mean(0.0, 0.0)
        # ky shifted by tilt (≈ scale·sin(3°) > 0) vs the no-tilt case.
        self.assertGreater(
            float(np.mean(out.ky_points)) - float(np.mean(base.ky_points)), 1e-3
        )

    def test_fs_tilt_corrected_not_disabled(self):
        # Opposite FS tilt → ky shift with opposite sign, no rejection.
        out = self._ky_mean(0.0, 4.0)
        self.assertNotEqual(out.quality, "incompatible")
        base = self._ky_mean(0.0, 0.0)
        self.assertLess(
            float(np.mean(out.ky_points)) - float(np.mean(base.ky_points)), -1e-3
        )

    def test_small_tilt_corrected_silently(self):
        # Zone < 10°: corrected without residual note.
        out = self._ky_mean(1.5, 0.0)
        self.assertNotEqual(out.quality, "incompatible")
        self.assertNotIn("residual", out.warning)

    def test_large_tilt_notes_residual(self):
        # tilt > 10°: corrected at the center + mismatch note far from center.
        out = self._ky_mean(15.0, 0.0)
        self.assertNotEqual(out.quality, "incompatible")
        self.assertIn("Ishida", out.warning)

    def test_tilt_none_regression_unchanged(self):
        # Missing tilt (None) must reproduce the exact historical behavior.
        bm = _bm_entry(polar=0.0, tilt=None)
        fs = _fs_entry(polar=0.0, tilt=None)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertEqual(out.quality, "exact")
        self.assertEqual(out.warning, "")
        np.testing.assert_allclose(out.ky_points, 0.0, atol=1e-10)

    def test_label_is_basename_stem(self):
        bm = _bm_entry()
        fs = _fs_entry()
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bna_s2/bm_03.txt", fs, "/d/bna_s2/fs1.txt", _fs_metadata(),
            work_func=self.WF, a_lattice=TEST_A,
        )
        self.assertEqual(out.label, "bm_03")


if __name__ == "__main__":
    unittest.main()
