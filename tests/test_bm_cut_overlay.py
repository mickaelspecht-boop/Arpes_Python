"""Tests B.1 — projection BM dans le repère FS."""
from __future__ import annotations

import unittest

import numpy as np

from arpes.core.session import FileEntry, FileMeta
from arpes.physics.bm_cut_overlay import (
    BMCutLine,
    _scale_factor,
    compute_bm_cut_in_fs_frame,
)


def _bm_entry(*, polar=0.0, azi=0.0, hv=60.0, pol="LH") -> FileEntry:
    return FileEntry(meta=FileMeta(
        hv=hv, polar=polar, azi=azi, polarization=pol, scan_kind="BM",
    ))


def _fs_entry(*, polar=0.0, azi=0.0, hv=60.0, pol="LH") -> FileEntry:
    return FileEntry(meta=FileMeta(
        hv=hv, polar=polar, azi=azi, polarization=pol, scan_kind="FS",
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
            work_func=self.WF,
        )
        self.assertIsNone(out)

    def test_polar_zero_gives_ky_zero_in_fs_center(self):
        bm = _bm_entry(polar=0.0)
        fs = _fs_entry(polar=0.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF,
        )
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out.ky_points, 0.0, atol=1e-10)
        self.assertEqual(out.quality, "exact")

    def test_polar_nonzero_shifts_ky_in_fs(self):
        bm = _bm_entry(polar=2.0)
        fs = _fs_entry(polar=0.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF,
        )
        self.assertIsNotNone(out)
        # toutes les ky points = même valeur ≠ 0
        ky0 = out.ky_points[0]
        self.assertGreater(abs(ky0), 1e-3)
        np.testing.assert_allclose(out.ky_points, ky0, atol=1e-12)

    def test_quality_rotated_for_azi_diff(self):
        bm = _bm_entry(azi=0.0)
        fs = _fs_entry(azi=30.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF,
        )
        self.assertEqual(out.quality, "rotated")
        self.assertIn("Δazi", out.warning)

    def test_quality_scaled_for_hv_diff(self):
        bm = _bm_entry(hv=60.0)
        fs = _fs_entry(hv=80.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF,
        )
        self.assertEqual(out.quality, "scaled")
        self.assertIn("Δhv", out.warning)

    def test_rotation_90deg_swaps_axes(self):
        bm = _bm_entry(polar=1.0, azi=0.0)
        fs = _fs_entry(polar=0.0, azi=90.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(scan_center=0.0),
            work_func=self.WF,
        )
        # ligne d'origine ky=ky0 (constante), kx varie en t
        # après rotation 90° : x_new = -ky0, y_new = t
        # → kx doit être constant, ky doit varier
        kx_unique = np.unique(np.round(out.kx_points, 6))
        self.assertEqual(len(kx_unique), 1)
        self.assertGreater(out.ky_points.max() - out.ky_points.min(), 0.1)

    def test_returns_incompatible_for_invalid_fs_hv(self):
        bm = _bm_entry(hv=60.0)
        fs = _fs_entry(hv=0.0)
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bm.txt", fs, "/d/fs.txt", _fs_metadata(),
            work_func=self.WF,
        )
        self.assertEqual(out.quality, "incompatible")

    def test_label_is_basename_stem(self):
        bm = _bm_entry()
        fs = _fs_entry()
        out = compute_bm_cut_in_fs_frame(
            bm, "/d/bna_s2/bm_03.txt", fs, "/d/bna_s2/fs1.txt", _fs_metadata(),
            work_func=self.WF,
        )
        self.assertEqual(out.label, "bm_03")


if __name__ == "__main__":
    unittest.main()
