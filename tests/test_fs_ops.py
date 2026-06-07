"""Tests fs_ops: common regrid, diff/sum/ratio, group_by_pol."""
from __future__ import annotations

import numpy as np
import pytest

from arpes.physics.fs_ops import (
    FsPair,
    find_pol_partner,
    fs_diff,
    fs_ratio,
    fs_sum,
    group_files_by_pol,
    regrid_to_common,
)


def _make_fs(n=21, half=1.0, shift=(0.0, 0.0)):
    kx = np.linspace(-half, half, n)
    ky = np.linspace(-half, half, n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    # Off-center Gaussian.
    fs = np.exp(-((KX - shift[0])**2 + (KY - shift[1])**2) / 0.1)
    return kx, ky, fs.T  # (ny, nx)


# ---- regrid_to_common ------------------------------------------------------


class TestRegridCommon:
    def test_identical_grids_pass(self):
        kx, ky, fs = _make_fs()
        pair = regrid_to_common(kx, ky, fs, kx, ky, fs)
        assert isinstance(pair, FsPair)
        assert pair.overlap_ratio == 1.0
        assert np.allclose(pair.fs_a, pair.fs_b, equal_nan=True)

    def test_disjoint_ranges_raise(self):
        kx_a = np.linspace(0.0, 1.0, 11)
        ky_a = np.linspace(0.0, 1.0, 11)
        fs_a = np.zeros((11, 11))
        kx_b = np.linspace(2.0, 3.0, 11)
        with pytest.raises(ValueError):
            regrid_to_common(kx_a, ky_a, fs_a, kx_b, ky_a, fs_a)

    def test_partial_overlap_reports_ratio(self):
        kx_a = np.linspace(-1.0, 1.0, 21); ky_a = kx_a.copy()
        kx_b = np.linspace(0.0, 2.0, 21); ky_b = kx_b.copy()
        fs_a = np.ones((21, 21)); fs_b = np.ones((21, 21))
        pair = regrid_to_common(kx_a, ky_a, fs_a, kx_b, ky_b, fs_b)
        # Overlap kx∈[0,1], ky∈[0,1] → expected ratio 1.0 (intersection covered).
        assert pair.overlap_ratio > 0.95

    def test_labels_propagated(self):
        kx, ky, fs = _make_fs()
        pair = regrid_to_common(kx, ky, fs, kx, ky, fs, label_a="LV", label_b="LH")
        assert pair.label_a == "LV"
        assert pair.label_b == "LH"

    def test_wrong_ndim_raises(self):
        kx, ky, fs = _make_fs()
        with pytest.raises(ValueError):
            regrid_to_common(kx, ky, fs[0], kx, ky, fs)


# ---- fs_diff/sum/ratio -----------------------------------------------------


class TestFsOperations:
    def test_diff_none(self):
        kx, ky, fs_a = _make_fs(shift=(0.1, 0.0))
        _, _, fs_b = _make_fs(shift=(-0.1, 0.0))
        pair = regrid_to_common(kx, ky, fs_a, kx, ky, fs_b)
        d = fs_diff(pair, normalize="none")
        assert d.shape == pair.fs_a.shape
        # Antisymmetric in kx (symmetric Gaussians shifted by ±0.1).
        assert d[d.shape[0]//2, 0] * d[d.shape[0]//2, -1] <= 0 or np.isnan(d).any()

    def test_diff_normalize_sum_dichroism(self):
        kx, ky, fs_a = _make_fs(shift=(0.1, 0.0))
        _, _, fs_b = _make_fs(shift=(-0.1, 0.0))
        pair = regrid_to_common(kx, ky, fs_a, kx, ky, fs_b)
        d = fs_diff(pair, normalize="sum")
        finite = d[np.isfinite(d)]
        # Normalized dichroism ∈ [-1, 1].
        assert finite.min() >= -1.001
        assert finite.max() <= 1.001

    def test_diff_normalize_max(self):
        kx, ky, fs_a = _make_fs()
        _, _, fs_b = _make_fs()
        pair = regrid_to_common(kx, ky, fs_a, kx, ky, fs_b)
        d = fs_diff(pair, normalize="max")
        # Identical FS → diff = 0 everywhere.
        finite = d[np.isfinite(d)]
        assert np.allclose(finite, 0.0, atol=1e-9)

    def test_diff_unknown_norm_raises(self):
        kx, ky, fs = _make_fs()
        pair = regrid_to_common(kx, ky, fs, kx, ky, fs)
        with pytest.raises(ValueError):
            fs_diff(pair, normalize="boom")

    def test_sum_simple(self):
        kx, ky, fs = _make_fs()
        pair = regrid_to_common(kx, ky, fs, kx, ky, fs)
        s = fs_sum(pair)
        assert np.allclose(s, 2.0 * pair.fs_a, equal_nan=True)

    def test_ratio_with_zero_protection(self):
        kx = np.linspace(-1, 1, 11); ky = kx.copy()
        fs_a = np.ones((11, 11))
        fs_b = np.zeros((11, 11))
        pair = regrid_to_common(kx, ky, fs_a, kx, ky, fs_b)
        r = fs_ratio(pair)
        assert np.isnan(r).all()


# ---- group_files_by_pol ----------------------------------------------------


class TestGroupByPol:
    def test_groups_same_run_distinct_pol(self):
        records = [
            {"path": "f1.ibw", "Pol": "LV", "material": "BaNi2As2", "run_id": "R1"},
            {"path": "f2.ibw", "Pol": "LH", "material": "BaNi2As2", "run_id": "R1"},
            {"path": "f3.ibw", "Pol": "LV", "material": "BaNi2As2", "run_id": "R2"},
        ]
        g = group_files_by_pol(records)
        assert len(g) == 2
        key_r1 = ("BaNi2As2", "R1")
        assert key_r1 in g
        assert set(g[key_r1].keys()) == {"LV", "LH"}

    def test_skip_empty_pol(self):
        records = [
            {"path": "f1.ibw", "Pol": "", "material": "X", "run_id": "R"},
            {"path": "f2.ibw", "Pol": "LH", "material": "X", "run_id": "R"},
            {"path": "f3.ibw", "Pol": None, "material": "X", "run_id": "R"},
        ]
        g = group_files_by_pol(records)
        assert len(g) == 1
        assert set(g[("X", "R")].keys()) == {"LH"}

    def test_skip_no_path(self):
        records = [
            {"Pol": "LV", "material": "X", "run_id": "R"},
            {"path": "f2.ibw", "Pol": "LH", "material": "X", "run_id": "R"},
        ]
        g = group_files_by_pol(records)
        assert len(g[("X", "R")]["LH"]) == 1
        assert "LV" not in g[("X", "R")]

    def test_pol_uppercased(self):
        records = [
            {"path": "f1.ibw", "Pol": "lv", "material": "X", "run_id": "R"},
        ]
        g = group_files_by_pol(records)
        assert "LV" in g[("X", "R")]

    def test_empty_records(self):
        g = group_files_by_pol([])
        assert g == {}


# ---- find_pol_partner ------------------------------------------------------


class TestFindPolPartner:
    def test_finds_lh_for_lv(self):
        records = [
            {"path": "lv.ibw", "Pol": "LV", "material": "M", "run_id": "1"},
            {"path": "lh.ibw", "Pol": "LH", "material": "M", "run_id": "1"},
        ]
        g = group_files_by_pol(records)
        assert find_pol_partner(g, "lv.ibw", other_pol="LH") == "lh.ibw"
        assert find_pol_partner(g, "lh.ibw", other_pol="LV") == "lv.ibw"

    def test_no_partner_returns_none(self):
        records = [
            {"path": "lv.ibw", "Pol": "LV", "material": "M", "run_id": "1"},
        ]
        g = group_files_by_pol(records)
        assert find_pol_partner(g, "lv.ibw", other_pol="LH") is None

    def test_unknown_path_returns_none(self):
        records = [
            {"path": "lv.ibw", "Pol": "LV", "material": "M", "run_id": "1"},
            {"path": "lh.ibw", "Pol": "LH", "material": "M", "run_id": "1"},
        ]
        g = group_files_by_pol(records)
        assert find_pol_partner(g, "nope.ibw", other_pol="LH") is None
