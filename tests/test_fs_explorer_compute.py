"""Headless tests for fs_explorer_compute (FS Explorer cuts)."""
import numpy as np
import pytest

from arpes.physics.fs_explorer_compute import (
    CutResult,
    downsample_volume,
    extract_bm_cut,
    extract_iso_e_slice,
    free_cut_allowed,
    native_cut,
    snap_to_native,
    volume_from_meta,
)


@pytest.fixture
def grid():
    """Analytic volume v(ky, kx, E) = 100*ky + 10*kx + E on known axes."""
    kx = np.linspace(-1.0, 1.0, 21)     # step 0.1
    ky = np.linspace(-0.5, 0.5, 11)     # step 0.1
    e_ax = np.linspace(-0.4, 0.1, 6)    # step 0.1
    vol = (100.0 * ky[:, None, None]
           + 10.0 * kx[None, :, None]
           + e_ax[None, None, :]).astype(np.float32)
    return vol, kx, ky, e_ax


class TestVolumeFromMeta:
    def test_roundtrip_and_float32(self, grid):
        vol, kx, ky, e_ax = grid
        meta = {"fs_data": vol.astype(np.float64), "fs_kx": kx,
                "fs_ky": ky, "fs_energy": e_ax}
        out = volume_from_meta(meta)
        assert out is not None
        v2, *_ = out
        assert v2.dtype == np.float32

    def test_absent_returns_none(self):
        assert volume_from_meta({}) is None

    def test_shape_mismatch_raises(self, grid):
        vol, kx, ky, e_ax = grid
        meta = {"fs_data": vol[:, :-1], "fs_kx": kx, "fs_ky": ky,
                "fs_energy": e_ax}
        with pytest.raises(ValueError, match="axes mismatch"):
            volume_from_meta(meta)


class TestFreeCutAllowed:
    def test_kxky_yes(self):
        assert free_cut_allowed({"fs_kind": "kxky"})

    def test_scan_axis_no(self):
        assert not free_cut_allowed({"fs_kind": "scan-kx-energy"})
        assert not free_cut_allowed({})


class TestIsoESlice:
    def test_exact_index(self, grid):
        vol, kx, ky, e_ax = grid
        sl = extract_iso_e_slice(vol, e_ax, -0.2)
        np.testing.assert_allclose(sl, vol[:, :, 2])

    def test_window_mean(self, grid):
        vol, kx, ky, e_ax = grid
        sl = extract_iso_e_slice(vol, e_ax, -0.2, width=0.1)
        np.testing.assert_allclose(sl, vol[:, :, 1:4].mean(axis=2), rtol=1e-6)


class TestExtractBMCut:
    def test_horizontal_matches_native(self, grid):
        vol, kx, ky, e_ax = grid
        cut = extract_bm_cut(vol, kx, ky, e_ax, cx=0.0, cy=ky[3],
                             angle_deg=0.0, length=2.0, n_pts=21)
        native = native_cut(vol, kx, e_ax, 3)
        np.testing.assert_allclose(cut.image, native.image, rtol=1e-5)
        assert cut.nan_fraction == 0.0

    def test_vertical_cut_values(self, grid):
        vol, kx, ky, e_ax = grid
        cut = extract_bm_cut(vol, kx, ky, e_ax, cx=0.5, cy=0.0,
                             angle_deg=90.0, length=1.0, n_pts=11)
        # along the line v = 100*ky + 10*0.5 + E, ky = t
        expected = (100.0 * cut.k_along[:, None] + 5.0 + e_ax[None, :])
        np.testing.assert_allclose(cut.image, expected, atol=1e-4)

    def test_diagonal_analytic(self, grid):
        vol, kx, ky, e_ax = grid
        cut = extract_bm_cut(vol, kx, ky, e_ax, cx=0.0, cy=0.0,
                             angle_deg=45.0, length=0.4, n_pts=9)
        c = np.cos(np.deg2rad(45.0))
        expected = (100.0 * cut.k_along[:, None] * c
                    + 10.0 * cut.k_along[:, None] * c + e_ax[None, :])
        np.testing.assert_allclose(cut.image, expected, atol=1e-3)

    def test_out_of_volume_is_nan(self, grid):
        vol, kx, ky, e_ax = grid
        # length 4 on a kx span of 2: half the samples fall outside
        cut = extract_bm_cut(vol, kx, ky, e_ax, cx=0.0, cy=0.0,
                             angle_deg=0.0, length=4.0, n_pts=100)
        assert 0.4 < cut.nan_fraction < 0.6
        assert np.isnan(cut.image[0]).all()
        assert np.isfinite(cut.image[50]).all()

    def test_fully_outside_all_nan(self, grid):
        vol, kx, ky, e_ax = grid
        cut = extract_bm_cut(vol, kx, ky, e_ax, cx=10.0, cy=10.0,
                             angle_deg=30.0, length=1.0, n_pts=16)
        assert cut.nan_fraction == 1.0
        assert np.isnan(cut.image).all()

    def test_zero_length_returns_none(self, grid):
        vol, kx, ky, e_ax = grid
        assert extract_bm_cut(vol, kx, ky, e_ax, cx=0.0, cy=0.0,
                              angle_deg=0.0, length=0.0) is None

    def test_output_float32_shape(self, grid):
        vol, kx, ky, e_ax = grid
        cut = extract_bm_cut(vol, kx, ky, e_ax, cx=0.0, cy=0.0,
                             angle_deg=10.0, length=0.5, n_pts=33)
        assert isinstance(cut, CutResult)
        assert cut.image.dtype == np.float32
        assert cut.image.shape == (33, e_ax.size)
        assert cut.k_along.shape == (33,)


class TestNativeAndSnap:
    def test_snap_to_native(self, grid):
        _, _, ky, _ = grid
        assert snap_to_native(ky, 0.06) == np.argmin(np.abs(ky - 0.06))

    def test_native_cut_is_exact_slice(self, grid):
        vol, kx, ky, e_ax = grid
        cut = native_cut(vol, kx, e_ax, 5)
        np.testing.assert_array_equal(cut.image, vol[5])
        np.testing.assert_array_equal(cut.k_along, kx)

    def test_native_cut_clips_index(self, grid):
        vol, kx, ky, e_ax = grid
        cut = native_cut(vol, kx, e_ax, 999)
        np.testing.assert_array_equal(cut.image, vol[-1])


class TestDownsample:
    def test_strided_no_copy(self, grid):
        vol, kx, ky, e_ax = grid
        v2, kx2, ky2 = downsample_volume(vol, kx, ky, 2)
        assert v2.shape == (6, 11, 6)
        assert v2.base is vol
        np.testing.assert_array_equal(kx2, kx[::2])

    def test_factor_one_identity(self, grid):
        vol, kx, ky, e_ax = grid
        v2, kx2, ky2 = downsample_volume(vol, kx, ky, 1)
        assert v2 is vol
