"""Tests for arpes.physics.norm and arpes.physics.norm_grid_fft.

Strategy: build clean synthetic maps, inject a known artifact (flux
modulation, periodic grid), then check the correction recovers the clean
data within tolerance. References are constructed independently of the
implementation under test.
"""
import warnings

import numpy as np
import pytest

from arpes.physics.norm import (
    _PROFILE_FACTOR_MAX,
    _PROFILE_FACTOR_MIN,
    _finite_ref_mask,
    _safe_profile,
    apply_fs_flux_factors_to_map,
    fs_flux_profile_factors,
    normalize_bandmap_above_ef,
    normalize_bandmap_flux_profile,
    normalize_fs_flux_profiles,
    remove_grid_artifact,
)
from arpes.physics.norm_grid_fft import (
    _dilate_mask,
    remove_grid_artifact_fft2_mask,
)


# ---------------------------------------------------------------- fixtures

NK, NE = 64, 80
ENERGY = np.linspace(-1.0, 0.3, NE)  # eV, EF at 0


def _clean_bandmap(nk: int = NK, ne: int = NE) -> np.ndarray:
    """Smooth k-independent spectrum: any k-profile structure is artifact."""
    spectrum = 1.0 + np.exp(-((ENERGY[:ne] + 0.4) / 0.15) ** 2)
    return np.tile(spectrum, (nk, 1))


def _flux_modulation(nk: int = NK, amplitude: float = 0.3) -> np.ndarray:
    """Slowly varying multiplicative flux, within the clamp bounds."""
    k = np.linspace(0, 1, nk)
    return 1.0 + amplitude * np.sin(2 * np.pi * k)


# ---------------------------------------------------------- _finite_ref_mask

class TestFiniteRefMask:
    def test_clamps_range_to_axis(self):
        mask, (lo, hi) = _finite_ref_mask(ENERGY, (-5.0, 5.0))
        assert lo == pytest.approx(ENERGY.min())
        assert hi == pytest.approx(ENERGY.max())
        assert mask.all()

    def test_unsorted_range_accepted(self):
        m1, b1 = _finite_ref_mask(ENERGY, (-0.2, -0.6))
        m2, b2 = _finite_ref_mask(ENERGY, (-0.6, -0.2))
        assert b1 == b2
        assert np.array_equal(m1, m2)

    def test_all_nan_axis_empty_mask(self):
        mask, _ = _finite_ref_mask(np.full(10, np.nan), (-0.6, -0.2))
        assert mask.sum() == 0

    def test_window_outside_axis_empty_mask(self):
        mask, _ = _finite_ref_mask(ENERGY, (5.0, 6.0))
        assert mask.sum() == 0


# ------------------------------------------------------------ _safe_profile

class TestSafeProfile:
    def test_factors_normalized_to_median(self):
        profile = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        factors = _safe_profile(profile, min_valid=2)
        assert factors is not None
        assert np.median(factors) == pytest.approx(1.0)

    def test_clamped_to_bounds(self):
        # 100x spread: unclamped factors would reach ~10 and ~0.1.
        profile = np.array([0.1, 1.0, 1.0, 1.0, 10.0])
        factors = _safe_profile(profile, min_valid=2)
        assert factors is not None
        assert factors.min() >= _PROFILE_FACTOR_MIN
        assert factors.max() <= _PROFILE_FACTOR_MAX

    def test_too_few_valid_returns_none(self):
        profile = np.array([np.nan, np.nan, np.nan, 1.0])
        assert _safe_profile(profile, min_valid=2) is None

    def test_zero_profile_returns_none(self):
        assert _safe_profile(np.zeros(8), min_valid=2) is None

    def test_nan_filled_with_median(self):
        profile = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        factors = _safe_profile(profile, min_valid=2)
        assert factors is not None
        assert np.isfinite(factors).all()
        assert factors[1] == pytest.approx(1.0)


# --------------------------------------------- normalize_bandmap_flux_profile

class TestNormalizeBandmapFluxProfile:
    def test_removes_known_flux_modulation(self):
        clean = _clean_bandmap()
        flux = _flux_modulation()
        out, label = normalize_bandmap_flux_profile(clean * flux[:, None], ENERGY)
        assert "norm flux k" in label
        # Recovered map proportional to the clean one (global scale free).
        ratio = out / clean
        assert np.nanstd(ratio) / np.nanmean(ratio) < 0.01

    def test_wrong_ndim_passthrough(self):
        arr = np.ones(NE)
        out, label = normalize_bandmap_flux_profile(arr, ENERGY)
        assert label == "without flux norm"
        np.testing.assert_array_equal(out, arr)

    def test_shape_mismatch_passthrough(self):
        arr = np.ones((NK, NE + 3))
        _, label = normalize_bandmap_flux_profile(arr, ENERGY)
        assert label == "without flux norm"

    def test_ref_window_outside_axis(self):
        _, label = normalize_bandmap_flux_profile(_clean_bandmap(), ENERGY, ref_range=(5.0, 6.0))
        assert label == "empty flux norm ref"

    def test_zero_data_passthrough(self):
        arr = np.zeros((NK, NE))
        out, label = normalize_bandmap_flux_profile(arr, ENERGY)
        assert label == "invalid flux norm profile"
        np.testing.assert_array_equal(out, arr)

    def test_extreme_column_clamped_not_amplified(self):
        # CLS angular-limit scenario: one nearly dead column must not be
        # amplified by more than 1/_PROFILE_FACTOR_MIN.
        data = _clean_bandmap()
        data[0, :] *= 1e-3
        out, _ = normalize_bandmap_flux_profile(data, ENERGY)
        amplification = np.nanmean(out[0]) / np.nanmean(data[0])
        assert amplification <= 1.0 / _PROFILE_FACTOR_MIN + 1e-9


# ---------------------------------------------- normalize_bandmap_above_ef

class TestNormalizeBandmapAboveEf:
    def test_removes_background_profile(self):
        clean = _clean_bandmap()
        flux = _flux_modulation(amplitude=0.2)
        out, label = normalize_bandmap_above_ef(
            clean * flux[:, None], ENERGY, ef_calibrated=True, smooth_k_sigma=0.0
        )
        assert "norm above-EF" in label
        ratio = out / clean
        assert np.nanstd(ratio) / np.nanmean(ratio) < 0.01

    def test_warns_when_ef_not_calibrated(self):
        with pytest.warns(RuntimeWarning, match="EF has not been calibrated"):
            _, label = normalize_bandmap_above_ef(_clean_bandmap(), ENERGY, ef_calibrated=False)
        assert "[EF not calibrated]" in label

    def test_no_warning_when_calibrated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalize_bandmap_above_ef(_clean_bandmap(), ENERGY, ef_calibrated=True)

    def test_axis_below_window_passthrough(self):
        energy = np.linspace(-1.0, -0.1, NE)  # everything below EF
        arr = np.ones((NK, NE))
        out, label = normalize_bandmap_above_ef(arr, energy, ef_calibrated=True)
        assert "impossible" in label
        np.testing.assert_array_equal(out, arr)

    def test_all_nan_energy_axis(self):
        _, label = normalize_bandmap_above_ef(
            np.ones((NK, NE)), np.full(NE, np.nan), ef_calibrated=True
        )
        assert label == "above-EF norm empty energy axis"

    def test_wrong_ndim_passthrough(self):
        _, label = normalize_bandmap_above_ef(np.ones(NE), ENERGY, ef_calibrated=True)
        assert label == "without above-EF norm"


# ------------------------------------------------ normalize_fs_flux_profiles

class TestNormalizeFsFluxProfiles:
    def _clean_volume(self, ny=16, nx=24):
        spectrum = 1.0 + np.exp(-((ENERGY + 0.4) / 0.15) ** 2)
        return np.tile(spectrum, (ny, nx, 1))

    def test_removes_y_and_x_modulation(self):
        clean = self._clean_volume()
        fy = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, clean.shape[0]))
        fx = 1.0 + 0.2 * np.cos(np.linspace(0, 2 * np.pi, clean.shape[1]))
        dirty = clean * fy[:, None, None] * fx[None, :, None]
        out, label = normalize_fs_flux_profiles(dirty, ENERGY)
        assert "norm flux y+x" in label
        ratio = out / clean
        assert np.nanstd(ratio) / np.nanmean(ratio) < 0.02

    def test_axis_toggles(self):
        dirty = self._clean_volume() * 1.5
        _, label = normalize_fs_flux_profiles(dirty, ENERGY, normalize_x=False)
        assert label.startswith("norm flux y ")
        _, label = normalize_fs_flux_profiles(dirty, ENERGY, normalize_y=False)
        assert label.startswith("norm flux x ")

    def test_wrong_ndim_passthrough(self):
        _, label = normalize_fs_flux_profiles(np.ones((4, NE)), ENERGY)
        assert label == "without flux norm"

    def test_factors_match_full_normalization(self):
        clean = self._clean_volume()
        fy = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, clean.shape[0]))
        dirty = clean * fy[:, None, None]
        full, _ = normalize_fs_flux_profiles(dirty, ENERGY)
        safe_y, safe_x, _ = fs_flux_profile_factors(dirty, ENERGY)
        rebuilt = dirty.copy()
        if safe_y is not None:
            rebuilt = rebuilt / safe_y[:, None, None]
        if safe_x is not None:
            rebuilt = rebuilt / safe_x[None, :, None]
        np.testing.assert_allclose(rebuilt, full, rtol=1e-10)

    def test_apply_factors_to_map_consistent(self):
        dirty = self._clean_volume()
        fy = 1.0 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, dirty.shape[0]))
        dirty = dirty * fy[:, None, None]
        safe_y, safe_x, _ = fs_flux_profile_factors(dirty, ENERGY)
        fs_map = dirty[:, :, 10]
        out = apply_fs_flux_factors_to_map(fs_map, safe_y, safe_x)
        expected = fs_map.copy()
        if safe_y is not None:
            expected = expected / safe_y[:, None]
        if safe_x is not None:
            expected = expected / safe_x[None, :]
        np.testing.assert_allclose(out, expected)

    def test_factors_none_on_invalid(self):
        safe_y, safe_x, label = fs_flux_profile_factors(np.zeros((8, 8, NE)), ENERGY)
        assert safe_y is None and safe_x is None
        assert label == "invalid flux norm profile"


# ------------------------------------------------------- remove_grid_artifact

class TestRemoveGridArtifactProfile:
    def _gridded(self, period=6, amp=0.25, n0=96, n1=64):
        rng = np.random.default_rng(7)
        clean = 10.0 + rng.normal(0, 0.05, (n0, n1))
        gain = 1.0 + amp * np.sin(2 * np.pi * np.arange(n0) / period)
        return clean, clean * gain[:, None]

    def test_profile_method_reduces_ripple(self):
        clean, dirty = self._gridded()
        out, info = remove_grid_artifact(dirty, method="profile", strength=1.0)
        assert info["method"] == "profile"
        assert info["grid_ripple_percent"] > 5.0  # artifact was detected
        rms_before = np.sqrt(np.mean((dirty - clean) ** 2))
        rms_after = np.sqrt(np.mean((out - clean) ** 2))
        assert rms_after < 0.3 * rms_before

    def test_fft_method_reduces_ripple(self):
        clean, dirty = self._gridded()
        out, info = remove_grid_artifact(dirty, method="fft", strength=1.0)
        assert info["method"] == "fft"
        assert info["grid_period_px"] == pytest.approx(6.0, rel=0.15)
        rms_before = np.sqrt(np.mean((dirty - clean) ** 2))
        rms_after = np.sqrt(np.mean((out - clean) ** 2))
        assert rms_after < 0.5 * rms_before

    def test_fft_with_known_period(self):
        _, dirty = self._gridded(period=8)
        _, info = remove_grid_artifact(dirty, method="fft", grid_period_px=8.0)
        assert info["grid_period_px"] == pytest.approx(8.0, rel=0.15)

    def test_strength_zero_is_identity(self):
        _, dirty = self._gridded()
        out, _ = remove_grid_artifact(dirty, method="profile", strength=0.0)
        np.testing.assert_allclose(out, dirty, rtol=1e-9)

    def test_median_preserved(self):
        _, dirty = self._gridded()
        out, _ = remove_grid_artifact(dirty, method="profile", strength=1.0)
        assert np.nanmedian(out) == pytest.approx(np.nanmedian(dirty), rel=1e-6)

    def test_nan_positions_preserved(self):
        _, dirty = self._gridded()
        dirty[3, 5] = np.nan
        out, _ = remove_grid_artifact(dirty, method="profile", strength=1.0)
        assert np.isnan(out[3, 5])
        assert np.isfinite(np.delete(out.ravel(), 3 * dirty.shape[1] + 5)).all()

    def test_axis_argument(self):
        clean, dirty = self._gridded()
        out0, _ = remove_grid_artifact(dirty, axis=0, method="profile", strength=1.0)
        out1, _ = remove_grid_artifact(dirty.T, axis=1, method="profile", strength=1.0)
        np.testing.assert_allclose(out1.T, out0, rtol=1e-9)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="profile"):
            remove_grid_artifact(np.ones((8, 8)), method="bogus")

    def test_1d_passthrough(self):
        out, info = remove_grid_artifact(np.ones(10))
        assert info["method"] == "none"
        np.testing.assert_array_equal(out, np.ones(10))

    def test_tiny_axis_passthrough(self):
        arr = np.ones((3, 8))
        out, info = remove_grid_artifact(arr, method="profile")
        assert info["method"] == "none"
        np.testing.assert_array_equal(out, arr)


class TestRemoveGridArtifactFft2Mask:
    def _gridded_2d(self, n=64, period=8, amp=0.4):
        rng = np.random.default_rng(3)
        clean = 10.0 + rng.normal(0, 0.02, (n, n))
        yy, xx = np.mgrid[:n, :n]
        grid = amp * np.sin(2 * np.pi * yy / period) * np.sin(2 * np.pi * xx / period)
        return clean, clean + grid

    def test_removes_2d_grid(self):
        clean, dirty = self._gridded_2d()
        out, info = remove_grid_artifact_fft2_mask(dirty, strength=1.0)
        assert info["removed_peak_count"] > 0
        rms_before = np.sqrt(np.mean((dirty - clean) ** 2))
        rms_after = np.sqrt(np.mean((out - clean) ** 2))
        assert rms_after < 0.3 * rms_before

    def test_clean_image_untouched(self):
        rng = np.random.default_rng(5)
        clean = 10.0 + rng.normal(0, 0.02, (48, 48))
        out, info = remove_grid_artifact_fft2_mask(clean, peak_sensitivity=12.0)
        if info["removed_peak_count"] == 0:
            np.testing.assert_array_equal(out, clean)
        else:
            # If a noise peak crossed threshold, change must stay marginal.
            assert info["rms_delta_percent"] < 5.0

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            remove_grid_artifact_fft2_mask(np.ones((4, 4, 4)))

    def test_tiny_image_passthrough(self):
        arr = np.ones((3, 3))
        out, info = remove_grid_artifact_fft2_mask(arr)
        assert info["removed_peak_count"] == 0
        np.testing.assert_array_equal(out, arr)

    def test_nan_positions_preserved(self):
        _, dirty = self._gridded_2d()
        dirty[2, 2] = np.nan
        out, _ = remove_grid_artifact_fft2_mask(dirty, strength=1.0)
        assert np.isnan(out[2, 2])

    def test_fs_volume_detector_plane(self):
        clean2d, dirty2d = self._gridded_2d(n=64)
        vol = np.stack([dirty2d] * 5, axis=0)  # (ny, nx, E)-like
        out, info = remove_grid_artifact(vol, method="fft2mask", fft2_plane="detector", strength=1.0)
        assert out.shape == vol.shape
        assert info["slice_count"] == 5
        assert info["removed_peak_count"] > 0
        rms_before = np.sqrt(np.mean((dirty2d - clean2d) ** 2))
        rms_after = np.sqrt(np.mean((out[0] - clean2d) ** 2))
        assert rms_after < 0.5 * rms_before

    def test_fs_volume_map_plane(self):
        _, dirty2d = self._gridded_2d(n=32)
        vol = np.stack([dirty2d] * 4, axis=2)  # grid lives in (ny, nx) plane
        out, info = remove_grid_artifact(vol, method="fft2mask", fft2_plane="map", strength=1.0)
        assert info["slice_axis"] == 2
        assert info["slice_count"] == 4
        assert out.shape == vol.shape

    def test_invalid_plane_raises(self):
        vol = np.ones((6, 6, 6))
        with pytest.raises(ValueError, match="fft2_plane"):
            remove_grid_artifact(vol, method="fft2mask", fft2_plane="bogus")


class TestDilateMask:
    def test_radius_zero_identity(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = _dilate_mask(mask, 0)
        np.testing.assert_array_equal(out, mask)

    def test_radius_one_cross(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = _dilate_mask(mask, 1)
        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 2] = True
        expected[2, 1:4] = True
        np.testing.assert_array_equal(out, expected)

    def test_edge_clipping(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        out = _dilate_mask(mask, 1)
        assert out[0, 0] and out[0, 1] and out[1, 0]
        assert out.sum() == 3
