"""Tests BM distortion propagation → FS volume (trapezoid only)."""
from __future__ import annotations

import numpy as np
import pytest

from arpes.physics.distortion import (
    _ky_drift_metric,
    apply_distortion_to_fs_volume,
    fs_domain_checksum,
)

try:
    from scipy.ndimage import map_coordinates  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy.ndimage missing")


def _make_volume(n_ky=15, n_kx=41, n_e=31):
    """Synthetic FS volume: 2D Gaussian centered at kx=0, independent of ky/e."""
    kx = np.linspace(-1.0, 1.0, n_kx)
    ky = np.linspace(-0.5, 0.5, n_ky)
    ev = np.linspace(-0.5, 0.05, n_e)
    KX, EV = np.meshgrid(kx, ev, indexing="ij")
    slice2d = np.exp(-(KX**2 / 0.2 + EV**2 / 0.5))
    vol = np.broadcast_to(slice2d[None, :, :], (n_ky, n_kx, n_e)).astype(np.float32)
    return vol.copy(), kx, ky, ev


def _trap_cfg(slope_l=0.0, slope_r=0.0):
    return {
        "enabled": True,
        "trapezoid": {
            "enabled": True,
            "slope_left": float(slope_l),
            "slope_right": float(slope_r),
            "pivot_ev": -0.2,
        },
        "parabola": {"enabled": False, "a": 0.0, "k0": 0.0},
    }


# ---- identity --------------------------------------------------------------


class TestIdentity:
    def test_no_cfg_returns_input(self):
        vol, kx, ky, ev = _make_volume()
        out, info = apply_distortion_to_fs_volume(vol, kx, ky, ev, None)
        assert info["applied"] is False
        assert np.array_equal(out, vol)

    def test_disabled_cfg_returns_input(self):
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.01)
        cfg["trapezoid"]["enabled"] = False
        out, info = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        assert info["applied"] is False
        assert np.array_equal(out, vol)

    def test_zero_slopes_no_effective_change(self):
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.0, slope_r=0.0)
        out, info = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        assert info["applied"] is False  # apply_distortion 2D returns identity
        assert np.array_equal(out, vol)


# ---- effective propagation -------------------------------------------------


@requires_scipy
class TestPropagation:
    def test_trap_applied_to_all_slices(self):
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.05, slope_r=-0.05)
        out, info = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        assert info["applied"] is True
        assert info["n_slices"] == vol.shape[0]
        # Volume changed (at least one slice differs).
        assert not np.array_equal(out, vol)

    def test_slice_consistency_matches_2d(self):
        """Slice-by-slice = same result as direct 2D apply_distortion."""
        from arpes.physics.distortion import apply_distortion
        vol, kx, ky, ev = _make_volume(n_ky=5)
        cfg = _trap_cfg(slope_l=0.03, slope_r=-0.04)
        out, _ = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        ref, _ = apply_distortion(vol[2], kx, ev, cfg)
        np.testing.assert_allclose(out[2], ref, equal_nan=True, rtol=1e-5)

    def test_parabola_skipped_on_fs_volume(self):
        """arpes-physicist decision: parabola forbidden on FS volume."""
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.05)
        cfg["parabola"] = {"enabled": True, "a": 0.5, "k0": 0.0}
        out, info = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        assert info["parabola_skipped"] is True
        # Remains applied via trapezoid.
        assert info["applied"] is True


# ---- NaN guard -------------------------------------------------------------


@requires_scipy
class TestNaN:
    def test_nan_slice_stays_nan(self):
        vol, kx, ky, ev = _make_volume(n_ky=5)
        vol[2] = np.nan  # ky=2 slice entirely NaN
        cfg = _trap_cfg(slope_l=0.05)
        out, _ = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        # Other slices processed normally; NaN slice remains NaN.
        assert np.isnan(out[2]).all()
        assert not np.isnan(out[0]).all()


# ---- ky drift guard --------------------------------------------------------


class TestDriftGuard:
    def test_high_drift_refuses(self):
        """σ(BM_by_ky)/⟨BM⟩ > 0.15 → rejection."""
        vol, kx, ky, ev = _make_volume(n_ky=5)
        # Creates strong drift: amplifies one ky slice by 10x.
        vol[0] *= 10.0
        cfg = _trap_cfg(slope_l=0.05)
        with pytest.raises(ValueError, match="drift ky"):
            apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)

    @requires_scipy
    def test_low_drift_passes(self):
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.05)
        # Volume uniform by construction → drift ≈ 0.
        out, info = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        assert info["drift_ratio"] < 0.01
        assert info["applied"] is True

    def test_drift_metric_zero_on_uniform(self):
        vol, _, _, _ = _make_volume()
        assert _ky_drift_metric(vol) < 1e-6

    def test_drift_metric_high_on_amplified(self):
        vol, _, _, _ = _make_volume()
        vol[0] *= 10
        assert _ky_drift_metric(vol) > 0.15


# ---- checksum guard --------------------------------------------------------


class TestChecksum:
    @requires_scipy
    def test_checksum_matches_passes(self):
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.05)
        chk = fs_domain_checksum(kx, ev)
        out, info = apply_distortion_to_fs_volume(
            vol, kx, ky, ev, cfg, bm_checksum=chk,
        )
        assert info["applied"] is True

    def test_checksum_mismatch_refuses(self):
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.05)
        # Checksum from another run (kpar shifted +0.5).
        wrong = (-0.5, 1.5, -0.5, 0.05)
        with pytest.raises(ValueError, match="outside FS domain"):
            apply_distortion_to_fs_volume(
                vol, kx, ky, ev, cfg, bm_checksum=wrong,
            )

    @requires_scipy
    def test_checksum_no_check_if_none(self):
        vol, kx, ky, ev = _make_volume()
        cfg = _trap_cfg(slope_l=0.05)
        # bm_checksum=None → no check.
        out, info = apply_distortion_to_fs_volume(vol, kx, ky, ev, cfg)
        assert info["applied"] is True


# ---- input validation ------------------------------------------------------


class TestInputValidation:
    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            apply_distortion_to_fs_volume(
                np.zeros((10, 10)),  # 2D instead of 3D
                np.arange(10), np.arange(10), np.arange(10),
                _trap_cfg(slope_l=0.05),
            )

    def test_axes_mismatch_raises(self):
        vol, kx, ky, ev = _make_volume(n_ky=5, n_kx=10, n_e=8)
        with pytest.raises(ValueError):
            apply_distortion_to_fs_volume(
                vol, kx[:5], ky, ev, _trap_cfg(slope_l=0.05),
            )

    def test_ky_size_mismatch_raises(self):
        vol, kx, ky, ev = _make_volume(n_ky=5)
        with pytest.raises(ValueError):
            apply_distortion_to_fs_volume(
                vol, kx, ky[:3], ev, _trap_cfg(slope_l=0.05),
            )


# ---- fs_domain_checksum ----------------------------------------------------


class TestChecksumHelper:
    def test_reproducible(self):
        kx = np.linspace(-1, 1, 21)
        ev = np.linspace(-0.5, 0.05, 31)
        assert fs_domain_checksum(kx, ev) == fs_domain_checksum(kx, ev)

    def test_shift_changes_checksum(self):
        kx = np.linspace(-1, 1, 21)
        ev = np.linspace(-0.5, 0.05, 31)
        c1 = fs_domain_checksum(kx, ev)
        c2 = fs_domain_checksum(kx + 0.5, ev)
        assert c1 != c2
