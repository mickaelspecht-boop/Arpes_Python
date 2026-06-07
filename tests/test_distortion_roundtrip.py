"""Round-trip diagnostic for distortion auto-detect → apply.

Verifies the *sign convention* of both trapezoid and parabola corrections:
- Build a synthetic BM with a known distortion (trapezoid widening at low E,
  or parabolic dispersion E=a·k²).
- Run auto-detect → apply.
- Check the corrected BM is STRAIGHTER than the input (smaller distortion).

If apply with auto-detected params makes the BM *worse*, the sign convention
is inverted somewhere — this test will fail and locate the bug.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy missing")

from arpes.physics.distortion import (
    apply_distortion,
    auto_detect_parabola,
    auto_detect_trapezoid,
)


def _build_hard_trapezoid(slope_phys: float = 0.5):
    """Build BM with hard-edged trapezoid wider at lower E (de<0).

    Bright region between k_left(e) = -0.3 + slope_phys·de and
    k_right(e) = +0.3 − slope_phys·de. Smoothed with Gaussian for realism.
    """
    from scipy.ndimage import gaussian_filter
    kpar = np.linspace(-1.0, 1.0, 160)
    ev = np.linspace(-0.30, 0.05, 120)
    pivot = 0.5 * (ev[0] + ev[-1])
    KK, EE = np.meshgrid(kpar, ev, indexing="ij")
    de = EE - pivot
    k_left = -0.3 + slope_phys * de
    k_right = +0.3 - slope_phys * de
    bright = ((KK >= k_left) & (KK <= k_right)).astype(np.float32)
    data = gaussian_filter(bright, sigma=2.0).astype(np.float32)
    return data, kpar, ev


def _measure_edge_slopes(data, kpar, ev, percentile=50):
    """Return (slope_left, slope_right) of the bright-region edges vs E.

    Slope = ∂k_edge/∂E. For a straightened BM, both ~ 0.
    """
    arr = np.asarray(data)
    thr = np.nanpercentile(arr, percentile)
    mask = arr > thr
    left = np.full(ev.size, np.nan)
    right = np.full(ev.size, np.nan)
    for j in range(ev.size):
        idx = np.where(mask[:, j])[0]
        if idx.size >= 3:
            left[j] = kpar[idx[0]]
            right[j] = kpar[idx[-1]]
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 4:
        return np.nan, np.nan
    de = ev[valid] - 0.5 * (ev[0] + ev[-1])
    sl = np.polyfit(de, left[valid], 1)[0]
    sr = np.polyfit(de, right[valid], 1)[0]
    return float(sl), float(sr)


@requires_scipy
class TestTrapezoidRoundTrip:
    def test_auto_apply_reduces_widening_trapezoid(self):
        # Trapezoid wider at lower E (de<0). Hard edges → auto-detect robust.
        data, kpar, ev = _build_hard_trapezoid(slope_phys=0.5)
        sl_in, sr_in = _measure_edge_slopes(data, kpar, ev)
        # Sanity: input has clearly opposite slopes (left>0, right<0 for widening at low E)
        assert sl_in > 0.1, f"input left slope expected >0.1, got {sl_in}"
        assert sr_in < -0.1, f"input right slope expected <-0.1, got {sr_in}"

        cfg = auto_detect_trapezoid(data, kpar, ev)
        assert cfg is not None, "auto-detect should succeed on sharp trapezoid"
        full_cfg = {
            "enabled": True,
            "trapezoid": {
                "enabled": True,
                "slope_left": cfg["slope_left"],
                "slope_right": cfg["slope_right"],
                "pivot_ev": cfg["pivot_ev"],
            },
        }
        corrected, info = apply_distortion(data, kpar, ev, full_cfg)
        assert info["applied"]

        sl_out, sr_out = _measure_edge_slopes(corrected, kpar, ev)
        # After correction, both edge slopes must be CLOSER to zero (any
        # reduction proves sign convention is correct).
        assert abs(sl_out) < abs(sl_in), (
            f"left slope NOT reduced: in={sl_in:.4f} out={sl_out:.4f}"
        )
        assert abs(sr_out) < abs(sr_in), (
            f"right slope NOT reduced: in={sr_in:.4f} out={sr_out:.4f}"
        )


@requires_scipy
class TestParabolaRoundTrip:
    def test_auto_apply_straightens_downward_band(self):
        # Synthetic band E_peak(k) = -a_true * k² with a_true > 0 (downward dispersion)
        a_true = 0.5
        kpar = np.linspace(-0.8, 0.8, 140)
        ev = np.linspace(-0.40, -0.01, 100)
        KK, EE = np.meshgrid(kpar, ev, indexing="ij")
        E_peak = -a_true * KK ** 2
        sigma = 0.020
        data = np.exp(-((EE - E_peak) ** 2) / (2 * sigma ** 2)).astype(np.float32)

        cfg = auto_detect_parabola(data, kpar, ev)
        assert cfg is not None
        # Detect should recover negative a (downward dispersion). Magnitude
        # may have ~30% bias from edge effects but sign must be right.
        assert cfg["a"] < -0.1, f"a expected <0, got {cfg['a']}"

        full_cfg = {
            "enabled": True,
            "parabola": {"enabled": True, "a": cfg["a"], "k0": cfg["k0"]},
        }
        corrected, info = apply_distortion(data, kpar, ev, full_cfg)
        assert info["applied"]

        # After straightening, argmax over E for each k should be ~constant
        # (band horizontal). Measure dispersion: std of argmax_E.
        arg_e_in = np.argmax(data, axis=1)
        arg_e_out = np.argmax(corrected, axis=1)
        # Restrict to valid k range (avoid edge nan after warp)
        mid = slice(20, -20)
        std_in = np.std(ev[arg_e_in[mid]])
        std_out = np.std(ev[arg_e_out[mid]])
        assert std_out < std_in, (
            f"parabola NOT flattened: std_in={std_in:.4f} std_out={std_out:.4f} "
            f"(sign convention inverted)"
        )
