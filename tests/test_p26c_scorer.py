"""Test P2.6c — angle-sign scorer tolerates an off-Γ band (gamma_expected)."""
from __future__ import annotations

import numpy as np

from arpes.physics.gamma import score_bm_gamma_residual_detail


def _loaded():
    return {
        "data": np.ones((10, 10), dtype=float),
        "kpar": np.linspace(-0.5, 0.5, 10),
        "ev_arr": np.linspace(-0.2, 0.05, 10),
    }


def _estimate_fixed(gamma_val):
    def _fn(data, kpar, ev, *, ev_range, k_range, center_guess,
            center_window, smooth_sigma, verbose):
        return {"gamma": gamma_val, "mad": 0.0, "n": 8}
    return _fn


_KW = dict(ev_range=(-0.2, 0.05), k_range=(-0.5, 0.5),
           center_window=0.5, smooth_sigma=1.0)


def test_off_gamma_band_scored_low_when_expected_matches():
    # Band actually at Γ=0.3: with gamma_expected=0.3 the residual ≈ 0.
    det = score_bm_gamma_residual_detail(
        _loaded(), estimate_fn=_estimate_fixed(0.3), gamma_expected=0.3, **_KW)
    assert abs(det["gamma_residual_after"]) < 1e-9
    assert det["score"] < 0.1


def test_off_gamma_band_penalized_when_assuming_zero():
    # Old behavior (gamma_expected=0) penalizes the true off-Γ band.
    det = score_bm_gamma_residual_detail(
        _loaded(), estimate_fn=_estimate_fixed(0.3), gamma_expected=0.0, **_KW)
    assert det["gamma_residual_after"] == 0.3
    assert det["score"] > 0.25


def test_default_expected_zero_backward_compatible():
    det = score_bm_gamma_residual_detail(
        _loaded(), estimate_fn=_estimate_fixed(0.05), **_KW)
    assert det["gamma_residual_after"] == 0.05  # = gamma (default g_exp 0)
