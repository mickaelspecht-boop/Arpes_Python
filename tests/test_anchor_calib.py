"""Manual DFT anchor calibration: local-k lookup + scale/shift fit."""
from __future__ import annotations

import numpy as np
import pytest

from arpes.theory.anchor_calib import fit_scale_shift, local_k_for_label


def _data():
    return {
        "labels": [{"label": "Γ", "k": 0.0}, {"label": "X", "k": 1.0}],
        "k_distance": [0.0, 0.5, 1.0],
        "bands": [[0.0, -0.1, -0.2]],
        "branches": [],
    }


def test_local_k_no_branch_is_raw():
    d = _data()
    assert local_k_for_label(d, {}, "Γ") == pytest.approx(0.0)
    assert local_k_for_label(d, {}, "X") == pytest.approx(1.0)


def test_local_k_label_normalization():
    d = _data()
    assert local_k_for_label(d, {}, "gamma") == pytest.approx(0.0)
    assert local_k_for_label(d, {}, "GAMMA") == pytest.approx(0.0)


def test_local_k_unknown_label_is_none():
    assert local_k_for_label(_data(), {}, "M") is None


def test_fit_two_points_exact():
    out = fit_scale_shift([(0.0, -0.05), (1.0, 0.78)], current_scale=1.0)
    assert out is not None
    scale, shift = out
    assert scale == pytest.approx(0.83)
    assert shift == pytest.approx(-0.05)


def test_fit_single_point_keeps_scale():
    out = fit_scale_shift([(0.5, 0.2)], current_scale=2.0)
    scale, shift = out
    assert scale == pytest.approx(2.0)
    assert shift == pytest.approx(-0.8)


def test_fit_three_points_least_squares():
    # Perfect line k = 1.5 u - 0.1 sampled at 3 points -> recovered exactly.
    pairs = [(0.0, -0.1), (1.0, 1.4), (2.0, 2.9)]
    scale, shift = fit_scale_shift(pairs, current_scale=1.0)
    assert scale == pytest.approx(1.5)
    assert shift == pytest.approx(-0.1)


def test_fit_scale_clamped():
    # Huge slope clamped to the spinbox max (5.0).
    scale, _ = fit_scale_shift([(0.0, 0.0), (0.01, 1.0)], current_scale=1.0)
    assert scale == pytest.approx(5.0)


def test_fit_degenerate_same_u_returns_none():
    assert fit_scale_shift([(0.5, 0.1), (0.5, 0.9)], current_scale=1.0) is None


def test_fit_no_points_returns_none():
    assert fit_scale_shift([], current_scale=1.0) is None
