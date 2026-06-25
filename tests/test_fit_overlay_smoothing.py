from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arpes.ui.controllers.fit_overlay_drawer import (
    draw_kf_overlay,
    kf_sigma_arrays,
    smooth_kf_for_display,
    subtle_uncertainty_indices,
)


def test_smooth_kf_for_display_preserves_deleted_points():
    raw = np.array([0.10, 0.12, np.nan, 0.16, 0.18], dtype=float)

    out = smooth_kf_for_display(raw, smooth_on=True, sigma=1.0)

    assert np.isnan(out[2])
    assert np.isfinite(out[[0, 1, 3, 4]]).all()


def test_smooth_kf_for_display_noop_when_disabled():
    raw = np.array([0.10, np.nan, 0.18], dtype=float)

    out = smooth_kf_for_display(raw, smooth_on=False, sigma=1.0)

    np.testing.assert_equal(out, raw)


def test_subtle_uncertainty_indices_downsamples_long_branches():
    idx = subtle_uncertainty_indices(np.arange(80), max_bars=18)

    assert idx.size == 18
    assert idx[0] == 0
    assert idx[-1] == 79
    assert np.all(np.diff(idx) > 0)


def test_kf_sigma_arrays_prefers_full_fit_sigma_keys():
    fr = {
        "sigma_kF_plus": [[0.003, 0.004]],
        "ensemble": {"kF_plus_std": [[0.1, 0.2]]},
    }

    assert kf_sigma_arrays(fr, "kF_plus") == [[0.003, 0.004]]


def test_kf_sigma_arrays_falls_back_to_legacy_ensemble_keys():
    fr = {"ensemble": {"kF_minus_std": [[0.005, 0.006]]}}

    assert kf_sigma_arrays(fr, "kF_minus") == [[0.005, 0.006]]


def test_draw_overlay_keeps_raw_points_when_smoothing_enabled():
    class _Value:
        def __init__(self, value):
            self._value = value
        def value(self):
            return self._value
        def isChecked(self):
            return bool(self._value)

    fig, ax = plt.subplots()
    fr = {
        "e_fitted": [-0.2, -0.1, 0.0],
        "kF_minus": [[-0.4, -0.1, -0.2]],
        "kF_plus": [[0.4, 0.1, 0.2]],
    }
    ctrl = type("Ctrl", (), {})()
    ctrl._fit_res = fr
    ctrl._params = type("Params", (), {
        "sp_np": _Value(1),
        "chk_smooth_kf": _Value(True),
        "sp_smooth_kf_sigma": _Value(1.0),
    })()
    ctrl._parent = type("Parent", (), {
        "_fit_selected": [],
        "_current_path": None,
    })()
    draw_kf_overlay(ctrl, ax)
    offsets = np.vstack([c.get_offsets() for c in ax.collections if len(c.get_offsets())])
    assert any(np.allclose(row, [-0.1, -0.1]) for row in offsets)
    assert any(np.allclose(row, [0.1, -0.1]) for row in offsets)
    plt.close(fig)
