from __future__ import annotations

import numpy as np

from arpes.ui.controllers.fit_overlay_drawer import (
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
