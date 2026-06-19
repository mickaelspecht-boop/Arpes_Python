"""FS autocorrelation C(q) for nesting analysis."""
from __future__ import annotations

import numpy as np
import pytest

from arpes.physics.nesting import (
    autocorrelation_peaks,
    autocorrelation_q_axes,
    fs_autocorrelation,
)


def test_autocorr_delta_peaks_at_center():
    fs = np.zeros((9, 9))
    fs[4, 4] = 1.0
    ac = fs_autocorrelation(fs)
    iy, ix = np.unravel_index(int(np.argmax(ac)), ac.shape)
    assert (iy, ix) == (4, 4)  # centered q=0
    assert ac[4, 4] == pytest.approx(1.0)


def test_autocorr_two_deltas_peak_at_separation():
    # Two bright points separated by dx=3 columns -> autocorr side peaks at ±3.
    fs = np.zeros((11, 11))
    fs[5, 3] = 1.0
    fs[5, 6] = 1.0
    ac = fs_autocorrelation(fs, normalize=True)
    center = 11 // 2
    # side peaks on the same row, offset ±3 from center
    assert ac[center, center + 3] == pytest.approx(ac[center, center - 3])
    assert ac[center, center + 3] > 0.4  # strong cross-overlap


def test_q_axes_centered_and_spaced():
    kx = np.linspace(-1.0, 1.0, 11)  # dk = 0.2
    ky = np.linspace(-2.0, 2.0, 21)
    qx, qy = autocorrelation_q_axes(kx, ky)
    assert qx[len(qx) // 2] == pytest.approx(0.0)
    assert np.mean(np.diff(qx)) == pytest.approx(0.2)
    assert qy.size == 21


def test_peaks_excludes_origin_finds_offcenter():
    fs = np.zeros((21, 21))
    fs[10, 6] = 1.0
    fs[10, 14] = 1.0  # separation 8 px
    ac = fs_autocorrelation(fs, normalize=True)
    qx, qy = autocorrelation_q_axes(np.arange(21) * 1.0, np.arange(21) * 1.0)
    peaks = autocorrelation_peaks(ac, qx, qy, n_peaks=2)
    assert peaks
    # strongest off-center peak at |qx| = 8 px
    assert abs(abs(peaks[0]["qx"]) - 8.0) < 1.5
    assert peaks[0]["value"] > 0.3


def test_subtract_mean_changes_background():
    rng = np.random.default_rng(0)
    fs = rng.random((16, 16)) + 5.0  # large DC offset
    raw = fs_autocorrelation(fs, subtract_mean=False, normalize=False)
    conn = fs_autocorrelation(fs, subtract_mean=True, normalize=False)
    # DC removal shrinks the q=0 peak dramatically
    assert conn[8, 8] < raw[8, 8]
