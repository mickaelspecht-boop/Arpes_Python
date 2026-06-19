"""P(k,E) linear-dichroism contrast + resampling."""
from __future__ import annotations

import numpy as np
import pytest

from arpes.physics.polarization import pkE_contrast, resample_map


def test_resample_identity():
    e = np.linspace(-1, 0, 5)
    k = np.linspace(-1, 1, 7)
    v = np.outer(e, k)
    out = resample_map(v, k, e, k, e)
    np.testing.assert_allclose(out, v, atol=1e-9)


def test_resample_midpoint_bilinear():
    e = np.array([0.0, 1.0])
    k = np.array([0.0, 1.0])
    v = np.array([[0.0, 1.0], [1.0, 2.0]])  # v = e + k
    out = resample_map(v, k, e, np.array([0.5]), np.array([0.5]))
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(1.0)  # 0.5 + 0.5


def test_resample_out_of_range_is_nan():
    e = np.array([0.0, 1.0]); k = np.array([0.0, 1.0])
    v = np.zeros((2, 2))
    out = resample_map(v, k, e, np.array([5.0]), np.array([0.5]))
    assert np.isnan(out[0, 0])


def test_pkE_contrast_basic():
    a = np.array([[3.0, 1.0]])
    b = np.array([[1.0, 1.0]])
    p = pkE_contrast(a, b, denom_floor_frac=0.0, clip=0.0)
    np.testing.assert_allclose(p, [[0.5, 0.0]])  # (3-1)/4, (1-1)/2


def test_pkE_contrast_floor_masks_low_sum():
    a = np.array([[10.0, 0.001]])
    b = np.array([[10.0, 0.0]])
    p = pkE_contrast(a, b, denom_floor_frac=0.5)  # floor = 0.5*20 = 10
    assert np.isnan(p[0, 1])  # tiny sum masked
    assert p[0, 0] == pytest.approx(0.0)


def test_pkE_contrast_clip():
    a = np.array([[1.0]]); b = np.array([[0.0]])
    p = pkE_contrast(a, b, denom_floor_frac=0.0, clip=0.5)
    assert p[0, 0] == pytest.approx(0.5)  # raw +1 clipped to 0.5


def test_pkE_contrast_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pkE_contrast(np.zeros((2, 2)), np.zeros((2, 3)))
