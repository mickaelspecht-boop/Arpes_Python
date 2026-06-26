"""Curvature anisotropy weight w (Igor parity): default (dk/dE)², scaling, effect."""
import numpy as np

from arpes.physics.plot_compute import DerivParams, compute_curvature


def _band(nk=40, ne=30):
    kpar = np.linspace(-1.0, 1.0, nk)        # dk = 2/(nk-1)
    ev = np.linspace(-0.20, 0.0, ne)         # dE = 0.20/(ne-1)
    k0 = 0.8 * ev                            # band dispersing in k vs E
    kk = kpar[:, None]
    data = np.exp(-((kk - k0[None, :]) / 0.15) ** 2) + 0.05
    return data, kpar, ev


def test_default_weight_equals_dk_over_dE_squared():
    data, kpar, ev = _band()
    c_auto = compute_curvature(data, kpar, ev)            # weight=None -> (dk/dE)^2
    dk = float(np.median(np.abs(np.diff(kpar))))
    de = float(np.median(np.abs(np.diff(ev))))
    c_expl = compute_curvature(data, kpar, ev, weight=(dk / de) ** 2)
    assert np.allclose(np.nan_to_num(c_auto), np.nan_to_num(c_expl))


def test_weight_scale_multiplies_base():
    data, kpar, ev = _band()
    dk = float(np.median(np.abs(np.diff(kpar))))
    de = float(np.median(np.abs(np.diff(ev))))
    c_scaled = compute_curvature(data, kpar, ev, weight_scale=2.0)
    c_expl = compute_curvature(data, kpar, ev, weight=2.0 * (dk / de) ** 2)
    assert np.allclose(np.nan_to_num(c_scaled), np.nan_to_num(c_expl))


def test_weight_changes_output():
    data, kpar, ev = _band()
    c1 = compute_curvature(data, kpar, ev, weight=1.0)
    c2 = compute_curvature(data, kpar, ev, weight=50.0)
    a = np.nan_to_num(c1)
    b = np.nan_to_num(c2)
    assert not np.allclose(a, b)


def test_curvature_finite_and_shaped():
    data, kpar, ev = _band()
    c = compute_curvature(data, kpar, ev, weight=1.0)
    assert c.shape == data.shape
    assert np.isfinite(c).any()


def test_derivparams_default_weight_scale():
    assert DerivParams().curv_weight_scale == 1.0
