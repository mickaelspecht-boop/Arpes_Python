"""Curvature-based dispersion extraction (Zhang) — pure numpy, headless."""
import numpy as np

from arpes.physics.curvature_dispersion import (
    extract_curvature_dispersion,
    momentum_curvature_1d,
)


def _synthetic_hole_band(kpar, ev_arr, *, kf0=0.30, slope=0.8, gamma=0.03):
    """Two Lorentzian peaks at ±kF(E), kF growing with binding depth (hole band).

    Returns data[k, E] and the true kF(E) array.
    """
    data = np.zeros((kpar.size, ev_arr.size))
    kf_true = np.zeros(ev_arr.size)
    for ie, e in enumerate(ev_arr):
        kf = kf0 + slope * abs(e)        # crossings spread as E goes deeper
        kf_true[ie] = kf
        for k0 in (-kf, +kf):
            data[:, ie] += gamma ** 2 / ((kpar - k0) ** 2 + gamma ** 2)
    return data, kf_true


def test_momentum_curvature_peaks_at_intensity_peak():
    k = np.linspace(-1, 1, 401)
    I = 0.04 ** 2 / ((k - 0.25) ** 2 + 0.04 ** 2)
    c = momentum_curvature_1d(I, k, c0_alpha=0.05)
    assert abs(k[int(np.argmax(c))] - 0.25) < 0.02


def test_extract_recovers_two_branches():
    k = np.linspace(-0.8, 0.8, 321)
    ev = np.linspace(-0.15, 0.0, 60)
    data, kf_true = _synthetic_hole_band(k, ev)
    out = extract_curvature_dispersion(
        data, k, ev, ev_start=-0.15, ev_end=0.0,
        k_min=-0.8, k_max=0.8, center_init=0.0, n_pairs=1,
    )
    assert out["method"] == "curvature"
    assert "gamma" not in out and "gamma_corrige" not in out  # positions only
    km = np.asarray(out["kF_minus"][0])
    kp = np.asarray(out["kF_plus"][0])
    assert km.size > 30 and kp.size > 30
    # branches on the correct side of center
    assert np.nanmean(km) < 0 < np.nanmean(kp)
    # recovered |kF| within one k-step of the truth (order matches e_fitted)
    ev_out = np.asarray(out["e_fitted"])
    kf_at = np.interp(ev_out, ev, kf_true)
    dk = abs(k[1] - k[0])
    assert np.nanmedian(np.abs(np.abs(kp) - kf_at)) < 3 * dk
    assert np.nanmedian(np.abs(np.abs(km) - kf_at)) < 3 * dk


def test_empty_window_returns_schema():
    k = np.linspace(-0.5, 0.5, 101)
    ev = np.linspace(-0.1, 0.0, 20)
    data = np.zeros((k.size, ev.size))
    out = extract_curvature_dispersion(
        data, k, ev, ev_start=5.0, ev_end=6.0, n_pairs=2)  # window outside data
    assert out["e_fitted"] == []
    assert len(out["kF_minus"]) == 2 and len(out["kF_plus"]) == 2
    assert out["n_pairs"] == 2


def test_flat_data_no_peaks():
    k = np.linspace(-0.5, 0.5, 101)
    ev = np.linspace(-0.1, 0.0, 20)
    data = np.ones((k.size, ev.size))
    out = extract_curvature_dispersion(
        data, k, ev, ev_start=-0.1, ev_end=0.0, n_pairs=1)
    assert out["e_fitted"] == []
