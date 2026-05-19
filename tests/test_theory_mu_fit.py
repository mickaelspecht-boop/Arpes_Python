"""Auto-μ adjustment (fit_mu_shift): closed-form alignment of DFT to ARPES."""
import numpy as np

from arpes.theory.models import TheoryBandData, TheoryOverlayConfig, fit_mu_shift


def _data(n=41):
    k = np.linspace(0.0, 1.0, n)
    band = -0.5 + 1.2 * k  # one linear DFT band
    return TheoryBandData.from_dict({
        "k_distance": k.tolist(),
        "bands": [band.tolist()],
        "material_id": "mp-test",
    }), k, band


def _cfg(**kw):
    base = {"enabled": True, "segment": "", "band_indices": "",
            "mu_shift": 0.0, "z_scale": 1.0, "k_scale": 1.0, "k_shift": 0.0}
    base.update(kw)
    return TheoryOverlayConfig.from_dict(base)


def test_recovers_known_offset():
    data, k, band = _data()
    offset = 0.18  # e_exp = E_DFT + offset  ->  μ should be -offset (Z=1, μ_cur=0)
    fr = {"e_fitted": (band + offset).tolist(), "kF_minus": [k.tolist()]}
    res = fit_mu_shift(data, _cfg(), fr)
    assert res is not None
    assert res["mu"] == res["mu"]  # finite
    assert abs(res["mu"] - (-offset)) < 1e-6
    assert res["rms_after"] < res["rms_before"] + 1e-9
    assert res["rms_after"] < 1e-6


def test_z_scale_divides_shift():
    data, k, band = _data()
    z = 2.0
    # E_overlay = z*(E_DFT - μ); pick e_exp = z*E_DFT + C  -> μ = -C/z
    C = 0.3
    fr = {"e_fitted": (z * band + C).tolist(), "kF_minus": [k.tolist()]}
    res = fit_mu_shift(data, _cfg(z_scale=z), fr)
    assert res is not None
    assert abs(res["mu"] - (-C / z)) < 1e-6


def test_robust_median_resists_outlier():
    data, k, band = _data()
    offset = 0.10
    e = band + offset
    e[5] += 5.0  # gross kF outlier
    fr = {"e_fitted": e.tolist(), "kF_minus": [k.tolist()]}
    robust = fit_mu_shift(data, _cfg(), fr, robust=True)
    mean = fit_mu_shift(data, _cfg(), fr, robust=False)
    assert abs(robust["mu"] - (-offset)) < 1e-3
    assert abs(mean["mu"] - (-offset)) > abs(robust["mu"] - (-offset))


def test_no_fit_returns_none():
    data, _, _ = _data()
    assert fit_mu_shift(data, _cfg(), None) is None
    assert fit_mu_shift(data, _cfg(), {"e_fitted": [], "kF_minus": []}) is None
