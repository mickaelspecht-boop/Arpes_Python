"""Tests for kink_analysis."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy missing")

from arpes.physics import kink_analysis as ka


@requires_scipy
class TestBareParabolic:
    def test_recovers_linear(self):
        # Pure linear data : v_F is uniquely defined but (k0, intercept) are
        # degenerate (model E = v_F·(k − k0) + alpha·(k − k0)²). Test v_F only.
        k = np.linspace(-0.5, 0.5, 50)
        E = -0.3 + 2.0 * (k - 0.1)
        E_full = np.where(E < -0.05, E, np.nan)
        ks = k[np.isfinite(E_full)]
        Es = E_full[np.isfinite(E_full)]
        res = ka.fit_bare_parabolic(ks, Es, window_eV=(-1.0, -0.05))
        assert abs(res["v_F"] - 2.0) < 1e-6
        assert abs(res["alpha"]) < 1e-6


@requires_scipy
class TestSelfEnergy:
    def test_re_sigma_zero_when_bare_equals_exp(self):
        k = np.linspace(-0.3, 0.3, 30)
        E = -0.1 + 1.5 * k
        re, im = ka.compute_self_energy(
            E, k, bare_fn=lambda kk: -0.1 + 1.5 * kk, bare_v_F=1.5
        )
        np.testing.assert_allclose(re, 0.0, atol=1e-12)
        assert im is None

    def test_im_sigma_from_gamma(self):
        k = np.linspace(-0.3, 0.3, 5)
        E = np.linspace(-0.2, 0.0, 5)
        gamma = np.full(5, 0.04)
        v_F = 2.0
        re, im = ka.compute_self_energy(
            E, k, bare_fn=lambda kk: 0.0 * kk, bare_v_F=v_F, gamma_mdc=gamma
        )
        # Im Σ = (v_F / 2) · Γ = (2/2)·0.04 = 0.04
        np.testing.assert_allclose(im, 0.04, atol=1e-12)


@requires_scipy
class TestLambda:
    def test_lambda_from_linear_re_sigma(self):
        omega = np.linspace(-0.05, 0.05, 20)
        re_sigma = -0.5 * omega  # slope = -0.5 → λ = +0.5
        lam, err = ka.extract_lambda(omega, re_sigma, window_eV=0.05)
        assert abs(lam - 0.5) < 1e-6
        assert err < 1e-6


@requires_scipy
class TestRunKinkPipeline:
    def test_synthetic_kink(self):
        # Build dispersion with a kink at ω = -0.07 eV
        # Bare: E_bare(k) = 2.0 * k. Renormalize near E_F: E_exp = E_bare / (1+λ)
        k = np.linspace(-0.2, 0.05, 50)
        E_bare = 2.0 * k
        # Smooth crossover λ=0.5 near E_F, λ→0 deep
        omega_b = E_bare
        renorm = 1.0 + 0.5 * np.exp(-(omega_b / 0.04) ** 2)
        E_exp = E_bare / renorm  # measured dispersion (kink-like)
        res = ka.run_kink_analysis(
            E_exp, k, bare="parabolic",
            bare_window_eV=(-0.5, -0.15),
            lambda_window_eV=0.03,
        )
        assert res.lambda_coupling is not None
        assert res.lambda_coupling > 0.0
        assert res.re_sigma.shape == E_exp.shape

    def test_custom_bare_requires_v_F(self):
        with pytest.raises(ValueError):
            ka.run_kink_analysis(
                np.array([-0.1, 0.0]), np.array([0.0, 0.05]),
                bare="custom",
                bare_model_fn=lambda k: 2.0 * k,
            )


class TestDispersionFromMDC:
    def test_extract_sorted(self):
        payload = [
            {"E": -0.05, "kF_minus": 0.10, "gamma_minus": 0.02},
            {"E": -0.20, "kF_minus": 0.05, "gamma_minus": 0.03},
            {"E": -0.10, "kF_minus": 0.08, "gamma_minus": 0.025},
        ]
        E, k, g = ka.dispersion_from_mdc_peaks(payload, branch_key="kF_minus")
        assert list(E) == [-0.20, -0.10, -0.05]
        assert list(k) == [0.05, 0.08, 0.10]
        assert g is not None and len(g) == 3

    def test_skip_missing(self):
        payload = [
            {"E": -0.05, "kF_minus": 0.10},
            {"E": -0.10},  # missing k
        ]
        E, k, g = ka.dispersion_from_mdc_peaks(payload)
        assert len(E) == 1
        assert g is None
