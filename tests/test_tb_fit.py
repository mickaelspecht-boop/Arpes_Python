"""Tests for tight-binding fit module."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy absent")

from arpes.physics import tb_fit


@requires_scipy
class TestTB1D:
    def test_recovers_synthetic_1d(self):
        a = 3.9
        eps0_true, t_true = 0.05, 0.30
        k = np.linspace(-np.pi / a, np.pi / a, 64)
        E = tb_fit.tb_1d_chain(k, eps0_true, t_true, a)
        res = tb_fit.fit_dispersion_1d(k, E, a)
        assert abs(res.params["eps0"] - eps0_true) < 1e-3
        assert abs(res.params["t"] - t_true) < 1e-3
        assert res.m_eff_over_me is not None and res.m_eff_over_me > 0
        assert res.bandwidth_eV == pytest.approx(4.0 * t_true, rel=1e-3)

    def test_noisy_1d_within_error(self):
        rng = np.random.default_rng(0)
        a = 4.0
        k = np.linspace(-0.8, 0.8, 40)
        E_clean = tb_fit.tb_1d_chain(k, -0.05, 0.25, a)
        noise = rng.normal(0, 0.005, size=k.shape)
        res = tb_fit.fit_dispersion_1d(k, E_clean + noise, a,
                                       sigma=np.full_like(k, 0.005))
        assert abs(res.params["t"] - 0.25) < 5 * res.perr["t"]

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            tb_fit.fit_dispersion_1d(np.array([0.0]), np.array([0.0]), 4.0)


@requires_scipy
class TestTB2DSquare:
    def test_recovers_synthetic_square(self):
        a = 3.9
        true = dict(eps0=0.0, t=0.30, tp=-0.09, tpp=0.02)
        kx = np.linspace(-np.pi / a, np.pi / a, 25)
        ky = np.linspace(-np.pi / a, np.pi / a, 25)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        E = tb_fit.tb_2d_square(KX.ravel(), KY.ravel(), **true, a=a)
        res = tb_fit.fit_dispersion_2d(KX.ravel(), KY.ravel(), E, a, lattice_type="square")
        for name, v in true.items():
            assert abs(res.params[name] - v) < 5e-3, (name, res.params[name], v)
        assert res.n_points == KX.size
        assert res.chi2_red < 1e-6

    def test_multi_band_warning(self):
        a = 3.9
        # Mock dispersion spanning much wider than 8t to trigger note
        kx = np.linspace(-1.0, 1.0, 20)
        ky = np.linspace(-1.0, 1.0, 20)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        # Single band but very wide → use very large E range artificially
        E_synthetic = tb_fit.tb_2d_square(KX.ravel(), KY.ravel(),
                                          0.0, 0.1, 0.0, 0.0, a)
        # Concatenate fake second-band span to force note
        E_wide = np.concatenate([E_synthetic, E_synthetic + 5.0])
        kx_wide = np.concatenate([KX.ravel(), KX.ravel()])
        ky_wide = np.concatenate([KY.ravel(), KY.ravel()])
        res = tb_fit.fit_dispersion_2d(kx_wide, ky_wide, E_wide, a,
                                       lattice_type="square")
        assert any("multi-band" in n.lower() or "pocket" in n.lower()
                   for n in res.notes)


@requires_scipy
class TestTB2DHex:
    def test_recovers_synthetic_hex(self):
        a = 2.46  # graphene-like
        true = dict(eps0=0.0, t=0.5, tp=-0.05)
        kx = np.linspace(-1.5, 1.5, 21)
        ky = np.linspace(-1.5, 1.5, 21)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        E = tb_fit.tb_2d_hex(KX.ravel(), KY.ravel(), **true, a=a)
        res = tb_fit.fit_dispersion_2d(KX.ravel(), KY.ravel(), E, a, lattice_type="hex")
        for name, v in true.items():
            assert abs(res.params[name] - v) < 5e-3


@requires_scipy
class TestEvaluate:
    def test_evaluate_matches_fit(self):
        a = 4.0
        k = np.linspace(-0.8, 0.8, 32)
        E = tb_fit.tb_1d_chain(k, 0.0, 0.2, a)
        res = tb_fit.fit_dispersion_1d(k, E, a)
        E_eval = tb_fit.evaluate_tb_model(res, k)
        np.testing.assert_allclose(E_eval, E, atol=1e-4)


class TestRenormalization:
    def test_basic(self):
        assert tb_fit.renormalization_vs_dft(0.2, 0.4) == pytest.approx(2.0)
        assert np.isnan(tb_fit.renormalization_vs_dft(0.0, 0.4))


class TestUnknownLattice:
    @requires_scipy
    def test_raises(self):
        with pytest.raises(ValueError):
            tb_fit.fit_dispersion_2d(
                np.array([0.0, 0.1]), np.array([0.0, 0.1]),
                np.array([0.0, 0.1]), 4.0, lattice_type="bogus",
            )
