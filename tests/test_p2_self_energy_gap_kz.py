"""Tests P2.3 (consistent self-energy) + P2.5 (kz/gap).

- Dynes: positive DOS without |Re| hack (fixed branch).
- Norman: positive ARPES spectral function, peaks at ±Δ.
- kz: warn on negative radicand + |E_center| > 0.05.
- λ unified through kink_analysis.extract_lambda; DFT double-counting note.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arpes.physics.gap_extraction import (
    dynes,
    norman_spectral,
    fit_norman_single,
    fit_dynes_single,
)
from arpes.physics import kz as kzmod
from arpes.analysis.self_energy import _estimate_kink, _double_counting_notes


# ---------------------------------------------------------------- P2.5 Dynes

class TestDynesPositive:
    def test_positive_everywhere_finite_gamma(self):
        w = np.linspace(-20, 20, 401)
        y = dynes(w, 5.0, 0.7)
        assert np.all(y >= 0.0)                # no more negative Re in the gap

    def test_symmetric(self):
        w = np.linspace(-15, 15, 301)
        assert np.allclose(dynes(w, 4.0, 0.3), dynes(-w, 4.0, 0.3), atol=1e-9)

    def test_peak_near_delta(self):
        w = np.linspace(-15, 15, 601)
        y = dynes(w, 6.0, 0.3)
        assert abs(abs(w[int(np.argmax(y))]) - 6.0) < 0.6


# --------------------------------------------------------------- P2.5 Norman

class TestNormanSpectral:
    def test_positive_definite(self):
        w = np.linspace(-30, 30, 601)
        assert np.all(norman_spectral(w, 8.0, 1.0, 1.0) > 0.0)

    def test_two_peaks_at_delta(self):
        w = np.linspace(-20, 20, 801)
        y = norman_spectral(w, 7.0, 0.3, 0.3)
        assert abs(abs(w[int(np.argmax(y))]) - 7.0) < 0.5

    def test_delta_zero_single_peak_at_zero(self):
        w = np.linspace(-20, 20, 401)
        y = norman_spectral(w, 0.0, 2.0, 1.0)
        assert abs(w[int(np.argmax(y))]) < 0.2

    def test_fit_recovers_delta(self):
        w = np.linspace(-30, 30, 241)
        I = 3.0 * norman_spectral(w, 8.0, 1.5, 1.0) + 0.05
        r = fit_norman_single(w, I, Delta_guess_meV=5.0)
        assert r.deltas_meV[0] == pytest.approx(8.0, abs=0.5)
        assert any("Norman" in n for n in r.notes)

    def test_filled_gap_flagged(self):
        w = np.linspace(-40, 40, 201)
        I = norman_spectral(w, 4.0, 6.0, 1.0)  # Γ/Δ = 1.5 > 0.5
        r = fit_norman_single(w, I, Delta_guess_meV=4.0, Gamma_guess_meV=6.0)
        assert any("Γ₁/Δ" in n for n in r.notes)

    def test_delta_bound_raised_to_100(self):
        # Δ guess 80 meV (pseudogap) accepted, no longer rejected at the 50 bound.
        w = np.linspace(-150, 150, 301)
        I = 2.0 * norman_spectral(w, 80.0, 5.0, 2.0) + 0.01
        r = fit_norman_single(w, I, Delta_guess_meV=70.0,
                              Gamma_guess_meV=5.0, resolution_meV=2.0)
        assert r.deltas_meV[0] > 50.0


# ------------------------------------------------------------------ P2.5 kz

class TestKzWarnings:
    def test_negative_radicand_warns(self):
        with pytest.warns(RuntimeWarning, match="radicand"):
            out = kzmod.kz_from_hv_kpar(
                20.0, np.array([5.0]),
                work_func=4.5, inner_potential=12.0, a_lattice=4.0,
            )
        assert math.isnan(out[0])

    def test_valid_kpar_no_warn(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = kzmod.kz_from_hv_kpar(
                100.0, np.array([0.0]),
                work_func=4.5, inner_potential=12.0, a_lattice=4.0,
            )
        assert np.isfinite(out[0])

    def test_energy_center_off_fs_warns(self):
        with pytest.warns(RuntimeWarning, match="Fermi surface"):
            kzmod._warn_energy_center(0.20)

    def test_energy_center_at_fs_silent(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kzmod._warn_energy_center(0.01)


# -------------------------------------------------------- P2.3 self-energy

class TestLambdaUnified:
    def test_lambda_from_linear_slope(self):
        # ReΣ = −0.6·ω near E_F → λ = −∂ReΣ/∂ω = +0.6.
        e = np.linspace(-0.10, 0.04, 30)
        re = -0.6 * e
        kink_e, lam, lam_err = _estimate_kink(e, re)
        assert lam == pytest.approx(0.6, abs=0.05)
        assert math.isfinite(lam_err)

    def test_returns_three_tuple(self):
        e = np.linspace(-0.1, 0.05, 20)
        out = _estimate_kink(e, np.zeros_like(e))
        assert len(out) == 3

    def test_too_few_points(self):
        out = _estimate_kink(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert len(out) == 3 and all(math.isnan(v) for v in out)


class TestDoubleCountingNote:
    def test_gga_plus_u_flagged(self):
        assert _double_counting_notes("Materials Project GGA+U") != ()

    def test_hybrid_flagged(self):
        assert any("renormalized" in n for n in _double_counting_notes("HSE06 hybrid"))

    def test_plain_pbe_no_note(self):
        assert _double_counting_notes("Materials Project PBE") == ()

    def test_empty_source_no_note(self):
        assert _double_counting_notes("") == ()
