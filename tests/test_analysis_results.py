"""Tests for arpes.analysis.results: kF/vF/m* extraction + σ propagation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from arpes.analysis.results import (
    HBAR2_OVER_ME_eV_A2,
    BranchResult,
    GammaFermiLiquid,
    LinearFit,
    compute_asymmetry,
    compute_results,
    extract_branch_result,
    fit_gamma_fermi_liquid,
    gamma_reliability_mask,
    weighted_linear_fit,
)


def test_gamma_reliability_mask_flags_merged_and_saturated():
    e = [-0.15, -0.10, -0.05, 0.0]
    kp = [0.30, 0.05, 0.20, 0.20]   # sep = 0.60, 0.10, 0.40, 0.40
    km = [-k for k in kp]
    g = [0.05, 0.05, 0.05, 0.30]    # last pinned at gamma_max
    fr = {
        "e_fitted": e,
        "kF_plus": [kp], "kF_minus": [km],
        "gamma_corrige": [g], "gamma_brut": [g],
    }
    m = gamma_reliability_mask(fr, pair_index=0, gamma_max=0.30)
    # 0.60>2·0.05 True ; 0.10 not > 0.10 merged False ; 0.40>0.10 True ;
    # last saturated (0.30 ≥ 0.98·0.30) False
    np.testing.assert_array_equal(m, [True, False, True, False])


def test_gamma_zero_uses_reliable_region_only():
    # Flat Γ=0.05 where resolved; a near-EF blow-up that must be ignored.
    e = np.linspace(-0.15, 0.0, 16)
    sep = 0.5 - 2.0 * (e + 0.15)                      # shrinks toward EF
    kp = (sep / 2).tolist(); km = (-sep / 2).tolist()
    g = np.where(e > -0.03, 0.30, 0.05).tolist()     # blow-up in last slices
    fr = {"e_fitted": e.tolist(), "kF_plus": [kp], "kF_minus": [km],
          "gamma_corrige": [g], "gamma_brut": [g]}
    res = fit_gamma_fermi_liquid(fr, pair_index=0, e_window=0.30, gamma_max=0.30)
    # Γ₀ extrapolates the resolved 0.05 plateau, not the 0.30 blow-up.
    assert res.gamma_zero < 0.12


def test_gamma_zero_honours_chosen_e_window():
    e = np.linspace(-0.15, 0.0, 16)
    kp = np.full_like(e, 0.25); km = -kp            # peaks always resolved
    g = np.where((e >= -0.10) & (e <= -0.04), 0.05, 0.20)
    fr = {"e_fitted": e.tolist(), "kF_plus": [kp.tolist()], "kF_minus": [km.tolist()],
          "gamma_corrige": [g.tolist()], "gamma_brut": [g.tolist()]}
    win = fit_gamma_fermi_liquid(fr, pair_index=0, e_lo=-0.10, e_hi=-0.04)
    full = fit_gamma_fermi_liquid(fr, pair_index=0, e_window=0.30)
    assert win.gamma_zero < 0.10        # only the 0.05 window
    assert full.gamma_zero > win.gamma_zero  # full range dragged up by the 0.20 region


def _synthetic_fit_result(*, slope=2.0, intercept=-0.5, sigma_k=0.003, n=25, seed=0):
    """Generates a synthetic fit_result: E = intercept + slope·k → kF=-intercept/slope."""
    rng = np.random.default_rng(seed)
    e = np.linspace(-0.20, 0.05, n)
    k_clean = (e - intercept) / slope
    k = k_clean + sigma_k * rng.standard_normal(n)
    g = 0.05 + 1.2 * e ** 2
    return {
        "e_fitted": e,
        "kF_plus": [k.tolist()],
        "kF_minus": [(-k).tolist()],
        "sigma_kF_plus": [np.full(n, sigma_k).tolist()],
        "sigma_kF_minus": [np.full(n, sigma_k).tolist()],
        "gamma": [g.tolist()],
        "gamma_corrige": [g.tolist()],
        "sigma_gamma": [np.full(n, 0.005).tolist()],
        "n_pairs": 1,
    }


class TestWeightedLinearFit:
    def test_recovers_slope_intercept(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 50)
        y = 3.0 * x + 2.0 + 0.05 * rng.standard_normal(50)
        fit = weighted_linear_fit(x, y, sigma=np.full(50, 0.05))
        assert math.isfinite(fit.slope)
        # Tolerance ≥ 3·expected σ_slope (~0.025 for this dataset).
        assert abs(fit.slope - 3.0) < 0.10
        assert abs(fit.intercept - 2.0) < 0.10
        assert fit.n_points == 50

    def test_skips_nan(self):
        x = np.array([0.0, 1.0, 2.0, np.nan, 3.0])
        y = np.array([0.0, 1.0, 2.0, 99.0, 3.0])
        fit = weighted_linear_fit(x, y, sigma=np.ones(5))
        assert fit.n_points == 4
        assert abs(fit.slope - 1.0) < 1e-9

    def test_too_few_points_returns_empty(self):
        fit = weighted_linear_fit(np.array([1.0]), np.array([2.0]))
        assert fit.n_points == 1
        assert math.isnan(fit.slope)

    def test_unit_sigma_when_none(self):
        x = np.linspace(0, 1, 10)
        y = 2.0 * x + 1.0
        fit = weighted_linear_fit(x, y)
        assert abs(fit.slope - 2.0) < 1e-6


class TestExtractBranchResult:
    def test_recovers_known_kF_and_vF(self):
        fr = _synthetic_fit_result(slope=2.0, intercept=-0.5)
        br = extract_branch_result(fr, branch="kF_plus", pair_index=0,
                                   e_window=0.10)
        assert abs(br.kF_at_EF - 0.25) < 0.02
        # vF tolerance ≥ 3σ: with σ_k=0.003, slope ~ 2 ± 0.2.
        assert abs(br.vF_eV_pi_a - 2.0) < 0.5
        assert br.kF_at_EF_sigma > 0
        assert br.vF_sigma > 0
        assert br.n_points_used >= 5

    def test_uses_ensemble_std_when_sigma_keys_missing(self):
        fr = _synthetic_fit_result(slope=2.0, intercept=-0.5)
        fr["ensemble"] = {
            "kF_plus_std": fr.pop("sigma_kF_plus"),
            "kF_minus_std": fr.pop("sigma_kF_minus"),
            "gamma_std": fr.pop("sigma_gamma"),
        }

        br = extract_branch_result(fr, branch="kF_plus", pair_index=0,
                                   e_window=0.10)
        bundle = compute_results(fr, e_window_kF=0.10, e_window_gamma=0.30)

        assert br.kF_at_EF_sigma > 0
        assert bundle.gamma_fl[0].gamma_zero_sigma > 0

    def test_m_star_with_crystal_a(self):
        fr = _synthetic_fit_result(slope=2.0, intercept=-0.5)
        br = extract_branch_result(fr, branch="kF_plus", pair_index=0,
                                   e_window=0.10, crystal_a_angstrom=4.143)
        kF_A = 0.25 * math.pi / 4.143
        vF_A = 2.0 * 4.143 / math.pi
        m_expected = HBAR2_OVER_ME_eV_A2 * kF_A / vF_A
        # 15% tolerance for kF·vF propagation with noisy slope.
        assert abs(br.m_star_over_me - m_expected) < 0.15 * m_expected
        assert br.luttinger_density_pi_a2 > 0
        assert br.luttinger_units == "A^-2"
        expected_luttinger = 2.0 * kF_A ** 2 / (2.0 * math.pi)
        assert br.luttinger_density_pi_a2 == pytest.approx(expected_luttinger, rel=0.2)

    def test_too_few_points_returns_empty(self):
        fr = _synthetic_fit_result(n=5)
        # Window too narrow → few points.
        br = extract_branch_result(fr, branch="kF_plus", pair_index=0,
                                   e_window=0.001)
        assert br.n_points_used < 3
        assert math.isnan(br.kF_at_EF)

    def test_invalid_pair_index(self):
        fr = _synthetic_fit_result()
        br = extract_branch_result(fr, branch="kF_plus", pair_index=5)
        assert br.n_points_used == 0


class TestGammaFermiLiquid:
    def test_recovers_gamma_zero_and_coef(self):
        fr = _synthetic_fit_result()
        # gamma = 0.05 + 1.2 * E²
        fl = fit_gamma_fermi_liquid(fr, pair_index=0, e_window=0.30)
        assert abs(fl.gamma_zero - 0.05) < 0.01
        assert abs(fl.coef_E2 - 1.2) < 0.1
        assert fl.n_points_used >= 10

    def test_handles_missing_gamma(self):
        fr = {"e_fitted": np.linspace(-0.1, 0.1, 5).tolist()}
        fl = fit_gamma_fermi_liquid(fr, pair_index=0)
        assert fl.n_points_used == 0


class TestAsymmetry:
    def test_perfectly_symmetric(self):
        fr = _synthetic_fit_result(slope=2.0, intercept=-0.5)
        asym = compute_asymmetry(fr, pair_index=0, e_window=0.10)
        assert abs(asym.delta_kF) < 5 * asym.delta_kF_sigma
        assert asym.is_symmetric is True

    def test_invalid_returns_nan(self):
        fr = {"e_fitted": [], "kF_plus": [], "kF_minus": []}
        asym = compute_asymmetry(fr, pair_index=0)
        assert math.isnan(asym.delta_kF)


class TestComputeResults:
    def test_full_bundle(self):
        fr = _synthetic_fit_result()
        bundle = compute_results(fr, crystal_a_angstrom=4.143)
        assert len(bundle.branches) == 2  # 1 pair × 2 branches
        assert len(bundle.gamma_fl) == 1
        assert len(bundle.asymmetry) == 1
        assert bundle.crystal_a_angstrom == 4.143

    def test_empty_input(self):
        assert compute_results(None) == compute_results({})
        bundle = compute_results({})
        assert bundle.branches == ()

    def test_to_dict_round_trip(self):
        fr = _synthetic_fit_result()
        bundle = compute_results(fr, crystal_a_angstrom=4.143)
        d = bundle.to_dict()
        assert "branches" in d and len(d["branches"]) == 2
        assert d["crystal_a_angstrom"] == 4.143
