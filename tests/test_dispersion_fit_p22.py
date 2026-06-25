"""Tests P2.2 — orthogonal regression (TLS) + linearity gate + σ covariance.

Covers the shared arpes.physics.dispersion_fit module and the two consumers:
analysis.results.extract_branch_result (table/export) and
physics.fit.compute_fermi_velocity_mstar (Im Σ path), plus the moving-block
bootstrap.
"""
from __future__ import annotations

import math

import numpy as np

from arpes.physics.dispersion_fit import (
    CURVATURE_MAX,
    MIN_DISP_POINTS,
    curvature_ratio,
    linear_dispersion_fit,
    local_k_of_e_fit,
)
from arpes.physics.fit import compute_fermi_velocity_mstar
from arpes.analysis.results import extract_branch_result
from arpes.analysis.bootstrap import (
    bootstrap_branch_result,
    _block_length,
    _moving_block_indices,
)


# ---------------------------------------------------------------- module TLS

class TestCurvatureRatio:
    def test_linear_band_near_zero(self):
        k = np.linspace(0.0, 0.5, 12)
        e = 2.0 * k - 0.5
        assert curvature_ratio(k, e) < 1e-6

    def test_curved_band_large(self):
        k = np.linspace(0.0, 0.5, 12)
        e = 2.0 * k - 0.5 + 8.0 * k ** 2
        assert curvature_ratio(k, e) > CURVATURE_MAX

    def test_constant_k_nan(self):
        assert math.isnan(curvature_ratio(np.full(8, 0.3), np.linspace(0, 1, 8)))


class TestLinearDispersionFit:
    def test_tls_recovers_slope_with_sk(self):
        rng = np.random.default_rng(1)
        e = np.linspace(-0.2, 0.05, 30)
        k = (e + 0.5) / 2.0 + 0.003 * rng.standard_normal(30)
        out = linear_dispersion_fit(k, e, np.full(30, 0.003))
        assert out["ok"] and out["method"] == "orthogonal_tls"
        assert abs(out["slope"] - 2.0) < 0.4
        assert np.all(np.isfinite(out["cov"]))
        assert out["cov"].shape == (2, 2)

    def test_ols_fallback_without_sk(self):
        e = np.linspace(-0.2, 0.05, 20)
        k = (e + 0.5) / 2.0
        out = linear_dispersion_fit(k, e, None)
        assert out["ok"] and out["method"] == "ols_regression"
        assert abs(out["slope"] - 2.0) < 1e-6

    def test_degenerate_constant_k_fails(self):
        out = linear_dispersion_fit(np.full(8, 0.3), np.linspace(0, 1, 8),
                                    np.full(8, 0.003))
        assert out["ok"] is False

    def test_tls_covariance_positive_diagonal(self):
        rng = np.random.default_rng(2)
        e = np.linspace(-0.15, 0.05, 20)
        k = (e + 0.5) / 2.0 + 0.004 * rng.standard_normal(20)
        out = linear_dispersion_fit(k, e, np.full(20, 0.004))
        assert out["cov"][0, 0] > 0 and out["cov"][1, 1] > 0


class TestLocalKOfEFit:
    def test_selects_linear_for_linear_dispersion(self):
        e = np.linspace(-0.10, 0.0, 8)
        k = 0.25 + 0.5 * e
        out = local_k_of_e_fit(e, k, np.full(e.size, 0.003))
        assert out["ok"]
        assert out["method"] == "local_linear_k_of_e"
        assert abs(out["k0"] - 0.25) < 1e-8
        assert abs(1.0 / out["dk_dE"] - 2.0) < 1e-8

    def test_selects_quadratic_and_evaluates_at_ef(self):
        e = np.linspace(-0.10, 0.0, 8)
        k = 0.25 + 0.5 * e + 8.0 * e ** 2
        out = local_k_of_e_fit(e, k, np.full(e.size, 0.001))
        assert out["ok"]
        assert out["method"] == "local_quadratic_k_of_e"
        assert abs(out["k0"] - 0.25) < 1e-8
        assert abs(1.0 / out["dk_dE"] - 2.0) < 1e-8


# ---------------------------------------------------- extract_branch_result

def _fr(slope=2.0, intercept=-0.5, *, n=25, curv=0.0, sigma_k=0.003, seed=0):
    rng = np.random.default_rng(seed)
    e = np.linspace(-0.20, 0.05, n)
    k_clean = (e - intercept) / slope + curv * e ** 2
    k = k_clean + sigma_k * rng.standard_normal(n)
    return {
        "e_fitted": e,
        "kF_plus": [k.tolist()],
        "sigma_kF_plus": [np.full(n, sigma_k).tolist()],
        "n_pairs": 1,
    }


class TestExtractBranchGate:
    def test_linear_accepted(self):
        br = extract_branch_result(_fr(), branch="kF_plus", pair_index=0, e_window=0.10)
        assert br.linear_ok is True and br.refused_reason == ""
        assert abs(br.kF_at_EF - 0.25) < 0.03
        assert br.kF_at_EF_sigma > 0

    def test_nonlinear_uses_local_quadratic(self):
        br = extract_branch_result(_fr(curv=40.0), branch="kF_plus",
                                   pair_index=0, e_window=0.10)
        assert br.linear_ok is True
        assert br.dispersion_model == "local_quadratic_k_of_e"
        assert math.isfinite(br.kF_at_EF)

    def test_too_few_points_refused(self):
        br = extract_branch_result(_fr(n=25), branch="kF_plus",
                                   pair_index=0, e_window=0.005)
        assert br.linear_ok is False
        assert br.n_points_used < MIN_DISP_POINTS
        assert "too few points" in br.refused_reason


# --------------------------------------------- compute_fermi_velocity_mstar

class TestFermiVelocityMstar:
    def test_linear_returns_values_and_sigmas(self):
        fr = _fr(slope=2.0, intercept=-0.5)
        out = compute_fermi_velocity_mstar(fr, crystal_a=4.0,
                                           branch="kF_plus", window_eV=0.10)
        assert out["linear_ok"] is True
        assert math.isfinite(out["vF_eV_A"]) and out["vF_eV_A"] > 0
        assert math.isfinite(out["kF_inv_A"]) and out["kF_inv_A"] > 0
        assert out["vF_sigma_eV_A"] >= 0 and out["kF_inv_A_sigma"] >= 0
        assert out["sigma_type"] in ("orthogonal_tls", "ols_regression")

    def test_curved_accepted_by_default(self):
        # Curvature is a physical observable: by default a curved band is kept
        # (flagged), with vF from the local slope — not rejected.
        out = compute_fermi_velocity_mstar(_fr(curv=40.0), crystal_a=4.0,
                                           branch="kF_plus", window_eV=0.10)
        assert out["linear_ok"] is True
        assert out["curved"] is True
        assert math.isfinite(out["vF_eV_A"]) and out["vF_eV_A"] > 0

    def test_curved_refused_when_enforce_linear(self):
        out = compute_fermi_velocity_mstar(_fr(curv=40.0), crystal_a=4.0,
                                           branch="kF_plus", window_eV=0.10,
                                           enforce_linear=True)
        assert out["linear_ok"] is False
        assert "nonlinear" in out["refused_reason"]
        assert math.isnan(out["vF_eV_A"]) and math.isnan(out["mstar_over_me"])

    def test_too_few_points_refused(self):
        out = compute_fermi_velocity_mstar(_fr(), crystal_a=4.0,
                                           branch="kF_plus", window_eV=0.003)
        assert out["linear_ok"] is False
        assert out["n_points"] < MIN_DISP_POINTS


# ----------------------------------------------------- moving-block bootstrap

class TestMovingBlock:
    def test_block_length_bounds(self):
        assert _block_length(4) == 3
        assert _block_length(30) == 5
        assert 3 <= _block_length(12) <= 5

    def test_indices_contiguous_blocks(self):
        rng = np.random.default_rng(0)
        idx = _moving_block_indices(rng, n=12, block_len=4)
        assert idx.size == 12
        # Chaque bloc de 4 est contigu (diff == 1 dans le bloc).
        for b in range(0, 12, 4):
            blk = idx[b:b + 4]
            assert np.all(np.diff(blk) == 1)

    def test_indices_in_range(self):
        rng = np.random.default_rng(3)
        idx = _moving_block_indices(rng, n=9, block_len=3)
        assert idx.min() >= 0 and idx.max() < 9

    def test_bootstrap_recovers_kF(self):
        br = bootstrap_branch_result(_fr(slope=2.0, intercept=-0.5),
                                     branch="kF_plus", pair_index=0,
                                     e_window=0.10, n_iter=200, seed=0)
        assert abs(br.kF_at_EF - 0.25) < 0.05
        assert br.kF_at_EF_sigma > 0
        assert br.sigma_method == "bootstrap"
