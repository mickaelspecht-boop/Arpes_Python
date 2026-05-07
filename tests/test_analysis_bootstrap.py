"""Tests pour arpes.analysis.bootstrap : robustesse outliers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from arpes.analysis.bootstrap import bootstrap_branch_result


def _fr_with_outlier(*, n=25, outlier_idx=3, outlier_val=0.6, seed=0):
    rng = np.random.default_rng(seed)
    e = np.linspace(-0.10, 0.05, n)
    k_clean = 0.25 + 0.10 * e
    k = k_clean + 0.003 * rng.standard_normal(n)
    if outlier_idx is not None:
        k[outlier_idx] = outlier_val
    return {
        "e_fitted": e,
        "kF_plus": [k.tolist()],
        "kF_minus": [(-k).tolist()],
        "sigma_kF_plus": [np.full(n, 0.003).tolist()],
        "sigma_kF_minus": [np.full(n, 0.003).tolist()],
        "n_pairs": 1,
    }


class TestBootstrapBranch:
    def test_basic_recovery(self):
        fr = _fr_with_outlier(outlier_idx=None)
        res = bootstrap_branch_result(
            fr, branch="kF_plus", pair_index=0,
            e_window=0.10, n_iter=500, seed=42,
        )
        assert math.isfinite(res.kF_at_EF)
        assert math.isfinite(res.vF_eV_pi_a)
        assert abs(res.kF_at_EF - 0.25) < 0.01
        assert res.kF_at_EF_sigma > 0
        assert res.n_iter == 500
        assert res.sigma_method == "bootstrap"

    def test_outlier_inflates_sigma(self):
        fr_clean = _fr_with_outlier(outlier_idx=None, seed=1)
        fr_outlier = _fr_with_outlier(outlier_idx=3, outlier_val=0.6, seed=1)
        s_clean = bootstrap_branch_result(
            fr_clean, branch="kF_plus", pair_index=0,
            e_window=0.10, n_iter=500, seed=42,
        )
        s_outlier = bootstrap_branch_result(
            fr_outlier, branch="kF_plus", pair_index=0,
            e_window=0.10, n_iter=500, seed=42,
        )
        assert s_outlier.kF_at_EF_sigma > 2 * s_clean.kF_at_EF_sigma

    def test_too_few_points(self):
        fr = _fr_with_outlier(n=4, outlier_idx=None)
        res = bootstrap_branch_result(
            fr, branch="kF_plus", pair_index=0,
            e_window=0.001, n_iter=100,  # window cuts most points
        )
        assert res.n_iter == 0
        assert math.isnan(res.kF_at_EF)

    def test_m_star_propagation(self):
        fr = _fr_with_outlier(outlier_idx=None, seed=7)
        res = bootstrap_branch_result(
            fr, branch="kF_plus", pair_index=0,
            e_window=0.10, n_iter=300, crystal_a_angstrom=4.143, seed=42,
        )
        assert math.isfinite(res.m_star_over_me)
        assert res.m_star_sigma > 0

    def test_seed_reproducibility(self):
        fr = _fr_with_outlier(outlier_idx=None)
        a = bootstrap_branch_result(fr, branch="kF_plus", pair_index=0,
                                    n_iter=200, seed=123)
        b = bootstrap_branch_result(fr, branch="kF_plus", pair_index=0,
                                    n_iter=200, seed=123)
        assert a.kF_at_EF == b.kF_at_EF
        assert a.kF_at_EF_sigma == b.kF_at_EF_sigma
