"""Tests for gap_extraction."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy missing")

from arpes.physics import gap_extraction as ge


class TestDynesModel:
    def test_dynes_peak_at_delta(self):
        D = 5.0
        w = np.linspace(-15, 15, 401)
        y = ge.dynes(w, D, 0.1)
        # Peak should be near ±Δ
        peak_idx = int(np.argmax(y))
        assert abs(abs(w[peak_idx]) - D) < 0.5

    def test_dynes_symmetric(self):
        w = np.linspace(-10, 10, 51)
        y = ge.dynes(w, 3.0, 0.2)
        # I(ω) = I(-ω)
        np.testing.assert_allclose(y, y[::-1], atol=1e-10)

    def test_dynes_zero_below_delta_no_broadening(self):
        # With Γ → 0, |ω| < Δ gives zero spectral weight
        w = np.linspace(-1, 1, 11)
        y = ge.dynes(w, 5.0, 1e-8)
        np.testing.assert_allclose(y, 0.0, atol=1e-6)


class TestMultiDynes:
    def test_two_gap_sums(self):
        w = np.linspace(-15, 15, 101)
        y1 = ge.dynes(w, 3.0, 0.2)
        y2 = ge.dynes(w, 8.0, 0.2)
        ymulti = ge.dynes_multi(w, [3.0, 8.0], [0.2, 0.2], [0.5, 0.5])
        np.testing.assert_allclose(ymulti, 0.5 * (y1 + y2), atol=1e-12)


@requires_scipy
class TestConvolution:
    def test_gaussian_norm(self):
        w = np.linspace(-50, 50, 201)
        k = ge.gaussian_kernel(w, fwhm_meV=5.0)
        assert abs(k.sum() - 1.0) < 1e-10

    def test_zero_resolution_passthrough(self):
        w = np.linspace(-10, 10, 21)
        s = np.ones_like(w)
        out = ge.convolve_resolution(s, w, 0.0)
        np.testing.assert_allclose(out, s)


class TestSymmetrize:
    def test_symmetric_input(self):
        E = np.linspace(-0.03, 0.03, 31)
        I = E ** 2  # already symmetric around 0
        w, sym = ge.symmetrize_edc(E, I, E_F=0.0, omega_max_meV=30.0)
        # I_sym should be ~ 2·I(ω) since I(-ω) = I(ω)
        idx_pos = w > 0
        np.testing.assert_allclose(sym[idx_pos], 2.0 * (w[idx_pos] * 1e-3) ** 2,
                                   atol=1e-6)


@requires_scipy
class TestFitDynesSingle:
    def test_recovers_delta(self):
        D_true, G_true = 4.0, 0.3
        w = np.linspace(-20, 20, 121)
        I = 2.0 * ge.dynes(w, D_true, G_true)
        res = ge.fit_dynes_single(w, I, Delta_guess_meV=3.0)
        assert abs(res.deltas_meV[0] - D_true) < 0.2
        assert res.n_gaps == 1
        assert res.chi2_red < 1e-3

    def test_warns_when_gamma_exceeds_delta(self):
        w = np.linspace(-30, 30, 121)
        # Strong broadening, weak gap → Γ > Δ
        I = 1.0 * ge.dynes(w, 2.0, 5.0)
        res = ge.fit_dynes_single(w, I, Delta_guess_meV=1.0,
                                   Gamma_guess_meV=4.0)
        assert any("Γ" in n or "gap filled" in n.lower() for n in res.notes)


@requires_scipy
class TestFitDynesTwoGap:
    def test_recovers_two_gaps(self):
        w = np.linspace(-20, 20, 161)
        I = 2.0 * ge.dynes_multi(w, [2.5, 7.0], [0.2, 0.3], [0.4, 0.6])
        res = ge.fit_dynes_two_gap(w, I)
        # Both gaps should be recovered (order may swap)
        recovered = sorted(res.deltas_meV)
        assert abs(recovered[0] - 2.5) < 0.5
        assert abs(recovered[1] - 7.0) < 0.5
        assert res.n_gaps == 2


@requires_scipy
class TestScanOverKF:
    def test_runs_over_multiple_edcs(self):
        E = np.linspace(-0.03, 0.03, 61)
        edcs = []
        for ang, D in [(0.0, 3.0), (15.0, 4.0), (30.0, 5.0)]:
            w = E * 1e3
            I = ge.dynes(w, D, 0.2) + 0.01
            edcs.append({"angle_deg": ang, "E": E, "I": I, "E_F": 0.0})
        out = ge.scan_gap_over_kf(edcs, resolution_meV=0.0, omega_max_meV=25.0)
        assert len(out["angle_deg"]) == 3
        assert np.all(out["delta_meV"] > 0)


@requires_scipy
class TestNormanSpectral:
    def test_zero_gap_is_single_lorentzian_peak(self):
        w = np.linspace(-20, 20, 401)
        A = ge.norman_spectral(w, Delta=0.0, Gamma1=1.0, Gamma0=1.0)
        assert np.argmax(A) == len(w) // 2  # peak at omega = 0
        assert (A > 0).all()

    def test_finite_gap_two_peaks_near_pm_delta(self):
        w = np.linspace(-20, 20, 801)
        A = ge.norman_spectral(w, Delta=5.0, Gamma1=0.3, Gamma0=0.3)
        # Peaks near ±sqrt(Δ²−Γ₀²) ≈ ±Δ; the minimum sits at ω = 0.
        pos = w[np.argmax(np.where(w > 1, A, 0))]
        neg = w[np.argmax(np.where(w < -1, A, 0))]
        assert abs(pos - 5.0) < 0.5
        assert abs(neg + 5.0) < 0.5
        assert A[len(w) // 2] < 0.5 * A.max()

    def test_positive_definite_everywhere(self):
        w = np.linspace(-50, 50, 1001)
        A = ge.norman_spectral(w, Delta=8.0, Gamma1=2.0, Gamma0=1.0)
        assert (A > 0).all()


@requires_scipy
class TestFitNormanSingle:
    def test_recovers_delta(self):
        D_true, G_true, g0 = 5.0, 0.8, 1.0
        w = np.linspace(-25, 25, 201)
        I = 3.0 * ge.norman_spectral(w, D_true, G_true, g0) + 0.05
        res = ge.fit_norman_single(w, I, Delta_guess_meV=4.0)
        assert abs(res.deltas_meV[0] - D_true) < 0.5
        assert res.n_gaps == 1

    def test_no_resolution_note(self):
        w = np.linspace(-25, 25, 201)
        I = ge.norman_spectral(w, 5.0, 0.5, 1.0)
        res = ge.fit_norman_single(w, I, resolution_meV=0.0)
        assert any("resolution" in n.lower() for n in res.notes)


@requires_scipy
class TestFitNormanTwoGap:
    def test_recovers_two_gaps(self):
        w = np.linspace(-25, 25, 301)
        I = 2.0 * ge.norman_multi(w, [3.0, 9.0], [0.3, 0.4], [0.5, 0.5],
                                  gamma0=1.0) + 0.02
        res = ge.fit_norman_two_gap(w, I, D1_guess_meV=2.5, D2_guess_meV=8.0)
        recovered = sorted(res.deltas_meV)
        assert abs(recovered[0] - 3.0) < 1.0
        assert abs(recovered[1] - 9.0) < 1.0
        assert res.n_gaps == 2
        assert sum(res.weights) == pytest.approx(1.0, abs=0.05)

    def test_degenerate_gaps_noted(self):
        w = np.linspace(-25, 25, 301)
        I = ge.norman_multi(w, [5.0, 5.0], [0.4, 0.4], [0.5, 0.5], gamma0=1.0)
        res = ge.fit_norman_two_gap(w, I, D1_guess_meV=5.0, D2_guess_meV=5.1)
        assert any("indistinguishable" in n for n in res.notes)


@requires_scipy
class TestScanOverKFTwoGapAndRobustness:
    def _edcs(self):
        E = np.linspace(-0.03, 0.03, 61)
        w = E * 1e3
        edcs = []
        for ang in (0.0, 20.0):
            I = ge.dynes_multi(w, [2.5, 7.0], [0.2, 0.3], [0.5, 0.5]) + 0.01
            edcs.append({"angle_deg": ang, "E": E, "I": I, "E_F": 0.0})
        return edcs

    def test_two_gap_scan(self):
        out = ge.scan_gap_over_kf(self._edcs(), omega_max_meV=25.0, n_gaps=2)
        assert len(out["angle_deg"]) == 2
        assert "delta2_meV" in out
        assert len(out["delta2_meV"]) == 2

    def test_failing_edc_skipped(self):
        edcs = self._edcs()
        # EDC with mismatched arrays must be skipped, not crash the scan.
        edcs.insert(1, {"angle_deg": 10.0, "E": np.linspace(-0.03, 0.03, 61),
                        "I": np.ones(5), "E_F": 0.0})
        out = ge.scan_gap_over_kf(edcs, omega_max_meV=25.0)
        assert len(out["angle_deg"]) == 2
        assert 10.0 not in out["angle_deg"]
