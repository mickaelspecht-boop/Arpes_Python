from __future__ import annotations

import unittest

import numpy as np

from arpes.analysis.aggregation import aggregate_session_entries
from arpes.core.session import FileEntry, FileMeta, FitParams, Session


def _fit_result(kf: float = 0.25) -> dict:
    e = np.linspace(-0.12, 0.08, 9)
    slope = 2.0
    intercept = -slope * kf
    k = (e - intercept) / slope
    gamma = 0.05 + 0.5 * e**2
    sigma = np.full(e.size, 0.003)
    return {
        "n_pairs": 1,
        "e_fitted": e.tolist(),
        "kF_plus": [k.tolist()],
        "kF_minus": [(-k).tolist()],
        "sigma_kF_plus": [sigma.tolist()],
        "sigma_kF_minus": [sigma.tolist()],
        "gamma_corrige": [gamma.tolist()],
        "gamma": [gamma.tolist()],
        "sigma_gamma": [np.full(e.size, 0.002).tolist()],
    }


class TestMultiFileAggregation(unittest.TestCase):
    def test_aggregate_sorts_by_temperature_and_skips_unfitted(self):
        session = Session()
        session.files["T30"] = FileEntry(
            fit_params=FitParams(n_pairs=1),
            fit_result=_fit_result(0.30),
            meta=FileMeta(temperature=30.0, hv=80.0, direction="Γ-M", crystal_a_angstrom=4.0),
        )
        session.files["T10"] = FileEntry(
            fit_params=FitParams(n_pairs=1),
            fit_result=_fit_result(0.20),
            meta=FileMeta(temperature=10.0, hv=80.0, direction="Gamma-M", crystal_a_angstrom=4.0),
        )
        session.files["raw"] = FileEntry(meta=FileMeta(temperature=20.0))

        series = aggregate_session_entries(session, x_axis="T (K)", direction_filter="G-M")

        self.assertEqual(series.skipped, 1)
        self.assertEqual([p.filename for p in series.points], ["T10", "T30"])
        self.assertLess(series.points[0].x_value, series.points[1].x_value)
        self.assertTrue(np.isfinite(series.points[0].kF))
        self.assertTrue(np.isfinite(series.points[0].gamma_zero))

    def test_aggregate_polarisation_uses_categories(self):
        session = Session()
        session.files["LH"] = FileEntry(
            fit_result=_fit_result(),
            meta=FileMeta(temperature=10.0, polarization="LH", crystal_a_angstrom=4.0),
        )
        session.files["LV"] = FileEntry(
            fit_result=_fit_result(),
            meta=FileMeta(temperature=10.0, polarization="LV", crystal_a_angstrom=4.0),
        )

        series = aggregate_session_entries(session, x_axis="polarisation")

        self.assertEqual([p.x_label for p in series.points], ["LH", "LV"])
        self.assertEqual([p.x_value for p in series.points], [0.0, 1.0])

    def test_aggregate_skips_physics_when_lattice_missing(self):
        session = Session()
        session.files["missing_a"] = FileEntry(
            fit_result=_fit_result(),
            meta=FileMeta(temperature=10.0, polarization="LH"),
        )

        series = aggregate_session_entries(session, x_axis="T (K)")

        self.assertEqual(series.points, ())
        self.assertEqual(series.skipped, 1)
        self.assertIn("Missing lattice parameter a", series.warning)


class TestRepresentativeBranch(unittest.TestCase):
    def test_picks_lowest_relative_mstar_sigma(self):
        from arpes.analysis.results import BranchResult, select_representative_branch
        bad = BranchResult(branch="kF_plus", kF_at_EF=0.13,
                           m_star_over_me=1.24, m_star_sigma=0.57)   # 46 %
        good = BranchResult(branch="kF_minus", kF_at_EF=-0.13,
                            m_star_over_me=1.24, m_star_sigma=0.03)  # 2 %
        chosen = select_representative_branch([bad, good])
        self.assertIs(chosen, good)

    def test_none_when_no_finite_kf(self):
        from arpes.analysis.results import BranchResult, select_representative_branch
        self.assertIsNone(select_representative_branch(
            [BranchResult(branch="kF_minus"), BranchResult(branch="kF_plus")]))

    def test_falls_back_to_finite_kf_without_mstar(self):
        from arpes.analysis.results import BranchResult, select_representative_branch
        only = BranchResult(branch="kF_plus", kF_at_EF=0.2)  # m* NaN
        self.assertIs(select_representative_branch([only]), only)

    def test_mstar_sigma_sane_for_offcentre_pocket(self):
        # Regression: an off-Γ pocket (center≠0) whose +branch sits near k=0
        # used to blow σ_m* up (old |α|/β² form divided by α≈0). σ_m* must now
        # stay comparable to σ_kF/σ_vF, not explode.
        from arpes.analysis.results import extract_branch_result
        center, kf, slope = -0.10, 0.13, 0.5
        e = np.linspace(-0.10, -0.005, 14)
        kp = (center + kf) + e / slope          # +branch crosses EF near k=+0.03
        km = 2.0 * center - kp                   # mirror about center
        sig = np.full(e.size, 0.003)
        fr = {
            "n_pairs": 1, "e_fitted": e.tolist(),
            "kF_plus": [kp.tolist()], "kF_minus": [km.tolist()],
            "sigma_kF_plus": [sig.tolist()], "sigma_kF_minus": [sig.tolist()],
        }
        br = extract_branch_result(fr, branch="kF_plus", pair_index=0,
                                   crystal_a_angstrom=4.0, center=center)
        self.assertTrue(np.isfinite(br.m_star_over_me))
        rel = br.m_star_sigma / br.m_star_over_me
        self.assertLess(rel, 0.10)              # would be ~0.5 with the old bug

    def test_pair_index_restricts_to_one_band(self):
        # Single-band fits: pair 0 keeps the point, a non-existent pair drops it.
        session = Session()
        session.files["A"] = FileEntry(
            fit_result=_fit_result(), fit_params=FitParams(n_pairs=1),
            meta=FileMeta(temperature=10.0, crystal_a_angstrom=4.0))
        self.assertEqual(len(aggregate_session_entries(session, pair_index=0).points), 1)
        self.assertEqual(len(aggregate_session_entries(session, pair_index=1).points), 0)


if __name__ == "__main__":
    unittest.main()
