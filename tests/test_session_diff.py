from __future__ import annotations

import unittest

import numpy as np

from arpes.analysis.session_diff import compare_session_payloads
from arpes.core.session import FileEntry, Session


def _fit_result_for_kf(kf: float, *, slope: float = 2.0) -> dict:
    e = np.linspace(-0.08, 0.08, 7)
    intercept = -slope * kf
    k_plus = (e - intercept) / slope
    k_minus = -k_plus
    sigma = np.full(e.size, 0.003)
    gamma = np.full(e.size, 0.04)
    return {
        "n_pairs": 1,
        "e_fitted": e.tolist(),
        "kF_plus": [k_plus.tolist()],
        "kF_minus": [k_minus.tolist()],
        "sigma_kF_plus": [sigma.tolist()],
        "sigma_kF_minus": [sigma.tolist()],
        "gamma_corrige": [gamma.tolist()],
        "gamma": [gamma.tolist()],
        "sigma_gamma": [np.full(e.size, 0.002).tolist()],
    }


def _payload(name: str, *, kf: float, include_extra: bool = False) -> dict:
    session = Session()
    entry = FileEntry(fit_result=_fit_result_for_kf(kf))
    entry.meta.crystal_a_angstrom = 4.0
    session.files[name] = entry
    if include_extra:
        session.files["only_here"] = FileEntry(fit_result=_fit_result_for_kf(kf + 0.1))
    return session.to_payload()


class TestSessionDiff(unittest.TestCase):
    def test_compare_payloads_reports_branch_deltas(self):
        rows = compare_session_payloads(
            _payload("BM1", kf=0.25),
            _payload("BM1", kf=0.27),
        )

        plus = [r for r in rows if r.filename == "BM1" and r.branch == "kF_plus"][0]
        self.assertEqual(plus.status, "OK")
        self.assertAlmostEqual(plus.delta_kF, 0.02, places=6)
        self.assertAlmostEqual(plus.delta_vF, 0.0, places=6)
        self.assertTrue(np.isfinite(plus.a_kF_sigma))
        self.assertTrue(np.isfinite(plus.b_vF_sigma))

    def test_compare_payloads_reports_missing_files(self):
        rows = compare_session_payloads(
            _payload("BM1", kf=0.25, include_extra=True),
            _payload("BM1", kf=0.25),
        )
        missing = [r for r in rows if r.filename == "only_here"][0]
        self.assertEqual(missing.status, "absent B")


if __name__ == "__main__":
    unittest.main()
