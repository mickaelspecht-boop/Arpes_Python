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
            meta=FileMeta(temperature=10.0, polarization="LH"),
        )
        session.files["LV"] = FileEntry(
            fit_result=_fit_result(),
            meta=FileMeta(temperature=10.0, polarization="LV"),
        )

        series = aggregate_session_entries(session, x_axis="polarisation")

        self.assertEqual([p.x_label for p in series.points], ["LH", "LV"])
        self.assertEqual([p.x_value for p in series.points], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
