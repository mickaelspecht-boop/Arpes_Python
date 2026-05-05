from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arpes_export import BASE_RESULT_COLUMNS, result_columns, result_rows, write_results_csv
from arpes_session import FileEntry, FileMeta, FitParams, Session


class TestResultsExport(unittest.TestCase):
    def _session_with_fit(self) -> Session:
        session = Session()
        session.files["fit"] = FileEntry(
            fit_params=FitParams(n_pairs=1),
            fit_result={
                "e_fitted": np.array([-0.1, 0.0]),
                "kF_minus": [np.array([-0.2, -0.1])],
                "kF_plus": [np.array([0.2, 0.1])],
                "gamma_brut": [np.array([0.05, 0.06])],
                "gamma_min": [np.array([0.01, 0.01])],
                "gamma_corrige": [np.array([0.049, 0.059])],
                "resolution": {
                    "dE_meV": 25.0,
                    "dk_inv_a": 0.006,
                    "source": "estime",
                },
            },
            meta=FileMeta(hv=48.0, temperature=20.0, direction="Γ-M"),
        )
        session.files["nofit"] = FileEntry(meta=FileMeta(hv=100.0))
        return session

    def test_result_rows_exports_fit_and_ignores_unfitted(self):
        rows = result_rows(self._session_with_fit())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["file"], "fit")
        self.assertEqual(rows[0]["hv"], 48.0)
        self.assertEqual(rows[0]["direction"], "Γ-M")
        self.assertEqual(rows[0]["dE_meV"], 25.0)
        self.assertEqual(rows[0]["resolution_source"], "estime")
        self.assertEqual(rows[0]["kF_minus_1"], -0.2)
        self.assertEqual(rows[1]["gamma_corrige_1"], 0.059)

    def test_legacy_fit_without_resolution_gets_empty_resolution_fields(self):
        session = Session()
        session.files["legacy"] = FileEntry(
            fit_params=FitParams(n_pairs=1),
            fit_result={
                "e_fitted": [-0.1],
                "kF_minus": [[-0.2]],
                "kF_plus": [[0.2]],
            },
        )
        rows = result_rows(session)
        self.assertEqual(rows[0]["dE_meV"], "")
        self.assertEqual(rows[0]["dk_inv_a"], "")
        self.assertEqual(rows[0]["resolution_source"], "")
        self.assertEqual(rows[0]["gamma_brut_1"], "")

    def test_result_columns_are_stable(self):
        rows = result_rows(self._session_with_fit())
        cols = result_columns(rows)
        self.assertEqual(cols[:len(BASE_RESULT_COLUMNS)], BASE_RESULT_COLUMNS)
        self.assertIn("kF_minus_1", cols)
        self.assertIn("gamma_corrige_1", cols)

    def test_write_results_csv(self):
        rows = result_rows(self._session_with_fit())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            write_results_csv(str(path), rows)
            with path.open(newline="") as f:
                out = list(csv.DictReader(f))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["file"], "fit")
        self.assertIn("gamma_brut_1", out[0])


if __name__ == "__main__":
    unittest.main()
