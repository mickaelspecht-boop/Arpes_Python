from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arpes.io.export import (
    BASE_RESULT_COLUMNS,
    export_provenance,
    physics_rows,
    result_columns,
    result_rows,
    write_provenance_sidecar,
    write_results_csv,
)
from arpes.core.session import FileEntry, FileMeta, FitParams, Session


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

    def test_export_provenance_tracks_inputs_and_sample_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data_path = folder / "fit"
            data_path.write_bytes(b"raw")
            session = self._session_with_fit()
            session.folder = folder
            session.current_sample = {
                "formula": "Bi2Se3",
                "a_angstrom": 4.14,
                "work_function_eV": 4.5,
            }

            provenance = export_provenance(session, content="results")

        self.assertEqual(provenance["app"], "ARPES Explorer")
        self.assertEqual(provenance["content"], "results")
        self.assertRegex(provenance["timestamp_utc"], r"Z$")
        self.assertRegex(provenance["input_hash"], r"^[0-9a-f]{64}$")
        self.assertIn("git_commit", provenance)
        self.assertEqual(provenance["sample_config"]["fit"]["formula"], "Bi2Se3")
        self.assertAlmostEqual(provenance["sample_config"]["fit"]["a_angstrom"], 4.14)
        self.assertIn("fit_params", provenance["files"]["fit"])
        self.assertEqual(provenance["files"]["fit"]["hv"], 48.0)
        self.assertTrue(any(item["file"] == "fit" and item["exists"] for item in provenance["inputs"]))

    def test_export_provenance_can_limit_files_for_filtered_figures(self):
        session = self._session_with_fit()
        provenance = export_provenance(session, content="figure", file_names={"fit"})

        self.assertEqual(list(provenance["sample_config"]), ["fit"])
        self.assertEqual([item["file"] for item in provenance["inputs"]], ["fit"])
        self.assertNotIn("nofit", provenance["files"])

    def test_write_results_csv_with_provenance_comment_header(self):
        rows = result_rows(self._session_with_fit())
        provenance = {
            "app": "ARPES Explorer",
            "session_version": 1,
            "git_commit": "abc123",
            "timestamp_utc": "2026-06-04T00:00:00Z",
            "input_hash": "0" * 64,
            "sample_config": {"fit": {"formula": "Bi2Se3"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            write_results_csv(str(path), rows, provenance=provenance)
            lines = path.read_text(encoding="utf-8").splitlines()
            data_lines = [line for line in lines if not line.startswith("#")]
            out = list(csv.DictReader(data_lines))

        self.assertEqual(lines[0], "# ARPES Explorer v1 commit abc123 exported 2026-06-04T00:00:00Z")
        self.assertTrue(lines[1].startswith("# provenance_json: "))
        self.assertIn('"git_commit":"abc123"', lines[1])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["file"], "fit")

    def test_write_provenance_sidecar(self):
        provenance = {
            "app": "ARPES Explorer",
            "git_commit": "abc123",
            "timestamp_utc": "2026-06-04T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            write_provenance_sidecar(str(path), provenance)
            sidecar = path.with_suffix(".meta.json")

            text = sidecar.read_text(encoding="utf-8")

        self.assertIn('"git_commit": "abc123"', text)

    def test_physics_rows_requires_lattice_a(self):
        with self.assertRaisesRegex(ValueError, "Paramètre de maille a manquant"):
            physics_rows(self._session_with_fit())

    def test_physics_rows_uses_sample_config_lattice(self):
        session = self._session_with_fit()
        entry = session.files["fit"]
        entry.meta.formula = "Bi2Se3"
        entry.meta.crystal_a_angstrom = 4.14
        rows = physics_rows(session)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["formula"], "Bi2Se3")
        self.assertAlmostEqual(rows[0]["crystal_a_angstrom"], 4.14)


if __name__ == "__main__":
    unittest.main()
