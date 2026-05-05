from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arpes.core.session import FileEntry, FileMeta, FitParams, Session


class TestSessionManager(unittest.TestCase):
    def test_session_round_trip_preserves_existing_json_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root, work_func=4.5)
            entry = FileEntry(
                ef_offset=0.012,
                edcnorm=False,
                view_mode="Raw",
                fit_params=FitParams(n_pairs=2, dE_meV=25.0, dk_inv_a=0.006),
                fit_result={
                    "e_fitted": np.array([-0.1, 0.0]),
                    "gamma_brut": [np.array([0.05, 0.06])],
                    "resolution": {
                        "dE_meV": 25.0,
                        "dk_inv_a": 0.006,
                        "source": "estime PE=50 DA30",
                    },
                },
                meta=FileMeta(hv=48.0, temperature=20.0, direction="G-M"),
            )
            session.files["BM1"] = entry
            session.gamma_reference = {"kx": np.float64(0.01)}
            session.save()

            raw = json.loads((root / ".arpes_session.json").read_text())
            self.assertIn("files", raw)
            self.assertIn("BM1", raw["files"])
            self.assertEqual(raw["files"]["BM1"]["fit_params"]["dE_meV"], 25.0)
            self.assertEqual(raw["files"]["BM1"]["fit_result"]["e_fitted"], [-0.1, 0.0])

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            out = restored.files["BM1"]
            self.assertEqual(out.ef_offset, 0.012)
            self.assertFalse(out.edcnorm)
            self.assertEqual(out.fit_params.n_pairs, 2)
            self.assertEqual(out.meta.direction, "G-M")
            self.assertEqual(out.fit_result["resolution"]["dk_inv_a"], 0.006)

    def test_session_load_ignores_unknown_legacy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({
                "version": 1,
                "folder": str(root),
                "work_func": 4.031,
                "files": {
                    "old": {
                        "fit_params": {
                            "n_pairs": 1,
                            "unknown_future_field": 123,
                        },
                        "meta": {
                            "hv": 100.0,
                            "unknown_meta_field": "kept out",
                        },
                        "fit_result": {
                            "gamma": [[0.04]],
                        },
                    }
                },
            }))

            session = Session(root)
            session.load(path)
            entry = session.files["old"]
            self.assertEqual(entry.fit_params.n_pairs, 1)
            self.assertEqual(entry.meta.hv, 100.0)
            self.assertEqual(entry.fit_result["gamma"], [[0.04]])

    def test_key_for_path_prefers_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "folder"
            sub.mkdir()
            file_path = sub / "BM1"
            file_path.write_text("dummy")

            session = Session(root)
            self.assertEqual(session.key_for_path(file_path), "folder/BM1")


if __name__ == "__main__":
    unittest.main()
