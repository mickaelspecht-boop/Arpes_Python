from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arpes.core.session import (
    FileEntry,
    FileMeta,
    FitParams,
    Session,
    normalize_tags,
    session_tags,
)


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
                meta=FileMeta(hv=48.0, temperature=20.0, direction="G-M", tags=["publi", "T-dep"]),
            )
            session.files["BM1"] = entry
            session.kz_logbook_path = str(root / "kz.xlsx")
            session.kz_logbook_sheet = "KZ"
            session.kz_logbook_mapping = {"file": "Scan", "hv": "Energy"}
            session.kz_logbook_records = [{"Scan": "BM1", "Energy": np.float64(48.0)}]
            session.gamma_reference = {"kx": np.float64(0.01)}
            session.save()

            raw = json.loads((root / ".arpes_session.json").read_text())
            self.assertIn("files", raw)
            self.assertIn("BM1", raw["files"])
            self.assertEqual(raw["files"]["BM1"]["fit_params"]["dE_meV"], 25.0)
            self.assertEqual(raw["files"]["BM1"]["fit_result"]["e_fitted"], [-0.1, 0.0])
            self.assertEqual(raw["kz_logbook_sheet"], "KZ")
            self.assertEqual(raw["kz_logbook_records"][0]["Energy"], 48.0)

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            out = restored.files["BM1"]
            self.assertEqual(out.ef_offset, 0.012)
            self.assertFalse(out.edcnorm)
            self.assertEqual(out.fit_params.n_pairs, 2)
            self.assertEqual(out.meta.direction, "G-M")
            self.assertEqual(out.meta.tags, ["publi", "T-dep"])
            self.assertEqual(out.theory_overlay, {})
            self.assertEqual(out.fit_result["resolution"]["dk_inv_a"], 0.006)
            self.assertEqual(restored.kz_logbook_sheet, "KZ")
            self.assertEqual(restored.kz_logbook_mapping["hv"], "Energy")

    def test_session_round_trip_preserves_theory_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            entry = session.get_or_create("BM1")
            entry.theory_overlay = {"enabled": True, "data": {"material_id": "mp-149"}}
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertEqual(
                restored.files["BM1"].theory_overlay["data"]["material_id"],
                "mp-149",
            )

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

    def test_normalize_and_collect_tags(self):
        self.assertEqual(normalize_tags(" publi, outliers, Publi,  "), ["publi", "outliers"])
        session = Session()
        session.files["a"] = FileEntry(meta=FileMeta(tags=["T-dep", "publi"]))
        session.files["b"] = FileEntry(meta=FileMeta(tags=["outliers", "publi"]))
        self.assertEqual(session_tags(session), ["outliers", "publi", "T-dep"])


if __name__ == "__main__":
    unittest.main()
