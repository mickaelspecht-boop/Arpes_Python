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
    FitZone,
    Session,
    SessionVersionError,
    normalize_fit_zones,
    normalize_tags,
    session_tags,
)
from arpes.core.sample import SampleConfig, sample_for_entry, work_function_for_entry


class TestFitZoneP34(unittest.TestCase):
    def test_from_dict_fills_defaults(self):
        z = FitZone.from_dict({"id": "a", "label": "Z1"})
        self.assertEqual((z.color_idx, z.active, z.fit_params, z.fit_result),
                         (0, True, {}, None))

    def test_from_dict_warns_on_unknown_key(self):
        with self.assertWarns(UserWarning):
            FitZone.from_dict({"id": "a", "label": "Z1", "bogus": 9})

    def test_normalize_roundtrips_canonical_zone(self):
        zone = {"id": "a", "label": "Z1", "color_idx": 2, "active": False,
                "fit_params": {"k_min": -0.5}, "fit_result": {"e": 1}}
        self.assertEqual(normalize_fit_zones([zone]), [zone])

    def test_normalize_drops_unknown_key_loudly(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            out = normalize_fit_zones([{"id": "a", "label": "Z1", "stale": 1}])
        self.assertNotIn("stale", out[0])
        self.assertEqual(set(out[0]), {"id", "label", "color_idx", "active",
                                       "fit_params", "fit_result"})


class TestSessionManager(unittest.TestCase):
    def test_session_default_work_function_is_unknown(self):
        session = Session()
        self.assertEqual(session.work_func, 0.0)

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

    def test_session_round_trip_preserves_fs_pockets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            entry = session.get_or_create("FS1")
            entry.fs_pockets = [{
                "centroid_kx": 0.1,
                "centroid_ky": -0.2,
                "area_pct_bz": 3.5,
                "topology": "electron",
            }]
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertEqual(restored.files["FS1"].fs_pockets[0]["topology"], "electron")
            self.assertAlmostEqual(restored.files["FS1"].fs_pockets[0]["area_pct_bz"], 3.5)

    def test_session_round_trip_preserves_fs_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            entry = session.get_or_create("FS1")
            entry.fs_rotation_deg = 17.5
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertAlmostEqual(restored.files["FS1"].fs_rotation_deg, 17.5)

    def test_session_round_trip_preserves_bz_mp_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            entry = session.get_or_create("FS1")
            entry.fs_bz_crystal_force_override = True
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertTrue(restored.files["FS1"].fs_bz_crystal_force_override)

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
            self.assertEqual(session.work_func, 4.031)

    def test_legacy_session_without_work_func_loads_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({
                "version": 1,
                "folder": str(root),
                "files": {"old": {"meta": {"hv": 100.0}}},
            }))

            session = Session(root)
            session.load(path)
            self.assertEqual(session.work_func, 0.0)

    def test_sample_config_round_trip_and_legacy_meta_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            session.current_sample = SampleConfig(
                formula="Sr2RuO4",
                a_angstrom=3.87,
                c_angstrom=12.74,
                work_function_eV=4.5,
                space_group="I4/mmm",
                mp_id="mp-123",
                lattice_source="manual",
            ).to_dict()
            entry = session.get_or_create("FS1")
            entry.meta.crystal_a_angstrom = 3.96
            entry.meta.formula = "BaNi2As2"
            entry.meta.lattice_source = "logbook"
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            sample = sample_for_entry(restored, restored.files["FS1"])
            self.assertEqual(sample.formula, "BaNi2As2")
            self.assertAlmostEqual(sample.a_angstrom, 3.96)
            self.assertAlmostEqual(sample.c_angstrom, 12.74)
            self.assertAlmostEqual(sample.work_function_eV, 4.5)
            self.assertEqual(sample.lattice_source, "logbook")

    def test_legacy_session_without_sample_config_stays_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({
                "version": 1,
                "work_func": 4.031,
                "files": {
                    "old": {
                        "meta": {
                            "formula": "Bi2Se3",
                            "crystal_a_angstrom": 4.14,
                        },
                    }
                },
            }))

            session = Session(root)
            session.load(path)
            sample = sample_for_entry(session, session.files["old"])
            self.assertEqual(sample.formula, "Bi2Se3")
            self.assertAlmostEqual(sample.a_angstrom, 4.14)
            self.assertFalse(sample.has_work_function)

    def test_work_function_prefers_sample_before_fallback(self):
        session = Session()
        entry = FileEntry()
        self.assertAlmostEqual(
            work_function_for_entry(session, entry, fallback=4.031),
            4.031,
        )
        session.current_sample = {"work_function_eV": 4.8}
        self.assertAlmostEqual(
            work_function_for_entry(session, entry, fallback=4.031),
            4.8,
        )
        entry.meta.work_function_eV = 4.6
        self.assertAlmostEqual(
            work_function_for_entry(session, entry, fallback=4.031),
            4.6,
        )
        entry.meta.work_function_eV = 0.0
        entry.meta.sample_config = {"work_function_eV": 4.7}
        self.assertAlmostEqual(
            work_function_for_entry(session, entry, fallback=4.031),
            4.7,
        )

    def test_save_writes_current_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            session.get_or_create("BM1")
            session.save()
            raw = json.loads((root / ".arpes_session.json").read_text())
            self.assertEqual(raw["version"], Session.VERSION)

    def test_load_legacy_v1_migrates_and_records_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({
                "version": 1,
                "files": {"old": {"meta": {"hv": 50.0}}},
            }))
            session = Session(root)
            session.load(path)
            self.assertEqual(session.loaded_version, 1)
            entry = session.files["old"]
            self.assertEqual(entry.band_analysis, {})
            self.assertEqual(entry.fit_zones, [])
            self.assertIsNone(entry.active_zone_id)

    def test_load_unversioned_payload_treated_as_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({"files": {"old": {"meta": {"hv": 1.0}}}}))
            session = Session(root)
            session.load(path)
            self.assertEqual(session.loaded_version, 1)

    def test_load_future_version_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({
                "version": Session.VERSION + 1,
                "files": {},
            }))
            session = Session(root)
            with self.assertRaises(SessionVersionError):
                session.load(path)

    def test_save_keeps_backup_of_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            session.get_or_create("BM1")
            session.save()
            session.get_or_create("BM2")
            session.save()
            bak = root / ".arpes_session.json.bak"
            self.assertTrue(bak.exists())
            prev = json.loads(bak.read_text())
            self.assertIn("BM1", prev["files"])
            self.assertNotIn("BM2", prev["files"])
            self.assertFalse((root / ".arpes_session.json.tmp").exists())

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


class TestBzLabelOverridesCompat(unittest.TestCase):
    def test_missing_keys_default_empty(self):
        # Sessions saved before the BZ label convention feature must load
        # with empty overrides (default square labels, original behaviour).
        entry = FileEntry(meta=FileMeta(scan_kind="FS"))
        d = entry.to_dict() if hasattr(entry, "to_dict") else None
        self.assertEqual(entry.fs_bz_label_overrides, {})
        self.assertEqual(entry.fs_bz_label_preset, "")

    def test_roundtrip_preserves_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            entry = session.get_or_create("FS1")
            entry.fs_bz_label_overrides = {"M": "Σ"}
            entry.fs_bz_label_preset = "i4mmm_sigma_diagonal"
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            out = restored.files["FS1"]
            self.assertEqual(out.fs_bz_label_overrides, {"M": "Σ"})
            self.assertEqual(out.fs_bz_label_preset, "i4mmm_sigma_diagonal")
