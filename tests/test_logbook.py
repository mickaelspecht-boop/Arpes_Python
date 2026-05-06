from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arpes.io.logbook import (
    LogbookManager,
    _cell_float,
    _extract_measurement_numbers,
    _format_direction_label,
    _infer_logbook_mapping,
    _normalize_mp_id,
    _record_matches_path,
)
from arpes.core.session import FileEntry


class TestLogbookHelpers(unittest.TestCase):
    def test_solaris_aliases_do_not_confuse_temperature_with_angles(self):
        columns = ["Filename", "Photon Energy", "Sample Temperature", "Light Polarization", "High symmetry path"]
        mapping = _infer_logbook_mapping(columns)
        self.assertEqual(mapping["file"], "Filename")
        self.assertEqual(mapping["hv"], "Photon Energy")
        self.assertEqual(mapping["temperature"], "Sample Temperature")
        self.assertEqual(mapping["polarization"], "Light Polarization")
        self.assertEqual(mapping["direction"], "High symmetry path")
        self.assertEqual(mapping["polar"], "")
        self.assertEqual(mapping["tilt"], "")

    def test_legacy_bessy_measurement_plan_mapping(self):
        columns = ["Num", "Energy", "Temp", "Pol", "Polar", "Direction"]
        mapping = _infer_logbook_mapping(columns)
        self.assertEqual(mapping["file"], "Num")
        self.assertEqual(mapping["hv"], "Energy")
        self.assertEqual(mapping["temperature"], "Temp")
        self.assertEqual(mapping["polarization"], "Pol")
        self.assertEqual(mapping["polar"], "Polar")
        self.assertEqual(mapping["direction"], "Direction")

    def test_direction_label_formats_gamma_variants(self):
        self.assertEqual(_format_direction_label("G"), "Γ")
        self.assertEqual(_format_direction_label("gamma"), "Γ")
        self.assertEqual(_format_direction_label("Gamma-X"), "Γ-X")
        self.assertEqual(_format_direction_label("G M"), "Γ M")
        self.assertEqual(_format_direction_label("GM"), "ΓM")

    def test_measurement_numbers_match_fixed_cut_ibw(self):
        self.assertEqual(_extract_measurement_numbers("BaNi2As2_0015.pxt,.ibw,"), {15})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "BaNi2As2_0015fixed cut.ibw"
            path.write_text("dummy")
            self.assertTrue(_record_matches_path("BaNi2As2_0015.pxt,.ibw,", path, root))

    def test_numeric_logbook_num_matches_bessy_w_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "Ba1220009w_Band Map B122_009.ibw"
            path.write_text("dummy")
            self.assertTrue(_record_matches_path("9", path, root))

    def test_cell_float_uses_first_number_only(self):
        self.assertEqual(_cell_float("hv = 48.5 eV"), 48.5)
        self.assertEqual(_cell_float("48,5"), 48.5)
        self.assertIsNone(_cell_float("LH"))

    def test_logbook_manager_extracts_values_without_ui(self):
        records = [{
            "file": "BaNi2As2_0015.pxt,.ibw,",
            "hv": "48.0",
            "T": "22",
            "Pol": "LH",
            "path": "Gamma-X",
            "azi": "9",
            "P": "1.5",
            "Tilt": "-0.5",
        }]
        mapping = {
            "file": "file",
            "hv": "hv",
            "temperature": "T",
            "polarization": "Pol",
            "direction": "path",
            "azi": "azi",
            "polar": "P",
            "tilt": "Tilt",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "BaNi2As2_0015fixed cut.ibw"
            path.write_text("dummy")
            manager = LogbookManager(records, mapping, root)
            values = manager.values_for_path(path)

        self.assertEqual(values.hv, 48.0)
        self.assertEqual(values.temperature, 22.0)
        self.assertEqual(values.polarization, "LH")
        self.assertEqual(values.direction, "Γ-X")
        self.assertEqual(values.azi, 9.0)
        self.assertEqual(values.polar, 1.5)
        self.assertEqual(values.tilt, -0.5)
        self.assertEqual(values.sources["hv"], "logbook")

    def test_logbook_manager_applies_to_file_entry(self):
        records = [{"Num": "9", "Energy": "100", "Temp": "12", "Pol": "LV", "Direction": "G"}]
        mapping = _infer_logbook_mapping(["Num", "Energy", "Temp", "Pol", "Direction"])
        entry = FileEntry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "Ba1220009w_Band Map B122_009.ibw"
            path.write_text("dummy")
            values = LogbookManager(records, mapping, root).apply_to_entry(entry, path)
        self.assertTrue(values.has_any())
        self.assertEqual(entry.meta.hv, 100.0)
        self.assertEqual(entry.meta.temperature, 12.0)
        self.assertEqual(entry.meta.polarization, "LV")
        self.assertEqual(entry.meta.direction, "Γ")


class TestMpIdLogbook(unittest.TestCase):
    def test_normalize_mp_id_variants(self):
        self.assertEqual(_normalize_mp_id("mp-149"), "mp-149")
        self.assertEqual(_normalize_mp_id("MP-149"), "mp-149")
        self.assertEqual(_normalize_mp_id("  mp-568280  "), "mp-568280")
        self.assertEqual(_normalize_mp_id("149"), "mp-149")
        self.assertEqual(_normalize_mp_id(""), "")
        self.assertEqual(_normalize_mp_id("not_an_id"), "")
        self.assertEqual(_normalize_mp_id("mp-abc"), "")

    def test_logbook_mp_id_and_formula_columns_picked(self):
        columns = ["Filename", "Photon Energy", "Formula", "MP-ID"]
        mapping = _infer_logbook_mapping(columns)
        self.assertEqual(mapping["formula"], "Formula")
        self.assertEqual(mapping["mp_id"], "MP-ID")

    def test_logbook_applies_mp_id_to_entry(self):
        records = [{
            "Num": "9", "Energy": "100", "Formula": "BaNi2As2", "MP-ID": "mp-568280",
        }]
        mapping = _infer_logbook_mapping(["Num", "Energy", "Formula", "MP-ID"])
        entry = FileEntry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "Ba1220009w_Band Map B122_009.ibw"
            path.write_text("dummy")
            values = LogbookManager(records, mapping, root).apply_to_entry(entry, path)
        self.assertEqual(entry.meta.formula, "BaNi2As2")
        self.assertEqual(entry.meta.mp_id, "mp-568280")
        self.assertEqual(values.sources["mp_id"], "logbook")


if __name__ == "__main__":
    unittest.main()
