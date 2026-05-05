"""Tests de lecture logbook pure (`arpes_logbook_io`)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from arpes_logbook_io import (
    best_excel_table,
    excel_header_candidates,
    inherit_logbook_context,
    read_logbook,
)


class TestLogbookContext(unittest.TestCase):
    def test_inherit_direction_pol_azi(self):
        records = [
            {"File": "BM1", "Direction": "G-M", "Pol": "LH", "Azi": 30.0},
            {"File": "BM2", "Direction": "", "Pol": "", "Azi": None},
        ]
        mapping = {"direction": "Direction", "polarization": "Pol", "azi": "Azi"}
        out = inherit_logbook_context(records, mapping)
        self.assertEqual(out[1]["Direction"], "G-M")
        self.assertEqual(out[1]["Pol"], "LH")
        self.assertEqual(out[1]["Azi"], 30.0)


class TestHeaderDetection(unittest.TestCase):
    def test_best_table_after_title_row(self):
        raw = pd.DataFrame([
            ["Measurement Plan", None, None],
            ["DONE", "Num", "Energy"],
            ["x", 9, 48.0],
        ])
        candidates = excel_header_candidates(raw)
        self.assertIn(1, candidates)
        guessed = best_excel_table(raw, candidates)
        self.assertIsNotNone(guessed)
        df, mapping = guessed
        self.assertEqual(mapping["file"], "Num")
        self.assertEqual(mapping["hv"], "Energy")
        self.assertEqual(len(df), 1)


class TestReadLogbook(unittest.TestCase):
    def test_simple_semicolon_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.csv"
            path.write_text("File;hv;Temp;Pol\nBM1;48.0;20;LH\n")
            result = read_logbook(path)
            self.assertEqual(result.mapping["file"], "File")
            self.assertEqual(result.mapping["hv"], "hv")
            self.assertEqual(len(result.records), 1)
            self.assertEqual(result.records[0]["File"], "BM1")

    def test_legacy_measurement_plan_csv_with_title(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "Measurements Plan.csv"
            path.write_text(
                "Measurement Plan (Green=Important)\n"
                "DONE;Num;Sample;Temp;Direction;Measurement Type;Pol;Energy;Polar\n"
                "x;9;Ba122;150;G-M;Band Map;LH;48;1.5\n"
            )
            result = read_logbook(path)
            self.assertEqual(result.mapping["file"], "Num")
            self.assertEqual(result.mapping["hv"], "Energy")
            self.assertEqual(result.mapping["temperature"], "Temp")
            self.assertEqual(str(result.records[0]["Num"]), "9")

    def test_mapping_selector_used_when_required_columns_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.csv"
            path.write_text("A;B\nBM1;48.0\n")
            called = []

            def select(columns, mapping):
                called.append((columns, mapping))
                return {"file": "A", "hv": "B"}

            result = read_logbook(path, mapping_selector=select)
            self.assertEqual(len(called), 1)
            self.assertEqual(result.mapping["file"], "A")
            self.assertEqual(result.mapping["hv"], "B")

    def test_raises_without_required_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.csv"
            path.write_text("A;B\nBM1;48.0\n")
            with self.assertRaises(ValueError):
                read_logbook(path)


if __name__ == "__main__":
    unittest.main()
