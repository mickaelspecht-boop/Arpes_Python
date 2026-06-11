"""Headless test: attach_scoped_silent must open ZERO dialog."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from arpes.core.session import Session
from arpes.io.logbook_io import get_xlsx_sheet_names
from arpes.ui.controllers.logbook_controller import LogbookIngestController


class _Browser:
    def refresh(self):
        pass


class _Parent:
    def __init__(self, session):
        self._session = session
        self._browser = _Browser()
        self._params = None
        self._current_path = None
        self.statuses = []

    def _status(self, msg):
        self.statuses.append(msg)


def _xlsx(folder: Path, sheets=("CA041", "CA046")) -> Path:
    p = folder / "logbook.xlsx"
    with pd.ExcelWriter(p) as xw:
        for sh in sheets:
            pd.DataFrame({"File": ["scan1", "scan2"], "hv": [21.2, 48.0]}).to_excel(
                xw, sheet_name=sh, index=False)
    return p


class TestGetXlsxSheetNames(unittest.TestCase):
    def test_lists_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _xlsx(Path(tmp))
            self.assertEqual(get_xlsx_sheet_names(p), ["CA041", "CA046"])

    def test_corrupt_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.xlsx"
            bad.write_bytes(b"junk")
            with self.assertRaises(ValueError):
                get_xlsx_sheet_names(bad)


class TestAttachScopedSilent(unittest.TestCase):
    def test_scoped_attach_no_dialog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            parent = _Parent(session)
            ctrl = LogbookIngestController(parent)
            path = _xlsx(root)
            n = ctrl.attach_scoped_silent(path, "CA041", ["CA041"])
            self.assertEqual(n, 2)
            self.assertIn("CA041", session.scoped_logbooks)
            self.assertEqual(session.scoped_logbooks["CA041"]["sheet"], "CA041")
            tagged = [r for r in session.logbook_records
                      if r.get("_subfolder_rel") == "CA041"]
            self.assertEqual(len(tagged), 2)

    def test_global_attach_rel_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            ctrl = LogbookIngestController(_Parent(session))
            path = _xlsx(root)
            ctrl.attach_scoped_silent(path, "CA046", [""])
            self.assertEqual(session.logbook_sheet, "CA046")
            self.assertEqual(session.logbook_path, str(path))
            self.assertEqual(session.scoped_logbooks, {})

    def test_reattach_replaces_scope_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            ctrl = LogbookIngestController(_Parent(session))
            path = _xlsx(root)
            ctrl.attach_scoped_silent(path, "CA041", ["CA041"])
            ctrl.attach_scoped_silent(path, "CA046", ["CA041"])  # switch sheet
            tagged = [r for r in session.logbook_records
                      if r.get("_subfolder_rel") == "CA041"]
            self.assertEqual(len(tagged), 2)  # replaced, not duplicated
            self.assertEqual(session.scoped_logbooks["CA041"]["sheet"], "CA046")

    def test_mapping_override_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            ctrl = LogbookIngestController(_Parent(session))
            path = _xlsx(root)
            ctrl.attach_scoped_silent(path, "CA041", ["CA041"],
                                      mapping_override={"file": "File", "hv": "hv"})
            self.assertEqual(session.scoped_logbooks["CA041"]["mapping"]["hv"], "hv")


class TestBrowseOnlyPersistence(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            session.browse_only = True
            session.save()
            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertTrue(restored.browse_only)

    def test_pre_feature_default_false(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({"version": 1, "folder": str(root), "files": {}}))
            session = Session(root)
            session.load(path)
            self.assertFalse(session.browse_only)


if __name__ == "__main__":
    unittest.main()
