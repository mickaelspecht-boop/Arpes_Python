"""Tests A.1 — inférence scan_kind depuis metadata loader."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from arpes.core.session import FileEntry, FileMeta, Session
from arpes.io.scan_kind_inference import infer_scan_kind


class TestInferScanKind(unittest.TestCase):
    def test_empty_or_none_returns_unknown(self):
        self.assertEqual(infer_scan_kind(None), "unknown")
        self.assertEqual(infer_scan_kind({}), "unknown")

    def test_explicit_metadata_value_wins(self):
        self.assertEqual(infer_scan_kind({"scan_kind": "BM"}), "BM")
        self.assertEqual(infer_scan_kind({"scan_kind": "FS"}), "FS")
        self.assertEqual(infer_scan_kind({"scan_kind": "KZ"}), "KZ")

    def test_kz_detected_from_kz_keys(self):
        self.assertEqual(infer_scan_kind({"kz_scan": True}), "KZ")
        self.assertEqual(infer_scan_kind({"kz_data": [1, 2]}), "KZ")

    def test_fs_detected_from_fs_data(self):
        self.assertEqual(infer_scan_kind({"fs_data": object()}), "FS")

    def test_fallback_on_ndim(self):
        self.assertEqual(infer_scan_kind({}, data_ndim=1), "EDC")
        self.assertEqual(infer_scan_kind({}, data_ndim=2), "BM")
        self.assertEqual(infer_scan_kind({}, data_ndim=3), "FS")

    def test_explicit_overrides_ndim(self):
        # Cas pathologique : metadata dit BM mais ndim=3 → respecte metadata.
        self.assertEqual(
            infer_scan_kind({"scan_kind": "BM"}, data_ndim=3),
            "BM",
        )

    def test_invalid_scan_kind_falls_through(self):
        self.assertEqual(
            infer_scan_kind({"scan_kind": "GARBAGE"}, data_ndim=2),
            "BM",
        )


class TestScanKindRoundtripInSession(unittest.TestCase):
    """A.1 — scan_kind dans FileMeta survit save/load JSON."""

    def test_meta_scan_kind_persists(self):
        session = Session(Path("/tmp/scan_kind_test"))
        entry = session.get_or_create("fs_run_01")
        entry.meta.scan_kind = "FS"
        entry.meta.hv = 60.0
        entry.meta.azi = 12.5

        payload = session.to_payload()
        reloaded = Session(Path("/tmp/scan_kind_test"))
        reloaded.load_from_payload(json.loads(json.dumps(payload)))

        restored = reloaded.files["fs_run_01"]
        self.assertEqual(restored.meta.scan_kind, "FS")
        self.assertAlmostEqual(restored.meta.hv, 60.0)
        self.assertAlmostEqual(restored.meta.azi, 12.5)

    def test_default_scan_kind_is_unknown(self):
        meta = FileMeta()
        self.assertEqual(meta.scan_kind, "unknown")


class TestParentFsPathRoundtrip(unittest.TestCase):
    """A.3 — entry.parent_fs_path survit save/load JSON."""

    def test_default_is_none(self):
        self.assertIsNone(FileEntry().parent_fs_path)

    def test_explicit_path_persists(self):
        session = Session(Path("/tmp/parent_fs_test"))
        bm = session.get_or_create("bm04")
        bm.parent_fs_path = "/data/bna_s2/fs1.txt"

        payload = session.to_payload()
        reloaded = Session(Path("/tmp/parent_fs_test"))
        reloaded.load_from_payload(json.loads(json.dumps(payload)))

        self.assertEqual(reloaded.files["bm04"].parent_fs_path,
                         "/data/bna_s2/fs1.txt")

    def test_legacy_payload_without_field_loads(self):
        session = Session(Path("/tmp/parent_fs_legacy"))
        session.get_or_create("bm04")
        payload = session.to_payload()
        # Simule ancien JSON : retire le champ
        for entry_dict in payload["files"].values():
            entry_dict.pop("parent_fs_path", None)

        reloaded = Session(Path("/tmp/parent_fs_legacy"))
        reloaded.load_from_payload(json.loads(json.dumps(payload)))
        self.assertIsNone(reloaded.files["bm04"].parent_fs_path)


if __name__ == "__main__":
    unittest.main()
