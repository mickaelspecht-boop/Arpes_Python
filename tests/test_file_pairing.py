"""Tests A.2 — pairing BM ↔ FS (M4 hybride)."""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace



from arpes.core.session import FileEntry, FileMeta
from arpes.io.file_pairing import (
    PairingCriteria,
    PairingMatch,
    find_bms_for_fs,
    find_fs_for_bm,
    group_files_by_fs,
)


def _entry(*, scan_kind: str, hv: float = 60.0, azi: float = 0.0,
           polarization: str = "LH", parent_fs_path: str | None = None,
           formula: str = "", mp_id: str = "") -> FileEntry:
    meta = FileMeta(
        hv=hv, azi=azi, polarization=polarization,
        scan_kind=scan_kind, formula=formula, mp_id=mp_id,
    )
    e = FileEntry(meta=meta)
    e.parent_fs_path = parent_fs_path
    return e


class TestFindBmsForFs(unittest.TestCase):
    def test_basic_match_same_folder(self):
        files = {
            "/d/bna_s2/fs1.txt": _entry(scan_kind="FS", hv=60.0, azi=0.0),
            "/d/bna_s2/bm03.txt": _entry(scan_kind="BM", hv=60.0, azi=0.0),
            "/d/bna_s2/bm04.txt": _entry(scan_kind="BM", hv=60.0, azi=1.0),
        }
        out = find_bms_for_fs(files["/d/bna_s2/fs1.txt"], "/d/bna_s2/fs1.txt", files)
        paths = [m.path for m in out]
        self.assertEqual(set(paths), {"/d/bna_s2/bm03.txt", "/d/bna_s2/bm04.txt"})
        # bm03 distance plus petite (azi=0)
        self.assertEqual(out[0].path, "/d/bna_s2/bm03.txt")

    def test_excludes_fs_files(self):
        files = {
            "/d/fs1.txt": _entry(scan_kind="FS"),
            "/d/fs2.txt": _entry(scan_kind="FS"),
        }
        out = find_bms_for_fs(files["/d/fs1.txt"], "/d/fs1.txt", files)
        self.assertEqual(out, [])

    def test_filter_hv_out_of_tolerance(self):
        files = {
            "/d/fs.txt": _entry(scan_kind="FS", hv=60.0),
            "/d/bm_close.txt": _entry(scan_kind="BM", hv=62.0),   # ~3.3% OK
            "/d/bm_far.txt": _entry(scan_kind="BM", hv=80.0),     # 25% OUT
        }
        out = find_bms_for_fs(files["/d/fs.txt"], "/d/fs.txt", files)
        self.assertEqual([m.path for m in out], ["/d/bm_close.txt"])

    def test_filter_azi_out_of_tolerance(self):
        files = {
            "/d/fs.txt": _entry(scan_kind="FS", azi=0.0),
            "/d/bm_a.txt": _entry(scan_kind="BM", azi=1.5),
            "/d/bm_b.txt": _entry(scan_kind="BM", azi=10.0),
        }
        out = find_bms_for_fs(files["/d/fs.txt"], "/d/fs.txt", files)
        self.assertEqual([m.path for m in out], ["/d/bm_a.txt"])

    def test_azi_wraparound_is_compatible(self):
        files = {
            "/d/fs.txt": _entry(scan_kind="FS", azi=1.0),
            "/d/bm_wrap.txt": _entry(scan_kind="BM", azi=359.0),
        }
        out = find_bms_for_fs(files["/d/fs.txt"], "/d/fs.txt", files)
        self.assertEqual([m.path for m in out], ["/d/bm_wrap.txt"])

    def test_filter_different_folder_rejected(self):
        files = {
            "/d/sample_A/fs.txt": _entry(scan_kind="FS"),
            "/d/sample_B/bm.txt": _entry(scan_kind="BM"),
        }
        criteria = PairingCriteria(folder_depth=0)
        out = find_bms_for_fs(files["/d/sample_A/fs.txt"], "/d/sample_A/fs.txt",
                              files, criteria)
        self.assertEqual(out, [])

    def test_filter_polarization_mismatch(self):
        files = {
            "/d/fs.txt": _entry(scan_kind="FS", polarization="LH"),
            "/d/bm_lh.txt": _entry(scan_kind="BM", polarization="LH"),
            "/d/bm_lv.txt": _entry(scan_kind="BM", polarization="LV"),
        }
        out = find_bms_for_fs(files["/d/fs.txt"], "/d/fs.txt", files)
        self.assertEqual([m.path for m in out], ["/d/bm_lh.txt"])

    def test_manual_override_first(self):
        files = {
            "/d/fs1.txt": _entry(scan_kind="FS", hv=60.0, azi=0.0),
            "/d/bm_auto.txt": _entry(scan_kind="BM", hv=60.0, azi=0.0),
            "/d/somewhere_else/bm_pinned.txt": _entry(
                scan_kind="BM", hv=90.0, azi=20.0,  # ne passerait pas l'auto
                parent_fs_path="/d/fs1.txt",
            ),
        }
        out = find_bms_for_fs(files["/d/fs1.txt"], "/d/fs1.txt", files,
                              PairingCriteria(folder_depth=2))
        self.assertEqual(out[0].path, "/d/somewhere_else/bm_pinned.txt")
        self.assertEqual(out[0].reason, "manual")
        # bm_auto vient après car distance plus grande que 0.0
        paths_auto = [m.path for m in out if m.reason == "auto"]
        self.assertIn("/d/bm_auto.txt", paths_auto)


class TestFindFsForBm(unittest.TestCase):
    def test_symmetric_basic(self):
        files = {
            "/d/fs.txt": _entry(scan_kind="FS", hv=60.0, azi=0.0),
            "/d/bm.txt": _entry(scan_kind="BM", hv=60.0, azi=0.0),
        }
        out = find_fs_for_bm(files["/d/bm.txt"], "/d/bm.txt", files)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].path, "/d/fs.txt")

    def test_pinned_fs_returns_manual_even_if_incompatible(self):
        files = {
            "/d/fs_far.txt": _entry(scan_kind="FS", hv=120.0, azi=45.0),
            "/d/bm.txt": _entry(
                scan_kind="BM", hv=60.0, azi=0.0,
                parent_fs_path="/d/fs_far.txt",
            ),
        }
        out = find_fs_for_bm(files["/d/bm.txt"], "/d/bm.txt", files,
                             PairingCriteria(folder_depth=2))
        self.assertEqual(out[0].path, "/d/fs_far.txt")
        self.assertEqual(out[0].reason, "manual")


class TestGroupFilesByFs(unittest.TestCase):
    def test_tree_with_orphans(self):
        files = {
            "/d/fs1.txt": _entry(scan_kind="FS", hv=60.0, azi=0.0),
            "/d/fs2.txt": _entry(scan_kind="FS", hv=80.0, azi=0.0),
            "/d/bm_a.txt": _entry(scan_kind="BM", hv=60.0, azi=0.0),    # → fs1
            "/d/bm_b.txt": _entry(scan_kind="BM", hv=80.0, azi=0.0),    # → fs2
            "/d/bm_orphan.txt": _entry(scan_kind="BM", hv=120.0, azi=30.0),
            "/d/kz_scan.txt": _entry(scan_kind="KZ"),
        }
        tree, orphans = group_files_by_fs(files)
        self.assertEqual(len(tree), 2)
        # fs1 a bm_a
        fs1_entry = [t for t in tree if t[0] == "/d/fs1.txt"][0]
        self.assertEqual([m.path for m in fs1_entry[2]], ["/d/bm_a.txt"])
        # fs2 a bm_b
        fs2_entry = [t for t in tree if t[0] == "/d/fs2.txt"][0]
        self.assertEqual([m.path for m in fs2_entry[2]], ["/d/bm_b.txt"])
        # orphans : bm_orphan + kz_scan
        orphan_paths = sorted([p for p, _ in orphans])
        self.assertEqual(orphan_paths, ["/d/bm_orphan.txt", "/d/kz_scan.txt"])


class TestPseudoEntriesFromLogbook(unittest.TestCase):
    """Synthèse FileEntry depuis logbook records (BMs candidates non chargées)."""

    def _stub_session(self, records, mapping, files=None, folder=None):
        s = SimpleNamespace()
        s.logbook_records = records
        s.logbook_mapping = mapping
        s.scoped_logbooks = {}
        s.files = files or {}
        s.folder = folder
        s.key_for_path = lambda p: str(p)
        return s

    def test_synthesize_from_real_files_in_folder(self):
        """Scan physique du folder + matching logbook par values_for_path."""
        import tempfile
        from arpes.io.file_pairing import build_pseudo_entries_from_logbook
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Crée un faux fichier BM (extension data) et un fichier ignoré
            (tmp_path / "BM99.pxt").write_text("")
            (tmp_path / "BM99_param.txt").write_text("")  # sidecar → skip
            records = [{"file": "BM99.pxt", "hv": "60", "Pol": "LH", "P": "1.2"}]
            mapping = {"file": "file", "hv": "hv", "polarization": "Pol", "polar": "P"}
            s = self._stub_session(records, mapping, folder=tmp_path)
            out = build_pseudo_entries_from_logbook(
                s, scan_kind_resolver=lambda p: "BM",
            )
            keys = list(out.keys())
            # Vérifie qu'au moins une entrée BM99 a été créée (path absolu)
            bm_key = next((k for k in keys if "BM99" in k), None)
            self.assertIsNotNone(bm_key, f"Aucune entry BM99 dans {keys}")
            e = out[bm_key]
            self.assertEqual(e.meta.scan_kind, "BM")
            self.assertAlmostEqual(e.meta.hv, 60.0)

    def test_skip_if_already_in_session_files(self):
        import tempfile
        from arpes.io.file_pairing import build_pseudo_entries_from_logbook
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "BM99.pxt").write_text("")
            records = [{"file": "BM99.pxt", "hv": "60"}]
            mapping = {"file": "file", "hv": "hv"}
            files = {str(tmp_path / "BM99.pxt"):
                     FileEntry(meta=FileMeta(scan_kind="BM"))}
            s = self._stub_session(records, mapping, files=files, folder=tmp_path)
            out = build_pseudo_entries_from_logbook(
                s, scan_kind_resolver=lambda p: "BM",
            )
            self.assertNotIn(str(tmp_path / "BM99.pxt"), out)


if __name__ == "__main__":
    unittest.main()
