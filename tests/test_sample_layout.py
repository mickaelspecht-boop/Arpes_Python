"""Tests for core/sample_layout + the per-subfolder sample resolution."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arpes.core.sample import SampleConfig, lattice_a_for_entry, sample_for_entry, work_function_for_entry
from arpes.core.sample_layout import (
    detect_sample_layout,
    sample_key_for_entry_key,
)
from arpes.core.session import FileEntry, FileMeta, Session
from arpes.io.scan_utils import is_data_file, is_scan_dataset_dir


def _make_cls_scan_dir(parent: Path, name: str, n_slices: int = 3) -> Path:
    """Synthetic CLS FS dataset: <prefix>_param.txt + Cycle/Step slices."""
    d = parent / name
    d.mkdir(parents=True)
    (d / f"{name}_param.txt").write_text("param")
    for i in range(n_slices):
        (d / f"{name}_Cycle_0_Step_{i}.txt").write_text("0 1 2")
    return d


class TestScanUtils(unittest.TestCase):
    def test_cls_scan_dir_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _make_cls_scan_dir(Path(tmp), "FS_001")
            self.assertTrue(is_scan_dataset_dir(d))

    def test_plain_dir_not_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "BNO"
            d.mkdir()
            (d / "scan.zip").write_bytes(b"x")
            self.assertFalse(is_scan_dataset_dir(d))

    def test_data_file_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.ibw").write_bytes(b"x")
            (root / "b_param.txt").write_text("p")
            self.assertTrue(is_data_file(root / "a.ibw"))
            self.assertFalse(is_data_file(root / "b_param.txt"))

    def test_cls_bm_extensionless(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "BM1").write_text("data")
            (root / "BM1_param.txt").write_text("p")
            self.assertTrue(is_data_file(root / "BM1"))


class TestDetectSampleLayout(unittest.TestCase):
    def test_two_sample_subfolders_multi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sample in ("BNO", "Au_ref"):
                d = root / sample
                d.mkdir()
                (d / "scan.zip").write_bytes(b"x")
            layout = detect_sample_layout(root)
            self.assertEqual(layout.mode, "multi")
            self.assertEqual({s.key for s in layout.subfolders}, {"BNO", "Au_ref"})

    def test_cls_scan_dirs_are_not_samples(self):
        # Redteam case 1: folder = ONE sample, subfolders = FS scans.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_cls_scan_dir(root, "FS_001")
            _make_cls_scan_dir(root, "FS_002")
            (root / "BM_scan.zip").write_bytes(b"x")
            layout = detect_sample_layout(root)
            self.assertEqual(layout.mode, "single")
            self.assertEqual(layout.subfolders, ())
            self.assertEqual(layout.n_root_files, 1)

    def test_sample_with_nested_fs_scan_counts_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "BNO"
            sample.mkdir()
            (sample / "bm1.zip").write_bytes(b"x")
            _make_cls_scan_dir(sample, "FS_001", n_slices=5)
            layout = detect_sample_layout(root)
            self.assertEqual(layout.mode, "multi")
            (sub,) = layout.subfolders
            self.assertEqual(sub.key, "BNO")
            # 1 BM file + 1 FS dataset (slices NOT counted individually).
            self.assertEqual(sub.n_files, 2)

    def test_empty_folder_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = detect_sample_layout(Path(tmp))
            self.assertEqual(layout.mode, "single")
            self.assertEqual(layout.n_root_files, 0)

    def test_hidden_dirs_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / ".arpes_cache"
            d.mkdir()
            (d / "x.zip").write_bytes(b"x")
            layout = detect_sample_layout(root)
            self.assertEqual(layout.mode, "single")


class TestSampleKeyForEntryKey(unittest.TestCase):
    def test_subfolder_key(self):
        self.assertEqual(sample_key_for_entry_key("BNO/scan.zip"), "BNO")
        self.assertEqual(sample_key_for_entry_key("BNO/FS_001/slice.txt"), "BNO")

    def test_root_file_empty_key(self):
        self.assertEqual(sample_key_for_entry_key("scan.zip"), "")

    def test_degenerate_inputs(self):
        self.assertEqual(sample_key_for_entry_key(""), "")


class TestPerSubfolderResolution(unittest.TestCase):
    def _session(self):
        session = Session()
        session.current_sample = SampleConfig(work_function_eV=4.5, a_angstrom=3.0).to_dict()
        session.sample_configs = {
            "BNO": SampleConfig(work_function_eV=4.3, a_angstrom=4.14).to_dict(),
        }
        return session

    def test_subfolder_config_wins_over_session_default(self):
        session = self._session()
        entry = FileEntry(meta=FileMeta())
        sample = sample_for_entry(session, entry, "BNO/scan.zip")
        self.assertAlmostEqual(sample.work_function_eV, 4.3)
        self.assertAlmostEqual(sample.a_angstrom, 4.14)

    def test_subfolder_config_wins_over_file_meta(self):
        # Explicit Samples-popup config is authoritative over file meta (which
        # is often just an echo of a prior UI value or a logbook default).
        session = self._session()
        entry = FileEntry(meta=FileMeta(work_function_eV=4.8))
        sample = sample_for_entry(session, entry, "BNO/scan.zip")
        self.assertAlmostEqual(sample.work_function_eV, 4.3)  # config beats meta
        self.assertAlmostEqual(sample.a_angstrom, 4.14)

    def test_file_meta_used_when_no_subfolder_config(self):
        # Without an explicit per-subfolder config the file meta still wins over
        # the session default (logbook-driven workflow preserved).
        session = self._session()
        entry = FileEntry(meta=FileMeta(work_function_eV=4.8, crystal_a_angstrom=3.9))
        sample = sample_for_entry(session, entry, "Au_ref/scan.zip")
        self.assertAlmostEqual(sample.work_function_eV, 4.8)
        self.assertAlmostEqual(sample.a_angstrom, 3.9)

    def test_other_subfolder_falls_back_to_session(self):
        session = self._session()
        entry = FileEntry(meta=FileMeta())
        sample = sample_for_entry(session, entry, "Au_ref/scan.zip")
        self.assertAlmostEqual(sample.work_function_eV, 4.5)

    def test_no_key_keeps_historical_behaviour(self):
        session = self._session()
        entry = FileEntry(meta=FileMeta())
        sample = sample_for_entry(session, entry)
        self.assertAlmostEqual(sample.work_function_eV, 4.5)

    def test_public_resolvers_accept_entry_key(self):
        session = self._session()
        entry = FileEntry(meta=FileMeta())
        self.assertAlmostEqual(
            work_function_for_entry(session, entry, fallback=0.0, entry_key="BNO/s.zip"),
            4.3,
        )
        self.assertAlmostEqual(
            lattice_a_for_entry(session, entry, entry_key="BNO/s.zip"), 4.14
        )


class TestSessionPersistence(unittest.TestCase):
    def test_roundtrip_sample_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            session.sample_configs = {"BNO": {"work_function_eV": 4.3, "a_angstrom": 4.14}}
            session.save()
            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertAlmostEqual(
                restored.sample_configs["BNO"]["work_function_eV"], 4.3
            )

    def test_pre_feature_session_defaults_empty(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({
                "version": 1, "folder": str(root), "files": {},
            }))
            session = Session(root)
            session.load(path)
            self.assertEqual(session.sample_configs, {})


if __name__ == "__main__":
    unittest.main()
