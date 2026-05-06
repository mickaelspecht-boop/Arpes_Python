from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from arpes.io.kz_dataset import discover_kz_inputs, load_kz_stack


class TestKzDataset(unittest.TestCase):
    def test_discover_and_load_orders_by_hv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("b.ibw", "a.ibw"):
                (root / name).write_text("fake")
            (root / "old [FS].zip").write_text("fake")
            (root / "fixed cut.ibw").write_text("fake")

            def fake_load(path, work_func, ef_offset, hv=None):
                name = Path(path).name
                hv_val = 80.0 if name == "b.ibw" else 60.0
                return {
                    "data": np.ones((2, 3)),
                    "kpar": np.asarray([-0.1, 0.1]),
                    "ev_arr": np.asarray([-0.1, 0.0, 0.1]),
                    "hv": hv_val,
                    "path": path,
                    "metadata": {"source_format": "fake"},
                }

            with patch("arpes.io.kz_dataset.detect_format", lambda _p: "fake"):
                paths = discover_kz_inputs(root)
                self.assertEqual([p.name for p in paths], ["a.ibw", "b.ibw"])
                dataset = load_kz_stack(root, work_func=4.0, ef_offset=0.0, load_func=fake_load)

        self.assertEqual(dataset.hv_values.tolist(), [60.0, 80.0])
        self.assertEqual(len(dataset.scans), 2)
        self.assertEqual(len(dataset.warnings), 2)
        self.assertTrue(any("[FS]" in warning for warning in dataset.warnings))

    def test_load_requires_varying_hv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.ibw", "b.ibw"):
                (root / name).write_text("fake")

            def fake_load(path, work_func, ef_offset, hv=None):
                return {
                    "data": np.ones((2, 3)),
                    "kpar": np.asarray([-0.1, 0.1]),
                    "ev_arr": np.asarray([-0.1, 0.0, 0.1]),
                    "hv": 60.0,
                    "path": path,
                    "metadata": {},
                }

            with patch("arpes.io.kz_dataset.detect_format", lambda _p: "fake"):
                with self.assertRaises(ValueError):
                    load_kz_stack(root, work_func=4.0, ef_offset=0.0, load_func=fake_load)

    def test_kz_logbook_supplies_per_file_hv_and_source_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.ibw", "b.ibw"):
                (root / name).write_text("fake")

            def fake_load(path, work_func, ef_offset, hv=None):
                return {
                    "data": np.ones((2, 3)),
                    "kpar": np.asarray([-0.1, 0.1]),
                    "ev_arr": np.asarray([-0.1, 0.0, 0.1]),
                    "hv": hv,
                    "path": path,
                    "metadata": {},
                }

            records = [
                {"file": "a.ibw", "hv": 60.0},
                {"file": "b.ibw", "hv": 80.0},
            ]
            mapping = {"file": "file", "hv": "hv"}
            with patch("arpes.io.kz_dataset.detect_format", lambda _p: "fake"):
                dataset = load_kz_stack(
                    root,
                    work_func=4.0,
                    ef_offset=0.0,
                    kz_logbook_records=records,
                    kz_logbook_mapping=mapping,
                    session_folder=root,
                    load_func=fake_load,
                )

        self.assertEqual(dataset.hv_values.tolist(), [60.0, 80.0])
        self.assertEqual({s.metadata["hv_source"] for s in dataset.scans}, {"kz_logbook"})

    def test_file_hv_overrides_conflicting_logbook_hv_visibly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.ibw", "b.ibw"):
                (root / name).write_text("fake")

            def fake_load(path, work_func, ef_offset, hv=None):
                name = Path(path).name
                hv_val = 61.0 if name == "a.ibw" else 81.0
                return {
                    "data": np.ones((2, 3)),
                    "kpar": np.asarray([-0.1, 0.1]),
                    "ev_arr": np.asarray([-0.1, 0.0, 0.1]),
                    "hv": hv_val,
                    "path": path,
                    "metadata": {},
                }

            records = [
                {"file": "a.ibw", "hv": 60.0},
                {"file": "b.ibw", "hv": 80.0},
            ]
            mapping = {"file": "file", "hv": "hv"}
            with patch("arpes.io.kz_dataset.detect_format", lambda _p: "fake"):
                dataset = load_kz_stack(
                    root,
                    work_func=4.0,
                    ef_offset=0.0,
                    kz_logbook_records=records,
                    kz_logbook_mapping=mapping,
                    session_folder=root,
                    load_func=fake_load,
                )

        self.assertEqual(dataset.hv_values.tolist(), [61.0, 81.0])
        self.assertEqual({s.metadata["hv_source"] for s in dataset.scans}, {"file"})
        self.assertTrue(any("remplace" in warning for warning in dataset.warnings))


if __name__ == "__main__":
    unittest.main()
