from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from arpes.core.session import FileEntry, Session
from arpes.io.artifact_cache import cache_size_mb, clear_cache_folder
from arpes.io.logbook import LogbookAppliedValues

try:
    import arpes.ui.controllers.load_controller as load_mod
    from arpes.ui.controllers.load_controller import LoadController, _PreparedEntry
    HAS_LOAD_CONTROLLER = True
except Exception:
    load_mod = None
    LoadController = None
    _PreparedEntry = None
    HAS_LOAD_CONTROLLER = False


class _Spin:
    def __init__(self, value: float):
        self._value = float(value)

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        self._value = float(value)

    def blockSignals(self, _blocked: bool) -> None:
        return None


class _Params:
    def __init__(self):
        self.sp_phi = _Spin(4.031)
        self.sp_ef = _Spin(0.052)
        self.sp_hv = _Spin(0.0)
        self.hv_source = ""

    def set_hv_value_with_source(self, value: float, source: str) -> None:
        self.sp_hv.setValue(value)
        self.hv_source = source

    def update_hv_source(self, source: str | None) -> None:
        self.hv_source = source or ""


class _LogbookCtrl:
    def __init__(self, *, hit: bool = False, hv: float | None = None):
        self.hit = hit
        self._last_applied_values = LogbookAppliedValues(hv=hv)

    def apply_to_controls(self, _path: str) -> bool:
        return self.hit


class _Parent:
    def __init__(self):
        self._params = _Params()
        self._raw_load_cache = OrderedDict()
        self._raw_load_cache_max = 3
        self._raw_disk_cache_enabled = False
        self._raw_disk_cache_async = False
        self._path_signature_cache = OrderedDict()
        self._path_signature_cache_max = 3
        self._last_load_cache_hit = False
        self._last_load_cache_source = ""
        self._last_path_signature_cache_hit = False
        self._current_path = None
        self._session = Session()
        self._logbook_ctrl = _LogbookCtrl()

    def _bessy_energy_reference_mode(self) -> str:
        return "auto"

    def _load_with_best_angle_offsets(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("best-angle path not used in this test")

    def _angle_offsets_for_load(self, *_args, **_kwargs):
        return {}


def _prepared(path: Path):
    return _PreparedEntry(
        key=str(path),
        entry=FileEntry(),
        is_new=False,
        fmt_guess="cls_txt",
        hv_for_load=50.0,
        hv_from_logbook=False,
        angle_offsets={},
        logbook_hit=False,
    )


@unittest.skipUnless(HAS_LOAD_CONTROLLER, "LoadController/PyQt6 indisponible")
class TestLoadControllerCache(unittest.TestCase):
    def test_clear_disk_cache_removes_raw_and_cls_fs_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / ".arpes_cache" / "raw_artifacts"
            raw_dir.mkdir(parents=True)
            raw_file = raw_dir / "raw_dummy.npz"
            fs_file = root / ".arpes_cache" / "FS_fs_mean_v2.npz"
            raw_file.write_bytes(b"raw-cache")
            fs_file.write_bytes(b"fs-cache")

            self.assertGreater(cache_size_mb(root), 0.0)
            n, total = clear_cache_folder(root)

            self.assertEqual(n, 2)
            self.assertEqual(total, len(b"raw-cache") + len(b"fs-cache"))
            self.assertFalse(raw_file.exists())
            self.assertFalse(fs_file.exists())

    def test_dispatch_loader_reuses_same_file_and_params(self):
        calls = []
        old_load = load_mod.load_arpes_file

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append((path, work_func, ef_offset, kwargs))
            return {
                "path": path,
                "data": np.ones((2, 3), dtype=np.float32),
                "kpar": np.array([0.0, 1.0]),
                "ev_arr": np.array([-0.1, 0.0, 0.1]),
                "hv": kwargs.get("hv"),
                "source_format": "cls_txt",
                "metadata": {"loader_label": "CLS"},
            }

        load_mod.load_arpes_file = fake_load
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "BM.txt"
                path.write_text("dummy")
                parent = _Parent()
                ctrl = LoadController(parent)
                prep = _prepared(path)

                first, _ = ctrl._dispatch_loader(str(path), prep)
                first.data["metadata"]["mutated_after_load"] = True
                second, _ = ctrl._dispatch_loader(str(path), prep)

                self.assertEqual(len(calls), 1)
                self.assertTrue(parent._last_load_cache_hit)
                self.assertNotIn("mutated_after_load", second.data["metadata"])
        finally:
            load_mod.load_arpes_file = old_load

    def test_dispatch_loader_misses_when_load_param_changes(self):
        calls = []
        old_load = load_mod.load_arpes_file

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append((work_func, ef_offset, kwargs.get("hv")))
            return {
                "path": path,
                "data": np.ones((1, 1), dtype=np.float32),
                "kpar": np.array([0.0]),
                "ev_arr": np.array([0.0]),
                "hv": kwargs.get("hv"),
                "metadata": {},
            }

        load_mod.load_arpes_file = fake_load
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "BM.txt"
                path.write_text("dummy")
                parent = _Parent()
                ctrl = LoadController(parent)
                prep = _prepared(path)

                ctrl._dispatch_loader(str(path), prep)
                parent._params.sp_ef.setValue(0.060)
                ctrl._dispatch_loader(str(path), prep)

                self.assertEqual(len(calls), 2)
                self.assertFalse(parent._last_load_cache_hit)
        finally:
            load_mod.load_arpes_file = old_load

    def test_dispatch_loader_uses_sample_work_function_before_ui_phi(self):
        calls = []
        old_load = load_mod.load_arpes_file

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append((work_func, kwargs))
            return {
                "path": path,
                "data": np.ones((1, 1), dtype=np.float32),
                "kpar": np.array([0.0]),
                "ev_arr": np.array([0.0]),
                "metadata": {},
            }

        load_mod.load_arpes_file = fake_load
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "BM.txt"
                path.write_text("dummy")
                parent = _Parent()
                parent._session.current_sample = {"work_function_eV": 4.8}
                parent._params.sp_phi.setValue(4.031)
                ctrl = LoadController(parent)

                ctrl._dispatch_loader(str(path), _prepared(path))

                self.assertEqual(len(calls), 1)
                self.assertAlmostEqual(calls[0][0], 4.8)
        finally:
            load_mod.load_arpes_file = old_load

    def test_dispatch_loader_misses_when_sample_work_function_changes(self):
        calls = []
        old_load = load_mod.load_arpes_file

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append(work_func)
            return {
                "path": path,
                "data": np.ones((1, 1), dtype=np.float32),
                "kpar": np.array([0.0]),
                "ev_arr": np.array([0.0]),
                "metadata": {},
            }

        load_mod.load_arpes_file = fake_load
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "BM.txt"
                path.write_text("dummy")
                parent = _Parent()
                parent._session.current_sample = {"work_function_eV": 4.8}
                ctrl = LoadController(parent)
                prep = _prepared(path)

                ctrl._dispatch_loader(str(path), prep)
                parent._session.current_sample = {"work_function_eV": 4.6}
                ctrl._dispatch_loader(str(path), prep)

                self.assertEqual(calls, [4.8, 4.6])
                self.assertFalse(parent._last_load_cache_hit)
        finally:
            load_mod.load_arpes_file = old_load

    def test_dispatch_loader_misses_when_hv_changes(self):
        calls = []
        old_load = load_mod.load_arpes_file

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append(kwargs.get("hv"))
            return {
                "path": path,
                "data": np.ones((1, 1), dtype=np.float32),
                "kpar": np.array([0.0]),
                "ev_arr": np.array([0.0]),
                "hv": kwargs.get("hv"),
                "metadata": {},
            }

        load_mod.load_arpes_file = fake_load
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "BM.txt"
                path.write_text("dummy")
                parent = _Parent()
                ctrl = LoadController(parent)
                prep = _prepared(path)

                ctrl._dispatch_loader(str(path), prep)
                prep.hv_for_load = 51.25
                ctrl._dispatch_loader(str(path), prep)

                self.assertEqual(calls, [50.0, 51.25])
                self.assertFalse(parent._last_load_cache_hit)
        finally:
            load_mod.load_arpes_file = old_load

    def test_prepare_entry_marks_logbook_hv_even_when_value_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "BM.txt"
            path.write_text("dummy")
            parent = _Parent()
            parent._session = Session(root)
            parent._params.sp_hv.setValue(78.0)
            parent._logbook_ctrl = _LogbookCtrl(hit=True, hv=78.0)

            prepared = LoadController(parent)._prepare_entry(str(path))

            self.assertTrue(prepared.hv_from_logbook)
            self.assertEqual(prepared.hv_for_load, 78.0)

    def test_prepare_entry_uses_session_hv_when_switching_files_without_logbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_path = root / "BM_old.txt"
            path = root / "BM_new.txt"
            old_path.write_text("dummy")
            path.write_text("dummy")
            parent = _Parent()
            parent._session = Session(root)
            parent._current_path = str(old_path)
            parent._params.sp_hv.setValue(80.0)
            parent._session.get_or_create(parent._session.key_for_path(path)).meta.hv = 66.0
            parent._logbook_ctrl = _LogbookCtrl(hit=False, hv=None)

            prepared = LoadController(parent)._prepare_entry(str(path))

            self.assertEqual(prepared.hv_for_load, 66.0)
            self.assertEqual(parent._params.hv_source, "session")

    def test_dispatch_loader_misses_when_cls_param_sidecar_changes(self):
        calls = []
        old_load = load_mod.load_arpes_file

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append((path, work_func, ef_offset, kwargs))
            return {
                "path": path,
                "data": np.ones((1, 1), dtype=np.float32) * len(calls),
                "kpar": np.array([0.0]),
                "ev_arr": np.array([0.0]),
                "hv": kwargs.get("hv"),
                "metadata": {},
            }

        load_mod.load_arpes_file = fake_load
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "BM.txt"
                sidecar = Path(tmp) / "BM.txt_param.txt"
                path.write_text("dummy")
                sidecar.write_text("param v1")
                parent = _Parent()
                ctrl = LoadController(parent)
                prep = _prepared(path)

                ctrl._dispatch_loader(str(path), prep)
                sidecar.write_text("param v2 changed")
                ctrl._dispatch_loader(str(path), prep)

                self.assertEqual(len(calls), 2)
                self.assertFalse(parent._last_load_cache_hit)
        finally:
            load_mod.load_arpes_file = old_load

    def test_path_signature_cache_reuses_file_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "BM.txt"
            path.write_text("dummy")
            parent = _Parent()
            ctrl = LoadController(parent)

            first = ctrl._path_signature(path)
            self.assertFalse(parent._last_path_signature_cache_hit)
            second = ctrl._path_signature(path)

            self.assertTrue(parent._last_path_signature_cache_hit)
            self.assertEqual(second, first)

    def test_path_signature_does_not_reuse_directory_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cycle1").mkdir()
            (root / "Cycle1" / "Step1.txt").write_text("dummy")
            parent = _Parent()
            ctrl = LoadController(parent)

            ctrl._path_signature(root)
            self.assertFalse(parent._last_path_signature_cache_hit)
            ctrl._path_signature(root)

            self.assertFalse(parent._last_path_signature_cache_hit)

    def test_dispatch_loader_reuses_disk_artifact_after_new_parent(self):
        calls = []
        old_load = load_mod.load_arpes_file

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append((path, work_func, ef_offset, kwargs))
            return {
                "path": path,
                "data": np.arange(6, dtype=np.float32).reshape(2, 3),
                "kpar": np.array([0.0, 1.0]),
                "ev_arr": np.array([-0.1, 0.0, 0.1]),
                "hv": kwargs.get("hv"),
                "source_format": "cls_txt",
                "metadata": {"loader_label": "CLS", "fs_data": np.ones((2, 2, 3), dtype=np.float32)},
            }

        load_mod.load_arpes_file = fake_load
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "BM.txt"
                path.write_text("dummy")

                first_parent = _Parent()
                first_parent._raw_disk_cache_enabled = True
                second_parent = _Parent()
                second_parent._raw_disk_cache_enabled = True
                LoadController(first_parent)._dispatch_loader(str(path), _prepared(path))
                second, _ = LoadController(second_parent)._dispatch_loader(str(path), _prepared(path))

                self.assertEqual(len(calls), 1)
                self.assertTrue(second_parent._last_load_cache_hit)
                self.assertEqual(second_parent._last_load_cache_source, "disque")
                np.testing.assert_allclose(second.data["metadata"]["fs_data"], np.ones((2, 2, 3)))
        finally:
            load_mod.load_arpes_file = old_load


if __name__ == "__main__":
    unittest.main()
