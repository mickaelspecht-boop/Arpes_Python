from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from arpes.core.session import FileEntry

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


class _Params:
    def __init__(self):
        self.sp_phi = _Spin(4.031)
        self.sp_ef = _Spin(0.052)


class _Parent:
    def __init__(self):
        self._params = _Params()
        self._raw_load_cache = OrderedDict()
        self._raw_load_cache_max = 3
        self._last_load_cache_hit = False

    def _bessy_energy_reference_mode(self) -> str:
        return "auto"

    def _load_with_best_angle_offsets(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("best-angle path not used in this test")


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


if __name__ == "__main__":
    unittest.main()
