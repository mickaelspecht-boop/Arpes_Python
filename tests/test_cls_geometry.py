"""Tests des helpers `arpes_cls_geometry`."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arpes.physics.cls_geometry import geometry_for_path, manipulator_from_param
from arpes.io.loaders.common import scan_axis_summary, static_polar_for_kx


def _write_param(path: Path, *, polar: float | None, tilt: float | None) -> None:
    motors = {}
    if polar is not None:
        motors["P"] = {"position": polar}
    if tilt is not None:
        motors["T"] = {"position": tilt}
    line = json.dumps({"d": motors})
    path.write_text("# header\n" + line + "\n")


class _FakeMeta:
    def __init__(self, **kw):
        self.polar = kw.get("polar")
        self.tilt = kw.get("tilt")
        self.azi = kw.get("azi")
        self.hv = kw.get("hv")


def _cell_float(v):
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None


class TestManipulatorFromParam(unittest.TestCase):
    def test_missing_path_returns_empty(self):
        self.assertEqual(manipulator_from_param("/no/such/path/1234"), {})

    def test_file_with_param_sibling(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "BM1"
            data.write_text("dummy")
            _write_param(data.parent / "BM1_param.txt", polar=1.5, tilt=-0.3)
            out = manipulator_from_param(data)
            self.assertAlmostEqual(out["polar"], 1.5)
            self.assertAlmostEqual(out["tilt"], -0.3)

    def test_directory_with_param_inside(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "FS1"
            sub.mkdir()
            _write_param(sub / "FS1_param.txt", polar=12.0, tilt=None)
            out = manipulator_from_param(sub)
            self.assertAlmostEqual(out["polar"], 12.0)
            self.assertNotIn("tilt", out)

    def test_fs_scan_start_polar_is_not_static_kx_polar(self):
        polar, raw, ignored = static_polar_for_kx(
            -15.0, [-15.0, -14.5, -14.0],
            is_fs=True,
            motor_present=True,
        )
        self.assertAlmostEqual(polar, 0.0)
        self.assertAlmostEqual(raw, -15.0)
        self.assertTrue(ignored)

    def test_fs_scan_in_range_polar_is_not_static_kx_polar(self):
        polar, raw, ignored = static_polar_for_kx(
            -10.2, [-14.0, -13.5, -13.0, -12.5, -12.0, -11.5, -11.0, -10.5, -10.0],
            is_fs=True,
            motor_present=True,
        )
        self.assertAlmostEqual(polar, 0.0)
        self.assertAlmostEqual(raw, -10.2)
        self.assertTrue(ignored)

    def test_bm_keeps_static_polar(self):
        polar, raw, ignored = static_polar_for_kx(
            -15.0, [-15.0, -14.5, -14.0],
            is_fs=False,
            motor_present=True,
        )
        self.assertAlmostEqual(polar, -15.0)
        self.assertAlmostEqual(raw, -15.0)
        self.assertFalse(ignored)

    def test_scan_axis_summary_records_range_center_and_step(self):
        summary = scan_axis_summary([-14.0, -13.5, -13.0, 10.0])
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertAlmostEqual(summary["min"], -14.0)
        self.assertAlmostEqual(summary["max"], 10.0)
        self.assertAlmostEqual(summary["center"], -2.0)
        self.assertAlmostEqual(summary["step"], 0.5)
        self.assertEqual(summary["n"], 4)


class TestGeometryForPath(unittest.TestCase):
    def test_param_overrides_nothing_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "BM1"
            data.write_text("x")
            _write_param(data.parent / "BM1_param.txt", polar=1.0, tilt=0.5)
            meta = _FakeMeta(polar=99.0, tilt=99.0, azi=30.0, hv=80.0)
            geom = geometry_for_path(data, entry_meta=meta)
            self.assertAlmostEqual(geom["polar"], 1.0)   # param gagne
            self.assertAlmostEqual(geom["tilt"], 0.5)
            self.assertAlmostEqual(geom["azi"], 30.0)    # azi vient de l'entry
            self.assertAlmostEqual(geom["hv"], 80.0)

    def test_entry_fills_missing_polar_tilt(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "BM1"
            data.write_text("x")
            # No param → entry must fill P/T.
            meta = _FakeMeta(polar=2.5, tilt=1.0, azi=15.0)
            geom = geometry_for_path(data, entry_meta=meta)
            self.assertAlmostEqual(geom["polar"], 2.5)
            self.assertAlmostEqual(geom["tilt"], 1.0)
            self.assertAlmostEqual(geom["azi"], 15.0)

    def test_logbook_fills_missing_only(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "BM1"
            data.write_text("x")
            meta = _FakeMeta(polar=2.5, tilt=None, azi=None)
            rec = {"col_polar": "999", "col_azi": "45"}
            mapping = {"polar": "col_polar", "azi": "col_azi"}
            geom = geometry_for_path(
                data, entry_meta=meta, logbook_record=rec,
                logbook_mapping=mapping, cell_float=_cell_float,
            )
            self.assertAlmostEqual(geom["polar"], 2.5)   # entry gagne sur logbook
            self.assertAlmostEqual(geom["azi"], 45.0)    # logbook remplit

    def test_no_entry_no_logbook_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "x"
            data.write_text("x")
            self.assertEqual(geometry_for_path(data), {})


if __name__ == "__main__":
    unittest.main()
