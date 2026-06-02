from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    import arpes.ui.controllers.pocket_controller as pocket_controller_mod
    from arpes.core.session import Session
    from arpes.physics.fs import FSParams
    from arpes.ui.controllers.pocket_controller import PocketController
    from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas

    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 indisponible")
class TestPocketController(unittest.TestCase):
    _qt_app = None

    @classmethod
    def setUpClass(cls):
        if UI_AVAILABLE:
            cls._qt_app = QApplication.instance() or QApplication([])

    def test_characterize_persists_pocket_and_redraws(self):
        class FakeDialog:
            def __init__(self, parent, pocket, **_kwargs):
                self.pocket = pocket
                self.delete_requested = False

            def exec(self):
                return 1

        old_dialog = pocket_controller_mod.PocketResultDialog
        pocket_controller_mod.PocketResultDialog = FakeDialog
        try:
            with tempfile.TemporaryDirectory() as tmp:
                parent = _Parent(Path(tmp))
                ctrl = PocketController(parent)

                pocket = ctrl._pocket_action("characterize", {"kx": 0.0, "ky": 0.0})

                entry = parent._current_entry()
                self.assertIsNotNone(pocket)
                self.assertEqual(len(entry.fs_pockets), 1)
                self.assertEqual(entry.fs_pockets[0]["topology"], "electron")
                self.assertGreater(entry.fs_pockets[0]["area_pct_bz"], 1.0)
                self.assertGreater(len(entry.fs_pockets[0]["contour"]), 10)
                self.assertEqual(parent.draws, 1)
        finally:
            pocket_controller_mod.PocketResultDialog = old_dialog

    def test_manual_level_show_and_clear(self):
        calls = []

        class FakeDialog:
            def __init__(self, parent, pocket, **_kwargs):
                calls.append(pocket)
                self.delete_requested = False

            def exec(self):
                return 1

        old_dialog = pocket_controller_mod.PocketResultDialog
        old_get_double = pocket_controller_mod.QInputDialog.getDouble
        pocket_controller_mod.PocketResultDialog = FakeDialog
        pocket_controller_mod.QInputDialog.getDouble = staticmethod(
            lambda *args, **kwargs: (0.5, True)
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                parent = _Parent(Path(tmp))
                parent._current_entry().fs_lattice = {"mp_id": "mp-test"}
                ctrl = PocketController(parent)

                pocket = ctrl._pocket_action(
                    "characterize_with_level", {"kx": 0.0, "ky": 0.0}
                )
                shown = ctrl._pocket_action("show", {"index": 0})
                ctrl._pocket_action("clear", {})

                self.assertIsNotNone(pocket)
                self.assertEqual(pocket["level"], 0.5)
                self.assertEqual(pocket["mp_label"], "mp-test:Γ")
                self.assertEqual(shown["topology"], "electron")
                self.assertEqual(len(calls), 2)
                self.assertEqual(parent._current_entry().fs_pockets, [])
                self.assertEqual(parent.draws, 2)
        finally:
            pocket_controller_mod.PocketResultDialog = old_dialog
            pocket_controller_mod.QInputDialog.getDouble = old_get_double

    def test_show_can_delete_single_pocket_and_export_csv(self):
        class DeleteDialog:
            def __init__(self, parent, pocket, **_kwargs):
                self.delete_requested = True

            def exec(self):
                return 1

        old_dialog = pocket_controller_mod.PocketResultDialog
        pocket_controller_mod.PocketResultDialog = DeleteDialog
        try:
            with tempfile.TemporaryDirectory() as tmp:
                parent = _Parent(Path(tmp))
                entry = parent._current_entry()
                entry.fs_pockets = [
                    {"hs_label_nearest": "Γ", "topology": "electron", "area_pct_bz": 1.0},
                    {"hs_label_nearest": "X", "topology": "hole", "area_pct_bz": 2.0},
                ]
                ctrl = PocketController(parent)

                ctrl._pocket_action("show", {"index": 0})
                out = ctrl._pocket_action("export_csv", {"path": str(Path(tmp) / "pockets.csv")})

                self.assertEqual([p["hs_label_nearest"] for p in entry.fs_pockets], ["X"])
                self.assertTrue(out.exists())
                text = out.read_text(encoding="utf-8")
                self.assertIn("index,centroid_kx", text)
                self.assertIn("hole", text)
        finally:
            pocket_controller_mod.PocketResultDialog = old_dialog

    def test_canvas_draw_pockets_adds_pickable_artists(self):
        canvas = FermiSurfaceCanvas()
        pocket = {
            "contour": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.0]],
            "centroid_kx": 0.1,
            "centroid_ky": 0.1,
            "hs_label_nearest": "Γ",
        }

        canvas.draw_pockets([pocket])

        self.assertEqual(len(canvas._pocket_artists), 2)
        self.assertEqual(getattr(canvas._pocket_artists[0], "pocket_index"), 0)


class _Parent:
    def __init__(self, folder: Path):
        self._current_path = str(folder / "fs")
        self._session = Session(folder)
        self._raw_data = _raw_fs()
        self._fs_controls = SimpleNamespace(params=lambda: FSParams(
            kx_center=0.0,
            ky_center=0.0,
            bz_shape="rectangle",
            bz_half_x=1.0,
            bz_half_y=1.0,
            bz_angle_deg=90.0,
        ))
        self.draws = 0
        self.statuses = []

    def _current_is_fs(self):
        return True

    def _current_entry(self):
        return self._session.get_or_create(self._session.key_for_path(self._current_path))

    def _draw_fs_tab(self):
        self.draws += 1

    def _status(self, text):
        self.statuses.append(str(text))


def _raw_fs():
    kx = np.linspace(-1.2, 1.2, 81)
    ky = np.linspace(-1.2, 1.2, 81)
    ev = np.array([-0.02, 0.0, 0.02])
    x, y = np.meshgrid(kx, ky)
    r = np.sqrt(x * x + y * y)
    fs = np.clip(1.0 - r / 0.6, 0.0, 1.0)
    volume = np.repeat(fs[:, :, None], ev.size, axis=2)
    return {
        "metadata": {
            "fs_data": volume,
            "fs_kx": kx,
            "fs_ky": ky,
            "fs_energy": ev,
            "fs_kind": "kxky",
        }
    }


if __name__ == "__main__":
    unittest.main()
