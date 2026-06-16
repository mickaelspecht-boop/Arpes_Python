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


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 unavailable")
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
                # Relaxed bound: reduced smoothing (guard σ·dk≤0.5·kF) → contour
                # slightly tighter than with the old σ_x=4.0. Still a real pocket.
                self.assertGreater(entry.fs_pockets[0]["area_pct_bz"], 0.5)
                self.assertGreater(len(entry.fs_pockets[0]["contour"]), 10)
                self.assertEqual(entry.fs_pockets[0]["processing"]["quality"], "Stable")
                self.assertEqual(entry.fs_pockets[0]["processing"]["contour_window"], 13)
                self.assertEqual(parent.draws, 1)
        finally:
            pocket_controller_mod.PocketResultDialog = old_dialog

    def test_manual_contour_persists_marked_pocket(self):
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
                theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
                points = np.column_stack([0.42 * np.cos(theta), 0.42 * np.sin(theta)])

                pocket = ctrl._pocket_action(
                    "manual_contour", {"points": points.tolist(), "snap": True}
                )

                self.assertIsNotNone(pocket)
                self.assertEqual(pocket["algo"], "manual_contour")
                self.assertEqual(pocket["processing"]["n_points"], 16)
                self.assertEqual(len(parent._current_entry().fs_pockets), 1)
                self.assertGreater(pocket["area_pct_bz"], 5.0)
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
        pocket_controller_mod.PocketResultDialog = FakeDialog
        try:
            with tempfile.TemporaryDirectory() as tmp:
                parent = _Parent(Path(tmp))
                parent._current_entry().fs_lattice = {"mp_id": "mp-test"}
                ctrl = PocketController(parent)

                ctrl._pocket_action("preview_start", {"kx": 0.0, "ky": 0.0})
                parent._fs_controls.sp_pocket_level.setValue(0.5)
                # New contract: Validate runs the radial-MDC fit; on failure
                # the pocket is NOT created and the preview is kept (no silent
                # ISO fallback). Quick ISO stays the explicit no-fit path.
                pocket = ctrl._pocket_action("preview_validate", {})
                if pocket is None:
                    self.assertIsNotNone(ctrl._preview_seed_plot)  # preview kept
                    ctrl._pocket_action("preview_cancel", {})
                    pocket = ctrl._pocket_action(
                        "characterize", {"kx": 0.0, "ky": 0.0, "level": 0.5})
                    self.assertIsNotNone(pocket)
                    self.assertEqual(pocket["level"], 0.5)
                    self.assertEqual(pocket["mp_label"], "mp-test:Γ")
                shown = ctrl._pocket_action("show", {"index": 0})
                ctrl._pocket_action("clear", {})

                self.assertIsNotNone(pocket)
                self.assertEqual(shown["topology"], "electron")
                self.assertEqual(parent._current_entry().fs_pockets, [])
        finally:
            pocket_controller_mod.PocketResultDialog = old_dialog

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

    def test_canvas_draw_pockets_uses_points_and_distinct_colors(self):
        canvas = FermiSurfaceCanvas()
        pockets = [
            {
                "contour": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.0]],
                "centroid_kx": 0.1,
                "centroid_ky": 0.1,
                "hs_label_nearest": "Γ",
            },
            {
                "contour": [[0.5, 0.5], [0.7, 0.5], [0.7, 0.7], [0.5, 0.5]],
                "centroid_kx": 0.6,
                "centroid_ky": 0.6,
                "hs_label_nearest": "X",
            },
        ]

        canvas.draw_pockets(pockets)

        first, second = canvas._pocket_artists[0], canvas._pocket_artists[2]
        self.assertEqual(first.__class__.__name__, "PathCollection")
        self.assertEqual(second.__class__.__name__, "PathCollection")
        self.assertNotEqual(
            tuple(first.get_facecolors()[0]),
            tuple(second.get_facecolors()[0]),
        )


class _Spin:
    def __init__(self, value: float = 0.0):
        self._value = float(value)

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        self._value = float(value)

    def blockSignals(self, _blocked: bool) -> None:
        return None


class _Chk:
    def __init__(self, checked: bool = False):
        self._checked = bool(checked)

    def setChecked(self, checked: bool) -> None:
        self._checked = bool(checked)

    def isChecked(self) -> bool:
        return self._checked


class _FakeCanvas:
    """No-op stand-in for FermiSurfaceCanvas used by the preview path."""

    def __init__(self):
        self.bar_calls: list[tuple] = []  # (visible, level, lo, hi)

    def draw_pocket_preview(self, *_a, **_k):
        return None

    def clear_pocket_preview(self, *_a, **_k):
        return None

    def draw_pockets(self, *_a, **_k):
        return None

    def set_pocket_bar_state(self, visible, level=None, lo=None, hi=None):
        self.bar_calls.append((bool(visible), level, lo, hi))


class _Parent:
    def __init__(self, folder: Path):
        self._current_path = str(folder / "fs")
        self._session = Session(folder)
        self._raw_data = _raw_fs()
        self._fs_canvas = _FakeCanvas()
        self._fs_controls = SimpleNamespace(params=lambda: FSParams(
            kx_center=0.0,
            ky_center=0.0,
            bz_shape="rectangle",
            bz_half_x=1.0,
            bz_half_y=1.0,
            bz_angle_deg=90.0,
        ), pocket_settings=lambda: {
            "quality": "Stable",
            "smooth_sigma_y": 1.0,
            "smooth_sigma_x": 1.5,  # σ·dk ≤ 0.5·kF (over-smoothing guard, pocket_quality)
            "contour_window": 13,
            "simplify_step": 0.025,
            "min_area_pct_bz": 0.2,
            "level": None,
        }, sp_pocket_level=_Spin(0.0), chk_pocket_level_manual=_Chk(False))
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


class TestPreviewActionBar(unittest.TestCase):
    """Inline Level/Validate/Cancel bar lifecycle (council 2026-06-10)."""

    def _ctrl(self, tmp):
        parent = _Parent(Path(tmp))
        return parent, PocketController(parent)

    def test_preview_start_shows_bar_with_dynamic_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, ctrl = self._ctrl(tmp)
            ctrl._pocket_action("preview_start", {"kx": 0.0, "ky": 0.0})
            shows = [c for c in parent._fs_canvas.bar_calls if c[0]]
            self.assertTrue(shows)
            _, level, lo, hi = shows[-1]
            self.assertIsNotNone(level)
            self.assertLess(lo, hi)  # calibrated to the map, not fixed 0-1

    def test_cancel_hides_bar(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, ctrl = self._ctrl(tmp)
            ctrl._pocket_action("preview_start", {"kx": 0.0, "ky": 0.0})
            ctrl._pocket_action("preview_cancel", {})
            self.assertFalse(parent._fs_canvas.bar_calls[-1][0])  # hidden

    def test_mdc_failure_keeps_bar_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, ctrl = self._ctrl(tmp)
            ctrl._pocket_action("preview_start", {"kx": 0.0, "ky": 0.0})
            n_calls = len(parent._fs_canvas.bar_calls)
            result = ctrl._pocket_action("preview_validate", {})
            if result is None:  # MDC failed on this synthetic map
                # No hide call may have been issued on the failure path.
                new = parent._fs_canvas.bar_calls[n_calls:]
                self.assertTrue(all(c[0] for c in new) or not new)
                self.assertIsNotNone(ctrl._preview_seed_plot)  # preview kept


if __name__ == "__main__":
    unittest.main()
