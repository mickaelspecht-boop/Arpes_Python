"""UI smoke test: instantiates ArpesExplorer headless and checks basic wiring.

Protects against silent regressions such as "forgot to import QLabel" or "proxy
dispatch does not resolve a method" — bugs that did not surface through
functional unit tests (see ν bugs fixed in ο).
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    from arpes.app import ArpesExplorer
    from arpes.core.session import Session
    from arpes.ui.controllers.gamma_controller import GammaController
    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


class _FakeMessageBoxBase:
    class ButtonRole:
        AcceptRole = object()
        RejectRole = object()

    def __init__(self, parent=None):
        self._buttons = []
        self._clicked = None

    def setWindowTitle(self, title):
        pass

    def setText(self, text):
        self.text = text

    def addButton(self, text, role):
        btn = object()
        self._buttons.append((btn, text, role))
        return btn

    def setDefaultButton(self, button):
        pass

    def clickedButton(self):
        return self._clicked


class _FakeMessageBoxReject(_FakeMessageBoxBase):
    def exec(self):
        self._clicked = self._buttons[-1][0]
        return 0


class _FakeMessageBoxAccept(_FakeMessageBoxBase):
    def exec(self):
        self._clicked = self._buttons[0][0]
        return 0


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 / Qt offscreen unavailable")
class TestUiSmoke(unittest.TestCase):
    _qt_app = None

    @classmethod
    def setUpClass(cls):
        cls._qt_app = QApplication.instance() or QApplication([])

    def _make_window(self) -> "ArpesExplorer":
        return ArpesExplorer()

    def test_window_instantiates(self):
        win = self._make_window()
        self.assertIsNotNone(win)
        from arpes.ui.tab_index import IDX_FS, TAB_TITLES
        self.assertEqual(win._tabs.count(), len(TAB_TITLES))
        self.assertEqual(
            [win._tabs.tabText(i) for i in range(win._tabs.count())],
            TAB_TITLES,
        )
        fs_tabs = win._tabs.widget(IDX_FS)
        self.assertEqual(
            [fs_tabs.tabText(i) for i in range(fs_tabs.count())],
            ["FS map"],  # "Compare pol" tab removed (commit 4d98163)
        )
        self.assertIsNotNone(win.ap, "arpes_plots must be loaded")

    def test_tab_right_panel_mapping(self):
        win = self._make_window()
        from arpes.ui import tab_index as ti
        expected = {
            ti.IDX_BM: 0,           # BM controls
            ti.IDX_MDC: 0,          # MDC controls
            ti.IDX_RESULTS: 0,      # Results: no dedicated right panel
            ti.IDX_FS: 1,           # FS controls
            ti.IDX_FS_EXPLORER: 0,  # FS Explorer: own control bar
            ti.IDX_KZ: 2,           # KZ controls
            ti.IDX_NOTES: 0,
            ti.IDX_HELP: 0,
        }
        for index, right_index in expected.items():
            win._on_tab_changed(index)
            self.assertEqual(
                win._right_stack.currentIndex(),
                right_index,
                f"tab {index} ({win._tabs.tabText(index)})",
            )

    def test_controllers_wired(self):
        win = self._make_window()
        for attr in (
            "_logbook_ctrl", "_load_ctrl", "_plot_ctrl",
            "_gamma_ctrl", "_norm_ctrl", "_fs_ctrl",
            "_interaction_ctrl", "_fit_runner_ctrl", "_kz_ctrl",
            "_theory_overlay_ctrl", "_distortion_ctrl",
        ):
            self.assertTrue(
                hasattr(win, attr),
                f"controller {attr} missing on ArpesExplorer",
            )

    def test_fs_explorer_no_data_no_crash(self):
        win = self._make_window()
        win._fs_explorer_action("draw", {})  # placeholder, no exception

    def test_fs_explorer_axes_raw_view_refuses(self):
        win = self._make_window()
        win._raw_data = {"metadata": {"axes_raw_view": True}}
        statuses = []
        win._status = statuses.append
        win._fs_explorer_action("draw", {})
        self.assertTrue(any("raw axes" in s for s in statuses))

    def test_fs_explorer_draws_synthetic_volume(self):
        win = self._make_window()
        kx = np.linspace(-1, 1, 12)
        ky = np.linspace(-0.5, 0.5, 8)
        e_ax = np.linspace(-0.3, 0.1, 5)
        vol = np.random.default_rng(0).random((8, 12, 5)).astype(np.float32)
        win._raw_data = {"metadata": {
            "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": e_ax,
            "fs_kind": "kxky",
        }}
        win._fs_explorer_action("draw", {})
        self.assertIsNotNone(win._fs_explorer_map._mesh)
        # native mode snaps + redraws without crash
        win._fs_explorer_action("mode_changed", {"mode": "native"})
        # animation step advances the line without crash, then stops cleanly
        win._fs_explorer_action("play_toggle", {"play": True})
        win._fs_explorer_ctrl._animation_step()
        win._fs_explorer_action("play_toggle", {"play": False})
        self.assertFalse(win._fs_explorer_ctrl._anim_timer.isActive())

    def _fs_explorer_window_with_volume(self):
        win = self._make_window()
        kx = np.linspace(-1, 1, 12)
        ky = np.linspace(-0.5, 0.5, 8)
        e_ax = np.linspace(-0.3, 0.1, 5)
        vol = np.random.default_rng(0).random((8, 12, 5)).astype(np.float32)
        win._raw_data = {"metadata": {
            "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": e_ax,
            "fs_kind": "kxky",
        }}
        win._fs_explorer_action("draw", {})
        return win

    def test_fs_explorer_line_never_autoscales_map(self):
        win = self._fs_explorer_window_with_volume()
        ax = win._fs_explorer_map.canvas.ax
        xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
        win._fs_explorer_map.set_line(5.0, 5.0, 30.0, 10.0)  # far outside
        self.assertEqual(ax.get_xlim(), xlim0)
        self.assertEqual(ax.get_ylim(), ylim0)

    def test_fs_explorer_drag_throttles_cut_redraw(self):
        win = self._fs_explorer_window_with_volume()
        ctrl = win._fs_explorer_ctrl
        win._fs_explorer_action("drag_state", {"dragging": True})
        win._fs_explorer_action(
            "line_changed", {"cx": 0.1, "cy": 0.0, "angle": 10.0, "length": 1.0})
        self.assertTrue(ctrl._throttle.isActive())
        self.assertIsNotNone(ctrl._pending_line)
        # release flushes the pending line at full resolution
        win._fs_explorer_action("drag_state", {"dragging": False})
        self.assertIsNone(ctrl._pending_line)
        self.assertEqual(ctrl._line[2], 10.0)

    def test_fs_explorer_cut_mesh_reused_during_drag(self):
        win = self._fs_explorer_window_with_volume()
        cut_view = win._fs_explorer_cut
        mesh0 = cut_view._mesh
        self.assertIsNotNone(mesh0)
        win._fs_explorer_action(
            "line_changed", {"cx": 0.05, "cy": 0.0, "angle": 0.0,
                             "length": win._fs_explorer_ctrl._line[3]})
        self.assertIs(cut_view._mesh, mesh0)  # updated in place, not rebuilt

    def test_fs_explorer_sweep_is_perpendicular_at_90deg(self):
        win = self._fs_explorer_window_with_volume()
        ctrl = win._fs_explorer_ctrl
        ctrl._line = [0.0, 0.0, 90.0, 0.6]   # vertical line
        win._fs_explorer_action("play_toggle", {"play": True})
        ctrl._animation_step()
        win._fs_explorer_action("play_toggle", {"play": False})
        self.assertAlmostEqual(ctrl._line[1], 0.0, places=12)  # no vertical drift
        self.assertGreater(abs(ctrl._line[0]), 1e-3)           # sweeps horizontally

    def test_fs_explorer_spin_click_is_deferred_not_blocking(self):
        win = self._fs_explorer_window_with_volume()
        ctrl = win._fs_explorer_ctrl
        win._fs_explorer_action("line_params", {"angle": 30.0, "length": 1.0})
        self.assertEqual(ctrl._line[2], 30.0)     # geometry applied at once
        self.assertTrue(ctrl._settle.isActive())  # full-res deferred to idle

    def test_fs_explorer_non_kxky_forces_native(self):
        win = self._make_window()
        kx = np.linspace(-1, 1, 12)
        ky = np.arange(8, dtype=float)
        e_ax = np.linspace(-0.3, 0.1, 5)
        vol = np.zeros((8, 12, 5), dtype=np.float32)
        win._raw_data = {"metadata": {
            "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": e_ax,
            "fs_kind": "scan-kx-energy",
        }}
        win._fs_explorer_action("draw", {})
        self.assertEqual(win._fs_explorer_ctrl._mode, "native")

    def test_mp_lattice_fetch_signal_is_wired_to_controller(self):
        win = self._make_window()
        self.assertTrue(hasattr(win._fs_controls, "mp_lattice_fetch_requested"))
        self.assertTrue(callable(getattr(win, "_on_mp_lattice_fetch")))

    def test_restore_fs_crystal_settings_from_entry(self):
        win = self._make_window()
        entry = win._session.get_or_create("FS1")
        entry.fs_v0 = 15.5
        entry.fs_kz_plane = "Z"
        entry.fs_phi_c_deg = 12.0
        entry.fs_bz_crystal_visible = True
        entry.fs_hs_crystal_visible = True
        entry.fs_lattice = {"mp_id": "mp-test", "a": 4.0}

        win._restore_fs_crystal_settings_from_entry(entry)

        self.assertAlmostEqual(win._fs_controls.sp_v0.value(), 15.5)
        self.assertEqual(win._fs_controls.cmb_kz_plane.currentText(), "Z")
        self.assertAlmostEqual(win._fs_controls.sp_phi_c.value(), 12.0)
        self.assertTrue(win._fs_controls.chk_bz_xtal.isChecked())
        self.assertTrue(win._fs_controls.chk_hs_xtal.isChecked())
        self.assertEqual(win._fs_controls.ed_mp_id.text(), "mp-test")

    def test_bz_crystal_overlay_without_mp_lattice_draws_no_polygon(self):
        from arpes.physics.fs import FSParams
        from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas

        canvas = FermiSurfaceCanvas()
        p = FSParams(overlay_bz_crystal=True, overlay_hs_crystal=True)
        canvas._overlay_bz_crystal(p, {"metadata": {}})

        self.assertEqual(len(canvas.ax.lines), 0)
        self.assertGreaterEqual(len(canvas.ax.texts), 1)

    def test_bz_mp_mismatch_disables_overlay_without_override(self):
        win = self._make_window()
        win._current_path = "/tmp/fs-mp"
        entry = win._current_entry()
        entry.fs_lattice = {"mp_id": "mp-x", "a": 3.0, "bravais": "tetragonal"}
        win._fs_controls.sp_a.setValue(4.0)
        win._fs_controls.chk_bz_xtal.blockSignals(True)
        win._fs_controls.chk_bz_xtal.setChecked(True)
        win._fs_controls.chk_bz_xtal.blockSignals(False)

        with patch("PyQt6.QtWidgets.QMessageBox", _FakeMessageBoxReject):
            ok = win._fs_ctrl._check_bz_crystal_consistency()

        self.assertFalse(ok)
        self.assertFalse(win._fs_controls.chk_bz_xtal.isChecked())
        self.assertFalse(entry.fs_bz_crystal_visible)
        self.assertFalse(entry.fs_bz_crystal_force_override)

    def test_bz_mp_mismatch_can_be_forced(self):
        win = self._make_window()
        win._current_path = "/tmp/fs-mp-force"
        entry = win._current_entry()
        entry.fs_lattice = {"mp_id": "mp-x", "a": 3.0, "bravais": "hexagonal"}
        win._fs_controls.sp_a.setValue(4.0)
        win._fs_controls.chk_bz_xtal.blockSignals(True)
        win._fs_controls.chk_bz_xtal.setChecked(True)
        win._fs_controls.chk_bz_xtal.blockSignals(False)

        with patch("PyQt6.QtWidgets.QMessageBox", _FakeMessageBoxAccept):
            ok = win._fs_ctrl._check_bz_crystal_consistency()

        self.assertTrue(ok)
        self.assertTrue(entry.fs_bz_crystal_force_override)

    def test_proxy_dispatch_resolves_every_entry(self):
        """`_PROXY_MAP` must point to a real controller method."""
        win = self._make_window()
        for name, ctrl_attr in win._PROXY_MAP.items():
            ctrl = getattr(win, ctrl_attr, None)
            self.assertIsNotNone(ctrl, f"controller {ctrl_attr} not found")
            self.assertTrue(
                callable(getattr(ctrl, name, None)),
                f"{ctrl_attr}.{name} does not exist or is not callable",
            )
            # Resolution through __getattr__ must return the same bound method.
            bound = getattr(win, name)
            self.assertTrue(callable(bound))
            self.assertEqual(
                bound.__qualname__,
                f"{type(ctrl).__name__}.{name}",
            )

    def test_widgets_built(self):
        win = self._make_window()
        for attr in ("_params", "_results", "_browser", "_bm_canvas",
                     "_mdc_edc", "_tabs", "_kz_canvas", "_kz_controls",
                     "_help_panel"):
            self.assertTrue(hasattr(win, attr), f"widget {attr} not built")
        self.assertTrue(hasattr(win._params, "_theory_widget"))
        self.assertTrue(hasattr(win._params, "btn_theory_pick_bands"))
        self.assertTrue(hasattr(win._params, "_distortion_widget"))
        self.assertTrue(callable(getattr(win._params, "bm_distortion_params")))
        cfg = win._params.bm_distortion_params()
        self.assertIn("trapezoid", cfg)
        self.assertIn("parabola", cfg)

    def test_distortion_coupled_slopes_are_bidirectional(self):
        win = self._make_window()
        p = win._params

        p.rb_distortion_trap_sym.setChecked(True)
        p.sp_distortion_slope_l.setValue(0.120)
        self.assertAlmostEqual(p.sp_distortion_slope_r.value(), 0.120)
        p.sp_distortion_slope_r.setValue(-0.080)
        self.assertAlmostEqual(p.sp_distortion_slope_l.value(), -0.080)

        p.rb_distortion_trap_anti.setChecked(True)
        p.sp_distortion_slope_l.setValue(0.070)
        self.assertAlmostEqual(p.sp_distortion_slope_r.value(), -0.070)
        p.sp_distortion_slope_r.setValue(0.040)
        self.assertAlmostEqual(p.sp_distortion_slope_l.value(), -0.040)

        p.rb_distortion_trap_free.setChecked(True)
        p.sp_distortion_slope_l.setValue(0.010)
        p.sp_distortion_slope_r.setValue(0.090)
        self.assertAlmostEqual(p.sp_distortion_slope_l.value(), 0.010)
        self.assertAlmostEqual(p.sp_distortion_slope_r.value(), 0.090)

    def test_reset_view_and_fast_path_wired(self):
        win = self._make_window()
        # A: BM reset wired to the canvas + proxy resolves.
        self.assertTrue(callable(getattr(win, "_reset_bm_view")))
        self.assertEqual(win._bm_canvas.reset_callback, win._reset_bm_view)
        # _reset_bm_view without data: safe no-op.
        win._reset_bm_view()
        # C: overlays-only fast path without data: safe no-op.
        win._draw_bm(overlays_only=True)
        win._draw_current_view(overlays_only=True)

    def test_fs_gamma_actions_keep_detected_center_in_panel(self):
        class FakeControls:
            def __init__(self):
                self.center = (0.0, 0.0)
                self.lbl_info = SimpleNamespace(setText=lambda text: None)

            def params(self):
                return SimpleNamespace(kx_center=self.center[0], ky_center=self.center[1])

            def set_center(self, kx, ky):
                self.center = (float(kx), float(ky))

        class FakeCanvas:
            def detect_gamma(self, raw_data, params):
                return {"kx": 0.23, "ky": -0.17, "gamma_kx_list": [1, 2, 3], "gamma_ky_list": [1, 2, 3]}

        class Parent:
            def __init__(self):
                self._raw_data = {
                    "path": "/tmp/fs",
                    "hv": 60.0,
                    "kpar": [-1.0, 0.0, 1.0],
                    "metadata": {
                        "fs_data": object(),
                        "fs_kx": [-1.0, 0.0, 1.0],
                        "fs_ky": [-1.0, 0.0, 1.0],
                        "fs_kind": "kxky",
                    },
                }
                self._current_path = "/tmp/fs"
                self._session = Session(Path("/tmp"))
                self._fs_controls = FakeControls()
                self._fs_canvas = FakeCanvas()
                self._params = SimpleNamespace(
                    sp_hv=SimpleNamespace(value=lambda: 60.0),
                    sp_phi=SimpleNamespace(value=lambda: 4.5),
                    mark_action_done=lambda text: None,
                )
                self._draws = 0

            def _current_entry(self):
                return self._session.get_or_create(self._session.key_for_path(self._current_path))

            def _same_path(self, a, b):
                return a == b

            def _current_is_fs(self):
                return True

            def _draw_fs_tab(self):
                self._draws += 1

            def _status(self, text):
                pass

        parent = Parent()
        ctrl = GammaController(parent)

        ctrl._detect_fs_gamma()

        self.assertEqual(parent._fs_controls.center, (0.0, 0.0))
        entry = parent._current_entry()
        self.assertAlmostEqual(entry.fs_center_kx, 0.0)
        self.assertAlmostEqual(entry.fs_center_ky, 0.0)
        self.assertTrue(parent._raw_data["metadata"].get("fs_gamma_axis_centered", False))
        self.assertEqual(parent._draws, 1)

    def test_manual_fs_gamma_updates_bm_center_parameter(self):
        class FakeSpin:
            def __init__(self):
                self.value_seen = None

            def setValue(self, value):
                self.value_seen = float(value)

        class FakeControls:
            def __init__(self):
                self.center = (0.10, -0.05)
                self.lbl_info = SimpleNamespace(setText=lambda text: None)

            def params(self):
                return SimpleNamespace(kx_center=self.center[0], ky_center=self.center[1])

            def set_center(self, kx, ky):
                self.center = (float(kx), float(ky))

        class FakeCanvas:
            ax = object()

        class Parent:
            def __init__(self):
                self._raw_data = {
                    "path": "/tmp/fs-manual",
                    "hv": 60.0,
                    "kpar": [-1.0, 0.0, 1.0],
                    "metadata": {
                        "fs_data": object(),
                        "fs_kx": [-1.0, 0.0, 1.0],
                        "fs_ky": [-1.0, 0.0, 1.0],
                        "fs_kind": "kxky",
                    },
                }
                self._current_path = "/tmp/fs-manual"
                self._session = Session(Path("/tmp"))
                self._fs_pick_center_active = True
                self._fs_controls = FakeControls()
                self._fs_canvas = FakeCanvas()
                self._sp_cx = FakeSpin()
                self._params = SimpleNamespace(
                    sp_hv=SimpleNamespace(value=lambda: 60.0),
                    sp_phi=SimpleNamespace(value=lambda: 4.5),
                    sp_cx=self._sp_cx,
                    mark_action_done=lambda text: None,
                )
                self._draws = 0

            def _current_entry(self):
                return self._session.get_or_create(self._session.key_for_path(self._current_path))

            def _current_is_fs(self):
                return True

            def _draw_fs_tab(self):
                self._draws += 1

            def _status(self, text):
                pass

        parent = Parent()
        ctrl = GammaController(parent)
        event = SimpleNamespace(inaxes=parent._fs_canvas.ax, xdata=0.20, ydata=0.10)

        ctrl._on_fs_map_click(event)

        self.assertEqual(parent._fs_controls.center, (0.0, 0.0))
        self.assertAlmostEqual(parent._sp_cx.value_seen, 0.0)
        self.assertAlmostEqual(parent._current_entry().fit_params.center_init, 0.0)
        self.assertEqual(parent._draws, 1)

    def test_stored_gamma_on_fs_does_not_shift_average_bm_axis(self):
        class FakeSpin:
            def __init__(self):
                self.value_seen = None

            def setValue(self, value):
                self.value_seen = float(value)

        class FakeControls:
            def __init__(self):
                self.center = (0.0, 0.0)

            def set_center(self, kx, ky):
                self.center = (float(kx), float(ky))

        class Parent:
            def __init__(self):
                self._raw_data = {
                    "path": "/tmp/fs-stored",
                    "hv": 60.0,
                    "kpar": np.array([-1.0, 0.0, 1.0]),
                    "metadata": {
                        "fs_data": object(),
                        "fs_kx": np.array([-1.0, 0.0, 1.0]),
                        "fs_ky": np.array([-0.5, 0.5]),
                        "fs_kind": "kxky",
                    },
                }
                self._current_path = "/tmp/fs-stored"
                self._session = Session(Path("/tmp"))
                self._session.gamma_reference = {
                    "kx": 0.25,
                    "ky": -0.10,
                    "path": "/tmp/fs-stored",
                    "source": "fs_manual",
                    "polar": 0.0,
                    "polar_already_applied_to_kx": False,
                }
                self._fs_controls = FakeControls()
                self._params = SimpleNamespace(sp_cx=FakeSpin())

            def _current_entry(self):
                return self._session.get_or_create(self._session.key_for_path(self._current_path))

            def _same_path(self, a, b):
                return a == b

            def _status(self, text):
                pass

        parent = Parent()
        ctrl = GammaController(parent)

        ctrl._apply_stored_gamma_to_current_file(save_entry=True)

        # P1.4 / P2: the FS branch now ALSO shifts the k axis to avoid drift on
        # reload. The raw axis (kpar, fs_kx) is translated by -kx_ref,
        # fs_gamma_axis_centered becomes True, and sp_cx shows 0 (fit center
        # relative to the new Γ-centered axis).
        np.testing.assert_allclose(parent._raw_data["kpar"], [-1.25, -0.25, 0.75])
        np.testing.assert_allclose(parent._raw_data["metadata"]["fs_kx"], [-1.25, -0.25, 0.75])
        self.assertTrue(parent._raw_data["metadata"].get("fs_gamma_axis_centered", False))
        self.assertEqual(parent._fs_controls.center, (0.0, 0.0))
        self.assertAlmostEqual(parent._params.sp_cx.value_seen, 0.0)

    def test_stored_gamma_on_bm_centers_display_axis(self):
        class FakeSpin:
            def __init__(self):
                self.value_seen = None

            def setValue(self, value):
                self.value_seen = float(value)

        class Parent:
            def __init__(self):
                self._raw_data = {
                    "path": "/tmp/bm04",
                    "hv": 66.0,
                    "kpar": np.array([-0.86, 0.0, 3.05]),
                    "metadata": {
                        "scan_kind": "BM",
                        "polar": -13.6,
                        "polar_already_applied_to_kx": True,
                    },
                }
                self._current_path = "/tmp/bm04"
                self._session = Session(Path("/tmp"))
                self._session.gamma_reference = {
                    "kx": 0.12,
                    "ky": 0.0,
                    "path": "/tmp/fs3",
                    "source": "fs_auto",
                    "polar": 0.0,
                    "azi": 0.0,
                    "polar_already_applied_to_kx": True,
                }
                self._params = SimpleNamespace(
                    sp_cx=FakeSpin(),
                    sp_phi=SimpleNamespace(value=lambda: 4.031),
                )

            def _current_entry(self):
                return self._session.get_or_create(self._session.key_for_path(self._current_path))

            def _status(self, text):
                pass

        parent = Parent()
        ctrl = GammaController(parent)

        ctrl._apply_stored_gamma_to_current_file(save_entry=True)

        np.testing.assert_allclose(parent._raw_data["kpar"], [-0.98, -0.12, 2.93])
        self.assertTrue(parent._raw_data["metadata"].get("bm_gamma_axis_centered", False))
        self.assertAlmostEqual(parent._params.sp_cx.value_seen, 0.0)
        self.assertAlmostEqual(parent._current_entry().fit_params.center_init, 0.0)


if __name__ == "__main__":
    unittest.main()
