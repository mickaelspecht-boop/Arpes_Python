"""Smoke test UI : instancie ArpesExplorer headless, vérifie wiring de base.

Sécurise contre les régressions silencieuses du type "j'ai oublié d'importer
QLabel" ou "le proxy dispatch ne résout pas une méthode" — bugs qui ne
sortaient pas via les tests unitaires fonctionnels (cf bugs ν corrigés en ο).
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

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


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 / Qt offscreen indisponible")
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
        self.assertEqual(win._tabs.count(), 7)
        self.assertEqual(
            [win._tabs.tabText(i) for i in range(win._tabs.count())],
            ["BM", "MDC Fit", "Résultats", "FS", "KZ", "Notes", "Aide"],
        )
        fs_tabs = win._tabs.widget(3)
        self.assertEqual(
            [fs_tabs.tabText(i) for i in range(fs_tabs.count())],
            ["Carte FS", "Compare pol"],
        )
        self.assertIsNotNone(win.ap, "arpes_plots doit être chargé")

    def test_tab_right_panel_mapping(self):
        win = self._make_window()
        expected = {
            0: 0,  # BM controls
            1: 0,  # MDC controls
            2: 0,  # Results: no dedicated right panel
            3: 1,  # FS controls
            4: 2,  # KZ controls
            5: 0,  # Notes
            6: 0,  # Help
        }
        for index, right_index in expected.items():
            win._on_tab_changed(index)
            self.assertEqual(
                win._right_stack.currentIndex(),
                right_index,
                f"onglet {index} ({win._tabs.tabText(index)})",
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
                f"controller {attr} manquant sur ArpesExplorer",
            )

    def test_proxy_dispatch_resolves_every_entry(self):
        """`_PROXY_MAP` doit pointer vers une vraie méthode du controller."""
        win = self._make_window()
        for name, ctrl_attr in win._PROXY_MAP.items():
            ctrl = getattr(win, ctrl_attr, None)
            self.assertIsNotNone(ctrl, f"controller {ctrl_attr} introuvable")
            self.assertTrue(
                callable(getattr(ctrl, name, None)),
                f"{ctrl_attr}.{name} n'existe pas ou n'est pas callable",
            )
            # Et la résolution via __getattr__ doit donner le même bound method
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
            self.assertTrue(hasattr(win, attr), f"widget {attr} non construit")
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
        # A : reset BM branché sur le canvas + proxy résout
        self.assertTrue(callable(getattr(win, "_reset_bm_view")))
        self.assertEqual(win._bm_canvas.reset_callback, win._reset_bm_view)
        # _reset_bm_view sans données : no-op sûr
        win._reset_bm_view()
        # C : fast path overlays-only sans données : no-op sûr
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

        self.assertEqual(parent._fs_controls.center, (0.23, -0.17))
        entry = parent._current_entry()
        self.assertAlmostEqual(entry.fs_center_kx, 0.23)
        self.assertAlmostEqual(entry.fs_center_ky, -0.17)
        self.assertFalse(parent._raw_data["metadata"].get("fs_gamma_axis_centered", False))
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

        self.assertEqual(parent._fs_controls.center, (0.30, 0.05))
        self.assertAlmostEqual(parent._sp_cx.value_seen, 0.30)
        self.assertAlmostEqual(parent._current_entry().fit_params.center_init, 0.30)
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

        # P1.4 / P2 : la branche FS shifte maintenant l'axe k AUSSI pour
        # éviter le drift au reload. L'axe brut (kpar, fs_kx) est translaté
        # de -kx_ref, fs_gamma_axis_centered devient True, et sp_cx affiche 0
        # (centre fit relatif au nouvel axe centré sur Γ).
        np.testing.assert_allclose(parent._raw_data["kpar"], [-1.25, -0.25, 0.75])
        np.testing.assert_allclose(parent._raw_data["metadata"]["fs_kx"], [-1.25, -0.25, 0.75])
        self.assertTrue(parent._raw_data["metadata"].get("fs_gamma_axis_centered", False))
        self.assertEqual(parent._fs_controls.center, (0.25, -0.10))
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
