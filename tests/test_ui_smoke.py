"""Smoke test UI : instancie ArpesExplorer headless, vérifie wiring de base.

Sécurise contre les régressions silencieuses du type "j'ai oublié d'importer
QLabel" ou "le proxy dispatch ne résout pas une méthode" — bugs qui ne
sortaient pas via les tests unitaires fonctionnels (cf bugs ν corrigés en ο).
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    from arpes.app import ArpesExplorer
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
        self.assertEqual(win._tabs.count(), 5)
        self.assertIsNotNone(win.ap, "arpes_plots doit être chargé")

    def test_controllers_wired(self):
        win = self._make_window()
        for attr in (
            "_logbook_ctrl", "_load_ctrl", "_plot_ctrl",
            "_gamma_ctrl", "_norm_ctrl", "_fs_ctrl",
            "_interaction_ctrl", "_fit_runner_ctrl", "_kz_ctrl",
            "_theory_overlay_ctrl",
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
                     "_mdc_edc", "_tabs", "_kz_canvas", "_kz_controls"):
            self.assertTrue(hasattr(win, attr), f"widget {attr} non construit")
        self.assertTrue(hasattr(win._params, "_theory_widget"))


if __name__ == "__main__":
    unittest.main()
