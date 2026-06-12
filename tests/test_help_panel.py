from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    from arpes.ui.widgets.help_panel import HelpPanel

    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 / Qt offscreen unavailable")
class TestHelpPanel(unittest.TestCase):
    _qt_app = None

    @classmethod
    def setUpClass(cls):
        cls._qt_app = QApplication.instance() or QApplication([])

    def test_help_panel_loads_markdown_sections(self):
        panel = HelpPanel()

        self.assertEqual(panel._index.count(), 4)
        self.assertIn("Workflow", panel._viewer.toPlainText())

        panel._index.setCurrentRow(1)
        self.assertIn("Features", panel._viewer.toPlainText())

        panel._index.setCurrentRow(2)
        self.assertIn("Shortcuts", panel._viewer.toPlainText())

    def test_help_panel_has_missing_file_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = HelpPanel(Path(tmp))

            self.assertIn("Documentation unavailable", panel._viewer.toPlainText())


if __name__ == "__main__":
    unittest.main()
