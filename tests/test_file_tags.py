from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from arpes.core.session import FileMeta, Session
    from arpes.ui.widgets.browsers.files import FileBrowserPanel

    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 / Qt offscreen indisponible")
class TestFileTagsBrowser(unittest.TestCase):
    _qt_app = None

    @classmethod
    def setUpClass(cls):
        cls._qt_app = QApplication.instance() or QApplication([])

    def test_browser_filters_items_by_session_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.pxt"
            b = root / "b.pxt"
            c = root / "c.pxt"
            for path in (a, b, c):
                path.write_text("dummy", encoding="utf-8")
            session = Session(root)
            session.get_or_create("a.pxt").meta = FileMeta(tags=["publi", "T-dep"])
            session.get_or_create("b.pxt").meta = FileMeta(tags=["outliers"])
            session.get_or_create("c.pxt")

            panel = FileBrowserPanel(session)
            panel.set_folder(root)
            panel._tag_filter.setText("publi")

            visible_paths = []
            for i in range(panel._list.count()):
                item = panel._list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    visible_paths.append(Path(path).name)

        self.assertEqual(visible_paths, ["a.pxt"])
        self.assertIn("publi", panel._tag_filter_model.stringList())


if __name__ == "__main__":
    unittest.main()
