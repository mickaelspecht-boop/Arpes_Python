"""Browser right-click tools: similar-parameter flags + fit-params clipboard."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    from arpes.core.session import FileMeta, Session
    from arpes.ui.widgets.browsers.files import FileBrowserPanel
    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 not available")
class TestBrowserSimilarFlag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _panel(self):
        tmp = Path(tempfile.mkdtemp())
        s = Session(folder=tmp)

        def mk(key, hv, d="GX", pol="LH"):
            e = s.get_or_create(key)
            e.meta = FileMeta(hv=hv, direction=d, polarization=pol)
            return e

        mk("CA046/ref.h5", 100)
        mk("CA046/a.h5", 100)      # same folder -> excluded
        mk("CA041/b.h5", 101)      # other folder, hv within tol -> flagged
        mk("CA052/c.h5", 120)      # hv too far -> excluded
        mk("CA041/d.h5", 100, d="GM")  # different direction -> excluded
        panel = FileBrowserPanel(s)
        panel._items_cache = [
            tmp / "CA046/ref.h5", tmp / "CA046/a.h5", tmp / "CA041/b.h5",
            tmp / "CA052/c.h5", tmp / "CA041/d.h5",
        ]
        return panel, s, tmp

    def test_flag_similar_other_folders_only(self):
        panel, s, tmp = self._panel()
        panel._browser_ctrl._flag_similar("CA046/ref.h5", str(tmp / "CA046/ref.h5"))
        self.assertEqual(panel._flagged_keys, {"CA041/b.h5"})

    def test_flag_marker_in_label(self):
        panel, s, tmp = self._panel()
        panel._flagged_keys.add("CA041/b.h5")
        label = panel._item_label(tmp / "CA041/b.h5", "loaded", "CA041/b.h5")
        self.assertIn("⚑", label)
        plain = panel._item_label(tmp / "CA052/c.h5", "loaded", "CA052/c.h5")
        self.assertNotIn("⚑", plain)

    def test_clear_flags(self):
        panel, s, tmp = self._panel()
        panel._flagged_keys.update({"CA041/b.h5"})
        panel._browser_ctrl._clear_flags()
        self.assertEqual(panel._flagged_keys, set())

    def test_copy_paste_fit_params_deepcopy(self):
        panel, s, tmp = self._panel()
        s.files["CA046/ref.h5"].fit_params.k_max = 0.77
        s.files["CA046/ref.h5"].fit_params.n_pairs = 3
        bc = panel._browser_ctrl
        bc._copy_fit_params("CA046/ref.h5")
        self.assertIsNotNone(panel._fit_params_clipboard)
        bc._paste_fit_params(["CA041/b.h5"])
        tgt = s.files["CA041/b.h5"].fit_params
        self.assertEqual(tgt.k_max, 0.77)
        self.assertEqual(tgt.n_pairs, 3)
        # Deep copy: mutating the source afterwards must not touch the target.
        s.files["CA046/ref.h5"].fit_params.k_max = 0.10
        self.assertEqual(s.files["CA041/b.h5"].fit_params.k_max, 0.77)

    def test_paste_to_multiple_flagged(self):
        panel, s, tmp = self._panel()
        s.files["CA046/ref.h5"].fit_params.k_min = -0.55
        bc = panel._browser_ctrl
        bc._copy_fit_params("CA046/ref.h5")
        panel._flagged_keys.update({"CA041/b.h5", "CA052/c.h5"})
        bc._paste_fit_params(sorted(panel._flagged_keys))
        self.assertEqual(s.files["CA041/b.h5"].fit_params.k_min, -0.55)
        self.assertEqual(s.files["CA052/c.h5"].fit_params.k_min, -0.55)


if __name__ == "__main__":
    unittest.main()
