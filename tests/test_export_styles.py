from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arpes.io.export_styles import PRESETS, apply_preset, savefig_with_preset


class TestExportStyles(unittest.TestCase):
    def test_apply_preset_sets_rcparams_inside_context(self):
        before = plt.rcParams["font.size"]
        with apply_preset("publication_npj"):
            self.assertEqual(plt.rcParams["font.size"], PRESETS["publication_npj"]["font.size"])
            self.assertFalse(plt.rcParams["text.usetex"])
        self.assertEqual(plt.rcParams["font.size"], before)

    def test_unknown_preset_falls_back_to_default(self):
        with apply_preset("missing"):
            self.assertEqual(plt.rcParams["savefig.dpi"], PRESETS["default"]["savefig.dpi"])

    def test_savefig_with_preset_writes_valid_png(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "figure.png"
            savefig_with_preset(fig, str(path), "publication_prb")
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            img = plt.imread(path)
            self.assertGreater(img.size, 0)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
