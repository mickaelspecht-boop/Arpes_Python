from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arpes.io.export_styles import (
    LIGHT_BG_PRESETS,
    PRESETS,
    apply_preset,
    figure_size_mm,
    savefig_with_preset,
)


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


class TestP4ExportFeatures(unittest.TestCase):
    def test_nature_science_presets_exist_and_vectorial(self):
        for name in ("publication_nature", "publication_science"):
            self.assertIn(name, PRESETS)
            self.assertEqual(PRESETS[name]["savefig.format"], "pdf")
            self.assertEqual(PRESETS[name]["font.size"], 7)
            self.assertIn(name, LIGHT_BG_PRESETS)

    def test_figure_size_mm_converts_to_inches(self):
        w, h = figure_size_mm(89.0, 60.0)
        self.assertAlmostEqual(w, 89.0 / 25.4)
        self.assertAlmostEqual(h, 60.0 / 25.4)

    def test_light_background_recolors_then_restores(self):
        # P4.1: light-background export must not permanently modify the
        # on-screen canvas (dark).
        fig = plt.figure(facecolor="#2b2b2b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a1a")
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            savefig_with_preset(fig, str(Path(tmp) / "f.pdf"), "publication_nature",
                                metadata={"Title": "x"})
            self.assertTrue((Path(tmp) / "f.pdf").exists())
        # Original facecolor restored (dark, R component < 0.3).
        self.assertLess(fig.get_facecolor()[0], 0.3)
        self.assertLess(ax.get_facecolor()[0], 0.3)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
