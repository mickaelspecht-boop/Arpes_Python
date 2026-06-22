"""Export figure: Γ(E) panel honours the chosen lifetime fit range."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    from arpes.core.session import FileMeta, Session
    from arpes.ui.widgets import results_export
    from arpes.ui.widgets.results import ResultsPanel
    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 not available")
class TestResultsExportRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _panel(self):
        tmp = Path(tempfile.mkdtemp())
        s = Session(folder=tmp)
        e = s.get_or_create("A/f.h5")
        e.meta = FileMeta(hv=48, temperature=150, direction="GX")
        ev = np.linspace(-0.15, 0.0, 40)
        kp = (0.30 + 0.5 * ev)
        km = (-0.30 - 0.5 * ev)
        g = np.full_like(ev, 0.05)
        e.fit_result = {
            "e_fitted": ev.tolist(), "kF_plus": [kp.tolist()],
            "kF_minus": [km.tolist()], "gamma_corrige": [g.tolist()],
            "n_pairs": 1, "width_convention": "HWHM",
        }
        e.fit_params.gamma_max = 0.30
        p = ResultsPanel(s)
        p.refresh()
        return p

    def test_export_gamma_clipped_to_chosen_range(self):
        p = self._panel()
        p._sp_gamma_emin.setValue(-0.10)
        p._sp_gamma_emax.setValue(-0.04)
        fig = results_export.build_scientific_export_figure(p)
        ax_g = fig.axes[1]
        lo, hi = ax_g.get_xlim()
        self.assertAlmostEqual(lo, -0.10, places=6)
        self.assertAlmostEqual(hi, -0.04, places=6)
        # Exported equations carry the chosen window.
        eqs = p._export_gamma_equations
        self.assertEqual(eqs[0]["e_range"], [-0.10, -0.04])


if __name__ == "__main__":
    unittest.main()
