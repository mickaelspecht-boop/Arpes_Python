"""P3.7: long fits (_fit_full / _fit_ensemble) reject re-entrancy.

``QApplication.processEvents()`` in the middle of a fit can deliver a repeated
"Fit" click, causing re-entry that would corrupt the shared ``_fit_res`` /
entry. The ``_fit_busy`` flag blocks the second call. Batch mode calls
``_fit_full`` sequentially (the flag is reset to False between calls), so it is
not blocked.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from arpes.ui.controllers.fit_runner_controller import FitRunnerController
    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 unavailable")
class TestFitReentrancyGuard(unittest.TestCase):
    def _parent(self):
        statuses: list[str] = []
        parent = SimpleNamespace(
            _fit_busy=True,  # a fit is already running
            _status=lambda m: statuses.append(str(m)),
            ap=object(),  # must never be reached
        )
        return parent, statuses

    def test_fit_full_blocked_when_busy(self):
        parent, statuses = self._parent()
        FitRunnerController(parent)._fit_full()
        self.assertTrue(parent._fit_busy)  # unchanged
        self.assertTrue(any("already running" in s for s in statuses), statuses)

    def test_fit_ensemble_blocked_when_busy(self):
        parent, statuses = self._parent()
        FitRunnerController(parent)._fit_ensemble()
        self.assertTrue(parent._fit_busy)
        self.assertTrue(any("already running" in s for s in statuses), statuses)


if __name__ == "__main__":
    unittest.main()
