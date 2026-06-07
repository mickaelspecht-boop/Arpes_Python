"""P3.1: forwarding controllers must fail loud on an unknown write.

The 8 god-object-camouflage controllers forward attribute access to a parent
(ArpesExplorer) or panel. Before P3.1 ``__setattr__`` blindly forwarded ANY
name, so a typo (``self._sel_kk = x``) silently created a junk attribute on the
parent while the real state went stale. Now writes are allow-listed via
``_PARENT_WRITES``; an unknown name raises ``AttributeError``.

Reads stay forwarded (a typo'd read already raises through the parent), so we
only pin the write contract here.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from arpes.ui.controllers.browser_controller import BrowserController
    from arpes.ui.controllers.distortion_controller import DistortionController
    from arpes.ui.controllers.fs_controller import FSController
    from arpes.ui.controllers.gamma_controller import GammaController
    from arpes.ui.controllers.norm_controller import NormController
    from arpes.ui.controllers.pairing_controller import PairingController
    from arpes.ui.controllers.plot_controller import PlotController
    from arpes.ui.controllers.pocket_controller import PocketController

    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


# (Controller, kwarg-name for the forward target)
_FORWARDERS = None
if UI_AVAILABLE:
    _FORWARDERS = [
        (DistortionController, "parent"),
        (BrowserController, "panel"),
        (PlotController, "parent"),
        (FSController, "parent"),
        (NormController, "parent"),
        (GammaController, "parent"),
        (PocketController, "parent"),
        (PairingController, "parent"),
    ]


@unittest.skipUnless(UI_AVAILABLE, "PyQt6 unavailable")
class TestForwarderFailLoud(unittest.TestCase):
    def test_unknown_write_raises_on_every_forwarder(self):
        for cls, _arg in _FORWARDERS:
            ctrl = cls(SimpleNamespace())
            with self.assertRaises(AttributeError, msg=f"{cls.__name__} swallowed typo"):
                ctrl._totally_bogus_attr_xyz = 1

    def test_allowlisted_write_reaches_parent(self):
        for cls, _arg in _FORWARDERS:
            parent = SimpleNamespace()
            ctrl = cls(parent)
            for name in cls._PARENT_WRITES:
                setattr(ctrl, name, "sentinel")
                self.assertEqual(
                    getattr(parent, name), "sentinel",
                    msg=f"{cls.__name__}.{name} did not forward to parent",
                )

    def test_read_still_forwards(self):
        for cls, _arg in _FORWARDERS:
            parent = SimpleNamespace(_some_state=123)
            ctrl = cls(parent)
            self.assertEqual(ctrl._some_state, 123)


if __name__ == "__main__":
    unittest.main()
