"""Tests A.4 — PairingController (pin / auto-pin / active FS).

Headless: PairingController is a thin wrapper with no direct Qt dependency, so
it can be instantiated with a minimal parent stub.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from arpes.core.session import FileEntry, FileMeta, Session
from arpes.ui.controllers.pairing_controller import PairingController


def _make_session(files: dict[str, FileEntry]) -> Session:
    s = Session(Path("/tmp/pairing_test"))
    for name, entry in files.items():
        s.files[name] = entry
    return s


def _bm(*, hv=60.0, azi=0.0, polar=0.0, pol="LH", parent=None) -> FileEntry:
    e = FileEntry(meta=FileMeta(
        hv=hv, azi=azi, polar=polar, polarization=pol, scan_kind="BM",
    ))
    e.parent_fs_path = parent
    return e


def _fs(*, hv=60.0, azi=0.0, pol="LH", crystal_a=3.96) -> FileEntry:
    # Explicit crystal_a: since P1.1 the default lattice is "unknown" (0), so a
    # BM↔FS overlay requires a provided lattice (otherwise a_lattice=0 → no cut).
    return FileEntry(meta=FileMeta(
        hv=hv, azi=azi, polarization=pol, scan_kind="FS",
        crystal_a_angstrom=crystal_a,
    ))


class _StubParent:
    """Minimal stub for PairingController (no Qt)."""
    def __init__(self, session: Session, current_path: str | None = None):
        self._session = session
        self._current_path = current_path
        self._pinned_fs_path: str | None = None
        self._status_msgs: list[str] = []

    def _status(self, msg: str) -> None:
        self._status_msgs.append(msg)


class TestPairingController(unittest.TestCase):
    def test_pin_and_unpin(self):
        s = _make_session({})
        parent = _StubParent(s)
        ctrl = PairingController(parent)

        ctrl._pin_fs_path("/d/fs1.txt")
        self.assertEqual(parent._pinned_fs_path, "fs1.txt")
        self.assertTrue(any("Context FS pinned" in m for m in parent._status_msgs))

        ctrl._unpin_fs_path()
        self.assertIsNone(parent._pinned_fs_path)

    def test_active_fs_returns_current_when_fs(self):
        s = _make_session({"/d/fs1.txt": _fs()})
        # Simple key_for_path stub: returns the path itself.
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/fs1.txt")
        ctrl = PairingController(parent)

        self.assertEqual(ctrl._active_fs_path(), "/d/fs1.txt")

    def test_active_fs_uses_session_key_for_absolute_current_path(self):
        s = _make_session({"fs1.txt": _fs()})
        parent = _StubParent(s, current_path="/tmp/pairing_test/fs1.txt")
        ctrl = PairingController(parent)

        self.assertEqual(ctrl._active_fs_path(), "fs1.txt")

    def test_active_fs_falls_back_to_pinned_when_current_is_bm(self):
        s = _make_session({"/d/fs1.txt": _fs(), "/d/bm.txt": _bm()})
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/bm.txt")
        parent._pinned_fs_path = "/d/fs1.txt"
        ctrl = PairingController(parent)

        self.assertEqual(ctrl._active_fs_path(), "/d/fs1.txt")

    def test_active_fs_none_when_no_current_no_pin(self):
        s = _make_session({})
        s.key_for_path = lambda p: p
        parent = _StubParent(s)
        ctrl = PairingController(parent)
        self.assertIsNone(ctrl._active_fs_path())

    def test_auto_pin_finds_fs_for_current_bm(self):
        s = _make_session({
            "fs1.txt": _fs(hv=60.0, azi=0.0),
            "bm.txt": _bm(hv=60.0, azi=0.0),
        })
        parent = _StubParent(s, current_path="/tmp/pairing_test/bm.txt")
        ctrl = PairingController(parent)

        chosen = ctrl._auto_pin_fs_for_current_bm()
        self.assertEqual(chosen, "fs1.txt")
        self.assertEqual(parent._pinned_fs_path, "fs1.txt")

    def test_auto_pin_respects_manual_override(self):
        s = _make_session({
            "/d/fs_compatible.txt": _fs(hv=60.0, azi=0.0),
            "/d/fs_pinned.txt": _fs(hv=120.0, azi=30.0),
            "/d/bm.txt": _bm(hv=60.0, azi=0.0, parent="/d/fs_pinned.txt"),
        })
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/bm.txt")
        ctrl = PairingController(parent)

        chosen = ctrl._auto_pin_fs_for_current_bm()
        # Manual override wins despite automatic incompatibility.
        self.assertEqual(chosen, "/d/fs_pinned.txt")

    def test_auto_pin_returns_none_when_current_is_fs(self):
        s = _make_session({"/d/fs1.txt": _fs()})
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/fs1.txt")
        ctrl = PairingController(parent)
        self.assertIsNone(ctrl._auto_pin_fs_for_current_bm())

    def test_bound_bms_lists_compatible(self):
        s = _make_session({
            "/d/fs1.txt": _fs(),
            "/d/bm_a.txt": _bm(),
            "/d/bm_b.txt": _bm(azi=10.0),  # excluded (azi)
        })
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/fs1.txt")
        ctrl = PairingController(parent)

        bms = ctrl._bound_bms_for_active_fs()
        self.assertEqual([m.path for m in bms], ["/d/bm_a.txt"])


class TestCollectBmCuts(unittest.TestCase):
    """B.2 — aggregation of BM projections in the FS frame."""

    def test_collect_returns_cuts_for_bound_bms(self):
        s = _make_session({
            "/d/fs1.txt": _fs(hv=60.0, azi=0.0),
            "/d/bm_a.txt": _bm(hv=60.0, azi=0.0),
            "/d/bm_far.txt": _bm(hv=120.0, azi=30.0),  # excluded
        })
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/fs1.txt")
        parent._raw_data = {"metadata": {"fs_scan_axis_deg": {"center": 0.0}}}
        parent._params = type("P", (), {
            "sp_phi": type("S", (), {"value": staticmethod(lambda: 4.5)})()
        })()
        ctrl = PairingController(parent)

        cuts = ctrl._collect_bm_cuts_for_active_fs()
        labels = [c.label for c in cuts]
        self.assertEqual(labels, ["bm_a"])
        self.assertEqual(cuts[0].quality, "exact")

    def test_collect_empty_when_no_active_fs(self):
        s = _make_session({"/d/bm.txt": _bm()})
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/bm.txt")
        ctrl = PairingController(parent)
        self.assertEqual(ctrl._collect_bm_cuts_for_active_fs(), [])


class TestPairingVerbDispatch(unittest.TestCase):
    def test_verbs_dispatch(self):
        s = _make_session({
            "/d/fs1.txt": _fs(),
            "/d/bm.txt": _bm(),
        })
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/bm.txt")
        ctrl = PairingController(parent)

        ctrl._pairing_action("pin", {"path": "/d/fs1.txt"})
        self.assertEqual(parent._pinned_fs_path, "/d/fs1.txt")

        self.assertEqual(ctrl._pairing_action("active_fs"), "/d/fs1.txt")
        bms = ctrl._pairing_action("bound_bms")
        self.assertEqual(len(bms), 1)

        ctrl._pairing_action("unpin")
        self.assertIsNone(parent._pinned_fs_path)

        with self.assertRaises(ValueError):
            ctrl._pairing_action("bogus")

    def test_toggle_cuts_sets_flag(self):
        s = _make_session({})
        s.key_for_path = lambda p: p
        parent = _StubParent(s)
        parent._show_bm_cuts = False
        parent._draw_fs_tab = lambda: None  # stub
        ctrl = PairingController(parent)
        ctrl._pairing_action("toggle_cuts", {"visible": True})
        self.assertTrue(parent._show_bm_cuts)
        ctrl._pairing_action("toggle_cuts", {"visible": False})
        self.assertFalse(parent._show_bm_cuts)


if __name__ == "__main__":
    unittest.main()
