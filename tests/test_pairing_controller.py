"""Tests A.4 — PairingController (pin / auto-pin / active FS).

Headless : PairingController est un mince wrapper sans Qt direct, on peut
l'instancier avec un parent stub minimaliste.
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


def _bm(*, hv=60.0, azi=0.0, pol="LH", parent=None) -> FileEntry:
    e = FileEntry(meta=FileMeta(hv=hv, azi=azi, polarization=pol, scan_kind="BM"))
    e.parent_fs_path = parent
    return e


def _fs(*, hv=60.0, azi=0.0, pol="LH") -> FileEntry:
    return FileEntry(meta=FileMeta(hv=hv, azi=azi, polarization=pol, scan_kind="FS"))


class _StubParent:
    """Stub minimaliste pour PairingController (pas de Qt)."""
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
        self.assertEqual(parent._pinned_fs_path, "/d/fs1.txt")
        self.assertTrue(any("FS contexte" in m for m in parent._status_msgs))

        ctrl._unpin_fs_path()
        self.assertIsNone(parent._pinned_fs_path)

    def test_active_fs_returns_current_when_fs(self):
        s = _make_session({"/d/fs1.txt": _fs()})
        # Stub key_for_path simple : retourne le path lui-même
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/fs1.txt")
        ctrl = PairingController(parent)

        self.assertEqual(ctrl._active_fs_path(), "/d/fs1.txt")

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
            "/d/fs1.txt": _fs(hv=60.0, azi=0.0),
            "/d/bm.txt": _bm(hv=60.0, azi=0.0),
        })
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/bm.txt")
        ctrl = PairingController(parent)

        chosen = ctrl._auto_pin_fs_for_current_bm()
        self.assertEqual(chosen, "/d/fs1.txt")
        self.assertEqual(parent._pinned_fs_path, "/d/fs1.txt")

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
        # Manual override gagne malgré incompatibilité auto
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
            "/d/bm_b.txt": _bm(azi=10.0),  # exclu (azi)
        })
        s.key_for_path = lambda p: p
        parent = _StubParent(s, current_path="/d/fs1.txt")
        ctrl = PairingController(parent)

        bms = ctrl._bound_bms_for_active_fs()
        self.assertEqual([m.path for m in bms], ["/d/bm_a.txt"])


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


if __name__ == "__main__":
    unittest.main()
