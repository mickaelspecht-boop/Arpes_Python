"""Tests for the manual Γ-center drag (marker-only, BM map).

The gesture moves only the pair-symmetry center (sp_cx / fit_params.center_init);
it must NOT shift the k// axis and must NOT touch the signal. These tests lock
that contract: the dropped display-k is written to sp_cx and persisted on the
entry, with clipping to the kpar range and a click (no move) being a no-op.
"""
from __future__ import annotations

import numpy as np

from arpes.ui.controllers import gamma_drag_handlers as gdh


class _FakeSpin:
    def __init__(self, value=0.0):
        self._v = float(value)
        self.set_calls = []

    def value(self):
        return self._v

    def setValue(self, v):
        self._v = float(v)
        self.set_calls.append(float(v))


class _ClampedSpin(_FakeSpin):
    def __init__(self, value=0.0, lo=-1.0, hi=1.0):
        super().__init__(value)
        self._lo = float(lo)
        self._hi = float(hi)

    def minimum(self):
        return self._lo

    def maximum(self):
        return self._hi

    def setRange(self, lo, hi):
        self._lo = float(lo)
        self._hi = float(hi)

    def setValue(self, v):
        v = min(max(float(v), self._lo), self._hi)
        super().setValue(v)


class _FakeFitParams:
    def __init__(self):
        self.center_init = 0.0


class _FakeEntry:
    def __init__(self):
        self.fit_params = _FakeFitParams()


class _FakeSession:
    def __init__(self, entry):
        self._entry = entry
        self.saved = 0

    def key_for_path(self, path):
        return path

    def get_or_create(self, key):
        return self._entry

    def save(self):
        self.saved += 1


class _FakeParams:
    def __init__(self):
        self.sp_cx = _FakeSpin()


class _FakeWindow:
    def __init__(self, raw, *, path="f1"):
        self._raw_data = raw
        self._params = _FakeParams()
        self._entry = _FakeEntry()
        self._session = _FakeSession(self._entry)
        self._current_path = path
        self.status_msgs = []

    def _status(self, msg):
        self.status_msgs.append(msg)


def _raw(kpar, meta=None):
    return {"kpar": np.asarray(kpar, dtype=float), "metadata": dict(meta or {})}


def test_commit_sets_center_no_axis_shift():
    w = _FakeWindow(_raw(np.linspace(-1.0, 1.0, 21)))
    raw_before = w._raw_data["kpar"].copy()
    gdh._commit(w, 0.3)
    np.testing.assert_allclose(w._params.sp_cx.value(), 0.3)
    np.testing.assert_allclose(w._entry.fit_params.center_init, 0.3)
    # Signal untouched: the kpar axis is not shifted.
    np.testing.assert_array_equal(w._raw_data["kpar"], raw_before)
    assert w._session.saved == 1


def test_commit_clips_to_kpar_range():
    w = _FakeWindow(_raw(np.linspace(-0.5, 0.5, 11)))
    gdh._commit(w, 9.0)
    np.testing.assert_allclose(w._params.sp_cx.value(), 0.5)
    np.testing.assert_allclose(w._entry.fit_params.center_init, 0.5)


def test_commit_negative_center():
    w = _FakeWindow(_raw(np.linspace(-1.0, 1.0, 21)))
    gdh._commit(w, -0.4)
    np.testing.assert_allclose(w._params.sp_cx.value(), -0.4)


def test_commit_expands_spinbox_range_for_wide_bm():
    w = _FakeWindow(_raw(np.linspace(-3.0, 3.0, 31)))
    w._params.sp_cx = _ClampedSpin(lo=-1.0, hi=1.0)
    gdh._commit(w, 1.8)
    gdh._commit(w, 2.4)
    np.testing.assert_allclose(w._params.sp_cx.value(), 2.4)
    np.testing.assert_allclose(w._entry.fit_params.center_init, 2.4)


def test_commit_no_path_still_sets_spinbox():
    w = _FakeWindow(_raw(np.linspace(-1.0, 1.0, 5)), path="")
    gdh._commit(w, 0.2)
    np.testing.assert_allclose(w._params.sp_cx.value(), 0.2)
    assert w._session.saved == 0  # no entry persistence without a path


def _event(x=None, xdata=None, inaxes=None, button=1):
    class _E:
        pass

    e = _E()
    e.x = x
    e.xdata = xdata
    e.inaxes = inaxes
    e.button = button

    class _CV:
        def setCursor(self, *_a):
            pass

        def unsetCursor(self):
            pass

        def draw_idle(self):
            pass

    e.canvas = _CV()
    return e


def test_release_without_move_is_noop():
    w = _FakeWindow(_raw(np.linspace(-1.0, 1.0, 5)))
    w._params.sp_cx.setValue(0.0)
    w._params.sp_cx.set_calls.clear()
    w._gamma_drag_active = True
    w._gamma_drag_moved = False
    # No BM canvas wired -> _bm_ax is None, but a non-moved release must not commit.
    gdh.on_release(w, _event(x=100.0, xdata=0.5))
    # center unchanged, nothing persisted.
    assert w._entry.fit_params.center_init == 0.0
    assert w._session.saved == 0
