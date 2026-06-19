"""kF drag snap behaviour: live snap, Shift bypass, checkbox off, free placement."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arpes.ui.controllers.kf_drag_handlers import (
    _build_snap_cache,
    _maybe_snap,
    _snap_cached,
)


def _ctrl(*, snap_on=True, gi=0.03, sfd=2.0):
    kpar = np.linspace(-0.6, 0.6, 400)
    mdc = np.exp(-((kpar - 0.30) / 0.02) ** 2) + np.exp(-((kpar + 0.30) / 0.02) ** 2)
    params = SimpleNamespace(
        sp_sfd=SimpleNamespace(value=lambda: sfd),
        sp_gi=SimpleNamespace(value=lambda: gi),
        chk_snap_kf=SimpleNamespace(isChecked=lambda: snap_on),
    )
    return SimpleNamespace(
        _get_mdc=lambda: (kpar, mdc),
        _params=params,
        _parent=SimpleNamespace(),
    )


def _ev(x, key=None):
    return SimpleNamespace(xdata=x, key=key, inaxes=object())


class TestSnapCache:
    def test_cache_finds_both_peaks(self):
        cache = _build_snap_cache(_ctrl())
        peaks = np.sort(cache["peaks"])
        assert len(peaks) == 2
        assert abs(peaks[0] + 0.30) < 0.02
        assert abs(peaks[1] - 0.30) < 0.02

    def test_snap_cached_in_window(self):
        cache = {"peaks": np.array([-0.30, 0.30])}
        assert abs(_snap_cached(cache, 0.27, 0.06) - 0.30) < 1e-9

    def test_snap_cached_outside_window_returns_none(self):
        cache = {"peaks": np.array([-0.30, 0.30])}
        assert _snap_cached(cache, 0.10, 0.06) is None

    def test_empty_peaks_returns_none(self):
        assert _snap_cached({"peaks": np.array([])}, 0.3, 0.06) is None


class TestMaybeSnap:
    def test_snaps_when_enabled_and_near(self):
        ctrl = _ctrl(snap_on=True)
        ctrl._parent._kf_snap_cache = _build_snap_cache(ctrl)
        assert abs(_maybe_snap(ctrl, _ev(0.28), 0.28) - 0.30) < 0.02

    def test_shift_bypasses_snap(self):
        ctrl = _ctrl(snap_on=True)
        ctrl._parent._kf_snap_cache = _build_snap_cache(ctrl)
        assert _maybe_snap(ctrl, _ev(0.28, key="shift"), 0.28) == 0.28

    def test_checkbox_off_disables_snap(self):
        ctrl = _ctrl(snap_on=False)
        ctrl._parent._kf_snap_cache = _build_snap_cache(ctrl)
        assert _maybe_snap(ctrl, _ev(0.28), 0.28) == 0.28

    def test_free_placement_away_from_peak(self):
        ctrl = _ctrl(snap_on=True)
        ctrl._parent._kf_snap_cache = _build_snap_cache(ctrl)
        # 0.10 is far from both peaks (>window) → unchanged
        assert _maybe_snap(ctrl, _ev(0.10), 0.10) == 0.10
