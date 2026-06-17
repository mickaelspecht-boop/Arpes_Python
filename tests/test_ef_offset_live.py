"""Live EF-offset shift: pure binding-axis shift, no reload (Qt offscreen)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from arpes.ui.controllers.ef_offset_live import apply_live_ef_offset

app = QApplication.instance() or QApplication([])


def _window(baked=0.0, ev=(-1.0, -0.5, 0.0)):
    entry = SimpleNamespace(ef_offset=baked)
    session = SimpleNamespace(
        key_for_path=lambda p: "k",
        get_or_create=lambda k: entry,
        save=lambda: None,
    )
    w = SimpleNamespace(
        _raw_data={"ev_arr": np.array(ev, dtype=float)},
        _data_disp_ev=np.array(ev, dtype=float),
        _ef_offset_applied=baked,
        _session=session,
        _current_path="/data/x",
        _draw_current_view=lambda: None,
    )
    return w, entry


def test_shifts_binding_axis_by_delta():
    w, entry = _window(baked=0.0)
    apply_live_ef_offset(w, 0.1)
    np.testing.assert_allclose(w._raw_data["ev_arr"], [-0.9, -0.4, 0.1])
    np.testing.assert_allclose(w._data_disp_ev, [-0.9, -0.4, 0.1])
    assert w._ef_offset_applied == pytest.approx(0.1)
    assert entry.ef_offset == pytest.approx(0.1)


def test_delta_relative_to_already_baked_offset():
    # ev already carries +0.1; moving to 0.15 must shift by only +0.05.
    w, entry = _window(baked=0.1, ev=(-0.9, -0.4, 0.1))
    apply_live_ef_offset(w, 0.15)
    np.testing.assert_allclose(w._raw_data["ev_arr"], [-0.85, -0.35, 0.15])
    assert w._ef_offset_applied == pytest.approx(0.15)


def test_noop_when_value_unchanged():
    w, _ = _window(baked=0.1, ev=(-0.9, -0.4, 0.1))
    before = w._raw_data["ev_arr"].copy()
    apply_live_ef_offset(w, 0.1)
    np.testing.assert_allclose(w._raw_data["ev_arr"], before)


def test_noop_without_data():
    w = SimpleNamespace(_raw_data=None)
    apply_live_ef_offset(w, 0.2)  # must not raise
