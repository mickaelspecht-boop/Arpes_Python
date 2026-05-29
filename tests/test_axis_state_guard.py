"""HIGH-1: kF overlay must detect grid/distortion state mismatch."""
from __future__ import annotations

from types import SimpleNamespace

from arpes.core.session import FileEntry, Session


def _make_parent(entry_attrs: dict) -> SimpleNamespace:
    sess = Session()
    sess.files["f.h5"] = FileEntry(**entry_attrs)
    p = SimpleNamespace()
    p._session = sess
    p._current_path = "f.h5"
    return p


def test_no_mismatch_when_tags_match_current_state():
    from arpes.ui.controllers.plot_controller import PlotController
    p = _make_parent({
        "bm_distortion": {},
        "grid_correction": {},
    })
    ctrl = PlotController.__new__(PlotController)
    ctrl._parent = p
    fr = {"distorted": False, "grid_active": False, "e_fitted": [0.0]}
    assert ctrl._axis_state_mismatch(fr) is False


def test_mismatch_when_distortion_was_on_but_now_off():
    from arpes.ui.controllers.plot_controller import PlotController
    p = _make_parent({
        "bm_distortion": {},
        "grid_correction": {},
    })
    ctrl = PlotController.__new__(PlotController)
    ctrl._parent = p
    fr = {"distorted": True, "grid_active": False, "e_fitted": [0.0]}
    assert ctrl._axis_state_mismatch(fr) is True


def test_legacy_fit_without_tags_assumed_consistent():
    from arpes.ui.controllers.plot_controller import PlotController
    p = _make_parent({
        "bm_distortion": {},
        "grid_correction": {"enabled": True, "strength": 0.5},
    })
    ctrl = PlotController.__new__(PlotController)
    ctrl._parent = p
    fr = {"e_fitted": [0.0]}  # no distorted/grid_active keys
    assert ctrl._axis_state_mismatch(fr) is False


def test_mismatch_when_grid_was_off_but_now_on():
    from arpes.ui.controllers.plot_controller import PlotController
    p = _make_parent({
        "bm_distortion": {},
        "grid_correction": {"enabled": True, "strength": 0.5},
    })
    ctrl = PlotController.__new__(PlotController)
    ctrl._parent = p
    fr = {"distorted": False, "grid_active": False, "e_fitted": [0.0]}
    assert ctrl._axis_state_mismatch(fr) is True
