from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arpes.ui.widgets import results_link
from arpes.ui.controllers.interaction_selection import handle_rect_selection
from arpes.core.undo import UndoStack


class _Canvas:
    def __init__(self, ax):
        self.ax = ax
        self.redraw_count = 0

    def redraw(self):
        self.redraw_count += 1


def _panel():
    fig, ax = plt.subplots()
    canvas = _Canvas(ax)
    return SimpleNamespace(
        _canvas=canvas,
        _session=SimpleNamespace(folder=None),
        _host=None,
        _result_point_refs=[],
        _linked_selection=None,
        _linked_result_artist=None,
    )


def test_append_branch_refs_skips_nan_points():
    panel = _panel()
    results_link.append_branch_refs(
        panel, "BM1", "kF_plus", 0,
        [0.1, float("nan"), 0.3],
        [-0.1, -0.2, float("nan")],
    )
    assert panel._result_point_refs == [{
        "file": "BM1", "branch": "kF_plus", "pair": 0,
        "index": 0, "k": 0.1, "e": -0.1,
    }]


def test_results_click_selects_nearest_ref():
    panel = _panel()
    results_link.append_branch_refs(panel, "BM1", "kF_minus", 1, [-0.2], [-0.05])
    event = SimpleNamespace(inaxes=panel._canvas.ax, xdata=-0.2, ydata=-0.05)
    results_link.on_results_click(panel, event)
    assert panel._linked_selection["file"] == "BM1"
    assert panel._linked_selection["branch"] == "kF_minus"
    assert panel._linked_selection["pair"] == 1
    assert panel._linked_selection["index"] == 0
    assert panel._linked_result_artist is not None


def test_sync_from_bm_selection_finds_matching_result_ref():
    panel = _panel()
    results_link.append_branch_refs(panel, "BM1", "kF_plus", 0, [0.25], [0.0])
    results_link.sync_from_bm_selection(panel, "BM1", ("kF_plus", 0, 0))
    assert panel._linked_selection["k"] == 0.25
    assert panel._linked_selection["e"] == 0.0


def test_sync_from_bm_selection_none_clears_highlight():
    panel = _panel()
    results_link.append_branch_refs(panel, "BM1", "kF_plus", 0, [0.25], [0.0])
    results_link.sync_from_bm_selection(panel, "BM1", ("kF_plus", 0, 0))
    results_link.sync_from_bm_selection(panel, "BM1", None)
    assert panel._linked_selection is None


def test_rect_selection_non_additive_still_selects_hits():
    parent = SimpleNamespace(
        _fit_res={
            "e_fitted": [-0.1, 0.0, 0.1],
            "kF_plus": [[0.2, 0.25, 0.3]],
            "kF_minus": [[-0.2, -0.25, -0.3]],
        },
        _fit_selected=[],
        _results=None,
        _current_path="BM1",
    )
    ctrl = SimpleNamespace(_parent=parent, _status=lambda _m: None)
    handle_rect_selection(ctrl, None, (0.19, 0.26), (-0.11, 0.01), additive=False)
    assert parent._fit_selected == [("kF_plus", 0, 0), ("kF_plus", 0, 1)]


def test_delete_marks_nan_without_rerunning_mdc_optimizer():
    from arpes.ui.controllers.interaction_controller import InteractionController

    statuses = []
    fit_calls = []
    fr = {
        "e_fitted": [-0.1, 0.0],
        "kF_minus": [[-0.2, -0.1]],
        "kF_plus": [[0.2, 0.1]],
    }
    parent = SimpleNamespace(
        _fit_res=fr,
        _fit_selected=[("kF_plus", 0, 1)],
        _undo_stack=UndoStack(),
        _params=SimpleNamespace(set_fit_undo_enabled=lambda _v: None),
        _status=statuses.append,
        _draw_current_view=lambda **_kwargs: None,
        _fit_guess=lambda: fit_calls.append(True),
    )
    ctrl = InteractionController(parent)
    ctrl._persist_fit_result = lambda _fr: None
    ctrl._delete_selected_fit_points()
    assert np.isnan(parent._fit_res["kF_plus"][0][1])
    assert fit_calls == []
    assert "fit MDC non relancé" in statuses[-1]
