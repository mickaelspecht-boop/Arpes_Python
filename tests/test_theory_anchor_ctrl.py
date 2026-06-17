"""Manual HS-point calibration controller: placement, fit, clear."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from arpes.ui.controllers import theory_anchor_ctrl as tac


class _Combo:
    def __init__(self, text=""):
        self._t = text
        self._items = []

    def currentText(self):
        return self._t

    def setCurrentText(self, t):
        self._t = t

    def clear(self):
        self._items = []

    def addItems(self, xs):
        self._items = list(xs)

    def blockSignals(self, _b):
        return False


class _Spin:
    def __init__(self, v=1.0):
        self._v = float(v)

    def value(self):
        return self._v

    def setValue(self, v):
        self._v = float(v)

    def blockSignals(self, _b):
        return False


class _TheoryCtrl:
    def __init__(self, overlay):
        self._overlay = overlay

    def _current_overlay(self):
        return self._overlay

    def _save_overlay(self, overlay):
        self._overlay = overlay


def _window(overlay, label="Γ"):
    params = SimpleNamespace(
        cmb_theory_anchor_label=_Combo(label),
        sp_theory_kscale=_Spin(1.0),
        sp_theory_dk=_Spin(0.0),
        theory_overlay_config=lambda: {},
    )
    ax = object()
    bm = SimpleNamespace(fig=SimpleNamespace(axes=[ax]),
                         canvas=SimpleNamespace(setCursor=lambda *_a: None,
                                                unsetCursor=lambda: None))
    w = SimpleNamespace(
        _params=params,
        _theory_overlay_ctrl=_TheoryCtrl(overlay),
        _bm_canvas=bm,
        _theory_anchor_pick_active=True,
        _on_theory_overlay_changed=lambda: None,
        _draw_current_view=lambda **k: None,
        status=[],
    )
    w._status = lambda m: w.status.append(m)
    return w, ax


def _overlay():
    return {"data": {
        "labels": [{"label": "Γ", "k": 0.0}, {"label": "X", "k": 1.0}],
        "k_distance": [0.0, 0.5, 1.0],
        "bands": [[0.0, -0.1, -0.2]],
        "branches": [],
    }, "anchors": []}


def _click(ax, x):
    return SimpleNamespace(inaxes=ax, xdata=x, button=1)


def test_click_places_anchor():
    w, ax = _window(_overlay(), label="Γ")
    tac.on_bm_click(w, _click(ax, -0.05))
    anchors = w._theory_overlay_ctrl._current_overlay()["anchors"]
    assert anchors == [{"label": "Γ", "k": -0.05}]


def test_click_replaces_same_label():
    w, ax = _window(_overlay(), label="Γ")
    tac.on_bm_click(w, _click(ax, -0.05))
    tac.on_bm_click(w, _click(ax, 0.02))
    anchors = w._theory_overlay_ctrl._current_overlay()["anchors"]
    assert anchors == [{"label": "Γ", "k": 0.02}]


def test_click_ignored_when_not_picking():
    w, ax = _window(_overlay(), label="Γ")
    w._theory_anchor_pick_active = False
    tac.on_bm_click(w, _click(ax, 0.3))
    assert w._theory_overlay_ctrl._current_overlay()["anchors"] == []


def test_click_ignored_during_roi_gesture():
    w, ax = _window(_overlay(), label="Γ")
    w._fit_roi_active = True
    tac.on_bm_click(w, _click(ax, 0.3))
    assert w._theory_overlay_ctrl._current_overlay()["anchors"] == []


def test_click_ignored_during_gamma_drag():
    w, ax = _window(_overlay(), label="Γ")
    w._gamma_drag_active = True
    tac.on_bm_click(w, _click(ax, 0.3))
    assert w._theory_overlay_ctrl._current_overlay()["anchors"] == []


def test_apply_two_points_sets_scale_and_shift():
    ov = _overlay()
    ov["anchors"] = [{"label": "Γ", "k": -0.05}, {"label": "X", "k": 0.78}]
    w, _ = _window(ov)
    tac.apply_calibration(w)
    assert w._params.sp_theory_kscale.value() == pytest.approx(0.83)
    assert w._params.sp_theory_dk.value() == pytest.approx(-0.05)


def test_apply_single_point_keeps_scale_sets_shift():
    ov = _overlay()
    ov["anchors"] = [{"label": "Γ", "k": 0.12}]
    w, _ = _window(ov)
    w._params.sp_theory_kscale.setValue(2.0)
    tac.apply_calibration(w)
    assert w._params.sp_theory_kscale.value() == pytest.approx(2.0)
    # Γ local-k is 0 -> shift = clicked k.
    assert w._params.sp_theory_dk.value() == pytest.approx(0.12)


def test_clear_anchors():
    ov = _overlay()
    ov["anchors"] = [{"label": "Γ", "k": 0.0}]
    w, _ = _window(ov)
    tac.clear_anchors(w)
    assert w._theory_overlay_ctrl._current_overlay()["anchors"] == []


def test_populate_labels_fills_combo():
    w, _ = _window(_overlay())
    tac.populate_labels(w)
    assert w._params.cmb_theory_anchor_label._items == ["Γ", "X"]
