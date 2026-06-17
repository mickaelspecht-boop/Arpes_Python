"""Manual HS-point calibration controller: placement, fit, clear."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from arpes.ui.controllers import theory_anchor_ctrl as tac


class _Combo:
    def __init__(self, text=""):
        self._t = text
        self._items = []
        self.tinted = {}

    def currentText(self):
        return self._t

    def setCurrentText(self, t):
        self._t = t

    def clear(self):
        self._items = []
        self.tinted = {}

    def addItems(self, xs):
        self._items = list(xs)

    def setItemData(self, i, value, role):
        self.tinted[i] = (value, role)

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


def _window(overlay, label="Γ", *, enabled=False):
    params = SimpleNamespace(
        cmb_theory_anchor_label=_Combo(label),
        sp_theory_kscale=_Spin(1.0),
        sp_theory_dk=_Spin(0.0),
        sp_cx=_Spin(0.0),
        theory_overlay_config=lambda: {"enabled": bool(enabled)},
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


def test_apply_sets_manual_center_from_gamma_anchor():
    ov = _overlay()
    ov["anchors"] = [{"label": "Γ", "k": 0.12}]
    w, _ = _window(ov)
    tac.apply_calibration(w)
    # Γ local-k is 0 → displayed Γ = shift = 0.12 → manual center locked to it.
    assert w._params.sp_cx.value() == pytest.approx(0.12)


def test_apply_without_gamma_anchor_leaves_center():
    ov = _overlay()
    ov["anchors"] = [{"label": "X", "k": 0.78}]
    w, _ = _window(ov)
    w._params.sp_cx.setValue(0.05)
    tac.apply_calibration(w)
    assert w._params.sp_cx.value() == pytest.approx(0.05)  # untouched


def test_track_gamma_center_shifts_dk_when_overlay_enabled():
    w, _ = _window(_overlay(), enabled=True)
    w._params.sp_theory_dk.setValue(0.10)
    tac.track_gamma_center_delta(w, 0.30)
    assert w._params.sp_theory_dk.value() == pytest.approx(0.40)


def test_track_gamma_center_noop_when_overlay_disabled():
    w, _ = _window(_overlay(), enabled=False)
    w._params.sp_theory_dk.setValue(0.10)
    tac.track_gamma_center_delta(w, 0.30)
    assert w._params.sp_theory_dk.value() == pytest.approx(0.10)


def test_track_gamma_center_noop_on_zero_delta():
    w, _ = _window(_overlay(), enabled=True)
    w._params.sp_theory_dk.setValue(0.10)
    tac.track_gamma_center_delta(w, 0.0)
    assert w._params.sp_theory_dk.value() == pytest.approx(0.10)


def test_placed_label_tinted_in_combo():
    w, ax = _window(_overlay(), label="Γ")
    tac.on_bm_click(w, _click(ax, -0.05))
    # Γ is index 0 and now placed → tinted green.
    assert 0 in w._params.cmb_theory_anchor_label.tinted


def test_use_manual_gamma_center_replaces_gamma_anchor_and_aligns():
    ov = _overlay()
    ov["anchors"] = [{"label": "Γ", "k": -0.2}, {"label": "X", "k": 0.8}]
    w, _ = _window(ov)
    w._params.sp_cx.setValue(0.15)
    tac.use_manual_gamma_center(w)
    anchors = w._theory_overlay_ctrl._current_overlay()["anchors"]
    assert {"label": "Γ", "k": 0.15} in anchors
    assert {"label": "Γ", "k": -0.2} not in anchors
    assert w._params.sp_theory_dk.value() == pytest.approx(0.15)
