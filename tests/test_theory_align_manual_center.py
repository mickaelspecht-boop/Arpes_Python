"""DFT realignment must honor the manual Γ center marker (sp_cx).

The manual center drag moves only the marker (no k// axis shift), so aligning
the DFT overlay must offset it by sp_cx → the DFT Γ lands on the chosen center,
not on k=0.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from arpes.ui.controllers.theory_overlay_controller import TheoryOverlayController


class _Spin:
    def __init__(self, value=0.0):
        self._v = float(value)

    def value(self):
        return self._v

    def setValue(self, v):
        self._v = float(v)

    def blockSignals(self, _b):
        return False


class _Combo:
    def __init__(self, text):
        self._t = text

    def currentText(self):
        return self._t


def _make_ctrl(k_center, segment="Γ-X", labels=None, branches=None):
    params = SimpleNamespace(
        sp_theory_kscale=_Spin(1.0),
        sp_theory_dk=_Spin(0.0),
        sp_crystal_a=_Spin(0.0),
        sp_cx=_Spin(k_center),
        cmb_theory_segment=_Combo(segment),
    )
    overlay = {"data": {
        "labels": labels if labels is not None else [
            {"label": "Γ", "k": 0.0}, {"label": "X", "k": 0.5},
        ],
        "branches": branches or [],
    }}
    entry = SimpleNamespace(theory_overlay=overlay)
    parent = SimpleNamespace(
        _params=params,
        _current_entry=lambda: entry,
        _status=lambda *_a: None,
    )
    ctrl = TheoryOverlayController(parent)
    ctrl._on_theory_overlay_changed = lambda: None  # avoid real redraw
    return ctrl, params


def test_label_align_offsets_dk_by_manual_center():
    ctrl, params = _make_ctrl(k_center=0.3)
    ctrl._align_theory_to_arpes()
    # Γ-X spans 0..0.5 -> scale=2; Γ should map to the manual center 0.3.
    assert params.sp_theory_kscale.value() == pytest.approx(2.0)
    assert params.sp_theory_dk.value() == pytest.approx(0.3)


def test_label_align_zero_center_is_origin():
    ctrl, params = _make_ctrl(k_center=0.0)
    ctrl._align_theory_to_arpes()
    assert params.sp_theory_dk.value() == pytest.approx(0.0)


def test_manual_gamma_center_reads_sp_cx():
    ctrl, _ = _make_ctrl(k_center=-0.25)
    assert ctrl._manual_gamma_center() == pytest.approx(-0.25)
