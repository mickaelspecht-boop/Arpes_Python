"""Helpers Qt partagés (spinboxes pré-configurés, séparateurs, palette pairs)."""
from __future__ import annotations

from PyQt6.QtWidgets import QDoubleSpinBox, QFrame, QSpinBox


PAIR_COLORS = ["#ff8c00", "#00e5ff", "#7fff00", "#ff44cc"]


def dspin(val, lo, hi, step, dec=3) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setSingleStep(step)
    w.setDecimals(dec); w.setValue(val); w.setFixedWidth(82)
    return w


def ispin(val, lo, hi) -> QSpinBox:
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); w.setFixedWidth(60)
    return w


def hsep() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken); return f
