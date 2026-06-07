"""Shared Qt helpers (preconfigured spinboxes, separators, pair palette)."""
from __future__ import annotations

from PyQt6.QtWidgets import QDoubleSpinBox, QFrame, QPushButton, QSizePolicy, QSpinBox


PAIR_COLORS = ["#ff8c00", "#00e5ff", "#7fff00", "#ff44cc"]


def dspin(val, lo, hi, step, dec=3) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setSingleStep(step)
    w.setDecimals(dec); w.setValue(val); w.setFixedWidth(82)
    w.setKeyboardTracking(False)
    return w


def ispin(val, lo, hi) -> QSpinBox:
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); w.setFixedWidth(60)
    w.setKeyboardTracking(False)
    return w


def hsep() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken); return f


def compact_button(button: QPushButton, max_width: int = 220) -> QPushButton:
    """Keep a button readable without stretching it across the full panel width."""
    button.setMaximumWidth(max_width)
    button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return button
