"""Menu builder for the ArpesExplorer main window."""
from __future__ import annotations

from PyQt6.QtWidgets import QMenuBar


def build_menubar(window) -> QMenuBar:
    return window.menuBar()
