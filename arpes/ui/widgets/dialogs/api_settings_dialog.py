"""Settings dialog for the Materials Project API key.

A discoverable, dedicated place to enter the per-user MP API key (stored via
``app_settings`` / QSettings, never in the repo or session). Also surfaces two
states that otherwise fail silently: whether a key is actually in effect, and
whether the ``mp-api`` library is even importable in this build — a packaged
binary without it cannot use MP regardless of the key.
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from arpes.ui.app_settings import get_mp_api_key, set_mp_api_key


def _mp_api_available() -> bool:
    try:
        import mp_api.client  # noqa: F401
        return True
    except Exception:
        return False


class ApiSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — Materials Project API key")
        self.resize(480, 300)

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Materials Project key (for DFT band import, MP search, lattice a).\n"
            "Get a free key at materialsproject.org → Dashboard → API.\n"
            "Stored locally for this user only — never uploaded or shared."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#cbd5e1;font-size:11px;")
        lay.addWidget(intro)

        self._field = QLineEdit()
        self._field.setEchoMode(QLineEdit.EchoMode.Password)
        self._field.setPlaceholderText("paste your Materials Project API key")
        self._field.setText(get_mp_api_key())
        lay.addWidget(self._field)

        self._chk_show = QCheckBox("Show key")
        self._chk_show.toggled.connect(
            lambda on: self._field.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        lay.addWidget(self._chk_show)

        btns = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._save)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear)
        btns.addWidget(btn_save)
        btns.addWidget(btn_clear)
        btns.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size:11px;")
        lay.addWidget(self._status)
        self._refresh_status()

    def _save(self) -> None:
        set_mp_api_key(self._field.text())
        # Keep a visible DFT-panel field (if built) in sync with the new value.
        parent = self.parent()
        field = getattr(parent, "txt_mp_api_key", None)
        if field is not None:
            field.setText(get_mp_api_key())
        self._refresh_status()

    def _clear(self) -> None:
        set_mp_api_key("")
        self._field.clear()
        parent = self.parent()
        field = getattr(parent, "txt_mp_api_key", None)
        if field is not None:
            field.clear()
        self._refresh_status()

    def _refresh_status(self) -> None:
        stored = bool(get_mp_api_key())
        env = bool(os.environ.get("MP_API_KEY"))
        lib = _mp_api_available()
        if stored:
            key_line = "Key: stored key in use ✓"
            key_col = "#7ec97e"
        elif env:
            key_line = "Key: none stored — falling back to MP_API_KEY env var ✓"
            key_col = "#7ec97e"
        else:
            key_line = "Key: none set — MP features disabled ✗"
            key_col = "#e0a05c"
        if lib:
            lib_line = "Library: mp-api available ✓"
        else:
            lib_line = ("Library: mp-api NOT available in this build ✗ — MP import/"
                        "search/lattice cannot run even with a valid key.")
            key_col = "#e05c5c"
        self._status.setText(f"{key_line}\n{lib_line}")
        self._status.setStyleSheet(f"color:{key_col};font-size:11px;")
