"""Session notes: persistent markdown editor stored in .arpes_session.json."""
from __future__ import annotations

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from arpes.core.session import Session


class NotesPanel(QWidget):
    """Markdown notes editor bound to `Session.session_notes`.

    Text is autosaved 800 ms after the last keystroke.
    A "Preview" tab displays the QTextBrowser.setMarkdown rendering.
    """

    notes_changed = pyqtSignal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._emit_notes)
        self._build()
        self.refresh_from_session()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        header.addWidget(QLabel("Session notes (markdown - persisted in .arpes_session.json)"))
        header.addStretch(1)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self._on_clear)
        header.addWidget(self._btn_clear)
        lay.addLayout(header)

        self._tabs = QTabWidget()
        self._editor = QTextEdit()
        self._editor.setAcceptRichText(False)
        self._editor.setPlaceholderText(
            "Add experiment notes here: material, conditions, observations, "
            "TODOs, bibliography references...\n\n"
            "Markdown format: # title, ## subtitle, **bold**, `code`, - list."
        )
        self._editor.setStyleSheet(
            "QTextEdit { background:#1f1f1f; color:#ddd; "
            "font-family:monospace; font-size:12px; padding:8px; }"
        )
        self._editor.textChanged.connect(self._on_text_changed)
        self._tabs.addTab(self._editor, "Edit")

        self._viewer = QTextBrowser()
        self._viewer.setOpenExternalLinks(True)
        self._viewer.setStyleSheet(
            "QTextBrowser { background:#1f1f1f; color:#ddd; font-size:13px; padding:10px; }"
            "h1 { color:#f8fafc; } h2 { color:#c7d2fe; } code { color:#fbbf24; }"
        )
        self._tabs.addTab(self._viewer, "Preview")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self._tabs, stretch=1)

    def refresh_from_session(self) -> None:
        text = str(getattr(self._session, "session_notes", "") or "")
        self._editor.blockSignals(True)
        self._editor.setPlainText(text)
        self._editor.blockSignals(False)
        self._viewer.setMarkdown(text)

    def _on_text_changed(self) -> None:
        self._save_timer.start()

    def _emit_notes(self) -> None:
        text = self._editor.toPlainText()
        self.notes_changed.emit(text)

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1:
            self._viewer.setMarkdown(self._editor.toPlainText())

    def _on_clear(self) -> None:
        self._editor.clear()
