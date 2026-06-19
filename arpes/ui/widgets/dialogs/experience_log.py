"""Dialog showing per-signal processing logs."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from arpes.core.experience_log import build_experience_log


class ExperienceLogDialog(QDialog):
    """Readable audit trail for each processed BM/FS signal."""

    def __init__(self, session, *, current_key: str | None = None, parent=None):
        super().__init__(parent)
        self._session = session
        self._current_key = current_key
        self.setWindowTitle("Processing log")
        self.resize(980, 680)
        self._build()
        self._populate()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("Processing log by signal"))
        header.addStretch(1)
        self._btn_save = QPushButton("Save Markdown...")
        self._btn_save.clicked.connect(self._save_current)
        header.addWidget(self._btn_save)
        root.addLayout(header)

        body = QHBoxLayout()
        self._files = QListWidget()
        self._files.currentItemChanged.connect(self._on_file_changed)
        self._files.setMinimumWidth(260)
        body.addWidget(self._files)

        self._viewer = QTextBrowser()
        self._viewer.setOpenExternalLinks(True)
        self._viewer.setStyleSheet(
            "QTextBrowser { background:#1f1f1f; color:#ddd; font-size:13px; padding:10px; }"
            "h1 { color:#f8fafc; } h2 { color:#c7d2fe; } code { color:#fbbf24; }"
            "li { margin-bottom:3px; }"
        )
        body.addWidget(self._viewer, stretch=1)
        root.addLayout(body, stretch=1)

    def _populate(self) -> None:
        self._files.clear()
        selected_row = 0
        for row, name in enumerate(sorted(getattr(self._session, "files", {}))):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._files.addItem(item)
            if name == self._current_key:
                selected_row = row
        if self._files.count() == 0:
            self._viewer.setMarkdown("No signal loaded in the current session.")
            self._btn_save.setEnabled(False)
            return
        self._files.setCurrentRow(selected_row)

    def _on_file_changed(self, current, _previous) -> None:
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        entry = self._session.files.get(key)
        if entry is None:
            return
        self._viewer.setMarkdown(build_experience_log(entry, name=key))

    def _save_current(self) -> None:
        item = self._files.currentItem()
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        entry = self._session.files.get(key)
        if entry is None:
            return
        suggested = Path(str(key).replace("/", "_")).with_suffix(".processing-log.md")
        if getattr(self._session, "folder", None):
            suggested = Path(self._session.folder) / suggested
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save processing log",
            str(suggested),
            "Markdown (*.md);;Text (*.txt)",
        )
        if not path:
            return
        try:
            Path(path).write_text(build_experience_log(entry, name=key), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Processing log", f"Write failed: {exc}")
