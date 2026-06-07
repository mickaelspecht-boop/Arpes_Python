"""In-app Markdown documentation panel."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QWidget,
)


@dataclass(frozen=True)
class HelpSection:
    title: str
    filename: str


DEFAULT_SECTIONS = (
    HelpSection("Workflow", "workflow.md"),
    HelpSection("Shortcuts", "shortcuts.md"),
    HelpSection("Physics", "physics.md"),
)


class HelpPanel(QWidget):
    """Standalone QWidget displaying Markdown files from ``arpes/docs``."""

    def __init__(self, docs_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self._docs_dir = docs_dir or Path(__file__).resolve().parents[2] / "docs"
        self._sections = list(DEFAULT_SECTIONS)
        self._build()
        self._populate_index()

    def _build(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self._index = QListWidget()
        self._index.setMaximumWidth(180)
        self._index.setStyleSheet(
            "QListWidget { background:#222; color:#ddd; font-size:12px; }"
            "QListWidget::item:selected { background:#2a6099; color:white; }"
        )
        self._viewer = QTextBrowser()
        self._viewer.setOpenExternalLinks(True)
        self._viewer.setStyleSheet(
            "QTextBrowser { background:#1f1f1f; color:#ddd; font-size:13px; padding:10px; }"
            "h1 { color:#f8fafc; } h2 { color:#c7d2fe; }"
            "code { color:#fbbf24; }"
        )
        self._index.currentItemChanged.connect(self._on_section_changed)
        lay.addWidget(self._index)
        lay.addWidget(self._viewer, stretch=1)

    def _populate_index(self) -> None:
        self._index.clear()
        for section in self._sections:
            item = QListWidgetItem(section.title)
            item.setData(Qt.ItemDataRole.UserRole, section.filename)
            self._index.addItem(item)
        if self._index.count():
            self._index.setCurrentRow(0)

    def _on_section_changed(self, item: QListWidgetItem | None, _previous=None) -> None:
        if item is None:
            self._viewer.setMarkdown("")
            return
        filename = item.data(Qt.ItemDataRole.UserRole)
        self._viewer.setMarkdown(self._read_doc(str(filename)))

    def _read_doc(self, filename: str) -> str:
        path = self._docs_dir / filename
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return f"# Documentation unavailable\n\nMissing file: `{filename}`."
