"""Unified dialog for content and format selection before exporting results."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


CONTENTS = {
    "Slice results (E, kF, γ, ...)": "slice",
    "Physical results (kF, vF, m*, Γ₀ ± σ)": "physics",
}

FORMATS = {
    "CSV (.csv)": "csv",
    "Aligned text (.txt)": "txt",
    "LaTeX booktabs (.tex)": "latex",
}

EXTENSIONS = {"csv": ".csv", "txt": ".txt", "latex": ".tex"}


class ExportDialog(QDialog):
    """Simple content and format choice before the file picker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.resize(420, 180)
        self.content_key: str = "physics"
        self.format_key: str = "csv"

        lay = QVBoxLayout(self)
        fl = QFormLayout()
        self._cmb_content = QComboBox()
        for label in CONTENTS:
            self._cmb_content.addItem(label)
        fl.addRow(QLabel("Content:"), self._cmb_content)
        self._cmb_format = QComboBox()
        for label in FORMATS:
            self._cmb_format.addItem(label)
        fl.addRow(QLabel("Format:"), self._cmb_format)
        lay.addLayout(fl)

        self._lbl_warn = QLabel(
            "Note: LaTeX is available only for physical results.\n"
            "Invalid combinations will fall back to CSV."
        )
        self._lbl_warn.setStyleSheet("color:#aaa;font-size:10px;")
        self._lbl_warn.setWordWrap(True)
        lay.addWidget(self._lbl_warn)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _on_accept(self) -> None:
        self.content_key = CONTENTS[self._cmb_content.currentText()]
        self.format_key = FORMATS[self._cmb_format.currentText()]
        if self.format_key == "latex" and self.content_key != "physics":
            self.format_key = "csv"
        self.accept()

    def extension(self) -> str:
        return EXTENSIONS.get(self.format_key, ".csv")

    def file_filter(self) -> str:
        ext = self.extension().lstrip(".")
        return f"{self.format_key.upper()} (*.{ext})"
