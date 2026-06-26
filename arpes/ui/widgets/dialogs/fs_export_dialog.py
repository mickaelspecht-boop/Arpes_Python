"""Options dialog for exporting the Fermi-surface figure.

Small chooser shown before the file dialog: the figure title (default
``FS : <file>``) and whether to add the high-symmetry markers.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class FsExportDialog(QDialog):
    def __init__(self, parent=None, *, default_title: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Export Fermi-surface figure")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Figure title"))
        self.txt_title = QLineEdit(default_title)
        lay.addWidget(self.txt_title)
        self.chk_hsym = QCheckBox("Add high-symmetry points")
        self.chk_hsym.setChecked(True)
        lay.addWidget(self.chk_hsym)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def options(self) -> dict:
        return {
            "add_hsym": self.chk_hsym.isChecked(),
            "title": self.txt_title.text().strip(),
        }
