"""Unified dialog: pick content + format, preview the rows, then export."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
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

# Cap the preview table — the written file always carries every row.
_PREVIEW_MAX_ROWS = 500


class ExportDialog(QDialog):
    """Content + format choice with a live preview before the file picker."""

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Export Results")
        self.resize(720, 460)
        self.content_key: str = "physics"
        self.format_key: str = "csv"

        lay = QVBoxLayout(self)
        fl = QFormLayout()
        self._cmb_content = QComboBox()
        for label in CONTENTS:
            self._cmb_content.addItem(label)
        self._cmb_content.setCurrentIndex(1)  # default = physical results
        self._cmb_content.currentTextChanged.connect(self._refresh_preview)
        fl.addRow(QLabel("Content:"), self._cmb_content)
        self._cmb_format = QComboBox()
        for label in FORMATS:
            self._cmb_format.addItem(label)
        fl.addRow(QLabel("Format:"), self._cmb_format)
        lay.addLayout(fl)

        lay.addWidget(QLabel("Preview (the rows that will be written):"))
        self._preview = QTableWidget(0, 0)
        self._preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._preview.setAlternatingRowColors(True)
        self._preview.setStyleSheet(
            "QTableWidget{background:#222;color:#ddd;font-size:11px;"
            "alternate-background-color:#2a2a2a;}"
            "QHeaderView::section{background:#333;color:#eee;font-weight:bold;}")
        self._preview.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        lay.addWidget(self._preview, stretch=1)
        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet("color:#9fd49f;font-size:11px;")
        lay.addWidget(self._lbl_count)

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
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Export…")
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        self._btn_box = bb
        lay.addWidget(bb)

        self._refresh_preview()

    # -- preview ---------------------------------------------------------------
    def _current_rows(self) -> list[dict]:
        """Rows for the selected content, or [] (empty/error shown to user)."""
        if self._session is None:
            return []
        from arpes.io.export import physics_rows, result_rows
        content = CONTENTS[self._cmb_content.currentText()]
        try:
            return physics_rows(self._session) if content == "physics" else result_rows(self._session)
        except ValueError:
            return []

    def _refresh_preview(self, *_args) -> None:
        rows = self._current_rows()
        ok_btn = self._btn_box.button(QDialogButtonBox.StandardButton.Ok)
        if not rows:
            self._preview.clear()
            self._preview.setRowCount(0)
            self._preview.setColumnCount(0)
            self._lbl_count.setText("No results to export (fit some files first).")
            self._lbl_count.setStyleSheet("color:#e0a05c;font-size:11px;")
            if ok_btn is not None:
                ok_btn.setEnabled(False)
            return
        cols = list(rows[0].keys())
        shown = rows[:_PREVIEW_MAX_ROWS]
        self._preview.setColumnCount(len(cols))
        self._preview.setHorizontalHeaderLabels(cols)
        self._preview.setRowCount(len(shown))
        for r, row in enumerate(shown):
            for c, key in enumerate(cols):
                val = row.get(key)
                txt = f"{val:.4f}" if isinstance(val, float) else ("" if val is None else str(val))
                self._preview.setItem(r, c, QTableWidgetItem(txt))
        self._preview.resizeColumnsToContents()
        extra = f"  (showing first {_PREVIEW_MAX_ROWS})" if len(rows) > _PREVIEW_MAX_ROWS else ""
        self._lbl_count.setText(f"{len(rows)} row(s) × {len(cols)} column(s){extra}")
        self._lbl_count.setStyleSheet("color:#9fd49f;font-size:11px;")
        if ok_btn is not None:
            ok_btn.setEnabled(True)

    # -- result accessors ------------------------------------------------------
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
