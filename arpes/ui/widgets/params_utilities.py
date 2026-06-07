"""Utilities section grouped into independent collapsible subsections.

Three subsections: grid filter, DFT/theory, BM distortion. Each section has a
clickable title button that opens/closes it. There is no exclusivity constraint
(unlike QToolBox), so all sections can be closed or open.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget


class _CollapsibleSection(QWidget):
    """Clickable header plus collapsible content widget."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        open_default: bool = False,
        summary: str = "",
    ):
        super().__init__()
        self._title = title
        self._summary = summary
        self._content = content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.btn = QPushButton()
        self.btn.setCheckable(True)
        self.btn.setStyleSheet(
            "QPushButton { background:#3a3a4a; color:#cde; padding:6px 8px;"
            " border-radius:3px; font-weight:bold; text-align:left; }"
            "QPushButton:checked { background:#4a4a6a; color:#fff; }"
            "QPushButton:hover { background:#454560; }"
        )
        self.btn.clicked.connect(self._on_toggle)
        lay.addWidget(self.btn)
        lay.addWidget(content)
        self.set_open(open_default)

    def _on_toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        arrow = "▼" if checked else "▶"
        suffix = f" · {self._summary}" if self._summary and not checked else ""
        self.btn.setText(f"{arrow}  {self._title}{suffix}")

    def set_open(self, opened: bool) -> None:
        self.btn.setChecked(bool(opened))
        self._on_toggle(bool(opened))


def build_utilities_section(panel, lay) -> None:
    from arpes.ui.widgets.params_distortion import build_bm_distortion_section
    from arpes.ui.widgets.params_ef import build_utils_section
    from arpes.ui.widgets.params_theory import build_theory_section

    panel._utilities_container = QWidget()
    cv = QVBoxLayout(panel._utilities_container)
    cv.setContentsMargins(0, 0, 0, 0)
    cv.setSpacing(4)

    def _wrap(builder, title, *, open_default=False, summary="") -> _CollapsibleSection:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(2, 2, 2, 2)
        builder(panel, page_lay)
        # vide le titre du QGroupBox interne pour éviter une double bordure
        for attr in ("_utils_widget", "_theory_widget", "_distortion_widget"):
            w = getattr(panel, attr, None)
            if w is not None and w.parent() is page:
                w.setTitle("")
                w.setFlat(True)
        sec = _CollapsibleSection(
            title, page, open_default=open_default, summary=summary
        )
        cv.addWidget(sec)
        return sec

    panel._sec_grid = _wrap(build_utils_section, "Grid filter (FFT)",
                             open_default=False, summary="force 0.85")
    panel._sec_theory = _wrap(build_theory_section, "DFT / Theory",
                               open_default=False, summary="off")
    panel._sec_distortion = _wrap(build_bm_distortion_section, "BM Distortion",
                                   open_default=False, summary="disabled")

    lay.addWidget(panel._utilities_container)
    # alias rétrocompat avec ancien attribut (set_context lit _utilities_toolbox)
    panel._utilities_toolbox = panel._utilities_container
