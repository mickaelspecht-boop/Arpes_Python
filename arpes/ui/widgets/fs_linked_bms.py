"""Widget « BMs reliées à la FS active » — A.5 file tree minimaliste.

Au lieu de refactor le file browser principal, on expose la hiérarchie
FS → BMs **dans l'onglet FS lui-même** (sous-panneau dédié). Double-click
sur une BM → load via load_controller. Couleur par qualité de projection.

Maintenu volontairement minimal : pas de drag-drop, pas de menu contextuel
au-delà de pin/unpin via parent_fs_path (P3.bis ultérieur si besoin).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)


_QUALITY_COLOR = {
    "exact": "#00d4ff",
    "rotated": "#ffae42",
    "scaled": "#ff5544",
    "incompatible": "#888888",
}


class FsLinkedBmsList(QWidget):
    """Liste plate des BMs rattachées à la FS active.

    Signals:
        bm_load_requested(str) : double-click sur une BM → path à charger.
    """
    bm_load_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

        self._title = QLabel("BMs reliées : (aucune FS active)")
        self._title.setStyleSheet("color:#9ab; font-size:11px;")
        lay.addWidget(self._title)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background:#1d1d1d; color:#ddd; font-size:11px; }"
            "QListWidget::item:selected { background:#2a6099; }"
        )
        self._list.itemDoubleClicked.connect(self._on_double_click)
        lay.addWidget(self._list)

    def refresh(self, active_fs_path: str | None, cuts: list) -> None:
        """Re-popule la liste pour la FS active.

        Args:
            active_fs_path: path FS (str) ou None si pas de FS active.
            cuts: list[BMCutLine] retournée par `_collect_bm_cuts_for_active_fs`.
        """
        self._list.clear()
        if not active_fs_path:
            self._title.setText("BMs reliées : (aucune FS active)")
            return
        from pathlib import Path as _P
        fs_name = _P(active_fs_path).name
        if not cuts:
            self._title.setText(f"BMs reliées à {fs_name} : aucune")
            return
        self._title.setText(f"BMs reliées à {fs_name} : {len(cuts)}")
        for cut in cuts:
            label = (
                f"{cut.label}  ·  polar={cut.polar_bm:+.2f}°"
                f"  ·  hv={cut.hv_bm:.1f} eV"
                f"  ·  [{cut.quality}]"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, cut.bm_path)
            color = QColor(_QUALITY_COLOR.get(cut.quality, "#dddddd"))
            item.setForeground(color)
            if cut.warning:
                item.setToolTip(cut.warning)
            self._list.addItem(item)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.bm_load_requested.emit(str(path))
