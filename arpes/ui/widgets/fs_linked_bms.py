"""Widget showing BMs linked to the active FS map.

The widget exposes the FS → BM relation inside the FS tab. Double-clicking a BM
requests loading it through the main load controller. Row colors indicate the
projection quality.
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
    """Flat list of BMs attached to the active FS.

    Signals:
        bm_load_requested(str): double-clicked BM path.
    """
    bm_load_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

        self._title = QLabel("Linked BMs: (no active FS)")
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
            self._title.setText("Linked BMs: (no active FS)")
            return
        from pathlib import Path as _P
        fs_name = _P(active_fs_path).name
        if not cuts:
            self._title.setText(f"BMs linked to {fs_name}: none")
            return
        self._title.setText(f"BMs linked to {fs_name}: {len(cuts)}")
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

    def refresh_matches(self, active_fs_path: str | None, matches: list) -> None:
        """Populate from pairing matches — no work function / lattice needed.

        Shows the FS↔BM links (BM name + hv/azi/direction) even when the cut
        geometry can't be computed yet (e.g. φ not set). ``matches`` is a list of
        PairingMatch (``.path``, ``.entry``, ``.reason``).
        """
        self._list.clear()
        if not active_fs_path:
            self._title.setText("Linked BMs: (no active FS)")
            return
        from pathlib import Path as _P
        fs_name = _P(active_fs_path).name
        if not matches:
            self._title.setText(f"BMs linked to {fs_name}: none")
            return
        self._title.setText(f"BMs linked to {fs_name}: {len(matches)}")
        for m in matches:
            meta = getattr(m.entry, "meta", None)
            hv = getattr(meta, "hv", 0.0) or 0.0
            azi = getattr(meta, "azi", None)
            direction = getattr(meta, "direction", "") or ""
            bits = [_P(m.path).name, f"hv={float(hv):.1f} eV"]
            if azi is not None:
                bits.append(f"azi={float(azi):+.1f}°")
            if direction:
                bits.append(direction)
            bits.append(f"[{m.reason}]")
            item = QListWidgetItem("  ·  ".join(bits))
            item.setData(Qt.ItemDataRole.UserRole, m.path)
            item.setForeground(QColor("#00d4ff" if m.reason == "manual" else "#bcd"))
            self._list.addItem(item)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.bm_load_requested.emit(str(path))
