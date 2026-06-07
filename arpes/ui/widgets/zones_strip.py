"""Compact zones strip widget: combo of zones + add/remove/run-all buttons.

Lives at the top of the Fit panel. Emits Qt signals consumed by the
FitZonesController and FitRunnerController.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)


def _color_swatch(hex_color: str, size: int = 14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(hex_color))
    return QIcon(pm)


class ZonesStrip(QWidget):
    add_zone_requested = pyqtSignal()
    remove_zone_requested = pyqtSignal(str)
    rename_zone_requested = pyqtSignal(str, str)
    active_zone_changed = pyqtSignal(str)
    toggle_zone_active = pyqtSignal(str, bool)
    run_all_zones_requested = pyqtSignal()
    clear_zone_results = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._zones: list[dict] = []
        self._suppress = False
        grp = QGroupBox("Zones MDC", self)
        lay = QHBoxLayout(grp)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        self.cmb = QComboBox()
        self.cmb.setMinimumWidth(140)
        self.cmb.setToolTip(
            "MDC zones. Selecting a zone loads its parameters into the "
            "panel. 'Full fit' updates the active zone."
        )
        self.cmb.currentIndexChanged.connect(self._on_combo_changed)
        self.chk_active = QCheckBox("Active")
        self.chk_active.setToolTip("Include this zone in 'Run all'.")
        self.chk_active.toggled.connect(self._on_toggle_active)
        self.btn_add = QToolButton()
        self.btn_add.setText("+")
        self.btn_add.setToolTip(
            "Create a zone from the current bounds (k_min/k_max/E_start/E_end)."
        )
        self.btn_add.clicked.connect(self.add_zone_requested.emit)
        self.btn_remove = QToolButton()
        self.btn_remove.setText("−")
        self.btn_remove.setToolTip("Remove the selected zone.")
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_run_all = QPushButton("▶ Run all")
        self.btn_run_all.setToolTip(
            "Run an independent MDC fit for each active zone."
        )
        self.btn_run_all.clicked.connect(self.run_all_zones_requested.emit)
        self.btn_run_all.setStyleSheet(
            "background:#1a5a99;color:white;padding:3px 8px;font-weight:bold;"
        )
        self.btn_clear = QToolButton()
        self.btn_clear.setText("⌫")
        self.btn_clear.setToolTip("Clear all zone fit results.")
        self.btn_clear.clicked.connect(self.clear_zone_results.emit)
        self.lbl_info = QLabel("0 zone")
        self.lbl_info.setStyleSheet("color:#888;")
        self.lbl_info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for w in (
            QLabel("Zone:"), self.cmb, self.chk_active,
            self.btn_add, self.btn_remove, self.btn_run_all, self.btn_clear,
            self.lbl_info,
        ):
            lay.addWidget(w)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(grp)

    def set_zones(self, zones: list[dict], active_id: str | None) -> None:
        self._suppress = True
        try:
            self._zones = list(zones or [])
            self.cmb.clear()
            for z in self._zones:
                label = z.get("label", "?")
                ok = z.get("fit_result") is not None
                marker = " ✓" if ok else ""
                self.cmb.addItem(_color_swatch(self._color_of(z)), f"{label}{marker}", z.get("id"))
            if active_id:
                idx = self.cmb.findData(active_id)
                if idx >= 0:
                    self.cmb.setCurrentIndex(idx)
            self._refresh_active_checkbox()
            self.lbl_info.setText(
                f"{len(self._zones)} zone{'s' if len(self._zones) > 1 else ''}"
            )
            self.btn_remove.setEnabled(bool(self._zones))
            self.chk_active.setEnabled(bool(self._zones))
            self.btn_run_all.setEnabled(any(z.get("active", True) for z in self._zones))
        finally:
            self._suppress = False

    def current_zone_id(self) -> str | None:
        return self.cmb.currentData()

    def _color_of(self, zone: dict) -> str:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
        ]
        idx = int(zone.get("color_idx", 0)) % len(palette)
        return palette[idx]

    def _on_combo_changed(self, _idx: int):
        if self._suppress:
            return
        zid = self.cmb.currentData()
        if zid:
            self._refresh_active_checkbox()
            self.active_zone_changed.emit(zid)

    def _on_remove(self):
        zid = self.cmb.currentData()
        if zid:
            self.remove_zone_requested.emit(zid)

    def _on_toggle_active(self, on: bool):
        if self._suppress:
            return
        zid = self.cmb.currentData()
        if zid:
            self.toggle_zone_active.emit(zid, bool(on))

    def _refresh_active_checkbox(self):
        zid = self.cmb.currentData()
        for z in self._zones:
            if z.get("id") == zid:
                self.chk_active.blockSignals(True)
                self.chk_active.setChecked(bool(z.get("active", True)))
                self.chk_active.blockSignals(False)
                return
