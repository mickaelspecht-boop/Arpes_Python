"""Compact multi-zone MDC table widget.

Lives at the top of the Fit panel. One row per zone: an "active" checkbox
(include in Run all), a color swatch + editable name, the k/E window, and the
fitted-point count. Row selection picks the zone whose parameters are loaded
into the panel; the checkbox is independent of selection.

Emits Qt signals consumed by FitZonesController / FitRunnerController. The
public API (signals + ``set_zones`` / ``current_zone_id``) is intentionally
identical to the previous combo-based widget so callers do not change.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.controllers.fit_zones_controller import ZONE_PALETTE

_COL_ON, _COL_NAME, _COL_MODEL, _COL_WINDOW, _COL_FIT = range(5)


def _color_swatch(hex_color: str, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(hex_color))
    return QIcon(pm)


def _color_of(zone: dict) -> str:
    idx = int(zone.get("color_idx", 0)) % len(ZONE_PALETTE)
    return ZONE_PALETTE[idx]


def _window_text(zone: dict) -> str:
    fp = zone.get("fit_params", {}) or {}
    try:
        return (
            f"k[{float(fp['k_min']):+.2f},{float(fp['k_max']):+.2f}] "
            f"E[{float(fp['ev_start']):+.2f},{float(fp['ev_end']):+.2f}]"
        )
    except (KeyError, TypeError, ValueError):
        return "—"


def _fit_text(zone: dict) -> str:
    fr = zone.get("fit_result")
    if not fr:
        return "—"
    e = fr.get("e_fitted")
    n = 0 if e is None else len(e)
    return f"✓ {n} pts"


class ZonesStrip(QWidget):
    add_zone_requested = pyqtSignal()
    remove_zone_requested = pyqtSignal(str)
    rename_zone_requested = pyqtSignal(str, str)
    model_changed = pyqtSignal(str, str)
    active_zone_changed = pyqtSignal(str)
    toggle_zone_active = pyqtSignal(str, bool)
    run_all_zones_requested = pyqtSignal()
    clear_zone_results = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._zones: list[dict] = []
        self._suppress = False
        self._build()

    # --------------------------------------------------------------- build UI
    def _build(self) -> None:
        grp = QGroupBox("Zones MDC", self)
        outer = QVBoxLayout(grp)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(4)
        self.btn_add = QPushButton("+ from ROI")
        self.btn_add.setToolTip(
            "Create a zone from the current analysis range "
            "(k_min/k_max/E_start/E_end). Edit the range to refine it."
        )
        self.btn_add.clicked.connect(self.add_zone_requested.emit)
        self.btn_run_all = QPushButton("▶ Run all")
        self.btn_run_all.setToolTip("Run an independent MDC fit for each active zone.")
        self.btn_run_all.setStyleSheet(
            "background:#1a5a99;color:white;padding:3px 8px;font-weight:bold;"
        )
        self.btn_run_all.clicked.connect(self.run_all_zones_requested.emit)
        self.btn_clear = QToolButton()
        self.btn_clear.setText("⌫")
        self.btn_clear.setToolTip("Clear all zone fit results.")
        self.btn_clear.clicked.connect(self.clear_zone_results.emit)
        header.addWidget(self.btn_add)
        header.addWidget(self.btn_run_all)
        header.addStretch(1)
        header.addWidget(self.btn_clear)
        outer.addLayout(header)

        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["On", "Zone", "Model", "k / E window", "Fit"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.tbl.setMaximumHeight(150)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(_COL_ON, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(_COL_MODEL, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_WINDOW, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_FIT, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.setToolTip(
            "Select a row to load that zone's parameters into the panel.\n"
            "'On' = include the zone in 'Run all'. Double-click the name to rename.\n"
            "Model: Peak pairs = Γ-symmetric pair fit; Free region = independent Lorentzians inside this zone."
        )
        self.tbl.itemSelectionChanged.connect(self._on_selection_changed)
        self.tbl.itemChanged.connect(self._on_item_changed)
        outer.addWidget(self.tbl)

        footer = QHBoxLayout()
        footer.setSpacing(4)
        self.lbl_info = QLabel("No zone — '+ from ROI' to create one.")
        self.lbl_info.setStyleSheet("color:#888;")
        self.btn_rename = QToolButton()
        self.btn_rename.setText("Rename")
        self.btn_rename.setToolTip("Rename the selected zone.")
        self.btn_rename.clicked.connect(self._begin_rename)
        self.btn_remove = QToolButton()
        self.btn_remove.setText("− Remove")
        self.btn_remove.setToolTip("Remove the selected zone.")
        self.btn_remove.clicked.connect(self._on_remove)
        footer.addWidget(self.lbl_info, 1)
        footer.addWidget(self.btn_rename)
        footer.addWidget(self.btn_remove)
        outer.addLayout(footer)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(grp)

    # ----------------------------------------------------------- public API
    def set_zones(self, zones: list[dict], active_id: str | None) -> None:
        self._suppress = True
        try:
            self._zones = list(zones or [])
            self.tbl.setRowCount(len(self._zones))
            sel_row = -1
            for row, z in enumerate(self._zones):
                zid = z.get("id")
                if zid == active_id:
                    sel_row = row

                on = QTableWidgetItem()
                on.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                on.setCheckState(
                    Qt.CheckState.Checked if z.get("active", True)
                    else Qt.CheckState.Unchecked
                )
                self.tbl.setItem(row, _COL_ON, on)

                name = QTableWidgetItem(str(z.get("label", "?")))
                name.setIcon(_color_swatch(_color_of(z)))
                name.setData(Qt.ItemDataRole.UserRole, zid)
                name.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEditable
                )
                self.tbl.setItem(row, _COL_NAME, name)

                cmb = QComboBox()
                cmb.addItem("Peak pairs", "peak_pair")
                cmb.addItem("Free region", "free_region")
                model = str(z.get("fit_model", "peak_pair") or "peak_pair")
                cmb.setCurrentIndex(1 if model == "free_region" else 0)
                cmb.setToolTip(
                    "Peak pairs: symmetric kF−/kF+ model with shared center Γ.\n"
                    "Free region: independent Lorentzian peak(s) inside this k window; use for one-sided/off-Γ bands."
                )
                cmb.currentIndexChanged.connect(
                    lambda _idx, zid=zid, combo=cmb: self._on_model_changed(str(zid), str(combo.currentData()))
                )
                self.tbl.setCellWidget(row, _COL_MODEL, cmb)

                self.tbl.setItem(row, _COL_WINDOW, self._readonly_item(_window_text(z)))
                self.tbl.setItem(row, _COL_FIT, self._readonly_item(_fit_text(z)))

            if sel_row >= 0:
                self.tbl.selectRow(sel_row)
            self._refresh_footer()
        finally:
            self._suppress = False

    def current_zone_id(self) -> str | None:
        row = self.tbl.currentRow()
        if row < 0:
            return None
        item = self.tbl.item(row, _COL_NAME)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    # ----------------------------------------------------------- internals
    @staticmethod
    def _readonly_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        return item

    def _refresh_footer(self) -> None:
        n = len(self._zones)
        has = n > 0
        self.btn_remove.setEnabled(has)
        self.btn_rename.setEnabled(has)
        self.btn_clear.setEnabled(has)
        self.btn_run_all.setEnabled(any(z.get("active", True) for z in self._zones))
        if not has:
            self.lbl_info.setText("No zone — '+ from ROI' to create one.")
            return
        zid = self.current_zone_id()
        label = next((z.get("label", "?") for z in self._zones if z.get("id") == zid), "—")
        n_active = sum(1 for z in self._zones if z.get("active", True))
        self.lbl_info.setText(f"selected: {label}   ·   {n} zone(s), {n_active} active")

    def _on_selection_changed(self) -> None:
        if self._suppress:
            return
        self._refresh_footer()
        zid = self.current_zone_id()
        if zid:
            self.active_zone_changed.emit(zid)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._suppress:
            return
        row = item.row()
        name_item = self.tbl.item(row, _COL_NAME)
        zid = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
        if not zid:
            return
        local = next((z for z in self._zones if z.get("id") == zid), None)
        if item.column() == _COL_ON:
            on = item.checkState() == Qt.CheckState.Checked
            if local is not None:
                local["active"] = on  # keep footer / Run-all enabled state live
            self.toggle_zone_active.emit(str(zid), on)
            self._refresh_footer()
        elif item.column() == _COL_NAME:
            new_label = item.text().strip()
            if new_label:
                if local is not None:
                    local["label"] = new_label
                self.rename_zone_requested.emit(str(zid), new_label)

    def _on_model_changed(self, zid: str, model: str) -> None:
        if self._suppress:
            return
        local = next((z for z in self._zones if z.get("id") == zid), None)
        if local is not None:
            local["fit_model"] = model
            local["fit_result"] = None
        self.model_changed.emit(zid, model)
        self._refresh_footer()

    def _begin_rename(self) -> None:
        row = self.tbl.currentRow()
        if row < 0:
            return
        item = self.tbl.item(row, _COL_NAME)
        if item is not None:
            self.tbl.editItem(item)

    def _on_remove(self) -> None:
        zid = self.current_zone_id()
        if zid:
            self.remove_zone_requested.emit(str(zid))
