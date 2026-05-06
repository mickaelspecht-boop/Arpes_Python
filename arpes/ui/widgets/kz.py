"""Widgets Qt pour l'onglet KZ."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets.canvas import MplCanvas


@dataclass(frozen=True)
class KzUiParams:
    inner_potential: float
    a_lattice: float
    c_lattice: float
    energy_center: float
    energy_window: float
    k_bins: int
    kz_bins: int
    kz_unit: str
    normalize: str
    display_mode: str


def _dspin(value, lo, hi, step, dec=3) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setDecimals(dec)
    sp.setValue(value)
    return sp


def _ispin(value, lo, hi) -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setValue(value)
    return sp


class KzControlPanel(QScrollArea):
    folder_requested = pyqtSignal()
    kz_logbook_requested = pyqtSignal()
    redraw_requested = pyqtSignal()
    params_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(root)

        btn_folder = QPushButton("Dossier KZ")
        btn_folder.setToolTip("Charge un dossier contenant une série de band maps à hν variable.")
        btn_folder.clicked.connect(self.folder_requested)
        lay.addWidget(btn_folder)

        btn_logbook = QPushButton("Logbook KZ")
        btn_logbook.setToolTip("Charge un logbook dédié à la série KZ. hν fichier reste prioritaire.")
        btn_logbook.clicked.connect(self.kz_logbook_requested)
        lay.addWidget(btn_logbook)

        btn_redraw = QPushButton("Redessiner KZ")
        btn_redraw.clicked.connect(self.redraw_requested)
        lay.addWidget(btn_redraw)

        grp_phys = QGroupBox("Paramètres KZ")
        form = QFormLayout(grp_phys)
        self.sp_v0 = _dspin(12.0, 0.1, 80.0, 0.5, dec=2)
        self.sp_v0.setToolTip("Potentiel interne V0 (eV). À calibrer par périodicité Γ-Z-Γ.")
        self.sp_a = _dspin(3.96, 1.0, 20.0, 0.01, dec=3)
        self.sp_c = _dspin(11.60, 1.0, 60.0, 0.01, dec=3)
        self.sp_e = _dspin(0.0, -5.0, 1.0, 0.01, dec=3)
        self.sp_ew = _dspin(0.03, 0.001, 0.5, 0.005, dec=3)
        form.addRow("V0 (eV):", self.sp_v0)
        form.addRow("a (Å):", self.sp_a)
        form.addRow("c (Å):", self.sp_c)
        form.addRow("E centre:", self.sp_e)
        form.addRow("± fenêtre E:", self.sp_ew)
        lay.addWidget(grp_phys)

        grp_grid = QGroupBox("Affichage")
        form2 = QFormLayout(grp_grid)
        self.sp_k_bins = _ispin(240, 32, 800)
        self.sp_kz_bins = _ispin(240, 32, 800)
        self.cmb_unit = QComboBox()
        self.cmb_unit.addItems(["A^-1", "pi/c"])
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["per_scan_median", "none"])
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["interpolated", "hv map", "MDC waterfall", "points", "binned"])
        form2.addRow("bins k:", self.sp_k_bins)
        form2.addRow("bins kz:", self.sp_kz_bins)
        form2.addRow("unité kz:", self.cmb_unit)
        form2.addRow("normalisation:", self.cmb_norm)
        form2.addRow("mode:", self.cmb_mode)
        lay.addWidget(grp_grid)

        self.lbl_info = QLabel("Aucun dossier KZ chargé.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#aaa;font-size:10px;")
        lay.addWidget(self.lbl_info)
        lay.addStretch(1)

        for widget in (
            self.sp_v0, self.sp_a, self.sp_c, self.sp_e, self.sp_ew,
            self.sp_k_bins, self.sp_kz_bins,
        ):
            widget.valueChanged.connect(self.params_changed)
        self.cmb_unit.currentIndexChanged.connect(self.params_changed)
        self.cmb_norm.currentIndexChanged.connect(self.params_changed)
        self.cmb_mode.currentIndexChanged.connect(self.params_changed)

    def params(self) -> KzUiParams:
        return KzUiParams(
            inner_potential=self.sp_v0.value(),
            a_lattice=self.sp_a.value(),
            c_lattice=self.sp_c.value(),
            energy_center=self.sp_e.value(),
            energy_window=self.sp_ew.value(),
            k_bins=self.sp_k_bins.value(),
            kz_bins=self.sp_kz_bins.value(),
            kz_unit=self.cmb_unit.currentText(),
            normalize=self.cmb_norm.currentText(),
            display_mode=self.cmb_mode.currentText(),
        )

    def set_info(self, text: str):
        self.lbl_info.setText(text)


class KzCanvas(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.map = MplCanvas(figsize=(7, 6), toolbar=True)
        lay.addWidget(self.map)
