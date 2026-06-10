"""Qt widgets for the KZ tab."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QLocale, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
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
from arpes.ui.widgets._qt_helpers import compact_button


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
    view: str
    show_points: bool
    show_planes: bool
    show_profile: bool


def _dspin(value, lo, hi, step, dec=3) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setLocale(QLocale(QLocale.Language.C))  # dot decimal regardless of system locale
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
    fit_v0_requested = pyqtSignal()
    params_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(root)

        btn_folder = compact_button(QPushButton("KZ Folder"), max_width=150)
        btn_folder.setToolTip("Load a folder containing a variable-hν band-map series.")
        btn_folder.clicked.connect(self.folder_requested)
        lay.addWidget(btn_folder)

        btn_logbook = compact_button(QPushButton("Logbook KZ"), max_width=150)
        btn_logbook.setToolTip("Load a logbook dedicated to the KZ series. File hν remains the priority.")
        btn_logbook.clicked.connect(self.kz_logbook_requested)
        lay.addWidget(btn_logbook)

        btn_redraw = compact_button(QPushButton("Redraw KZ"), max_width=150)
        btn_redraw.clicked.connect(self.redraw_requested)
        lay.addWidget(btn_redraw)

        btn_fit_v0 = compact_button(QPushButton("Fit V0"), max_width=150)
        btn_fit_v0.setToolTip(
            "Fit the inner potential V0 from the kz periodicity at normal "
            "emission (aligns E_F maxima with the Γ planes). Needs lattice c."
        )
        btn_fit_v0.clicked.connect(self.fit_v0_requested)
        lay.addWidget(btn_fit_v0)

        grp_phys = QGroupBox("KZ Parameters")
        form = QFormLayout(grp_phys)
        self.sp_v0 = _dspin(12.0, 0.1, 80.0, 0.5, dec=2)
        self.sp_v0.setToolTip(
            "Inner potential V0 (eV). Tune until periodic features in the kz "
            "view line up with the Γ/Z planes."
        )
        self.sp_a = _dspin(0.0, 0.0, 20.0, 0.01, dec=3)
        self.sp_c = _dspin(11.60, 1.0, 60.0, 0.01, dec=3)
        self.sp_e = _dspin(0.0, -5.0, 1.0, 0.01, dec=3)
        self.sp_ew = _dspin(0.03, 0.001, 0.5, 0.005, dec=3)
        form.addRow("V0 (eV):", self.sp_v0)
        form.addRow("a (Å):", self.sp_a)
        form.addRow("c (Å):", self.sp_c)
        form.addRow("E center:", self.sp_e)
        form.addRow("± E window:", self.sp_ew)
        lay.addWidget(grp_phys)

        grp_grid = QGroupBox("Display")
        form2 = QFormLayout(grp_grid)
        self.sp_k_bins = _ispin(240, 32, 800)
        self.sp_kz_bins = _ispin(240, 32, 800)
        self.cmb_unit = QComboBox()
        self.cmb_unit.addItems(["A^-1", "pi/c"])
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["per_scan_median", "none"])
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["k// vs kz", "k// vs hν (raw)"])
        self.cmb_view.setToolTip(
            "k// vs kz: converted I(k//, kz) map (needs lattice a & c).\n"
            "k// vs hν (raw): I(k//, hν) before any kz conversion (always available)."
        )
        self.chk_points = QCheckBox("sample points")
        self.chk_points.setToolTip("Overlay the raw (k//, kz) measured points on the map.")
        self.chk_planes = QCheckBox("Γ/Z planes")
        self.chk_planes.setChecked(True)
        self.chk_planes.setToolTip("Draw kz high-symmetry planes (Γ solid, Z dashed).")
        self.chk_profile = QCheckBox("I(kz) profile")
        self.chk_profile.setToolTip(
            "Overlay the normal-emission intensity profile I(kz) at k//≈0 "
            "(white curve); reports the FFT-implied lattice c."
        )
        form2.addRow("bins k:", self.sp_k_bins)
        form2.addRow("bins kz:", self.sp_kz_bins)
        form2.addRow("kz unit:", self.cmb_unit)
        form2.addRow("normalization:", self.cmb_norm)
        form2.addRow("view:", self.cmb_view)
        form2.addRow("overlays:", self.chk_points)
        form2.addRow("", self.chk_planes)
        form2.addRow("", self.chk_profile)
        lay.addWidget(grp_grid)

        self.lbl_info = QLabel("No KZ folder loaded.")
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
        self.cmb_view.currentIndexChanged.connect(self.params_changed)
        self.chk_points.stateChanged.connect(self.params_changed)
        self.chk_planes.stateChanged.connect(self.params_changed)
        self.chk_profile.stateChanged.connect(self.params_changed)

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
            view="raw" if "raw" in self.cmb_view.currentText().lower() else "kz",
            show_points=self.chk_points.isChecked(),
            show_planes=self.chk_planes.isChecked(),
            show_profile=self.chk_profile.isChecked(),
        )

    def set_info(self, text: str):
        self.lbl_info.setText(text)

    def autofill_lattice(self, a: float, c: float) -> None:
        """Fill a/c spinboxes from the sample when the user left them at 0."""
        for spin, value in ((self.sp_a, a), (self.sp_c, c)):
            if float(value) > 0 and float(spin.value()) <= 0:
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

    def force_raw_view(self) -> None:
        idx = next(
            (i for i in range(self.cmb_view.count())
             if "raw" in self.cmb_view.itemText(i).lower()),
            -1,
        )
        if idx >= 0 and self.cmb_view.currentIndex() != idx:
            self.cmb_view.blockSignals(True)
            self.cmb_view.setCurrentIndex(idx)
            self.cmb_view.blockSignals(False)


class KzCanvas(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.map = MplCanvas(figsize=(7, 6), toolbar=True)
        lay.addWidget(self.map)
