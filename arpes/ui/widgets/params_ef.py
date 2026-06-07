"""Energy / EF-Loading / Utilities sections of FitParamsPanel.

External builders to stay under the 700 LOC limit in params.py.
Each function receives the parent panel and the vertical layout, instantiates
widgets by attaching them to `panel.*`, and connects signals directly
to panel signals.
"""
from __future__ import annotations

from arpes.core.session import DEFAULT_EF_OFFSET_EV
from PyQt6.QtCore import Qt, QStringListModel
from PyQt6.QtWidgets import (
    QCompleter,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import compact_button, dspin


def build_energy_section(panel, lay) -> None:
    panel._energy_widget = QGroupBox("Selected energy")
    fl = QFormLayout(panel._energy_widget)
    panel.sp_ev = dspin(-0.30, -3.0, 0.2, 0.01)
    panel.sp_int_win = dspin(0.010, 0.001, 0.200, 0.005, dec=3)
    panel.sp_int_win.setToolTip(
        "Integration window ±eV for the MDC\n"
        "Wider = less noise, less energy resolution\n"
        "Equivalent to the 'range' parameter of a cut in Igor"
    )
    panel.sp_int_win.valueChanged.connect(panel.fit_only_changed)
    fl.addRow("E (eV):", panel.sp_ev)
    fl.addRow("± integr. (eV):", panel.sp_int_win)
    fl.addRow(QLabel("Click on the map or type here"))
    lay.addWidget(panel._energy_widget)


def build_ef_section(panel, lay) -> None:
    panel._ef_widget = QGroupBox("EF / Loading")
    ef_root = QVBoxLayout(panel._ef_widget)
    ef_root.setContentsMargins(6, 6, 6, 6)
    ef_root.setSpacing(6)

    grp_calib = QGroupBox("Energy calibration")
    fl_calib = QFormLayout(grp_calib)
    grp_session = QGroupBox("Source / session")
    fl_session = QFormLayout(grp_session)

    panel.sp_phi = dspin(0.0, 0.0, 7.0, 0.01)
    panel.sp_phi.setToolTip(
        "Work function φ (eV). Used to compute E_kin → E−EF.\n"
        "0 = unknown: set φ or SampleConfig.work_function_eV before physical loading."
    )
    panel.sp_phi.valueChanged.connect(panel.work_function_changed)
    panel.sp_hv = dspin(0.0, 0.0, 500.0, 0.01, dec=4)
    panel.sp_hv.setFixedWidth(96)
    panel.sp_hv.setToolTip(
        "Incident photon energy (eV).\n"
        "→ CLS/LNLS: enter manually BEFORE loading (required).\n"
        "→ Solaris/DA30: read automatically from the file.\n"
        "→ BESSY/SES: kept for diagnostic/kz; E−EF uses Center Energy automatically."
    )
    panel.sp_ef = dspin(DEFAULT_EF_OFFSET_EV, -5.0, 5.0, 0.005)
    panel.sp_ef.setToolTip(
        "Scalar EF offset in eV. Adjusts the energy zero.\n"
        "Use 'Auto EF calibration' to compute it via Fermi-Dirac fit.\n"
        "Ignored when a polynomial EF calibration (ef_correction) is active: "
        "the polynomial takes precedence over the scalar."
    )
    btn_ef = compact_button(QPushButton("Auto EF calibration"))
    btn_ef.clicked.connect(panel.ef_calib_requested)
    panel.btn_ef_ref = compact_button(QPushButton("No EF ref"), max_width=240)
    panel.btn_ef_ref.clicked.connect(panel.ef_apply_reference_requested)
    panel.btn_ef_ref.setEnabled(False)
    btn_log = compact_button(QPushButton("Load logbook"))
    btn_log.clicked.connect(panel.logbook_requested)
    panel.btn_copy = compact_button(QPushButton("Propagate fit params (0 targets)"), max_width=240)
    panel.btn_copy.clicked.connect(panel.copy_params_requested)
    panel.btn_copy.setEnabled(False)
    panel.update_ef_reference_button(None)
    panel.update_copy_params_button(0)
    panel.lbl_hv_src = QLabel("Unknown")
    panel.lbl_hv_src.setToolTip(
        "hν provenance:\n"
        "File = read from the file\n"
        "Logbook = read from the logbook\n"
        "Manual = entered manually\n"
        "Unknown = source unknown"
    )
    panel.lbl_hv_src.setMinimumWidth(58)
    hv_row = QWidget()
    hv_lay = QHBoxLayout(hv_row)
    hv_lay.setContentsMargins(0, 0, 0, 0)
    hv_lay.addWidget(panel.sp_hv, 1)
    hv_lay.addWidget(panel.lbl_hv_src)
    panel.sp_hv.valueChanged.connect(lambda _v: panel._mark_hv_manual_if_user_edit())
    panel._hv_source_lock = False
    panel.txt_file_tags = QLineEdit()
    panel.txt_file_tags.setPlaceholderText("outliers, publi, T-dep")
    panel.txt_file_tags.setToolTip(
        "Free-form tags for the current file. Separate with commas.\n"
        "Saved in the session and usable in the browser filter."
    )
    panel._tag_completer_model = QStringListModel([])
    tag_completer = QCompleter(panel._tag_completer_model, panel.txt_file_tags)
    tag_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    panel.txt_file_tags.setCompleter(tag_completer)
    panel.txt_file_tags.editingFinished.connect(panel.file_tags_changed)
    fl_calib.addRow("φ (eV):", panel.sp_phi)
    fl_calib.addRow("hν (eV):", hv_row)
    fl_calib.addRow("EF offset:", panel.sp_ef)
    fl_calib.addRow(btn_ef)
    fl_calib.addRow(panel.btn_ef_ref)
    fl_session.addRow(btn_log)
    fl_session.addRow("Tags:", panel.txt_file_tags)
    fl_session.addRow(panel.btn_copy)
    panel.lbl_action = QLabel("Last action: none")
    panel.lbl_action.setWordWrap(True)
    panel.lbl_action.setStyleSheet("color:#9fc;font-size:10px;")
    fl_session.addRow(panel.lbl_action)
    ef_root.addWidget(grp_calib)
    ef_root.addWidget(grp_session)
    lay.addWidget(panel._ef_widget)


def build_utils_section(panel, lay) -> None:
    panel._utils_widget = QGroupBox("Utilities")
    fl_ut = QFormLayout(panel._utils_widget)
    panel.sp_grid_strength = dspin(0.85, 0.0, 1.0, 0.05, dec=2)
    panel.sp_grid_strength.setToolTip(
        "Suppression strength of the displayed grid artefact.\n"
        "0 = no effect, 1 = full correction. Recommended: 0.8–0.9."
    )
    btn_grid = compact_button(QPushButton("Remove grid artefact"))
    btn_grid.setToolTip(
        "Activates an automatic 2D Fourier mask on the displayed BM map.\n"
        "The raw data remains unchanged."
    )
    btn_grid.clicked.connect(panel.grid_requested)
    btn_grid_reset = compact_button(QPushButton("Reload raw"))
    btn_grid_reset.setToolTip("Disables the saved grid correction for this file.")
    btn_grid_reset.clicked.connect(panel.grid_reset_requested)
    panel.lbl_grid = QLabel("BM correction: automatic 2D Fourier mask on the display.")
    panel.lbl_grid.setWordWrap(True)
    panel.lbl_grid.setStyleSheet("color:#aaa; font-size:10px;")
    fl_ut.addRow("Strength:", panel.sp_grid_strength)
    fl_ut.addRow(btn_grid)
    fl_ut.addRow(btn_grid_reset)
    fl_ut.addRow(panel.lbl_grid)
    lay.addWidget(panel._utils_widget)
