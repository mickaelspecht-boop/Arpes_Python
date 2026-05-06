"""Section DFT / Théorie du FitParamsPanel.

Builder externe pour le groupbox theory overlay (MP-ID, segment, alignement
manuel, boutons import/comparer/vider, recherche par formule).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import dspin, ispin


def build_theory_section(panel, lay) -> None:
    panel._theory_widget = QGroupBox("DFT / Théorie")
    fl_th = QFormLayout(panel._theory_widget)
    panel.chk_theory = QCheckBox("Afficher overlay DFT")
    panel.chk_theory.setToolTip(
        "Affiche des bandes DFT importées comme guide visuel.\n"
        "Aucun fit ni correction n'utilise ces bandes automatiquement."
    )
    panel.txt_theory_mpid = QLineEdit()
    panel.txt_theory_mpid.setPlaceholderText("mp-149")
    panel.txt_theory_mpid.setToolTip("Materials Project ID. Nécessite mp-api + MP_API_KEY.")
    panel.btn_theory_search = QPushButton("Chercher MP")
    panel.btn_theory_search.setToolTip(
        "Recherche par formule chimique sur Materials Project (réseau).\n"
        "Ouvre un dialog avec candidats et leur MPID."
    )
    panel.btn_theory_search.clicked.connect(panel.theory_search_requested)
    mpid_row = QWidget()
    mpid_lay = QHBoxLayout(mpid_row)
    mpid_lay.setContentsMargins(0, 0, 0, 0)
    mpid_lay.addWidget(panel.txt_theory_mpid, 1)
    mpid_lay.addWidget(panel.btn_theory_search)
    panel.cmb_theory_segment = QComboBox()
    panel.cmb_theory_segment.setEditable(True)
    panel.cmb_theory_segment.setToolTip("Segment DFT proposé depuis la direction logbook, modifiable.")
    panel.sp_theory_de = dspin(0.0, -5.0, 5.0, 0.05, dec=3)
    panel.sp_theory_de.setToolTip("Décalage énergie manuel appliqué aux bandes DFT (eV).")
    panel.sp_theory_dk = dspin(0.0, -5.0, 5.0, 0.02, dec=3)
    panel.sp_theory_dk.setToolTip("Décalage k manuel appliqué aux bandes DFT (π/a affiché).")
    panel.sp_theory_kscale = dspin(1.0, 0.1, 5.0, 0.05, dec=3)
    panel.sp_theory_kscale.setToolTip("Facteur d'échelle k manuel pour rapprocher axe DFT et axe ARPES.")
    panel.sp_theory_alpha = dspin(0.65, 0.05, 1.0, 0.05, dec=2)
    panel.sp_theory_max = ispin(10, 1, 80)
    btn_theory_import = QPushButton("Importer MP")
    btn_theory_import.clicked.connect(panel.theory_import_requested)
    btn_theory_clear = QPushButton("Vider")
    btn_theory_clear.clicked.connect(panel.theory_clear_requested)
    btn_theory_compare = QPushButton("Comparer au fit")
    btn_theory_compare.setToolTip(
        "Score les bandes DFT contre les points kF fittés.\n"
        "Diagnostic visuel uniquement, sans modifier le fit."
    )
    btn_theory_compare.clicked.connect(panel.theory_compare_requested)
    btn_theory_align = QPushButton("Aligner π/a")
    btn_theory_align.setToolTip(
        "Calcule scale k et Δk pour mapper le segment choisi sur [0, 1] (π/a).\n"
        "Premier label du segment → 0, second → 1."
    )
    btn_theory_align.clicked.connect(panel.theory_align_requested)
    theory_btns = QWidget()
    theory_btns_lay = QHBoxLayout(theory_btns)
    theory_btns_lay.setContentsMargins(0, 0, 0, 0)
    theory_btns_lay.addWidget(btn_theory_import)
    theory_btns_lay.addWidget(btn_theory_align)
    theory_btns_lay.addWidget(btn_theory_compare)
    theory_btns_lay.addWidget(btn_theory_clear)
    panel.lbl_theory_status = QLabel("Guide visuel uniquement.")
    panel.lbl_theory_status.setWordWrap(True)
    panel.lbl_theory_status.setStyleSheet("color:#aaa;font-size:10px;")
    for w in (
        panel.chk_theory, panel.cmb_theory_segment, panel.sp_theory_de,
        panel.sp_theory_dk, panel.sp_theory_kscale, panel.sp_theory_alpha,
        panel.sp_theory_max,
    ):
        signal = w.stateChanged if isinstance(w, QCheckBox) else (
            w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
        )
        signal.connect(panel.theory_overlay_changed)
    fl_th.addRow(panel.chk_theory)
    fl_th.addRow("MP-ID:", mpid_row)
    fl_th.addRow("Segment:", panel.cmb_theory_segment)
    fl_th.addRow("ΔE (eV):", panel.sp_theory_de)
    fl_th.addRow("Δk:", panel.sp_theory_dk)
    fl_th.addRow("Scale k:", panel.sp_theory_kscale)
    fl_th.addRow("Opacité:", panel.sp_theory_alpha)
    fl_th.addRow("Max bandes:", panel.sp_theory_max)
    fl_th.addRow(theory_btns)
    fl_th.addRow(panel.lbl_theory_status)
    lay.addWidget(panel._theory_widget)
