"""Sections Énergie / EF-Chargement / Utilitaires du FitParamsPanel.

Builders externes pour rester sous le plafond 700 LOC dans params.py.
Chaque fonction reçoit le panel parent et le layout vertical, instancie les
widgets en les attachant à `panel.*` et connecte les signaux directement
sur les signaux du panel.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import dspin


def build_energy_section(panel, lay) -> None:
    panel._energy_widget = QGroupBox("Énergie sélectionnée")
    fl = QFormLayout(panel._energy_widget)
    panel.sp_ev = dspin(-0.30, -3.0, 0.2, 0.01)
    panel.sp_int_win = dspin(0.010, 0.001, 0.200, 0.005, dec=3)
    panel.sp_int_win.setToolTip(
        "Fenêtre d'intégration ±eV pour la MDC\n"
        "Élargir = moins de bruit, moins de résolution en énergie\n"
        "Correspond au 'range' d'extraction d'une coupe dans Igor"
    )
    panel.sp_int_win.valueChanged.connect(panel.fit_only_changed)
    fl.addRow("E (eV):", panel.sp_ev)
    fl.addRow("± intég. (eV):", panel.sp_int_win)
    fl.addRow(QLabel("Clic sur la carte ou ici"))
    lay.addWidget(panel._energy_widget)


def build_ef_section(panel, lay) -> None:
    panel._ef_widget = QGroupBox("EF / Chargement")
    fl_ef = QFormLayout(panel._ef_widget)
    panel.sp_phi = dspin(4.031, 3.0, 6.0, 0.01)
    panel.sp_phi.setToolTip("Fonction de travail φ (eV). Utilisée pour calculer E_kin → E−EF.")
    panel.sp_hv = dspin(0.0, 0.0, 500.0, 1.0)
    panel.sp_hv.setToolTip(
        "Énergie du photon incident (eV).\n"
        "→ CLS/LNLS : entrer manuellement AVANT de charger (obligatoire).\n"
        "→ Solaris/DA30 : lu automatiquement depuis le fichier.\n"
        "→ BESSY/SES : gardé pour diagnostic/kz; E−EF utilise automatiquement Center Energy."
    )
    panel.sp_ef = dspin(0.052, -0.3, 0.3, 0.005)
    panel.sp_ef.setToolTip(
        "Décalage EF en eV. Ajuste le zéro d'énergie.\n"
        "Utiliser 'Calibrer EF auto' pour le calculer par fit Fermi-Dirac."
    )
    btn_ef = QPushButton("Calibrer EF auto")
    btn_ef.clicked.connect(panel.ef_calib_requested)
    panel.btn_ef_ref = QPushButton("Aucune réf EF (calibrer un Au d'abord)")
    panel.btn_ef_ref.clicked.connect(panel.ef_apply_reference_requested)
    panel.btn_ef_ref.setEnabled(False)
    btn_log = QPushButton("Charger logbook")
    btn_log.clicked.connect(panel.logbook_requested)
    panel.btn_copy = QPushButton("Propager fit params (0 cible)")
    panel.btn_copy.clicked.connect(panel.copy_params_requested)
    panel.btn_copy.setEnabled(False)
    panel.update_ef_reference_button(None)
    panel.update_copy_params_button(0)
    panel.lbl_hv_src = QLabel("Inconnu")
    panel.lbl_hv_src.setToolTip(
        "Provenance de hν :\n"
        "Fichier = lue depuis le fichier\n"
        "Logbook = lue depuis le logbook\n"
        "Manuel = saisie manuelle\n"
        "Inconnu = source inconnue"
    )
    panel.lbl_hv_src.setMinimumWidth(58)
    hv_row = QWidget()
    hv_lay = QHBoxLayout(hv_row)
    hv_lay.setContentsMargins(0, 0, 0, 0)
    hv_lay.addWidget(panel.sp_hv, 1)
    hv_lay.addWidget(panel.lbl_hv_src)
    panel.sp_hv.valueChanged.connect(lambda _v: panel._mark_hv_manual_if_user_edit())
    panel._hv_source_lock = False
    fl_ef.addRow("φ (eV):", panel.sp_phi)
    fl_ef.addRow("hν (eV):", hv_row)
    fl_ef.addRow("EF offset:", panel.sp_ef)
    fl_ef.addRow(btn_log)
    fl_ef.addRow(btn_ef)
    fl_ef.addRow(panel.btn_ef_ref)
    fl_ef.addRow(panel.btn_copy)
    panel.lbl_action = QLabel("Dernière action : aucune")
    panel.lbl_action.setWordWrap(True)
    panel.lbl_action.setStyleSheet("color:#9fc;font-size:10px;")
    fl_ef.addRow(panel.lbl_action)
    lay.addWidget(panel._ef_widget)


def build_utils_section(panel, lay) -> None:
    panel._utils_widget = QGroupBox("Utilitaires")
    fl_ut = QFormLayout(panel._utils_widget)
    panel.sp_grid_strength = dspin(0.85, 0.0, 1.0, 0.05, dec=2)
    panel.sp_grid_strength.setToolTip(
        "Force de suppression de la trame affichée.\n"
        "0 = aucun effet, 1 = correction complète. Valeur conseillée : 0.8-0.9."
    )
    btn_grid = QPushButton("Retirer effet grille")
    btn_grid.setToolTip(
        "Active un masque Fourier 2D automatique sur la carte BM affichée.\n"
        "La donnée brute reste inchangée."
    )
    btn_grid.clicked.connect(panel.grid_requested)
    btn_grid_reset = QPushButton("Recharger brut")
    btn_grid_reset.setToolTip("Désactive la correction grille sauvegardée pour ce fichier.")
    btn_grid_reset.clicked.connect(panel.grid_reset_requested)
    panel.lbl_grid = QLabel("Correction BM : masque Fourier 2D automatique sur l'affichage.")
    panel.lbl_grid.setWordWrap(True)
    panel.lbl_grid.setStyleSheet("color:#aaa; font-size:10px;")
    fl_ut.addRow("Force:", panel.sp_grid_strength)
    fl_ut.addRow(btn_grid)
    fl_ut.addRow(btn_grid_reset)
    fl_ut.addRow(panel.lbl_grid)
    lay.addWidget(panel._utils_widget)
