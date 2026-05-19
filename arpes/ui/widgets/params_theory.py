"""Section DFT / Théorie du FitParamsPanel.

Builder externe pour le groupbox theory overlay (MP-ID, segment, alignement
manuel, boutons import/comparer/vider, recherche par formule).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import compact_button, dspin, ispin


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
    panel.btn_theory_search = compact_button(QPushButton("Chercher MP"), max_width=130)
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
    panel.sp_crystal_a = dspin(4.143, 0.5, 50.0, 0.001, dec=4)
    panel.sp_crystal_a.setToolTip(
        "Paramètre de maille a (Å) du cristal. Sert à convertir kF (π/a) en\n"
        "Å⁻¹ et à calculer m*/m_e dans la table résultats. 0 = unités réduites."
    )
    panel.sp_crystal_a.valueChanged.connect(panel.crystal_a_changed)
    panel.txt_theory_bands = QLineEdit()
    panel.txt_theory_bands.setPlaceholderText("ex: 1,3,5-8 (vide = top-N)")
    panel.txt_theory_bands.setToolTip(
        "Indices bandes DFT à afficher (0-based). Ranges OK : `1,3,5-8`.\n"
        "Vide → sélection automatique top-N par overlap fenêtre Y."
    )
    panel.txt_theory_bands.editingFinished.connect(panel._on_theory_bands_text_edited)
    # --- filtre fenêtre E_F (B+A) ---
    panel.sp_theory_efwin = dspin(0.0, 0.0, 10.0, 0.05, dec=3)
    panel.sp_theory_efwin.setToolTip(
        "Fenêtre ±ΔE autour de E_F (eV). 0 = désactivé.\n"
        "Filtre l'overlay et la liste : seules les bandes traversant\n"
        "±ΔE de E_F (E=0) sont gardées."
    )
    panel.chk_theory_ef_only = QCheckBox("Seulement bandes croisant E_F")
    panel.chk_theory_ef_only.setToolTip(
        "Pré-coche uniquement les bandes traversant la fenêtre E_F\n"
        "ci-dessus et restreint l'overlay à celles-ci."
    )
    panel.chk_theory_color = QCheckBox("Couleur par bande + légende")
    panel.chk_theory_color.setChecked(True)
    panel.chk_theory_color.setToolTip(
        "Une couleur stable par index de bande + légende.\n"
        "Décoché → toutes les bandes en blanc (ancien rendu)."
    )
    panel.chk_theory_projections = QCheckBox("Récupérer projections orbitales (réseau)")
    panel.chk_theory_projections.setToolTip(
        "À l'import MP : tente de récupérer les projections orbitales\n"
        "pour afficher le caractère (ex Ti-d) par bande. Plus lent.\n"
        "Sans effet si Materials Project ne fournit pas les projections."
    )
    # --- table bandes cochable (remplace la saisie aveugle) ---
    panel.tbl_theory_bands = QTableWidget(0, 5)
    panel.tbl_theory_bands.setHorizontalHeaderLabels(
        ["#", "E min", "E max", "E_F", "caractère"]
    )
    panel.tbl_theory_bands.verticalHeader().setVisible(False)
    panel.tbl_theory_bands.setEditTriggers(
        QAbstractItemView.EditTrigger.NoEditTriggers
    )
    panel.tbl_theory_bands.setSelectionMode(
        QAbstractItemView.SelectionMode.NoSelection
    )
    panel.tbl_theory_bands.setAlternatingRowColors(True)
    panel.tbl_theory_bands.setMaximumHeight(190)
    _hdr = panel.tbl_theory_bands.horizontalHeader()
    _hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    _hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
    panel.tbl_theory_bands.setToolTip(
        "Coche les bandes DFT à afficher. Vide = sélection auto top-N.\n"
        "Se synchronise avec le champ « Bandes idx » (format 1,3,5-8)."
    )
    panel.tbl_theory_bands.itemChanged.connect(panel._on_theory_band_table_toggled)
    for w in (panel.sp_theory_efwin, panel.chk_theory_ef_only,
              panel.chk_theory_color):
        sig = w.stateChanged if isinstance(w, QCheckBox) else w.valueChanged
        sig.connect(panel.theory_overlay_changed)
    panel.chk_theory_mirror = QCheckBox("Miroir Γ (k → -k)")
    panel.chk_theory_mirror.setToolTip(
        "Duplique les bandes DFT en miroir autour de Γ (k → -k).\n"
        "Utile pour scans symétriques [-X, Γ, +X] où le path Setyawan-Curtarolo\n"
        "ne couvre que la moitié droite. Valable pour cristaux à symétrie\n"
        "d'inversion (ex BaNi₂As₂ I4/mmm)."
    )
    btn_theory_import = QPushButton("Importer MP")
    btn_theory_import.setToolTip("Importe les bandes DFT depuis le MP-ID saisi (réseau).")
    btn_theory_import.clicked.connect(panel.theory_import_requested)
    btn_theory_refresh = QPushButton("Rafraîchir MP")
    btn_theory_refresh.setToolTip(
        "Ré-importe le MP-ID en ignorant le cache disque.\n"
        "Nécessaire pour récupérer le vrai chemin de bandes MP\n"
        "(branches) si l'import a été mis en cache avant cette version."
    )
    btn_theory_refresh.clicked.connect(panel.theory_refresh_requested)
    btn_theory_local_import = QPushButton("Importer local")
    btn_theory_local_import.setToolTip(
        "Importe des bandes DFT locales depuis vasprun.xml, table QE .dat/.txt,\n"
        "ou schema YAML/JSON minimal."
    )
    btn_theory_local_import.clicked.connect(panel.theory_local_import_requested)
    btn_theory_clear = QPushButton("Vider")
    btn_theory_clear.setToolTip("Retire l'overlay DFT courant.")
    btn_theory_clear.clicked.connect(panel.theory_clear_requested)
    btn_theory_compare = QPushButton("Comparer au fit")
    btn_theory_compare.setToolTip(
        "Score les bandes DFT contre les points kF fittés.\n"
        "Diagnostic visuel uniquement, sans modifier le fit."
    )
    btn_theory_compare.clicked.connect(panel.theory_compare_requested)
    btn_self_energy = QPushButton("Re Σ(E)")
    btn_self_energy.setToolTip(
        "Calcule Re Σ(E) = E_exp − E_DFT(k_exp) avec la meilleure bande DFT\n"
        "contre le fit MDC courant, puis ouvre un plot diagnostic."
    )
    btn_self_energy.clicked.connect(panel.self_energy_requested)
    btn_theory_align = QPushButton("Aligner π/a")
    btn_theory_align.setToolTip(
        "Calcule scale k et Δk pour mapper le segment choisi sur [0, 1] (π/a).\n"
        "Premier label du segment → 0, second → 1."
    )
    btn_theory_align.clicked.connect(panel.theory_align_requested)
    btn_theory_efalign = QPushButton("Aligner E_F")
    btn_theory_efalign.setToolTip(
        "Force ΔE = 0. Bandes DFT centrées sur E_F = 0 (efermi MP déjà soustrait).\n"
        "Utiliser après calibration EF ARPES pour vérifier le matching d'énergie."
    )
    btn_theory_efalign.clicked.connect(panel.theory_efalign_requested)
    theory_btns = QWidget()
    theory_btns_lay = QGridLayout(theory_btns)
    theory_btns_lay.setContentsMargins(0, 0, 0, 0)
    theory_btns_lay.setHorizontalSpacing(4)
    theory_btns_lay.setVerticalSpacing(3)
    # grille 3 colonnes : largeur min ≈ 3 boutons au lieu de 7 → la colonne de
    # paramètres ne déborde plus l'écran.
    _grid_btns = [
        btn_theory_import, btn_theory_refresh, btn_theory_local_import,
        btn_theory_clear, btn_theory_align, btn_theory_efalign,
        btn_theory_compare, btn_self_energy,
    ]
    for i, b in enumerate(_grid_btns):
        b.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        b.setMaximumWidth(170)
        theory_btns_lay.addWidget(b, i // 3, i % 3)
    for c in range(3):
        theory_btns_lay.setColumnStretch(c, 1)
    panel.lbl_theory_status = QLabel("Guide visuel uniquement.")
    panel.lbl_theory_status.setWordWrap(True)
    panel.lbl_theory_status.setStyleSheet("color:#aaa;font-size:10px;")
    for w in (
        panel.chk_theory, panel.cmb_theory_segment, panel.sp_theory_de,
        panel.sp_theory_dk, panel.sp_theory_kscale, panel.sp_theory_alpha,
        panel.sp_theory_max, panel.chk_theory_mirror,
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
    fl_th.addRow("Fenêtre E_F (eV):", panel.sp_theory_efwin)
    fl_th.addRow(panel.chk_theory_ef_only)
    fl_th.addRow(panel.tbl_theory_bands)
    fl_th.addRow("Bandes idx:", panel.txt_theory_bands)
    fl_th.addRow(panel.chk_theory_color)
    fl_th.addRow(panel.chk_theory_projections)
    fl_th.addRow("a cristal (Å):", panel.sp_crystal_a)
    fl_th.addRow(panel.chk_theory_mirror)
    fl_th.addRow(theory_btns)
    fl_th.addRow(panel.lbl_theory_status)
    lay.addWidget(panel._theory_widget)
