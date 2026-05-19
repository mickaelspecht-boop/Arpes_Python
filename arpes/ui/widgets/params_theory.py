"""Section DFT / Théorie du FitParamsPanel.

Builder externe pour le groupbox theory overlay (MP-ID, segment, alignement
manuel, boutons import/comparer/vider, recherche par formule).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
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
    panel.txt_theory_mpid.setToolTip(
        "Materials Project ID. Entrée importe directement.\n"
        "Nécessite mp-api + MP_API_KEY."
    )
    panel.txt_theory_mpid.returnPressed.connect(panel.theory_import_requested)
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
    panel.cmb_theory_convention = QComboBox()
    panel.cmb_theory_convention.addItem("MP bulk 3D", "mp_bulk")
    panel.cmb_theory_convention.addItem("ARPES pnictides 2D", "arpes_pnictides")
    panel.cmb_theory_convention.setToolTip(
        "Convention d'affichage des labels dans le picker.\n"
        "MP bulk = chemin 3D Materials Project brut.\n"
        "ARPES pnictides = ajoute les alias 2D usuels Γ/X/M/S en annotation."
    )
    panel.sp_theory_mu = dspin(0.0, -5.0, 5.0, 0.05, dec=3)
    panel.sp_theory_mu.setKeyboardTracking(False)
    panel.sp_theory_mu.setToolTip(
        "Shift chimique DFT μ (eV), avant renormalisation.\n"
        "Transformation overlay: E = Z × (E_DFT - μ).\n"
        "μ déplace les croisements de Fermi DFT; Z ajuste la dispersion."
    )
    # Alias legacy: d'anciens controllers/tests utilisent encore sp_theory_de.
    panel.sp_theory_de = panel.sp_theory_mu
    panel.sp_theory_z = dspin(1.0, 0.05, 5.0, 0.05, dec=3)
    panel.sp_theory_z.setKeyboardTracking(False)
    panel.sp_theory_z.setToolTip(
        "Renormalisation globale de dispersion Z.\n"
        "Z=1: DFT inchangée. Z<1: bandes plus plates.\n"
        "Appliquée globalement à toutes les bandes sélectionnées."
    )
    panel.sp_theory_dk = dspin(0.0, -5.0, 5.0, 0.02, dec=3)
    panel.sp_theory_dk.setKeyboardTracking(False)
    panel.sp_theory_dk.setToolTip("Décalage k manuel appliqué aux bandes DFT (π/a affiché).")
    panel.sp_theory_kscale = dspin(1.0, 0.1, 5.0, 0.05, dec=3)
    panel.sp_theory_kscale.setKeyboardTracking(False)
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
    panel.btn_theory_pick_bands = QPushButton("Choisir les bandes...")
    panel.btn_theory_pick_bands.setToolTip(
        "Ouvre le diagramme DFT et permet de cliquer les bandes a afficher."
    )
    panel.btn_theory_pick_bands.clicked.connect(panel.theory_band_picker_requested)
    # --- filtre fenêtre E_F (B+A) ---
    panel.sp_theory_efwin = dspin(0.0, 0.0, 10.0, 0.05, dec=3)
    panel.sp_theory_efwin.setKeyboardTracking(False)
    panel.sp_theory_efwin.setToolTip(
        "Fenêtre ±E autour de E_F (eV). 0 = désactivé.\n"
        "Filtre l'overlay et la liste : seules les bandes traversant\n"
        "cette fenêtre autour de E_F (E=0) sont gardées."
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
    for w in (panel.sp_theory_efwin, panel.chk_theory_ef_only,
              panel.chk_theory_color):
        sig = w.stateChanged if isinstance(w, QCheckBox) else w.valueChanged
        sig.connect(panel._schedule_theory_overlay_changed)
    panel.chk_theory_mirror = QCheckBox("Miroir Γ (k → -k)")
    panel.chk_theory_mirror.setToolTip(
        "Duplique les bandes DFT en miroir autour de Γ (k → -k).\n"
        "Utile pour scans symétriques [-X, Γ, +X] où le path Setyawan-Curtarolo\n"
        "ne couvre que la moitié droite. Valable pour cristaux à symétrie\n"
        "d'inversion (ex BaNi₂As₂ I4/mmm)."
    )
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
    btn_theory_efalign = QPushButton("Forcer μ=0")
    btn_theory_efalign.setToolTip(
        "Remet le shift chimique à μ = 0.\n"
        "Overlay: E = Z × E_DFT.\n"
        "C'est une référence DFT brute, pas un alignement ARPES/DFT calculé."
    )
    btn_theory_efalign.clicked.connect(panel.theory_efalign_requested)
    def _btn_grid(buttons):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(3)
        for i, b in enumerate(buttons):
            b.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            b.setMaximumWidth(170)
            g.addWidget(b, i // 3, i % 3)
        for c in range(3):
            g.setColumnStretch(c, 1)
        return w

    def _section(title):
        lbl = QLabel(title)
        lbl.setStyleSheet(
            "color:#9fc;font-size:10px;font-weight:bold;"
            "margin-top:6px;border-top:1px solid #444;padding-top:3px;"
        )
        return lbl

    # boutons regroupés par rôle (source vs diagnostic) au lieu d'une grille
    # fourre-tout : l'utilisateur retrouve l'action dans sa phase de travail.
    source_btns = _btn_grid([
        btn_theory_refresh, btn_theory_local_import, btn_theory_clear,
    ])
    align_btns = _btn_grid([btn_theory_align, btn_theory_efalign])
    diag_btns = _btn_grid([btn_theory_compare, btn_self_energy])

    panel.lbl_theory_status = QLabel("Guide visuel uniquement.")
    panel.lbl_theory_status.setWordWrap(True)
    panel.lbl_theory_status.setStyleSheet("color:#aaa;font-size:10px;")
    for w in (
        panel.chk_theory, panel.cmb_theory_segment, panel.cmb_theory_convention,
        panel.sp_theory_mu, panel.sp_theory_z,
        panel.sp_theory_dk, panel.sp_theory_kscale, panel.sp_theory_alpha,
        panel.sp_theory_max, panel.chk_theory_mirror,
    ):
        signal = w.stateChanged if isinstance(w, QCheckBox) else (
            w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
        )
        signal.connect(panel._schedule_theory_overlay_changed)

    # 1 · Source DFT
    fl_th.addRow(_section("1 · Source DFT"))
    fl_th.addRow(panel.chk_theory)
    fl_th.addRow("MP-ID:", mpid_row)
    fl_th.addRow("Convention:", panel.cmb_theory_convention)
    fl_th.addRow(source_btns)
    # 2 · Bandes affichées
    fl_th.addRow(_section("2 · Bandes affichées"))
    fl_th.addRow(panel.btn_theory_pick_bands)
    fl_th.addRow("Bandes idx:", panel.txt_theory_bands)
    fl_th.addRow("Fenêtre E_F (eV):", panel.sp_theory_efwin)
    fl_th.addRow(panel.chk_theory_ef_only)
    fl_th.addRow("Max bandes:", panel.sp_theory_max)
    fl_th.addRow(panel.chk_theory_color)
    fl_th.addRow(panel.chk_theory_projections)
    # 3 · Alignement DFT ↔ ARPES
    fl_th.addRow(_section("3 · Alignement DFT ↔ ARPES"))
    fl_th.addRow("Segment:", panel.cmb_theory_segment)
    fl_th.addRow("Shift μ (eV):", panel.sp_theory_mu)
    fl_th.addRow("Renorm Z:", panel.sp_theory_z)
    fl_th.addRow("Δk:", panel.sp_theory_dk)
    fl_th.addRow("Scale k:", panel.sp_theory_kscale)
    fl_th.addRow("a cristal (Å):", panel.sp_crystal_a)
    fl_th.addRow(panel.chk_theory_mirror)
    fl_th.addRow(align_btns)
    # 4 · Rendu & diagnostic
    fl_th.addRow(_section("4 · Rendu & diagnostic"))
    fl_th.addRow("Opacité:", panel.sp_theory_alpha)
    fl_th.addRow(diag_btns)
    fl_th.addRow(panel.lbl_theory_status)
    lay.addWidget(panel._theory_widget)
