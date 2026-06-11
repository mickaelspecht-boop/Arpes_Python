"""Construction des groupes ZDB/BZ du panneau FS (extrait de fs_panel.py, cap 700).

Free-functions prenant le ``FSControlPanel`` (``panel``) en 1er argument : elles
posent les widgets sur ``panel`` (mêmes noms d'attributs qu'avant) et les
ajoutent à ``lay`` via ``panel._add_collapsible_group``. Aucune logique métier.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
)

from arpes.ui.widgets._qt_helpers import compact_button


def build_bz_theoretical_group(panel, lay) -> None:
    grp_bz = QGroupBox("Theoretical BZ")
    fl3 = QFormLayout(grp_bz)
    panel.chk_bz = QCheckBox("Show BZ"); panel.chk_bz.setChecked(True)
    panel.chk_hsym = QCheckBox("Points Γ/X/M"); panel.chk_hsym.setChecked(True)
    panel.cmb_bz_shape = QComboBox(); panel.cmb_bz_shape.addItems(["square", "rectangle", "hexagon", "trigonal", "centered_rect", "oblique"])
    panel.sp_bzx = panel._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
    panel.sp_bzy = panel._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
    panel.sp_bz_angle = panel._dspin(90.0, 20.0, 160.0, 1.0, dec=1)
    panel.sp_klim = panel._dspin(1.3, 0.1, 10.0, 0.05, dec=2)
    panel.cmb_bz_shape.currentIndexChanged.connect(panel.params_changed)
    panel.cmb_bz_shape.currentIndexChanged.connect(panel._update_bz_angle_visibility)
    panel.chk_bz.stateChanged.connect(panel.params_changed)
    panel.chk_hsym.stateChanged.connect(panel.params_changed)
    btn_bz = compact_button(QPushButton("Choose BZ..."), max_width=160)
    btn_bz.setToolTip("Opens a selector with a diagram for choosing a 2D Bravais BZ.")
    btn_bz.clicked.connect(panel.bz_preset_requested)
    panel.btn_bz_labels = compact_button(QPushButton("Label conventions..."), max_width=160)
    panel.btn_bz_labels.setToolTip(
        "Rename the high-symmetry point labels (e.g. M → Σ) to match the\n"
        "convention used in a given article. Display-only: the geometry does\n"
        "not change, and logbook directions (Γ-Σ…) match the renamed labels."
    )
    panel.btn_bz_labels.clicked.connect(panel.bz_labels_requested)
    fl3.addRow(panel.chk_bz)
    fl3.addRow(panel.chk_hsym)
    fl3.addRow("Forme:", panel.cmb_bz_shape)
    fl3.addRow("demi-ZDB x:", panel.sp_bzx)
    fl3.addRow("demi-ZDB y:", panel.sp_bzy)
    fl3.addRow("lattice angle:", panel.sp_bz_angle)
    fl3.addRow("display limit:", panel.sp_klim)
    fl3.addRow(btn_bz)
    fl3.addRow(panel.btn_bz_labels)
    panel._update_bz_angle_visibility()
    panel._add_collapsible_group(lay, "Theoretical BZ", grp_bz, open_default=False)


def build_bz_crystal_group(panel, lay) -> None:
    # --- Groupbox : overlay BZ cristal réel (lattice MP) ---------------
    grp_xb = QGroupBox("Crystal BZ mapping (Materials Project)")
    fx = QFormLayout(grp_xb)
    panel.ed_mp_id = QLineEdit()
    panel.ed_mp_id.setPlaceholderText("mp-xxxx (optional: auto from logbook)")
    panel.btn_mp_fetch = compact_button(QPushButton("Fetch MP symmetry"), max_width=160)
    panel.btn_mp_fetch.setToolTip(
        "Fetches lattice parameters (a,b,c,α,β,γ) + space group\n"
        "from Materials Project. Disk cache + 10 s timeout.\n"
        "Reuses the logbook mp_id if left empty.\n"
        "DFT/GGA: useful for symmetry and FS/Luttinger area; masses and "
        "band positions may be shifted."
    )
    panel.btn_mp_fetch.clicked.connect(panel.mp_lattice_fetch_requested)
    panel.sp_v0 = panel._dspin(12.0, 0.5, 50.0, 0.5, dec=2)
    panel.sp_v0.setToolTip("V0 (eV) free-electron model for kz. ≠ φ. Typical 8–15 eV.")
    panel.cmb_kz_plane = QComboBox(); panel.cmb_kz_plane.addItems(["Auto", "Gamma", "Z"])
    panel.cmb_kz_plane.setToolTip("kz plane for HS labels (Γ/X/M or Z/R/A). Auto = via kz(hν,V0).")
    panel.cmb_kz_plane.currentIndexChanged.connect(panel.bz_crystal_overlay_changed)
    panel.sp_phi_c = panel._dspin(0.0, -180.0, 180.0, 0.5, dec=2)
    panel.sp_phi_c.setToolTip("Crystal a* rotation vs detector kx. Run Δazi handled via Γ FS.")
    panel.sp_phi_c.valueChanged.connect(panel.bz_crystal_overlay_changed)
    panel.sp_v0.valueChanged.connect(panel.bz_crystal_overlay_changed)
    panel.chk_bz_xtal = QCheckBox("Crystal BZ contours"); panel.chk_bz_xtal.setChecked(False)
    panel.chk_bz_xtal.stateChanged.connect(panel.bz_crystal_overlay_changed)
    panel.chk_hs_xtal = QCheckBox("Crystal HS points"); panel.chk_hs_xtal.setChecked(False)
    panel.chk_hs_xtal.setToolTip("Γ/X/M (Γ plane) or Z/R/A (Z plane), from MP lattice.")
    panel.chk_hs_xtal.stateChanged.connect(panel.bz_crystal_overlay_changed)
    panel.lbl_kz = QLabel("kz: — | crystal: —")
    panel.lbl_kz.setStyleSheet("color:#aaa; font-size:10px;"); panel.lbl_kz.setWordWrap(True); panel.lbl_kz.setMaximumWidth(240)
    # Labels courts → garde la colonne form étroite, évite que le panel
    # déborde sur le canvas central et coupe le panneau droit.
    fx.addRow("MP :", panel.ed_mp_id)
    fx.addRow(panel.btn_mp_fetch)
    fx.addRow("V0 (eV) :", panel.sp_v0)
    fx.addRow("Plan :", panel.cmb_kz_plane)
    fx.addRow("φc (°) :", panel.sp_phi_c)
    fx.addRow(panel.chk_bz_xtal)
    fx.addRow(panel.chk_hs_xtal)
    fx.addRow(panel.lbl_kz)
    panel.btn_dft_load = compact_button(QPushButton("Load 3D DFT npz..."), max_width=180)
    panel.btn_dft_load.setToolTip("npz: kx, ky, kz (1/Å), energies (n_kz,n_ky,n_kx) eV, optional a_lattice.")
    panel.btn_dft_load.clicked.connect(panel.dft_grid_load_requested)
    panel.btn_dft_clear = compact_button(QPushButton("Forget DFT"), max_width=120)
    panel.btn_dft_clear.clicked.connect(panel.dft_grid_clear_requested)
    panel.lbl_dft = QLabel("DFT: none"); panel.lbl_dft.setStyleSheet("color:#aaa; font-size:10px;")
    fx.addRow(panel.btn_dft_load); fx.addRow(panel.btn_dft_clear); fx.addRow(panel.lbl_dft)
    panel._add_collapsible_group(
        lay, "Crystal BZ mapping (Materials Project)", grp_xb, open_default=False
    )
