"""FS UI widgets (panel + canvas) — extracted from arpes/physics/fs.py.

Pure-physics helpers (FSParams, extract_fs_map, _robust_norm, cache helpers,
remove_detector_grid_artifact, _axis_signature, _fs_cache_key) remain in
arpes/physics/fs.py to keep the layering rule (no PyQt in physics/).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from arpes.physics.bz import bz_high_symmetry_points, bz_polygon, resolve_bz_preset
from arpes.physics.fs import (
    FSParams,
    _axis_signature,
    _fs_cache_key,
    detect_gamma_from_fs_map,
    extract_fs_map,
)
from arpes.ui.widgets._qt_helpers import compact_button


class FSControlPanel(QScrollArea):
    params_changed = pyqtSignal()
    redraw_requested = pyqtSignal()
    gamma_requested = pyqtSignal()
    manual_center_requested = pyqtSignal(bool)
    forget_gamma_requested = pyqtSignal()
    bm_cuts_visibility_changed = pyqtSignal(bool)
    pockets_clear_requested = pyqtSignal()
    pockets_export_requested = pyqtSignal()
    pocket_preview_level_changed = pyqtSignal(float)
    bz_preset_requested = pyqtSignal()
    distortion_fs_toggled = pyqtSignal(bool)
    # --- Overlay BZ cristal (MP) -----------------------------------------
    mp_lattice_fetch_requested = pyqtSignal()   # bouton "Récup symétrie MP"
    bz_crystal_overlay_changed = pyqtSignal()   # toggles / V0 / plan / phi_c
    dft_grid_load_requested = pyqtSignal()      # bouton "Charger DFT 3D"
    dft_grid_clear_requested = pyqtSignal()     # bouton "Oublier DFT"

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        w = QWidget()
        self._lay = QVBoxLayout(w)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(w)
        self._build()

    def _dspin(self, value, lo, hi, step, dec=3):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec); sp.setValue(value)
        sp.setKeyboardTracking(False)
        sp.valueChanged.connect(self.params_changed)
        return sp

    def _ispin(self, value, lo, hi, step=1):
        sp = QSpinBox()
        sp.setRange(lo, hi); sp.setSingleStep(step); sp.setValue(value)
        sp.setKeyboardTracking(False)
        sp.valueChanged.connect(self.params_changed)
        return sp

    def _add_collapsible_group(
        self,
        parent_lay: QVBoxLayout,
        title: str,
        group: QGroupBox,
        *,
        open_default: bool,
        highlight: bool = False,
    ) -> QPushButton:
        group.setTitle("")
        group.setFlat(True)
        wrapper = QWidget()
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        btn = QPushButton()
        btn.setCheckable(True)
        if highlight:
            btn.setStyleSheet(
                "QPushButton { background:#5a3a18; color:#ffd089; padding:8px 10px;"
                " border:1px solid #ffae42; border-radius:4px; font-weight:bold;"
                " text-align:left; font-size:13px; }"
                "QPushButton:checked { background:#7a4e22; color:#fff; }"
                "QPushButton:hover { background:#6c4520; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background:#3a3a4a; color:#cde; padding:6px 8px;"
                " border-radius:3px; font-weight:bold; text-align:left; }"
                "QPushButton:checked { background:#4a4a6a; color:#fff; }"
                "QPushButton:hover { background:#454560; }"
            )
        lay.addWidget(btn)
        lay.addWidget(group)

        def _set_open(opened: bool) -> None:
            group.setVisible(bool(opened))
            arrow = "▼" if opened else "▶"
            btn.setText(f"{arrow}  {title}")

        btn.clicked.connect(_set_open)
        btn.setChecked(bool(open_default))
        _set_open(bool(open_default))
        parent_lay.addWidget(wrapper)
        return btn

    def _build(self):
        lay = self._lay

        grp_lat = QGroupBox("Réseau / unités π/a")
        fl = QFormLayout(grp_lat)
        self.sp_a = self._dspin(3.960, 1.0, 20.0, 0.01)
        self.sp_a.setToolTip("Paramètre de maille a (Å), utilisé pour les unités π/a et résultats physiques.")
        self.sp_b = self._dspin(3.960, 1.0, 20.0, 0.01)
        self.sp_b.setToolTip("Paramètre de maille b (Å), utilisé pour les unités π/a en carte FS.")
        self.sp_kx0 = self._dspin(0.0, -5.0, 5.0, 0.01)
        self.sp_kx0.setToolTip("Centre Γ en kx (π/a) pour recentrer la carte FS.")
        self.sp_ky0 = self._dspin(0.0, -5.0, 5.0, 0.01)
        self.sp_ky0.setToolTip("Centre Γ en ky (π/a) pour recentrer la carte FS.")
        fl.addRow("a (Å):", self.sp_a)
        fl.addRow("b (Å):", self.sp_b)
        fl.addRow("centre kx:", self.sp_kx0)
        fl.addRow("centre ky:", self.sp_ky0)
        self._add_collapsible_group(lay, "Réseau / unités π/a", grp_lat, open_default=False)

        grp_fs = QGroupBox("Carte FS")
        fl2 = QFormLayout(grp_fs)
        self.sp_win = self._dspin(0.030, 0.001, 0.500, 0.005)
        self.sp_win.setToolTip("Fenêtre d'intégration autour de EF pour construire l'intensité FS.")
        self.sp_ref_lo = self._dspin(-0.600, -5.000, 1.000, 0.050)
        self.sp_ref_lo.setToolTip("Borne basse de référence pour la normalisation de flux.")
        self.sp_ref_hi = self._dspin(-0.200, -5.000, 1.000, 0.050)
        self.sp_ref_hi.setToolTip("Borne haute de référence pour la normalisation de flux.")
        self.sp_sm = self._dspin(1.0, 0.0, 8.0, 0.25, dec=2)
        self.sp_sm.setToolTip("Lissage gaussien appliqué à la carte FS affichée.")
        self.cmb_cmap = QComboBox(); self.cmb_cmap.addItems(["inferno", "viridis", "magma", "gray", "hot"])
        self.cmb_cmap.setToolTip("Palette de couleur de la carte FS.")
        self.cmb_cmap.currentIndexChanged.connect(self.params_changed)
        self.chk_norm = QCheckBox("Normalisation flux par slice"); self.chk_norm.setChecked(True)
        self.chk_norm.setToolTip(
            "Corrige le flux slice par slice (axe ky) et le profil détecteur (axe kx).\n"
            "Utile pour les FS CLS où l'intensité varie entre les steps et aux bords du détecteur."
        )
        self.chk_norm.stateChanged.connect(self.params_changed)
        fl2.addRow("Fenêtre EF ±eV:", self.sp_win)
        fl2.addRow("Norm ref min:", self.sp_ref_lo)
        fl2.addRow("Norm ref max:", self.sp_ref_hi)
        fl2.addRow("Lissage σ:", self.sp_sm)
        fl2.addRow("Colormap:", self.cmb_cmap)
        fl2.addRow(self.chk_norm)
        self._add_collapsible_group(lay, "Carte FS", grp_fs, open_default=False)

        grp_bz = QGroupBox("ZDB théorique")
        fl3 = QFormLayout(grp_bz)
        self.chk_bz = QCheckBox("Afficher ZDB"); self.chk_bz.setChecked(True)
        self.chk_hsym = QCheckBox("Points Γ/X/M"); self.chk_hsym.setChecked(True)
        self.cmb_bz_shape = QComboBox(); self.cmb_bz_shape.addItems(["square", "rectangle", "hexagon", "centered_rect", "oblique"])
        self.sp_bzx = self._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
        self.sp_bzy = self._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
        self.sp_bz_angle = self._dspin(90.0, 20.0, 160.0, 1.0, dec=1)
        self.sp_klim = self._dspin(1.3, 0.1, 10.0, 0.05, dec=2)
        self.cmb_bz_shape.currentIndexChanged.connect(self.params_changed)
        self.cmb_bz_shape.currentIndexChanged.connect(self._update_bz_angle_visibility)
        self.chk_bz.stateChanged.connect(self.params_changed)
        self.chk_hsym.stateChanged.connect(self.params_changed)
        btn_bz = compact_button(QPushButton("Choisir ZDB..."), max_width=160)
        btn_bz.setToolTip("Ouvre un sélecteur avec schéma pour choisir une ZDB 2D Bravais.")
        btn_bz.clicked.connect(self.bz_preset_requested)
        fl3.addRow(self.chk_bz)
        fl3.addRow(self.chk_hsym)
        fl3.addRow("Forme:", self.cmb_bz_shape)
        fl3.addRow("demi-ZDB x:", self.sp_bzx)
        fl3.addRow("demi-ZDB y:", self.sp_bzy)
        fl3.addRow("angle réseau:", self.sp_bz_angle)
        fl3.addRow("limite affichage:", self.sp_klim)
        fl3.addRow(btn_bz)
        self._update_bz_angle_visibility()
        self._add_collapsible_group(lay, "ZDB théorique", grp_bz, open_default=False)

        # --- Groupbox : overlay BZ cristal réel (lattice MP) ---------------
        grp_xb = QGroupBox("Mapping BZ cristal (Materials Project)")
        fx = QFormLayout(grp_xb)
        self.ed_mp_id = QLineEdit()
        self.ed_mp_id.setPlaceholderText("mp-xxxx (optionnel : auto depuis logbook)")
        self.btn_mp_fetch = compact_button(QPushButton("Récup symétrie MP"), max_width=160)
        self.btn_mp_fetch.setToolTip(
            "Récupère paramètres de maille (a,b,c,α,β,γ) + groupe d'espace\n"
            "depuis Materials Project. Cache disque + timeout 10 s.\n"
            "Réutilise mp_id du logbook si laissé vide.\n"
            "DFT/GGA : utile pour symétrie et aire FS/Luttinger ; masses et "
            "positions de bandes peuvent être décalées."
        )
        self.btn_mp_fetch.clicked.connect(self.mp_lattice_fetch_requested)
        self.sp_v0 = self._dspin(12.0, 0.5, 50.0, 0.5, dec=2)
        self.sp_v0.setToolTip("V0 (eV) modèle électron libre pour kz. ≠ φ. Typique 8–15 eV.")
        self.cmb_kz_plane = QComboBox(); self.cmb_kz_plane.addItems(["Auto", "Gamma", "Z"])
        self.cmb_kz_plane.setToolTip("Plan kz pour labels HS (Γ/X/M ou Z/R/A). Auto = via kz(hν,V0).")
        self.cmb_kz_plane.currentIndexChanged.connect(self.bz_crystal_overlay_changed)
        self.sp_phi_c = self._dspin(0.0, -180.0, 180.0, 0.5, dec=2)
        self.sp_phi_c.setToolTip("Rotation a* cristal vs kx détecteur. Δazi manip géré via Γ FS.")
        self.sp_phi_c.valueChanged.connect(self.bz_crystal_overlay_changed)
        self.sp_v0.valueChanged.connect(self.bz_crystal_overlay_changed)
        self.chk_bz_xtal = QCheckBox("Contours BZ cristal"); self.chk_bz_xtal.setChecked(False)
        self.chk_bz_xtal.stateChanged.connect(self.bz_crystal_overlay_changed)
        self.chk_hs_xtal = QCheckBox("Points HS cristal"); self.chk_hs_xtal.setChecked(False)
        self.chk_hs_xtal.setToolTip("Γ/X/M (plan Γ) ou Z/R/A (plan Z), depuis lattice MP.")
        self.chk_hs_xtal.stateChanged.connect(self.bz_crystal_overlay_changed)
        self.lbl_kz = QLabel("kz : — | cristal : —")
        self.lbl_kz.setStyleSheet("color:#aaa; font-size:10px;"); self.lbl_kz.setWordWrap(True); self.lbl_kz.setMaximumWidth(240)
        # Labels courts → garde la colonne form étroite, évite que le panel
        # déborde sur le canvas central et coupe le panneau droit.
        fx.addRow("MP :", self.ed_mp_id)
        fx.addRow(self.btn_mp_fetch)
        fx.addRow("V0 (eV) :", self.sp_v0)
        fx.addRow("Plan :", self.cmb_kz_plane)
        fx.addRow("φc (°) :", self.sp_phi_c)
        fx.addRow(self.chk_bz_xtal)
        fx.addRow(self.chk_hs_xtal)
        fx.addRow(self.lbl_kz)
        self.btn_dft_load = compact_button(QPushButton("Charger DFT 3D npz..."), max_width=180)
        self.btn_dft_load.setToolTip("npz : kx, ky, kz (1/Å), energies (n_kz,n_ky,n_kx) eV, optionnel a_lattice.")
        self.btn_dft_load.clicked.connect(self.dft_grid_load_requested)
        self.btn_dft_clear = compact_button(QPushButton("Oublier DFT"), max_width=120)
        self.btn_dft_clear.clicked.connect(self.dft_grid_clear_requested)
        self.lbl_dft = QLabel("DFT : aucun"); self.lbl_dft.setStyleSheet("color:#aaa; font-size:10px;")
        fx.addRow(self.btn_dft_load); fx.addRow(self.btn_dft_clear); fx.addRow(self.lbl_dft)
        self._add_collapsible_group(
            lay, "Mapping BZ cristal (Materials Project)", grp_xb, open_default=False
        )

        # --- Construit widgets workflow (ajout au layout reporté à la fin) ---
        self.lbl_info = QLabel("Charge un fast map Solaris ou un dossier FS CLS.")
        self.lbl_info.setWordWrap(True); self.lbl_info.setStyleSheet("color:#aaa; font-size:10px;")
        self.chk_distortion_fs = QCheckBox("Appliquer distorsion BM au volume FS")
        self.chk_distortion_fs.setChecked(False)
        self.chk_distortion_fs.setToolTip(
            "Applique au volume FS la calibration de distorsion BM partagée (trapèze)."
        )
        self.chk_distortion_fs.toggled.connect(self.distortion_fs_toggled)
        self._btn_redraw_fs = compact_button(QPushButton("Redessiner FS"), max_width=160)
        self._btn_redraw_fs.clicked.connect(self.redraw_requested)
        grp_gamma = QGroupBox()
        gv = QVBoxLayout(grp_gamma); gv.setContentsMargins(6, 6, 6, 6); gv.setSpacing(4)
        btn_g = compact_button(QPushButton("Détecter Γ FS"), max_width=200)
        btn_g.setToolTip("Détecte Γ par milieux de paires MDC sur la FS et recentre la carte.")
        btn_g.clicked.connect(self.gamma_requested); gv.addWidget(btn_g)
        self.btn_pick_center = compact_button(QPushButton("Viser Γ manuel"), max_width=200)
        self.btn_pick_center.setCheckable(True)
        self.btn_pick_center.setToolTip("Active curseur. Clique = nouveau Γ recentré et sauvé.")
        self.btn_pick_center.toggled.connect(self.manual_center_requested)
        gv.addWidget(self.btn_pick_center)
        btn_forget = compact_button(QPushButton("Oublier Γ"), max_width=200)
        btn_forget.setToolTip("Réinitialise tout l'état Γ (référence session, axe, fit_result).")
        btn_forget.clicked.connect(self.forget_gamma_requested); gv.addWidget(btn_forget)
        self._grp_gamma = grp_gamma
        self.chk_show_bm_cuts = QCheckBox("Afficher BM cuts")
        self.chk_show_bm_cuts.setToolTip(
            "Projette les BMs compatibles. Couleurs : cyan=exact, orange=Δazi, rouge=Δhv."
        )
        self.chk_show_bm_cuts.toggled.connect(self.bm_cuts_visibility_changed)
        self.sp_pairing_hv_tol = self._dspin(5.0, 0.5, 50.0, 0.5, dec=1)
        self.sp_pairing_hv_tol.setToolTip("Tol Δhv (%) pour pairing FS↔BMs. 5% = même hv ; 30% lie scans kz.")
        self.sp_pairing_azi_tol = self._dspin(2.0, 0.0, 30.0, 0.5, dec=1)
        self.sp_pairing_azi_tol.setToolTip("Tol Δazi (°) pour pairing.")

        grp_pocket = QGroupBox("Poches FS")
        fp = QFormLayout(grp_pocket)
        self.lbl_pocket_count = QLabel("0 poche")
        self.lbl_pocket_count.setStyleSheet("color:#aaa; font-size:10px;")
        self.cmb_pocket_quality = QComboBox()
        self.cmb_pocket_quality.addItems(["Fin", "Standard", "Stable"])
        self.cmb_pocket_quality.setCurrentText("Standard")
        self.cmb_pocket_quality.setToolTip(
            "Qualité du contour : Fin suit plus les détails, Stable résiste mieux aux stries/bruit."
        )
        self.cmb_pocket_quality.currentIndexChanged.connect(self._on_pocket_quality_changed)
        self.sp_pocket_smooth_y = self._dspin(1.0, 0.0, 6.0, 0.25, dec=2); self.sp_pocket_smooth_y.setToolTip("Lissage ky avant extraction.")
        self.sp_pocket_smooth_x = self._dspin(3.0, 0.0, 12.0, 0.25, dec=2); self.sp_pocket_smooth_x.setToolTip("Lissage kx avant extraction (anti-stries CLS).")
        self.sp_pocket_contour_window = self._ispin(9, 3, 25, 2); self.sp_pocket_contour_window.setToolTip("Fenêtre lissage contour fermé (impair).")
        self.sp_pocket_simplify = self._dspin(0.015, 0.0, 0.100, 0.005, dec=3); self.sp_pocket_simplify.setToolTip("Distance minimale entre points du contour stocké.")
        self.sp_pocket_min_area = self._dspin(0.20, 0.0, 20.0, 0.10, dec=2); self.sp_pocket_min_area.setToolTip("Aire minimale en % BZ.")
        self.sp_pocket_n_bands = self._ispin(1, 1, 12, 1); self.sp_pocket_n_bands.setToolTip("Luttinger : nombre de bandes occupant la poche. Défaut : 1.")
        self.sp_pocket_spin = self._ispin(2, 1, 2, 1); self.sp_pocket_spin.setToolTip("Dégénérescence spin (1 polarisé, 2 sinon).")
        self.sp_pocket_hs_x_deg = self._dspin(0.0, -180.0, 180.0, 1.0, dec=1); self.sp_pocket_hs_x_deg.setToolTip("Γ-X (deg).")
        self.sp_pocket_hs_m_deg = self._dspin(45.0, -180.0, 180.0, 1.0, dec=1); self.sp_pocket_hs_m_deg.setToolTip("Γ-M (deg).")
        self.sp_pocket_hs_tol_deg = self._dspin(10.0, 1.0, 45.0, 1.0, dec=1); self.sp_pocket_hs_tol_deg.setToolTip("Tolérance secteur kF(Γ-X/M).")
        self.chk_pocket_bootstrap = QCheckBox("Bootstrap incertitude")
        self.chk_pocket_bootstrap.setChecked(False)
        self.chk_pocket_bootstrap.setToolTip(
            "Active le bootstrap : N tirages (level ±10%, smoothing ±25%) → "
            "médiane + écart-type par champ. Coût ≈ N× temps caractérisation."
        )
        self.sp_pocket_bootstrap_n = self._ispin(20, 4, 100, 1)
        self.sp_pocket_bootstrap_n.setToolTip("Nombre de tirages bootstrap. Défaut : 20.")
        self.sp_pocket_mdc_n = self._ispin(36, 8, 180, 4)
        self.sp_pocket_mdc_n.setToolTip("MDC-radial : nombre de directions échantillonnées (deg = 360/N).")
        self.sp_pocket_mdc_r2 = self._dspin(0.5, 0.0, 1.0, 0.05, dec=2)
        self.sp_pocket_mdc_r2.setToolTip("MDC-radial : R² minimum pour valider un fit Lorentzien.")
        self.chk_pocket_level_manual = QCheckBox("Level manuel")
        self.chk_pocket_level_manual.setToolTip("Utilise le level ci-dessous pour le prochain clic droit au lieu du seuil auto.")
        # Level slider : signal dédié (pas params_changed) pour preview live sans
        # déclencher un redraw complet de la FS à chaque pas du slider.
        self.sp_pocket_level = QDoubleSpinBox()
        self.sp_pocket_level.setRange(0.0, 1.0)
        self.sp_pocket_level.setSingleStep(0.01)
        self.sp_pocket_level.setDecimals(3)
        self.sp_pocket_level.setValue(0.50)
        self.sp_pocket_level.setKeyboardTracking(False)
        self.sp_pocket_level.setToolTip(
            "Seuil iso-intensité. Slider live : pilote le contour preview "
            "quand une poche est en mode aperçu."
        )
        self.sp_pocket_level.valueChanged.connect(
            lambda v: self.pocket_preview_level_changed.emit(float(v))
        )
        fp.addRow(self.lbl_pocket_count)
        fp.addRow("Qualité :", self.cmb_pocket_quality)
        fp.addRow(self.chk_pocket_level_manual)
        fp.addRow("Level :", self.sp_pocket_level)
        fp.addRow("Lissage ky :", self.sp_pocket_smooth_y)
        fp.addRow("Lissage kx :", self.sp_pocket_smooth_x)
        fp.addRow("Contour :", self.sp_pocket_contour_window)
        fp.addRow("Simplifier :", self.sp_pocket_simplify)
        fp.addRow("Aire min :", self.sp_pocket_min_area)
        fp.addRow("n bandes :", self.sp_pocket_n_bands)
        fp.addRow("Spin :", self.sp_pocket_spin)
        fp.addRow("Γ-X (°) :", self.sp_pocket_hs_x_deg)
        fp.addRow("Γ-M (°) :", self.sp_pocket_hs_m_deg)
        fp.addRow("Tol HS (°) :", self.sp_pocket_hs_tol_deg)
        fp.addRow(self.chk_pocket_bootstrap)
        fp.addRow("Bootstrap N :", self.sp_pocket_bootstrap_n)
        fp.addRow("MDC dirs :", self.sp_pocket_mdc_n)
        fp.addRow("MDC R²min :", self.sp_pocket_mdc_r2)
        btn_export_pockets = compact_button(QPushButton("Exporter poches CSV"), max_width=160)
        btn_export_pockets.clicked.connect(self.pockets_export_requested)
        btn_clear_pockets = compact_button(QPushButton("Effacer poches FS"), max_width=160)
        btn_clear_pockets.clicked.connect(self.pockets_clear_requested)
        fp.addRow(btn_export_pockets); fp.addRow(btn_clear_pockets)
        self._add_collapsible_group(lay, "Poches FS", grp_pocket, open_default=True)
        # --- Workflow ordre user : Γ → distorsion → Redessiner → BM cuts ---
        lay.addWidget(self.lbl_info)
        self._add_collapsible_group(
            lay, "★  Centrage Γ  ★", self._grp_gamma, open_default=True, highlight=True,
        )
        lay.addWidget(self.chk_distortion_fs)
        lay.addWidget(self._btn_redraw_fs)
        lay.addWidget(self.chk_show_bm_cuts)
        _hl = QHBoxLayout(); _hl.setContentsMargins(0, 0, 0, 0); _hl.setSpacing(4)
        _hl.addWidget(QLabel("Tol hv % :")); _hl.addWidget(self.sp_pairing_hv_tol)
        _hl.addWidget(QLabel("Tol azi° :")); _hl.addWidget(self.sp_pairing_azi_tol)
        _w = QWidget(); _w.setLayout(_hl); lay.addWidget(_w)
        self.sp_pairing_hv_tol.valueChanged.connect(self.bm_cuts_visibility_changed.emit)
        self.sp_pairing_azi_tol.valueChanged.connect(self.bm_cuts_visibility_changed.emit)
        lay.addStretch(1)

    def params(self) -> FSParams:
        return FSParams(
            a_lattice=self.sp_a.value(), b_lattice=self.sp_b.value(),
            ef_window=self.sp_win.value(),
            norm_ref_lo=self.sp_ref_lo.value(), norm_ref_hi=self.sp_ref_hi.value(),
            smooth_sigma=self.sp_sm.value(),
            klim=self.sp_klim.value(), kx_center=self.sp_kx0.value(), ky_center=self.sp_ky0.value(),
            bz_shape=self.cmb_bz_shape.currentText(),
            bz_half_x=self.sp_bzx.value(), bz_half_y=self.sp_bzy.value(),
            bz_angle_deg=self.sp_bz_angle.value(),
            normalize_profile=self.chk_norm.isChecked(), overlay_bz=self.chk_bz.isChecked(),
            show_hsym=self.chk_hsym.isChecked(), cmap=self.cmb_cmap.currentText(),
            v0_eV=self.sp_v0.value(),
            kz_plane=self.cmb_kz_plane.currentText(),
            phi_c_deg=self.sp_phi_c.value(),
            overlay_bz_crystal=self.chk_bz_xtal.isChecked(),
            overlay_hs_crystal=self.chk_hs_xtal.isChecked(),
            mp_id=self.ed_mp_id.text().strip(),
        )

    def set_center(self, kx: float, ky: float):
        self.sp_kx0.blockSignals(True); self.sp_ky0.blockSignals(True)
        self.sp_kx0.setValue(float(kx)); self.sp_ky0.setValue(float(ky))
        self.sp_kx0.blockSignals(False); self.sp_ky0.blockSignals(False)
        self.params_changed.emit()

    def set_dft_status(self, label: str) -> None:
        self.lbl_dft.setText(f"DFT : {label}" if label else "DFT : aucun")

    def set_manual_center_active(self, active: bool):
        self.btn_pick_center.blockSignals(True)
        self.btn_pick_center.setChecked(bool(active))
        self.btn_pick_center.blockSignals(False)

    def pocket_settings(self) -> dict[str, float | int | bool | None]:
        manual = bool(self.chk_pocket_level_manual.isChecked())
        return {
            "quality": self.cmb_pocket_quality.currentText(),
            "smooth_sigma_y": float(self.sp_pocket_smooth_y.value()),
            "smooth_sigma_x": float(self.sp_pocket_smooth_x.value()),
            "contour_window": int(self.sp_pocket_contour_window.value()),
            "simplify_step": float(self.sp_pocket_simplify.value()),
            "min_area_pct_bz": float(self.sp_pocket_min_area.value()),
            "level": float(self.sp_pocket_level.value()) if manual else None,
            "n_bands": int(self.sp_pocket_n_bands.value()),
            "spin": int(self.sp_pocket_spin.value()),
            "hs_dir_x_deg": float(self.sp_pocket_hs_x_deg.value()),
            "hs_dir_m_deg": float(self.sp_pocket_hs_m_deg.value()),
            "hs_dir_tol_deg": float(self.sp_pocket_hs_tol_deg.value()),
            "bootstrap": bool(self.chk_pocket_bootstrap.isChecked()),
            "bootstrap_n": int(self.sp_pocket_bootstrap_n.value()),
            "mdc_n_directions": int(self.sp_pocket_mdc_n.value()),
            "mdc_r2_min": float(self.sp_pocket_mdc_r2.value()),
        }

    def set_pocket_count(self, count: int) -> None:
        n = int(count)
        self.lbl_pocket_count.setText(f"{n} poche" + ("" if n <= 1 else "s"))

    def _on_pocket_quality_changed(self, _idx: int = 0) -> None:
        presets = {
            "Fin": (0.5, 1.0, 5, 0.008),
            "Standard": (1.0, 3.0, 9, 0.015),
            "Stable": (1.5, 4.0, 13, 0.025),
        }
        y, x, window, step = presets.get(self.cmb_pocket_quality.currentText(), presets["Standard"])
        widgets_values = (
            (self.sp_pocket_smooth_y, y),
            (self.sp_pocket_smooth_x, x),
            (self.sp_pocket_contour_window, window),
            (self.sp_pocket_simplify, step),
        )
        for widget, value in widgets_values:
            old = widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(old)

    def apply_bz_preset(self, key: str) -> None:
        preset = resolve_bz_preset(key)
        self.cmb_bz_shape.blockSignals(True)
        self.sp_bzx.blockSignals(True)
        self.sp_bzy.blockSignals(True)
        self.sp_bz_angle.blockSignals(True)
        self.cmb_bz_shape.setCurrentText(preset.shape)
        self.sp_bzx.setValue(preset.half_x)
        self.sp_bzy.setValue(preset.half_y)
        self.sp_bz_angle.setValue(preset.angle_deg)
        self.cmb_bz_shape.blockSignals(False)
        self.sp_bzx.blockSignals(False)
        self.sp_bzy.blockSignals(False)
        self.sp_bz_angle.blockSignals(False)
        self._update_bz_angle_visibility()
        self.chk_bz.setChecked(True)
        self.params_changed.emit()

    def _update_bz_angle_visibility(self) -> None:
        show = self.cmb_bz_shape.currentText() == "oblique"
        label = self.sp_bz_angle.parentWidget().layout().labelForField(self.sp_bz_angle)
        if label is not None:
            label.setVisible(show)
        self.sp_bz_angle.setVisible(show)


class FermiSurfaceCanvas(QWidget):
    pocket_requested = pyqtSignal(float, float)
    pocket_mdc_requested = pyqtSignal(float, float)
    pocket_wizard_requested = pyqtSignal(float, float)
    pairing_diagnose_requested = pyqtSignal()
    pocket_level_requested = pyqtSignal(float, float)
    pocket_preview_requested = pyqtSignal(float, float)
    pocket_preview_validate_requested = pyqtSignal()
    pocket_preview_cancel_requested = pyqtSignal()
    pockets_clear_requested = pyqtSignal()
    pockets_export_requested = pyqtSignal()
    pocket_open_requested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(80, 80)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Expanding)
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(80, 80)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self._fs_map_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = OrderedDict()
        self._fs_map_cache_max = 8
        self._mesh = None
        self._mesh_signature = None
        self._overlay_artists: list = []
        self._bm_cut_artists: list = []
        self._pocket_artists: list = []
        self._bm_cut_center = (0.0, 0.0)
        self._pocket_preview_artists: list = []
        self._pocket_preview_active = False
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.toolbar = NavToolbar(self.canvas, self)
        act = self.toolbar.addAction("⤢ Vue init")
        act.setToolTip("Réinitialise les axes aux limites des données "
                       "(le graphe garde sa taille).")
        act.triggered.connect(self.reset_view)
        lay.addWidget(self.toolbar); lay.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        self.canvas.mpl_connect("pick_event", self._on_pick_event)
        self._dark()

    def reset_view(self):
        try:
            self.ax.set_aspect("auto")
            self.ax.relim()
            self.ax.autoscale(enable=True, axis="both", tight=False)
        except Exception:
            pass
        try:
            self.fig.set_layout_engine("tight")
        except Exception:
            pass
        self.canvas.draw_idle()

    def _dark(self):
        self.fig.set_facecolor("#2b2b2b"); self.ax.set_facecolor("#1a1a1a")

    def draw_fs(self, raw_data: dict[str, Any] | None, params: FSParams):
        self._clear_bm_cut_artists()
        if raw_data is None:
            self.ax.cla(); self._dark()
            self._mesh = None
            self._mesh_signature = None
            self._overlay_artists = []
            self._clear_pocket_artists()
            self.ax.text(0.5, 0.5, "Charge une FS", transform=self.ax.transAxes,
                         ha="center", va="center", color="w")
            self.canvas.draw_idle(); return "Aucune donnée"
        try:
            key = _fs_cache_key(raw_data, params)
            cached = self._fs_map_cache.pop(key, None)
            if cached is None:
                kx, ky, fs, title = extract_fs_map(raw_data, params)
                self._fs_map_cache[key] = (kx, ky, fs, title)
                while len(self._fs_map_cache) > self._fs_map_cache_max:
                    self._fs_map_cache.popitem(last=False)
            else:
                kx, ky, fs, title = cached
                self._fs_map_cache[key] = cached
            meta = raw_data.get("metadata", {}) or {}
            fs_kind = meta.get("fs_kind", "")
            x = kx - params.kx_center
            y = ky - params.ky_center
            self._bm_cut_center = (float(params.kx_center), float(params.ky_center))
            # Center explicitly in signature : redondant avec _axis_signature(x/y)
            # mais immune aux collisions et explicite à la relecture. Garantit
            # qu'un changement de Γ (via set_center / detect_gamma) force toujours
            # fresh_draw → recadrage xlim/ylim sur nouveau centre.
            signature = (
                tuple(np.asarray(fs).shape),
                _axis_signature(x),
                _axis_signature(y),
                round(float(params.kx_center), 8),
                round(float(params.ky_center), 8),
            )
            for artist in list(self._overlay_artists):
                try:
                    artist.remove()
                except Exception:
                    pass
            self._overlay_artists = []
            self._clear_pocket_artists()
            if self._mesh is not None and self._mesh_signature != signature:
                try:
                    self._mesh.remove()
                except Exception:
                    pass
                self._mesh = None
            fresh_draw = self._mesh is None
            if fresh_draw:
                self.ax.cla(); self._dark()
                self._mesh = self.ax.pcolormesh(x, y, fs, cmap=params.cmap, shading="auto", vmin=0, vmax=1)
                self._mesh_signature = signature
            else:
                self._mesh.set_array(np.asarray(fs).ravel())
                self._mesh.set_cmap(params.cmap)
                self._mesh.set_clim(0, 1)
            has_kxky_axes = fs_kind == "kxky"
            self.ax.set_aspect("equal" if has_kxky_axes else "auto")
            self.ax.set_xlabel("kx (π/a)", color="w")
            ylabel = "ky (π/a)" if has_kxky_axes else "tilt (deg)"
            self.ax.set_ylabel(ylabel, color="w")
            self.ax.set_title(title, color="w", fontsize=10)
            if has_kxky_axes:
                self._overlay_bz(params)
                if params.overlay_bz_crystal or params.overlay_hs_crystal:
                    self._overlay_bz_crystal(params, raw_data)
            # Limites aux données UNIQUEMENT sur fresh_draw (nouvelle data /
            # nouveau fichier). Sur simple refresh (toggle overlay, redraw
            # layout), on PRESERVE le zoom courant — sinon chaque resize
            # splitter écrase le zoom user.
            if fresh_draw:
                self.ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
                self.ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
            self.ax.tick_params(colors="w")
            for sp in self.ax.spines.values(): sp.set_edgecolor("#555")
            self.canvas.draw_idle()
            return f"{title} | shape={fs.shape}"
        except Exception as exc:
            self.ax.cla(); self._dark()
            self._mesh = None
            self._mesh_signature = None
            self._overlay_artists = []
            self._clear_pocket_artists()
            self.ax.text(0.5, 0.5, str(exc), transform=self.ax.transAxes,
                         ha="center", va="center", color="tomato", wrap=True)
            self.canvas.draw_idle(); return f"Erreur FS: {exc}"

    def _on_canvas_button_press(self, event) -> None:
        if getattr(event, "button", None) != 3:
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        from arpes.ui.widgets.fs_panel_pockets import handle_canvas_right_click
        handle_canvas_right_click(self, event)

    def _on_pick_event(self, event) -> None:
        artist = getattr(event, "artist", None)
        idx = getattr(artist, "pocket_index", None)
        if idx is None:
            return
        self.pocket_open_requested.emit(int(idx))

    def _clear_pocket_artists(self) -> None:
        from arpes.ui.widgets.fs_panel_pockets import clear_pocket_artists
        clear_pocket_artists(self)

    def draw_pockets(self, pockets: list[dict] | None) -> None:
        from arpes.ui.widgets.fs_panel_pockets import draw_pockets
        draw_pockets(self, pockets)

    def draw_pocket_preview(self, contour) -> None:
        from arpes.ui.widgets.fs_panel_pockets import draw_pocket_preview
        draw_pocket_preview(self, contour)
        self._pocket_preview_active = True

    def clear_pocket_preview(self) -> None:
        from arpes.ui.widgets.fs_panel_pockets import clear_pocket_preview
        clear_pocket_preview(self)
        self._pocket_preview_active = False

    def _clear_bm_cut_artists(self) -> None:
        from arpes.ui.widgets.fs_panel_bm_cuts import clear_bm_cut_artists
        clear_bm_cut_artists(self)

    def draw_bm_cuts(self, cuts: list) -> None:
        from arpes.ui.widgets.fs_panel_bm_cuts import draw_bm_cuts
        draw_bm_cuts(self, cuts)

    def detect_gamma(self, raw_data: dict[str, Any] | None, params: FSParams):
        kx, ky, fs, _ = extract_fs_map(raw_data, params)
        if len(ky) < 3:
            raise ValueError("Détection Γ FS impossible sans volume FS 2D.")
        meta = raw_data.get("metadata", {}) or {}
        if meta.get("fs_kind") != "kxky":
            raise ValueError("Détection Γ FS disponible seulement avec deux axes en π/a.")
        return detect_gamma_from_fs_map(kx, ky, fs, params).as_dict()

    def _overlay_bz_crystal(self, p: FSParams, raw_data):
        from arpes.ui.widgets.fs_panel_bz_crystal import overlay_bz_crystal
        overlay_bz_crystal(self, p, raw_data)

    def _overlay_bz(self, p: FSParams):
        if not p.overlay_bz: return
        bx, by = p.bz_half_x, p.bz_half_y
        corners = bz_polygon(p.bz_shape, bx, by, p.bz_angle_deg)
        line, = self.ax.plot(corners[:,0], corners[:,1], color="white", lw=1.2, ls="--", alpha=0.85)
        self._overlay_artists.append(line)
        self._overlay_artists.append(self.ax.axhline(0, color="white", lw=0.5, ls=":", alpha=0.5))
        self._overlay_artists.append(self.ax.axvline(0, color="white", lw=0.5, ls=":", alpha=0.5))
        if p.show_hsym:
            def dot(x,y,name,color):
                scat = self.ax.scatter([x],[y], c=color, s=35, zorder=5, linewidths=0)
                ann = self.ax.annotate(name, (x,y), xytext=(4,4), textcoords="offset points", color=color, fontsize=9, fontweight="bold")
                self._overlay_artists.extend([scat, ann])
            for x, y, name, color in bz_high_symmetry_points(p.bz_shape, bx, by, p.bz_angle_deg):
                dot(x, y, name, color)
        # NE PAS recadrer aux limites BZ : on garde le signal entier visible,
        # le user zoome via la toolbar matplotlib si besoin. (Demandé par user
        # après ajout du zoom — éviter perte de signal hors BZ théorique.)
