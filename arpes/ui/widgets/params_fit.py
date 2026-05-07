"""Section Fit MDC + Plage analyse + Waterfall + Boutons + Outils Γ.

Builder externe pour les contrôles fit cachés sur l'onglet BM.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import dspin, hsep, ispin


def build_fit_controls(panel, lay) -> None:
    panel._fit_controls_widget = QWidget()
    _fcl = QVBoxLayout(panel._fit_controls_widget)
    _fcl.setContentsMargins(0, 0, 0, 0)
    _fcl.setSpacing(4)

    _build_roi_group(panel, _fcl)
    _build_fit_mdc_group(panel, _fcl)
    _build_waterfall_group(panel, _fcl)
    _build_fit_buttons(panel, _fcl)

    lay.addWidget(panel._fit_controls_widget)


def _build_roi_group(panel, _fcl) -> None:
    grp_r = QGroupBox("Plage d'analyse")
    fl2 = QFormLayout(grp_r)
    panel.sp_evs = dspin(-0.90, -5.0, 1.0, 0.05)
    panel.sp_eve = dspin(-0.005, -5.0, 1.0, 0.005)
    panel.sp_kmin = dspin(-0.80, -5.0, 5.0, 0.05)
    panel.sp_kmax = dspin(0.80, -5.0, 5.0, 0.05)
    for w in (panel.sp_evs, panel.sp_eve, panel.sp_kmin, panel.sp_kmax):
        w.valueChanged.connect(panel.params_changed)
    panel.btn_fit_roi = QPushButton("Sélectionner sur carte")
    panel.btn_fit_roi.setCheckable(True)
    panel.btn_fit_roi.setToolTip(
        "Active une sélection rectangulaire par cliquer-glisser sur la carte BM/MDC Fit.\n"
        "La zone choisie remplit k_min/k_max et ev_start/ev_end."
    )
    panel.btn_fit_roi.toggled.connect(panel.fit_roi_requested)
    btn_fit_roi_reset = QPushButton("Pleine BM")
    btn_fit_roi_reset.setToolTip("Remet la plage d'analyse sur toute la carte chargée.")
    btn_fit_roi_reset.clicked.connect(panel.fit_roi_reset_requested)
    panel.btn_fit_delete = QPushButton("Supprimer points")
    panel.btn_fit_delete.setCheckable(True)
    panel.btn_fit_delete.setToolTip(
        "Active la suppression de points de fit par clic sur la carte.\n"
        "Clic gauche : retire le point kF le plus proche (mis à NaN).\n"
        "Re-cliquer le bouton ou Échap pour quitter."
    )
    panel.btn_fit_delete.toggled.connect(panel.fit_delete_requested)
    roi_row = QWidget()
    roi_lay = QHBoxLayout(roi_row)
    roi_lay.setContentsMargins(0, 0, 0, 0)
    roi_lay.setSpacing(4)
    roi_lay.addWidget(panel.btn_fit_roi)
    roi_lay.addWidget(btn_fit_roi_reset)
    roi_lay.addWidget(panel.btn_fit_delete)
    fl2.addRow("ev_start:", panel.sp_evs)
    fl2.addRow("ev_end:", panel.sp_eve)
    fl2.addRow("k_min:", panel.sp_kmin)
    fl2.addRow("k_max:", panel.sp_kmax)
    fl2.addRow(roi_row)
    _fcl.addWidget(grp_r)


def _build_fit_mdc_group(panel, _fcl) -> None:
    from arpes.ui.widgets.params import ClickablePairLabel

    grp_f = QGroupBox("Fit MDC (Lorentzien)")
    fl3 = QFormLayout(grp_f)
    panel.sp_np = ispin(1, 1, 8)
    panel.sp_np.setToolTip("Nombre de paires de Lorentziennes (= nombre de bandes croisées).")
    panel.sp_np.valueChanged.connect(panel._on_n_pairs_changed)
    panel.sp_sff = dspin(2.0, 0.0, 10.0, 0.5, dec=1)
    panel.sp_sff.setToolTip(
        "Sigma du lissage gaussien appliqué à la MDC avant l'optimisation scipy.\n"
        "Augmenter pour données bruitées. Voir la courbe orange dans le graphique MDC."
    )
    panel.sp_sfd = dspin(3.0, 0.0, 10.0, 0.5, dec=1)
    panel.sp_sfd.setToolTip(
        "Sigma du lissage gaussien utilisé pour détecter les pics initiaux.\n"
        "Voir la courbe grise dans le graphique MDC."
    )

    panel._pair_lbl = ClickablePairLabel()
    panel._pair_lbl.pair_changed.connect(panel._on_pair_changed)
    panel.sp_kfi = dspin(0.30, 0.0, 3.0, 0.01)
    panel.sp_kfi.setToolTip(
        "Position initiale kF (π/a) pour cette paire, comptée depuis centre Γ.\n"
        "Voir les lignes tiret-point colorées dans le graphique MDC."
    )
    panel.sp_gi = dspin(0.08, 0.01, 0.5, 0.01)
    panel.sp_gi.setToolTip(
        "Demi-largeur initiale de la Lorentzienne (π/a).\n"
        "Valeur de départ pour l'optimiseur. Voir les courbes colorées dans le graphique MDC."
    )
    panel.sp_gm = dspin(0.30, 0.05, 1.0, 0.05)
    panel.sp_gm.setToolTip(
        "Demi-largeur maximale autorisée (π/a) — contrainte de l'optimiseur scipy.\n"
        "Voir les zones colorées translucides autour des pics dans le graphique MDC."
    )

    panel.sp_xg = dspin(0.10, 0.0, 0.5, 0.01)
    panel.sp_xg.setToolTip(
        "Demi-largeur de la zone de contrainte autour du centre Γ (π/a).\n"
        "L'optimiseur limite xg dans [centre − xg_range, centre + xg_range].\n"
        "Voir le rectangle cyan dans le graphique MDC."
    )
    panel.sp_cx = dspin(0.0, -1.0, 1.0, 0.01)
    panel.sp_cx.setToolTip(
        "Centre de symétrie des paires (position Γ, en π/a).\n"
        "Voir la ligne cyan pointillée dans le graphique MDC.\n"
        "Utiliser 'Auto Γ BM' ou 'Γ FS → BM' pour le calculer automatiquement."
    )
    panel.sp_k0m = dspin(0.0, 0.0, 2.0, 0.05)
    panel.sp_k0m.setToolTip(
        "Distance maximale autorisée de kF par rapport à Γ (π/a).\n"
        "Voir les lignes magenta dans le graphique MDC si actif."
    )
    panel.chk_k0a = QCheckBox("auto")
    panel.chk_k0a.setChecked(True)
    panel.chk_k0a.setToolTip("Si coché, pas de limite sur kF. Décocher pour activer kF max.")
    panel.sp_k0m.setEnabled(False)
    panel.chk_k0a.stateChanged.connect(
        lambda: panel.sp_k0m.setEnabled(not panel.chk_k0a.isChecked())
    )
    panel.cmb_wm = QComboBox()
    panel.cmb_wm.addItems(["symmetric", "asymmetric"])
    panel.cmb_wm.setFixedWidth(110)
    panel.cmb_wm.setToolTip(
        "symmetric : les deux pics de la paire ont le même γ.\n"
        "asymmetric : γ gauche et droit peuvent différer (pics asymétriques)."
    )
    panel.sp_ma = dspin(0.01, 0.0, 1.0, 0.01)
    panel.sp_ma.setToolTip(
        "Amplitude minimale relative d'un pic pour être accepté (0–1).\n"
        "Rejette les pics dont l'amplitude est < ampl_min × max(MDC).\n"
        "Augmenter pour éliminer les faux pics dus au bruit."
    )
    panel.sp_mj = dspin(0.20, 0.0, 1.0, 0.05)
    panel.sp_mj.setToolTip(
        "Saut maximal autorisé entre positions kF consécutives (π/a).\n"
        "Contrôle la continuité de la dispersion lors du fit complet.\n"
        "Réduire si la dispersion saute d'un point à l'autre."
    )
    panel.cmb_sd = QComboBox()
    panel.cmb_sd.addItems(["up", "down"])
    panel.cmb_sd.setFixedWidth(80)
    panel.cmb_sd.setToolTip(
        "up : parcourt la BM de ev_start (bas) vers ev_end (proche EF).\n"
        "down : sens inverse. Choisir le sens où les pics sont les plus nets en départ."
    )

    for w in (panel.sp_sff, panel.sp_sfd, panel.sp_kfi, panel.sp_gi, panel.sp_gm,
              panel.sp_xg, panel.sp_cx, panel.sp_k0m, panel.sp_ma, panel.sp_mj):
        w.valueChanged.connect(panel.fit_only_changed)
    panel.cmb_wm.currentIndexChanged.connect(panel.fit_only_changed)

    panel.sp_dE_meV = dspin(15.0, 1.0, 200.0, 1.0, dec=1)
    panel.sp_dE_meV.setToolTip(
        "FWHM énergie instrumentale estimée ou saisie manuellement (meV).\n"
        "Utilisée pour calculer Γ corrigé après fit MDC."
    )
    panel.sp_dk_inv_a = dspin(0.005, 0.001, 0.1, 0.001, dec=4)
    panel.sp_dk_inv_a.setToolTip(
        "FWHM k instrumentale en π/a, estimée depuis angle_step si disponible.\n"
        "Utilisée pour calculer Γ corrigé après fit MDC."
    )
    panel.lbl_dE_src = QLabel("—")
    panel.lbl_dk_src = QLabel("—")
    for lbl in (panel.lbl_dE_src, panel.lbl_dk_src):
        lbl.setToolTip("Provenance résolution : Estimée, Manuelle ou Défaut")
    panel.sp_dE_meV.valueChanged.connect(panel._mark_resolution_manual_if_user_edit)
    panel.sp_dk_inv_a.valueChanged.connect(panel._mark_resolution_manual_if_user_edit)
    panel.sp_dE_meV.valueChanged.connect(panel.fit_only_changed)
    panel.sp_dk_inv_a.valueChanged.connect(panel.fit_only_changed)

    k0w = QWidget()
    k0l = QHBoxLayout(k0w)
    k0l.setContentsMargins(0, 0, 0, 0)
    k0l.addWidget(panel.sp_k0m)
    k0l.addWidget(panel.chk_k0a)

    fl3.addRow("Nb paires:", panel.sp_np)
    fl3.addRow("Lissage fit σ:", panel.sp_sff)
    fl3.addRow("Lissage détect σ:", panel.sp_sfd)
    fl3.addRow(hsep())
    fl3.addRow(panel._pair_lbl)
    fl3.addRow("kF init (π/a):", panel.sp_kfi)
    fl3.addRow("γ init (π/a):", panel.sp_gi)
    fl3.addRow("γ max (π/a):", panel.sp_gm)
    fl3.addRow(hsep())
    fl3.addRow("Fenêtre Γ (π/a):", panel.sp_xg)
    fl3.addRow("Centre Γ (π/a):", panel.sp_cx)
    fl3.addRow("kF max (π/a):", k0w)
    fl3.addRow("Symétrie paire:", panel.cmb_wm)
    fl3.addRow(hsep())
    fl3.addRow("Ampl. min:", panel.sp_ma)
    fl3.addRow("Saut max (π/a):", panel.sp_mj)
    fl3.addRow("Sens scan:", panel.cmb_sd)
    fl3.addRow(hsep())
    de_row = QWidget()
    de_lay = QHBoxLayout(de_row)
    de_lay.setContentsMargins(0, 0, 0, 0)
    de_lay.addWidget(panel.sp_dE_meV, 1)
    de_lay.addWidget(panel.lbl_dE_src)
    dk_row = QWidget()
    dk_lay = QHBoxLayout(dk_row)
    dk_lay.setContentsMargins(0, 0, 0, 0)
    dk_lay.addWidget(panel.sp_dk_inv_a, 1)
    dk_lay.addWidget(panel.lbl_dk_src)
    fl3.addRow("ΔE FWHM (meV):", de_row)
    fl3.addRow("Δk FWHM (π/a):", dk_row)
    _fcl.addWidget(grp_f)


def _build_waterfall_group(panel, _fcl) -> None:
    panel._waterfall_controls_widget = QGroupBox("Waterfall MDC")
    fl_wf = QFormLayout(panel._waterfall_controls_widget)
    panel.sp_wf_n = ispin(32, 10, 80)
    panel.sp_wf_n.setToolTip(
        "Nombre cible de MDCs affichées dans le waterfall.\n"
        "Moins de courbes = plus de relief et moins de surcharge."
    )
    panel.sp_wf_relief = dspin(1.8, 0.5, 4.0, 0.1, dec=1)
    panel.sp_wf_relief.setToolTip(
        "Amplitude visuelle des MDCs dans le waterfall.\n"
        "Augmenter pour mieux voir les pics ; trop haut crée du chevauchement."
    )
    panel.sp_wf_n.valueChanged.connect(panel.fit_only_changed)
    panel.sp_wf_relief.valueChanged.connect(panel.fit_only_changed)
    fl_wf.addRow("Courbes:", panel.sp_wf_n)
    fl_wf.addRow("Relief:", panel.sp_wf_relief)
    panel._waterfall_controls_widget.setVisible(False)
    _fcl.addWidget(panel._waterfall_controls_widget)


def _build_fit_buttons(panel, _fcl) -> None:
    _fcl.addWidget(hsep())
    btn_g = QPushButton("Guess  (fit MDC ici)  [Ctrl+G]")
    btn_g.setStyleSheet("background:#1a6b3a;color:white;font-weight:bold;padding:6px;")
    btn_g.clicked.connect(panel.guess_requested)
    _fcl.addWidget(btn_g)

    panel._gamma_tools_widget = QWidget()
    gamma_lay = QVBoxLayout(panel._gamma_tools_widget)
    gamma_lay.setContentsMargins(0, 0, 0, 0)
    gamma_lay.setSpacing(4)
    btn_gamma = QPushButton("Auto Γ BM")
    btn_gamma.setToolTip("Estime le centre Γ par la médiane des milieux de paires MDC.")
    btn_gamma.clicked.connect(panel.gamma_bm_requested)
    gamma_lay.addWidget(btn_gamma)

    btn_ref = QPushButton("Γ FS → BM")
    btn_ref.setToolTip("Applique le Γ de référence mesuré sur une FS à la BM courante.")
    btn_ref.clicked.connect(panel.gamma_ref_requested)
    gamma_lay.addWidget(btn_ref)
    _fcl.addWidget(panel._gamma_tools_widget)

    btn_f = QPushButton("Fit complet  [Ctrl+F]")
    btn_f.setStyleSheet("background:#2a6099;color:white;font-weight:bold;padding:6px;")
    btn_f.clicked.connect(panel.full_fit_requested)
    _fcl.addWidget(btn_f)

    btn_cl = QPushButton("Effacer kF")
    btn_cl.clicked.connect(panel.clear_kf_requested)
    _fcl.addWidget(btn_cl)

    panel.lbl_res = QLabel("")
    panel.lbl_res.setWordWrap(True)
    panel.lbl_res.setStyleSheet("color:#8fc;font-family:monospace;font-size:11px;")
    _fcl.addWidget(panel.lbl_res)
