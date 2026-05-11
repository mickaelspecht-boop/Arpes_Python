"""Section Fit MDC + Plage analyse + Waterfall + Boutons + Outils Γ.

Builder externe pour les contrôles fit cachés sur l'onglet BM. Les sous-groupes
"Initiaux", "Contraintes", "Détection / scan" et "Résolution" sont des
QGroupBox checkable (collapsibles). L'état est persisté via le signal
`fit_section_toggled` du panneau parent.
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

from arpes.ui.widgets._qt_helpers import compact_button, dspin, hsep, ispin


# Presets matériau : applique un set de paramètres "détection / scan / largeur".
# "Custom" = aucune modification. La sélection est persistée dans la session.
MATERIAL_PRESETS: dict[str, dict | None] = {
    "Custom": None,
    "Métal léger": {
        "smooth_fit": 1.5, "smooth_detect": 2.0,
        "gamma_init": 0.05, "gamma_max": 0.20,
        "min_amplitude": 0.05, "max_jump": 0.15,
        "width_mode": "symmetric",
    },
    "Métal lourd": {
        "smooth_fit": 2.5, "smooth_detect": 3.5,
        "gamma_init": 0.12, "gamma_max": 0.40,
        "min_amplitude": 0.05, "max_jump": 0.25,
        "width_mode": "symmetric",
    },
    "SC dopé": {
        "smooth_fit": 2.0, "smooth_detect": 3.0,
        "gamma_init": 0.08, "gamma_max": 0.30,
        "min_amplitude": 0.02, "max_jump": 0.20,
        "width_mode": "asymmetric",
    },
    "Bruité (lissage++)": {
        "smooth_fit": 4.0, "smooth_detect": 5.0,
        "gamma_init": 0.10, "gamma_max": 0.30,
        "min_amplitude": 0.08, "max_jump": 0.25,
        "width_mode": "symmetric",
    },
}


def _make_collapsible(panel, title: str, key: str) -> tuple[QGroupBox, QFormLayout]:
    grp = QGroupBox(title)
    grp.setCheckable(True)
    grp.setChecked(True)
    content = QWidget()
    fl = QFormLayout(content)
    fl.setContentsMargins(2, 2, 2, 2)
    outer = QVBoxLayout(grp)
    outer.setContentsMargins(6, 18, 6, 6)
    outer.addWidget(content)
    grp.toggled.connect(content.setVisible)
    grp.toggled.connect(lambda v, k=key: panel.fit_section_toggled.emit(k, bool(v)))
    panel._fit_sections[key] = grp
    return grp, fl


def build_fit_controls(panel, lay) -> None:
    panel._fit_controls_widget = QWidget()
    _fcl = QVBoxLayout(panel._fit_controls_widget)
    _fcl.setContentsMargins(0, 0, 0, 0)
    _fcl.setSpacing(4)

    panel._fit_sections = {}

    _build_roi_group(panel, _fcl)
    _build_preset_combo(panel, _fcl)
    _build_init_section(panel, _fcl)
    _build_constraint_section(panel, _fcl)
    _build_detect_section(panel, _fcl)
    _build_resolution_section(panel, _fcl)
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
    panel.btn_fit_roi = compact_button(QPushButton("Sélectionner sur carte"), max_width=180)
    panel.btn_fit_roi.setCheckable(True)
    panel.btn_fit_roi.setToolTip(
        "Active une sélection rectangulaire par cliquer-glisser sur la carte BM/MDC Fit.\n"
        "La zone choisie remplit k_min/k_max et ev_start/ev_end."
    )
    panel.btn_fit_roi.toggled.connect(panel.fit_roi_requested)
    btn_fit_roi_reset = compact_button(QPushButton("Pleine BM"), max_width=120)
    btn_fit_roi_reset.setToolTip("Remet la plage d'analyse sur toute la carte chargée.")
    btn_fit_roi_reset.clicked.connect(panel.fit_roi_reset_requested)
    roi_row = QWidget()
    roi_lay = QHBoxLayout(roi_row)
    roi_lay.setContentsMargins(0, 0, 0, 0)
    roi_lay.setSpacing(4)
    roi_lay.addWidget(panel.btn_fit_roi)
    roi_lay.addWidget(btn_fit_roi_reset)
    roi_lay.addStretch(1)
    fl2.addRow("ev_start:", panel.sp_evs)
    fl2.addRow("ev_end:", panel.sp_eve)
    fl2.addRow("k_min:", panel.sp_kmin)
    fl2.addRow("k_max:", panel.sp_kmax)
    fl2.addRow(roi_row)
    _fcl.addWidget(grp_r)


def _build_preset_combo(panel, _fcl) -> None:
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 2, 0, 2)
    h.addWidget(QLabel("Preset :"))
    panel.cmb_fit_preset = QComboBox()
    panel.cmb_fit_preset.addItems(list(MATERIAL_PRESETS.keys()))
    panel.cmb_fit_preset.setToolTip(
        "Applique un set de paramètres (lissage, γ, ampl., saut, symétrie).\n"
        "Custom : aucune modification. Le choix est sauvegardé dans la session."
    )
    panel.cmb_fit_preset.currentTextChanged.connect(panel._on_preset_chosen)
    h.addWidget(panel.cmb_fit_preset, 1)
    _fcl.addWidget(row)


def _build_init_section(panel, _fcl) -> None:
    from arpes.ui.widgets.params import ClickablePairLabel
    grp, fl = _make_collapsible(panel, "Initiaux paire", "init")
    panel.sp_np = ispin(1, 1, 8)
    panel.sp_np.setToolTip("Nombre de paires de Lorentziennes (= nombre de bandes croisées).")
    panel.sp_np.valueChanged.connect(panel._on_n_pairs_changed)
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
    for w in (panel.sp_kfi, panel.sp_gi, panel.sp_gm):
        w.valueChanged.connect(panel.fit_only_changed)
    fl.addRow("Nb paires:", panel.sp_np)
    fl.addRow(panel._pair_lbl)
    fl.addRow("kF init (π/a):", panel.sp_kfi)
    fl.addRow("γ init (π/a):", panel.sp_gi)
    fl.addRow("γ max (π/a):", panel.sp_gm)
    _fcl.addWidget(grp)


def _build_constraint_section(panel, _fcl) -> None:
    grp, fl = _make_collapsible(panel, "Contraintes optimiseur", "constraints")
    panel.sp_xg = dspin(0.10, 0.0, 0.5, 0.01)
    panel.sp_xg.setToolTip(
        "Demi-largeur de la zone de contrainte autour du centre Γ (π/a).\n"
        "L'optimiseur limite xg dans [centre − xg_range, centre + xg_range].\n"
        "Voir le rectangle cyan dans le graphique MDC."
    )
    panel.sp_cx = dspin(0.0, -1.0, 1.0, 0.01)
    panel.sp_cx.setToolTip(
        "Centre de symétrie des paires (position Γ, en π/a).\n"
        "Halo cyan tireté sur la carte BM en temps réel pendant l'édition.\n"
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
    for w in (panel.sp_xg, panel.sp_cx, panel.sp_k0m):
        w.valueChanged.connect(panel.fit_only_changed)
    panel.cmb_wm.currentIndexChanged.connect(panel.fit_only_changed)
    panel.sp_cx.valueChanged.connect(
        lambda v: panel.gamma_center_preview.emit(float(v))
    )
    k0w = QWidget()
    k0l = QHBoxLayout(k0w)
    k0l.setContentsMargins(0, 0, 0, 0)
    k0l.addWidget(panel.sp_k0m)
    k0l.addWidget(panel.chk_k0a)
    fl.addRow("Fenêtre Γ (π/a):", panel.sp_xg)
    fl.addRow("Centre Γ (π/a):", panel.sp_cx)
    fl.addRow("kF max (π/a):", k0w)
    fl.addRow("Symétrie paire:", panel.cmb_wm)
    _fcl.addWidget(grp)


def _build_detect_section(panel, _fcl) -> None:
    grp, fl = _make_collapsible(panel, "Détection / scan", "detect")
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
    panel.sp_ma = dspin(0.01, 0.0, 1.0, 0.01)
    panel.sp_ma.setToolTip(
        "Amplitude minimale relative d'un pic pour être accepté (0–1).\n"
        "Rejette les pics dont l'amplitude est < ampl_min × max(MDC)."
    )
    panel.sp_mj = dspin(0.20, 0.0, 1.0, 0.05)
    panel.sp_mj.setToolTip(
        "Saut maximal autorisé entre positions kF consécutives (π/a).\n"
        "Contrôle la continuité de la dispersion lors du fit complet."
    )
    panel.sp_chi2_threshold = dspin(5.0, 0.1, 1_000.0, 0.5, dec=1)
    panel.sp_chi2_threshold.setToolTip(
        "Seuil chi2_red pour marquer les slices de fit douteuses en orange.\n"
        "N'agit que sur l'affichage si le fit_result contient chi2_red."
    )
    panel.cmb_sd = QComboBox()
    panel.cmb_sd.addItems(["up", "down"])
    panel.cmb_sd.setFixedWidth(80)
    panel.cmb_sd.setToolTip(
        "up : parcourt la BM de ev_start (bas) vers ev_end (proche EF).\n"
        "down : sens inverse."
    )
    for w in (panel.sp_sff, panel.sp_sfd, panel.sp_ma, panel.sp_mj, panel.sp_chi2_threshold):
        w.valueChanged.connect(panel.fit_only_changed)
    panel.cmb_sd.currentIndexChanged.connect(panel.fit_only_changed)
    fl.addRow("Lissage fit σ:", panel.sp_sff)
    fl.addRow("Lissage détect σ:", panel.sp_sfd)
    fl.addRow("Ampl. min:", panel.sp_ma)
    fl.addRow("Saut max (π/a):", panel.sp_mj)
    fl.addRow("Seuil chi2_red:", panel.sp_chi2_threshold)
    fl.addRow("Sens scan:", panel.cmb_sd)
    _fcl.addWidget(grp)


def _build_resolution_section(panel, _fcl) -> None:
    grp, fl = _make_collapsible(panel, "Résolution instrumentale", "resolution")
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
    fl.addRow("ΔE FWHM (meV):", de_row)
    fl.addRow("Δk FWHM (π/a):", dk_row)
    _fcl.addWidget(grp)


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
    btn_g = compact_button(QPushButton("Fit slice (E courante)  [Ctrl+G]"), max_width=260)
    btn_g.setStyleSheet("background:#1a6b3a;color:white;font-weight:bold;padding:6px;")
    btn_g.setToolTip(
        "Fit MDC à l'énergie courante (E sélectionnée) avec les paramètres actuels.\n"
        "Sert à calibrer les initiaux avant un fit complet."
    )
    btn_g.clicked.connect(panel.guess_requested)
    _fcl.addWidget(btn_g)

    panel._gamma_tools_widget = QWidget()
    gamma_lay = QVBoxLayout(panel._gamma_tools_widget)
    gamma_lay.setContentsMargins(0, 0, 0, 0)
    gamma_lay.setSpacing(4)
    btn_gamma = compact_button(QPushButton("Auto Γ BM"), max_width=160)
    btn_gamma.setToolTip("Estime le centre Γ par la médiane des milieux de paires MDC.")
    btn_gamma.clicked.connect(panel.gamma_bm_requested)
    gamma_lay.addWidget(btn_gamma)

    btn_ref = compact_button(QPushButton("Γ FS → BM"), max_width=160)
    btn_ref.setToolTip("Applique le Γ de référence mesuré sur une FS à la BM courante.")
    btn_ref.clicked.connect(panel.gamma_ref_requested)
    gamma_lay.addWidget(btn_ref)
    _fcl.addWidget(panel._gamma_tools_widget)

    btn_f = compact_button(QPushButton("Fit complet  [Ctrl+F]"), max_width=220)
    btn_f.setStyleSheet("background:#2a6099;color:white;font-weight:bold;padding:6px;")
    btn_f.clicked.connect(panel.full_fit_requested)
    _fcl.addWidget(btn_f)

    btn_batch = compact_button(QPushButton("Batch fit dossier"), max_width=200)
    btn_batch.setToolTip(
        "Lance Fit complet sur tous les fichiers du dossier qui n'ont pas\n"
        "encore de fit_result. Utilise les paramètres MDC actuels.\n"
        "Boîte de progression annulable."
    )
    btn_batch.clicked.connect(panel.batch_fit_requested)
    _fcl.addWidget(btn_batch)

    actions_row = QWidget()
    actions_lay = QHBoxLayout(actions_row)
    actions_lay.setContentsMargins(0, 0, 0, 0)
    btn_cl = compact_button(QPushButton("Effacer kF"), max_width=120)
    btn_cl.clicked.connect(panel.clear_kf_requested)
    panel.btn_fit_undo = compact_button(QPushButton("↶ Annuler suppression"), max_width=180)
    panel.btn_fit_undo.setEnabled(False)
    panel.btn_fit_undo.setToolTip(
        "Restaure les points de fit retirés par la dernière suppression "
        "(touche Suppr/Backspace après sélection)."
    )
    panel.btn_fit_undo.clicked.connect(panel.fit_undo_requested)
    actions_lay.addWidget(btn_cl)
    actions_lay.addWidget(panel.btn_fit_undo)
    actions_lay.addStretch(1)
    _fcl.addWidget(actions_row)

    panel.lbl_fit_quality = QLabel("")
    panel.lbl_fit_quality.setWordWrap(True)
    panel.lbl_fit_quality.setStyleSheet("color:#888;font-family:monospace;font-size:10px;")
    _fcl.addWidget(panel.lbl_fit_quality)

    panel.lbl_res = QLabel("")
    panel.lbl_res.setWordWrap(True)
    panel.lbl_res.setStyleSheet("color:#8fc;font-family:monospace;font-size:11px;")
    _fcl.addWidget(panel.lbl_res)
