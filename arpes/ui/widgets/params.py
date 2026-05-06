"""Panneau paramètres de fit + sélecteur de paires (FitParamsPanel)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from arpes.core.session import FitParams
from arpes.ui.widgets._qt_helpers import PAIR_COLORS, dspin, hsep, ispin


class ClickablePairLabel(QLabel):
    """Label cliquable pour naviguer entre les paires de Lorentziennes.
    Clic gauche → paire suivante.  Clic droit → paire précédente."""
    pair_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._current = 0
        self._n = 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "background:#3a3a4a; color:#cde; font-weight:bold;"
            " padding:4px 8px; border-radius:3px; border:1px solid #556;"
        )
        self._update()

    def setup(self, n: int, current: int = 0):
        self._n = max(1, n)
        self._current = max(0, min(current, self._n - 1))
        self._update()

    @property
    def current(self) -> int:
        return self._current

    def _update(self):
        if self._n == 1:
            self.setText("Paire 1 / 1")
        else:
            self.setText(f"<  Paire {self._current + 1} / {self._n}  >")

    def mousePressEvent(self, event):
        if self._n < 2:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._current = (self._current + 1) % self._n
        elif event.button() == Qt.MouseButton.RightButton:
            self._current = (self._current - 1) % self._n
        else:
            super().mousePressEvent(event)
            return
        self._update()
        self.pair_changed.emit(self._current)


class FitParamsPanel(QScrollArea):
    params_changed = pyqtSignal()
    fit_only_changed = pyqtSignal()
    guess_requested = pyqtSignal()
    full_fit_requested = pyqtSignal()
    clear_kf_requested = pyqtSignal()
    copy_params_requested = pyqtSignal()
    ef_calib_requested = pyqtSignal()
    ef_apply_reference_requested = pyqtSignal()
    logbook_requested = pyqtSignal()
    gamma_bm_requested = pyqtSignal()
    gamma_ref_requested = pyqtSignal()
    grid_requested = pyqtSignal()
    grid_reset_requested = pyqtSignal()
    fit_roi_requested = pyqtSignal(bool)
    fit_roi_reset_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        w = QWidget()
        self._lay = QVBoxLayout(w)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(w)
        self._pair_params: list[dict] = [{"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}]
        self._current_pair: int = 0
        self._resolution_source_lock = False
        self._resolution_source = "default"
        self._resolution_source_detail = "defaut"
        self._build()

    def _build(self):
        lay = self._lay

        # ── énergie ──────────────────────────────────────────────────────────
        self._energy_widget = QGroupBox("Énergie sélectionnée")
        fl = QFormLayout(self._energy_widget)
        self.sp_ev = dspin(-0.30, -3.0, 0.2, 0.01)
        # sp_ev est connecté dans ArpesExplorer._build_ui (→ _on_ev_spinbox_changed)
        self.sp_int_win = dspin(0.010, 0.001, 0.200, 0.005, dec=3)
        self.sp_int_win.setToolTip(
            "Fenêtre d'intégration ±eV pour la MDC\n"
            "Élargir = moins de bruit, moins de résolution en énergie\n"
            "Correspond au 'range' d'extraction d'une coupe dans Igor")
        self.sp_int_win.valueChanged.connect(self.fit_only_changed)
        fl.addRow("E (eV):", self.sp_ev)
        fl.addRow("± intég. (eV):", self.sp_int_win)
        fl.addRow(QLabel("Clic sur la carte ou ici"))
        lay.addWidget(self._energy_widget)

        # ── calibration EF ────────────────────────────────────────────────────
        self._ef_widget = QGroupBox("EF / Chargement")
        fl_ef = QFormLayout(self._ef_widget)
        self.sp_phi = dspin(4.031, 3.0, 6.0, 0.01)
        self.sp_phi.setToolTip("Fonction de travail φ (eV). Utilisée pour calculer E_kin → E−EF.")
        self.sp_hv  = dspin(0.0, 0.0, 500.0, 1.0)
        self.sp_hv.setToolTip(
            "Énergie du photon incident (eV).\n"
            "→ CLS/LNLS : entrer manuellement AVANT de charger (obligatoire).\n"
            "→ Solaris/DA30 : lu automatiquement depuis le fichier.\n"
            "→ BESSY/SES : gardé pour diagnostic/kz; E−EF utilise automatiquement Center Energy."
        )
        self.sp_ef  = dspin(0.052, -0.3, 0.3, 0.005)
        self.sp_ef.setToolTip(
            "Décalage EF en eV. Ajuste le zéro d'énergie.\n"
            "Utiliser 'Calibrer EF auto' pour le calculer par fit Fermi-Dirac."
        )
        btn_ef = QPushButton("Calibrer EF auto")
        btn_ef.clicked.connect(self.ef_calib_requested)
        self.btn_ef_ref = QPushButton("Aucune réf EF (calibrer un Au d'abord)")
        self.btn_ef_ref.clicked.connect(self.ef_apply_reference_requested)
        self.btn_ef_ref.setEnabled(False)
        btn_log = QPushButton("Charger logbook")
        btn_log.clicked.connect(self.logbook_requested)
        self.btn_copy = QPushButton("Propager fit params (0 cible)")
        self.btn_copy.clicked.connect(self.copy_params_requested)
        self.btn_copy.setEnabled(False)
        self.update_ef_reference_button(None)
        self.update_copy_params_button(0)
        self.lbl_hv_src = QLabel("Inconnu")
        self.lbl_hv_src.setToolTip(
            "Provenance de hν :\n"
            "Fichier = lue depuis le fichier\n"
            "Logbook = lue depuis le logbook\n"
            "Manuel = saisie manuelle\n"
            "Inconnu = source inconnue"
        )
        self.lbl_hv_src.setMinimumWidth(58)
        hv_row = QWidget()
        hv_lay = QHBoxLayout(hv_row); hv_lay.setContentsMargins(0, 0, 0, 0)
        hv_lay.addWidget(self.sp_hv, 1)
        hv_lay.addWidget(self.lbl_hv_src)
        # éditer la spinbox manuellement marque la source comme manuelle
        self.sp_hv.valueChanged.connect(lambda _v: self._mark_hv_manual_if_user_edit())
        self._hv_source_lock = False  # True quand on set par code (file/logbook), pour ne pas marquer "manual"
        fl_ef.addRow("φ (eV):",       self.sp_phi)
        fl_ef.addRow("hν (eV):", hv_row)
        fl_ef.addRow("EF offset:",    self.sp_ef)
        fl_ef.addRow(btn_log)
        fl_ef.addRow(btn_ef)
        fl_ef.addRow(self.btn_ef_ref)
        fl_ef.addRow(self.btn_copy)
        self.lbl_action = QLabel("Dernière action : aucune")
        self.lbl_action.setWordWrap(True)
        self.lbl_action.setStyleSheet("color:#9fc;font-size:10px;")
        fl_ef.addRow(self.lbl_action)
        lay.addWidget(self._ef_widget)

        # ── utilitaires BM ────────────────────────────────────────────────────
        self._utils_widget = QGroupBox("Utilitaires")
        fl_ut = QFormLayout(self._utils_widget)
        self.sp_grid_strength = dspin(0.85, 0.0, 1.0, 0.05, dec=2)
        self.sp_grid_strength.setToolTip(
            "Force de suppression de la trame affichée.\n"
            "0 = aucun effet, 1 = correction complète. Valeur conseillée : 0.8-0.9."
        )
        btn_grid = QPushButton("Retirer effet grille")
        btn_grid.setToolTip(
            "Active un masque Fourier 2D automatique sur la carte BM affichée.\n"
            "La donnée brute reste inchangée."
        )
        btn_grid.clicked.connect(self.grid_requested)
        btn_grid_reset = QPushButton("Recharger brut")
        btn_grid_reset.setToolTip("Désactive la correction grille sauvegardée pour ce fichier.")
        btn_grid_reset.clicked.connect(self.grid_reset_requested)
        self.lbl_grid = QLabel("Correction BM : masque Fourier 2D automatique sur l'affichage.")
        self.lbl_grid.setWordWrap(True)
        self.lbl_grid.setStyleSheet("color:#aaa; font-size:10px;")
        fl_ut.addRow("Force:", self.sp_grid_strength)
        fl_ut.addRow(btn_grid)
        fl_ut.addRow(btn_grid_reset)
        fl_ut.addRow(self.lbl_grid)
        lay.addWidget(self._utils_widget)

        # ── contrôles fit (cachés sur l'onglet BM) ────────────────────────────
        self._fit_controls_widget = QWidget()
        _fcl = QVBoxLayout(self._fit_controls_widget)
        _fcl.setContentsMargins(0, 0, 0, 0)
        _fcl.setSpacing(4)

        # ── plage d'analyse ───────────────────────────────────────────────────
        grp_r = QGroupBox("Plage d'analyse")
        fl2 = QFormLayout(grp_r)
        self.sp_evs  = dspin(-0.90, -5.0, 1.0, 0.05)
        self.sp_eve  = dspin(-0.005, -5.0, 1.0, 0.005)
        self.sp_kmin = dspin(-0.80, -5.0, 5.0, 0.05)
        self.sp_kmax = dspin( 0.80, -5.0, 5.0, 0.05)
        for w in (self.sp_evs, self.sp_eve, self.sp_kmin, self.sp_kmax):
            w.valueChanged.connect(self.params_changed)
        self.btn_fit_roi = QPushButton("Sélectionner sur carte")
        self.btn_fit_roi.setCheckable(True)
        self.btn_fit_roi.setToolTip(
            "Active une sélection rectangulaire par cliquer-glisser sur la carte BM/MDC Fit.\n"
            "La zone choisie remplit k_min/k_max et ev_start/ev_end."
        )
        self.btn_fit_roi.toggled.connect(self.fit_roi_requested)
        btn_fit_roi_reset = QPushButton("Pleine BM")
        btn_fit_roi_reset.setToolTip("Remet la plage d'analyse sur toute la carte chargée.")
        btn_fit_roi_reset.clicked.connect(self.fit_roi_reset_requested)
        roi_row = QWidget()
        roi_lay = QHBoxLayout(roi_row)
        roi_lay.setContentsMargins(0, 0, 0, 0)
        roi_lay.setSpacing(4)
        roi_lay.addWidget(self.btn_fit_roi)
        roi_lay.addWidget(btn_fit_roi_reset)
        fl2.addRow("ev_start:", self.sp_evs)
        fl2.addRow("ev_end:",   self.sp_eve)
        fl2.addRow("k_min:",    self.sp_kmin)
        fl2.addRow("k_max:",    self.sp_kmax)
        fl2.addRow(roi_row)
        _fcl.addWidget(grp_r)

        # ── fit MDC ───────────────────────────────────────────────────────────
        grp_f = QGroupBox("Fit MDC (Lorentzien)")
        fl3 = QFormLayout(grp_f)
        self.sp_np   = ispin(1,   1, 8)
        self.sp_np.setToolTip("Nombre de paires de Lorentziennes (= nombre de bandes croisées).")
        self.sp_np.valueChanged.connect(self._on_n_pairs_changed)
        self.sp_sff  = dspin(2.0,  0.0, 10.0, 0.5, dec=1)
        self.sp_sff.setToolTip(
            "Sigma du lissage gaussien appliqué à la MDC avant l'optimisation scipy.\n"
            "Augmenter pour données bruitées. Voir la courbe orange dans le graphique MDC."
        )
        self.sp_sfd  = dspin(3.0,  0.0, 10.0, 0.5, dec=1)
        self.sp_sfd.setToolTip(
            "Sigma du lissage gaussien utilisé pour détecter les pics initiaux.\n"
            "Voir la courbe grise dans le graphique MDC."
        )

        # ── paramètres par paire (navigables) ────────────────────────────────
        self._pair_lbl = ClickablePairLabel()
        self._pair_lbl.pair_changed.connect(self._on_pair_changed)
        self.sp_kfi  = dspin(0.30,  0.0,  3.0, 0.01)
        self.sp_kfi.setToolTip(
            "Position initiale kF (π/a) pour cette paire, comptée depuis centre Γ.\n"
            "Voir les lignes tiret-point colorées dans le graphique MDC."
        )
        self.sp_gi   = dspin(0.08, 0.01,  0.5, 0.01)
        self.sp_gi.setToolTip(
            "Demi-largeur initiale de la Lorentzienne (π/a).\n"
            "Valeur de départ pour l'optimiseur. Voir les courbes colorées dans le graphique MDC."
        )
        self.sp_gm   = dspin(0.30, 0.05,  1.0, 0.05)
        self.sp_gm.setToolTip(
            "Demi-largeur maximale autorisée (π/a) — contrainte de l'optimiseur scipy.\n"
            "Voir les zones colorées translucides autour des pics dans le graphique MDC."
        )

        # ── paramètres globaux ────────────────────────────────────────────────
        self.sp_xg   = dspin(0.10, 0.0,  0.5,  0.01)
        self.sp_xg.setToolTip(
            "Demi-largeur de la zone de contrainte autour du centre Γ (π/a).\n"
            "L'optimiseur limite xg dans [centre − xg_range, centre + xg_range].\n"
            "Voir le rectangle cyan dans le graphique MDC."
        )
        self.sp_cx   = dspin(0.0, -1.0,  1.0,  0.01)
        self.sp_cx.setToolTip(
            "Centre de symétrie des paires (position Γ, en π/a).\n"
            "Voir la ligne cyan pointillée dans le graphique MDC.\n"
            "Utiliser 'Auto Γ BM' ou 'Γ FS → BM' pour le calculer automatiquement."
        )
        self.sp_k0m  = dspin(0.0,  0.0,  2.0,  0.05)
        self.sp_k0m.setToolTip(
            "Distance maximale autorisée de kF par rapport à Γ (π/a).\n"
            "Voir les lignes magenta dans le graphique MDC si actif."
        )
        self.chk_k0a = QCheckBox("auto"); self.chk_k0a.setChecked(True)
        self.chk_k0a.setToolTip("Si coché, pas de limite sur kF. Décocher pour activer kF max.")
        self.sp_k0m.setEnabled(False)
        self.chk_k0a.stateChanged.connect(
            lambda: self.sp_k0m.setEnabled(not self.chk_k0a.isChecked()))
        self.cmb_wm  = QComboBox(); self.cmb_wm.addItems(["symmetric","asymmetric"])
        self.cmb_wm.setFixedWidth(110)
        self.cmb_wm.setToolTip(
            "symmetric : les deux pics de la paire ont le même γ.\n"
            "asymmetric : γ gauche et droit peuvent différer (pics asymétriques)."
        )
        self.sp_ma   = dspin(0.01, 0.0, 1.0, 0.01)
        self.sp_ma.setToolTip(
            "Amplitude minimale relative d'un pic pour être accepté (0–1).\n"
            "Rejette les pics dont l'amplitude est < ampl_min × max(MDC).\n"
            "Augmenter pour éliminer les faux pics dus au bruit."
        )
        self.sp_mj   = dspin(0.20, 0.0, 1.0, 0.05)
        self.sp_mj.setToolTip(
            "Saut maximal autorisé entre positions kF consécutives (π/a).\n"
            "Contrôle la continuité de la dispersion lors du fit complet.\n"
            "Réduire si la dispersion saute d'un point à l'autre."
        )
        self.cmb_sd  = QComboBox(); self.cmb_sd.addItems(["up","down"])
        self.cmb_sd.setFixedWidth(80)
        self.cmb_sd.setToolTip(
            "up : parcourt la BM de ev_start (bas) vers ev_end (proche EF).\n"
            "down : sens inverse. Choisir le sens où les pics sont les plus nets en départ."
        )

        for w in (self.sp_sff, self.sp_sfd, self.sp_kfi, self.sp_gi, self.sp_gm,
                  self.sp_xg, self.sp_cx, self.sp_k0m, self.sp_ma, self.sp_mj):
            w.valueChanged.connect(self.fit_only_changed)
        self.cmb_wm.currentIndexChanged.connect(self.fit_only_changed)

        self.sp_dE_meV = dspin(15.0, 1.0, 200.0, 1.0, dec=1)
        self.sp_dE_meV.setToolTip(
            "FWHM énergie instrumentale estimée ou saisie manuellement (meV).\n"
            "Utilisée pour calculer Γ corrigé après fit MDC."
        )
        self.sp_dk_inv_a = dspin(0.005, 0.001, 0.1, 0.001, dec=4)
        self.sp_dk_inv_a.setToolTip(
            "FWHM k instrumentale en π/a, estimée depuis angle_step si disponible.\n"
            "Utilisée pour calculer Γ corrigé après fit MDC."
        )
        self.lbl_dE_src = QLabel("—")
        self.lbl_dk_src = QLabel("—")
        for lbl in (self.lbl_dE_src, self.lbl_dk_src):
            lbl.setToolTip("Provenance résolution : Estimée, Manuelle ou Défaut")
        self.sp_dE_meV.valueChanged.connect(self._mark_resolution_manual_if_user_edit)
        self.sp_dk_inv_a.valueChanged.connect(self._mark_resolution_manual_if_user_edit)
        self.sp_dE_meV.valueChanged.connect(self.fit_only_changed)
        self.sp_dk_inv_a.valueChanged.connect(self.fit_only_changed)

        k0w = QWidget(); k0l = QHBoxLayout(k0w); k0l.setContentsMargins(0,0,0,0)
        k0l.addWidget(self.sp_k0m); k0l.addWidget(self.chk_k0a)

        fl3.addRow("Nb paires:",        self.sp_np)
        fl3.addRow("Lissage fit σ:",    self.sp_sff)
        fl3.addRow("Lissage détect σ:", self.sp_sfd)
        fl3.addRow(hsep())
        fl3.addRow(self._pair_lbl)
        fl3.addRow("kF init (π/a):",    self.sp_kfi)
        fl3.addRow("γ init (π/a):",     self.sp_gi)
        fl3.addRow("γ max (π/a):",      self.sp_gm)
        fl3.addRow(hsep())
        fl3.addRow("Fenêtre Γ (π/a):",  self.sp_xg)
        fl3.addRow("Centre Γ (π/a):",   self.sp_cx)
        fl3.addRow("kF max (π/a):",     k0w)
        fl3.addRow("Symétrie paire:",   self.cmb_wm)
        fl3.addRow(hsep())
        fl3.addRow("Ampl. min:",        self.sp_ma)
        fl3.addRow("Saut max (π/a):",   self.sp_mj)
        fl3.addRow("Sens scan:",        self.cmb_sd)
        fl3.addRow(hsep())
        de_row = QWidget(); de_lay = QHBoxLayout(de_row); de_lay.setContentsMargins(0,0,0,0)
        de_lay.addWidget(self.sp_dE_meV, 1); de_lay.addWidget(self.lbl_dE_src)
        dk_row = QWidget(); dk_lay = QHBoxLayout(dk_row); dk_lay.setContentsMargins(0,0,0,0)
        dk_lay.addWidget(self.sp_dk_inv_a, 1); dk_lay.addWidget(self.lbl_dk_src)
        fl3.addRow("ΔE FWHM (meV):", de_row)
        fl3.addRow("Δk FWHM (π/a):", dk_row)
        _fcl.addWidget(grp_f)

        # ── waterfall MDC (visible seulement dans le sous-onglet Waterfall) ────
        self._waterfall_controls_widget = QGroupBox("Waterfall MDC")
        fl_wf = QFormLayout(self._waterfall_controls_widget)
        self.sp_wf_n = ispin(32, 10, 80)
        self.sp_wf_n.setToolTip(
            "Nombre cible de MDCs affichées dans le waterfall.\n"
            "Moins de courbes = plus de relief et moins de surcharge."
        )
        self.sp_wf_relief = dspin(1.8, 0.5, 4.0, 0.1, dec=1)
        self.sp_wf_relief.setToolTip(
            "Amplitude visuelle des MDCs dans le waterfall.\n"
            "Augmenter pour mieux voir les pics ; trop haut crée du chevauchement."
        )
        self.sp_wf_n.valueChanged.connect(self.fit_only_changed)
        self.sp_wf_relief.valueChanged.connect(self.fit_only_changed)
        fl_wf.addRow("Courbes:", self.sp_wf_n)
        fl_wf.addRow("Relief:", self.sp_wf_relief)
        self._waterfall_controls_widget.setVisible(False)
        _fcl.addWidget(self._waterfall_controls_widget)

        # ── boutons ───────────────────────────────────────────────────────────
        _fcl.addWidget(hsep())
        btn_g = QPushButton("Guess  (fit MDC ici)  [Ctrl+G]")
        btn_g.setStyleSheet("background:#1a6b3a;color:white;font-weight:bold;padding:6px;")
        btn_g.clicked.connect(self.guess_requested)
        _fcl.addWidget(btn_g)

        self._gamma_tools_widget = QWidget()
        gamma_lay = QVBoxLayout(self._gamma_tools_widget)
        gamma_lay.setContentsMargins(0, 0, 0, 0)
        gamma_lay.setSpacing(4)
        btn_gamma = QPushButton("Auto Γ BM")
        btn_gamma.setToolTip("Estime le centre Γ par la médiane des milieux de paires MDC.")
        btn_gamma.clicked.connect(self.gamma_bm_requested)
        gamma_lay.addWidget(btn_gamma)

        btn_ref = QPushButton("Γ FS → BM")
        btn_ref.setToolTip("Applique le Γ de référence mesuré sur une FS à la BM courante.")
        btn_ref.clicked.connect(self.gamma_ref_requested)
        gamma_lay.addWidget(btn_ref)
        _fcl.addWidget(self._gamma_tools_widget)

        btn_f = QPushButton("Fit complet  [Ctrl+F]")
        btn_f.setStyleSheet("background:#2a6099;color:white;font-weight:bold;padding:6px;")
        btn_f.clicked.connect(self.full_fit_requested)
        _fcl.addWidget(btn_f)

        btn_cl = QPushButton("Effacer kF")
        btn_cl.clicked.connect(self.clear_kf_requested)
        _fcl.addWidget(btn_cl)

        self.lbl_res = QLabel("")
        self.lbl_res.setWordWrap(True)
        self.lbl_res.setStyleSheet("color:#8fc;font-family:monospace;font-size:11px;")
        _fcl.addWidget(self.lbl_res)

        lay.addWidget(self._fit_controls_widget)
        lay.addStretch()

    # ── accès params ──────────────────────────────────────────────────────────
    def update_ef_reference_button(self, ref: dict | None):
        """Met à jour le label/état du bouton EF réf selon la session."""
        if not ref:
            self.btn_ef_ref.setText("Aucune réf EF (calibrer un Au d'abord)")
            self.btn_ef_ref.setEnabled(False)
            self.btn_ef_ref.setToolTip(
                "Aucune référence EF enregistrée dans cette session.\n"
                "Pour en créer une : 'Calibrer EF auto' sur un scan Au, "
                "puis cocher 'Enregistrer comme référence' dans le dialog."
            )
            return
        mode = ref.get("mode", "?")
        src_path = ref.get("source_file", "")
        src_name = Path(src_path).name if src_path else "(source inconnue)"
        if mode == "scalar":
            shift_meV = float(ref.get("ef_shift", 0.0)) * 1000.0
            label = f"Appliquer EF réf : {src_name} (delta={shift_meV:+.1f} meV)"
        elif mode == "poly":
            n_valid = int(ref.get("n_valid", 0))
            fwhm = float(ref.get("fwhm_res", 0.0)) * 1000.0
            label = f"Appliquer EF réf poly : {src_name} (n={n_valid}, FWHM≈{fwhm:.0f} meV)"
        else:
            label = f"Appliquer EF réf : {src_name}"
        self.btn_ef_ref.setText(label)
        self.btn_ef_ref.setEnabled(True)
        self.btn_ef_ref.setToolTip(
            f"Référence EF enregistrée :\n"
            f"  mode = {mode}\n"
            f"  source = {src_path or '?'}\n"
            f"Applique cette correction au fichier courant."
        )

    def update_hv_source(self, source: str | None):
        """Affiche la provenance de hν : 'file', 'logbook', 'manual', None."""
        labels = {"file": "Fichier", "logbook": "Logbook", "manual": "Manuel"}
        self.lbl_hv_src.setText(labels.get(source or "", "Inconnu"))

    def _mark_hv_manual_if_user_edit(self):
        if not getattr(self, "_hv_source_lock", False):
            self.update_hv_source("manual")

    def set_hv_value_with_source(self, value: float, source: str):
        """Set la spinbox hν sans déclencher le marquage 'manuel'."""
        self._hv_source_lock = True
        try:
            self.sp_hv.blockSignals(True)
            self.sp_hv.setValue(float(value))
            self.sp_hv.blockSignals(False)
            self.update_hv_source(source)
        finally:
            self._hv_source_lock = False

    def update_resolution_source(self, source: str | None):
        """Affiche la provenance de la resolution : 'estimated', 'manual', 'default'."""
        self._resolution_source = source or "default"
        self._resolution_source_detail = self._resolution_source
        label = {"estimated": "Estimée", "manual": "Manuelle", "default": "Défaut"}.get(self._resolution_source, "Défaut")
        self.lbl_dE_src.setText(label)
        self.lbl_dk_src.setText(label)

    def mark_action_done(self, text: str):
        self.lbl_action.setText(f"Dernière action : {text}")

    def _mark_resolution_manual_if_user_edit(self):
        if not getattr(self, "_resolution_source_lock", False):
            self.update_resolution_source("manual")
            self._resolution_source_detail = "manual"

    def set_resolution_with_source(self, dE_meV: float, dk_inv_a: float, source: str, detail: str | None = None):
        """Set les spinboxes resolution sans déclencher le marquage manuel."""
        self._resolution_source_lock = True
        try:
            for sp, value in ((self.sp_dE_meV, dE_meV), (self.sp_dk_inv_a, dk_inv_a)):
                sp.blockSignals(True)
                sp.setValue(float(value))
                sp.blockSignals(False)
            self.update_resolution_source(source)
            self._resolution_source_detail = detail or source
        finally:
            self._resolution_source_lock = False

    def update_copy_params_button(self, n_targets: int):
        """Met à jour le label/état du bouton 'Propager fit params'."""
        if n_targets <= 0:
            self.btn_copy.setText("Propager fit params (0 cible)")
            self.btn_copy.setEnabled(False)
            self.btn_copy.setToolTip(
                "Aucun fichier non-fitté dans le dossier (hors fichier courant).\n"
                "Tous les autres ont déjà un fit_result enregistré : ils ne seront pas écrasés."
            )
        else:
            self.btn_copy.setText(f"Propager fit params ({n_targets} cible{'s' if n_targets > 1 else ''})")
            self.btn_copy.setEnabled(True)
            self.btn_copy.setToolTip(
                f"Copie les paramètres de fit MDC actuels vers les {n_targets} "
                f"fichier(s) du dossier qui n'ont pas encore été fittés.\n"
                f"Les fichiers déjà fittés ne sont jamais écrasés."
            )

    def get_fit_params(self) -> FitParams:
        self._save_pair()
        p0 = self._pair_params[0] if self._pair_params else {}
        return FitParams(
            n_pairs       = self.sp_np.value(),
            ev_start      = self.sp_evs.value(),
            ev_end        = self.sp_eve.value(),
            k_min         = self.sp_kmin.value(),
            k_max         = self.sp_kmax.value(),
            smooth_fit    = self.sp_sff.value(),
            smooth_detect = self.sp_sfd.value(),
            gamma_init    = p0.get("gamma_init", 0.08),
            gamma_max     = p0.get("gamma_max",  0.30),
            xg_range      = self.sp_xg.value(),
            center_init   = self.sp_cx.value(),
            k0_max        = None if self.chk_k0a.isChecked() else self.sp_k0m.value(),
            width_mode    = self.cmb_wm.currentText(),
            min_amplitude = self.sp_ma.value(),
            max_jump      = self.sp_mj.value(),
            scan_direction= self.cmb_sd.currentText(),
            dE_meV        = self.sp_dE_meV.value(),
            dk_inv_a      = self.sp_dk_inv_a.value(),
            pairs         = [dict(p) for p in self._pair_params],
        )

    def set_fit_controls_visible(self, visible: bool):
        self._fit_controls_widget.setVisible(visible)

    def set_utilities_visible(self, visible: bool):
        self._utils_widget.setVisible(visible)

    def set_fit_roi_active(self, active: bool):
        self.btn_fit_roi.blockSignals(True)
        self.btn_fit_roi.setChecked(bool(active))
        self.btn_fit_roi.blockSignals(False)

    def set_context(self, context: str):
        """Adapte le panneau droit à l'onglet actif."""
        is_bm = context == "bm"
        is_mdc = context == "mdc"
        self._energy_widget.setVisible(is_bm)
        self._ef_widget.setVisible(is_bm)
        self._utils_widget.setVisible(is_bm)
        self._fit_controls_widget.setVisible(is_mdc)
        self._gamma_tools_widget.setVisible(False)
        if not is_mdc:
            self.set_waterfall_controls_visible(False)

    def set_waterfall_controls_visible(self, visible: bool):
        self._waterfall_controls_widget.setVisible(bool(visible))

    def grid_params(self) -> dict:
        return {
            "enabled": True,
            "method": "display_fft2mask",
            "grid_period_px": None,
            "grid_freq": None,
            "notch_width": 2,
            "notch_sigma": 0.8,
            "strength": float(self.sp_grid_strength.value()),
            "fft2_center_radius": 18.0,
            "fft2_peak_sensitivity": 2.5,
            "fft2_plane": "display",
        }

    def load_fit_params(self, fp: FitParams):
        for sp, val in [
            (self.sp_evs,  fp.ev_start),  (self.sp_eve,  fp.ev_end),
            (self.sp_kmin, fp.k_min),     (self.sp_kmax, fp.k_max),
            (self.sp_sff,  fp.smooth_fit),(self.sp_sfd,  fp.smooth_detect),
            (self.sp_xg,   fp.xg_range),  (self.sp_cx,   fp.center_init),
            (self.sp_ma,   fp.min_amplitude),(self.sp_mj, fp.max_jump),
            (self.sp_dE_meV, getattr(fp, "dE_meV", 15.0)),
            (self.sp_dk_inv_a, getattr(fp, "dk_inv_a", 0.005)),
        ]:
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        if fp.k0_max is not None:
            self.chk_k0a.setChecked(False); self.sp_k0m.setValue(fp.k0_max)
        else:
            self.chk_k0a.setChecked(True)
        self.cmb_wm.setCurrentText(fp.width_mode)
        self.cmb_sd.setCurrentText(fp.scan_direction)

        # ── paires ────────────────────────────────────────────────────────────
        n = fp.n_pairs
        raw = list(getattr(fp, "pairs", None) or [])
        if not raw:
            raw = [{"kF_init": 0.30, "gamma_init": fp.gamma_init, "gamma_max": fp.gamma_max}]
        while len(raw) < n:
            raw.append(dict(raw[-1]))
        self._pair_params = raw[:max(n, 1)]
        self._current_pair = 0
        self.sp_np.blockSignals(True); self.sp_np.setValue(n); self.sp_np.blockSignals(False)
        self._pair_lbl.setup(n, 0)
        self._load_pair(0)

    # ── gestion par paire ─────────────────────────────────────────────────────
    def _save_pair(self):
        i = self._current_pair
        if i < len(self._pair_params):
            self._pair_params[i] = {
                "kF_init":    self.sp_kfi.value(),
                "gamma_init": self.sp_gi.value(),
                "gamma_max":  self.sp_gm.value(),
            }

    def _load_pair(self, i: int):
        i = max(0, min(i, len(self._pair_params) - 1))
        p = self._pair_params[i]
        for sp, key, default in [
            (self.sp_kfi, "kF_init",    0.30),
            (self.sp_gi,  "gamma_init", 0.08),
            (self.sp_gm,  "gamma_max",  0.30),
        ]:
            sp.blockSignals(True); sp.setValue(p.get(key, default)); sp.blockSignals(False)

    def _on_n_pairs_changed(self, n: int):
        self._save_pair()
        default = dict(self._pair_params[-1]) if self._pair_params else \
                  {"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}
        while len(self._pair_params) < n:
            self._pair_params.append(dict(default))
        self._pair_params = self._pair_params[:max(n, 1)]
        self._current_pair = min(self._current_pair, n - 1)
        self._pair_lbl.setup(n, self._current_pair)
        self._load_pair(self._current_pair)
        self.params_changed.emit()

    def _on_pair_changed(self, i: int):
        self._save_pair()
        self._current_pair = i
        self._load_pair(i)
