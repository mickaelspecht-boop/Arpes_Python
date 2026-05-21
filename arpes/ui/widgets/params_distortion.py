"""Section 'Distorsion BM' du FitParamsPanel.

Builder externe pour le groupbox de correction trapèze + parabole.
La logique vit dans `arpes.ui.controllers.distortion_controller`.

Ergonomie : sous-blocs Trapèze / Parabole avec un header explicatif
chacun, libellés en mots simples, et boutons d'action en bas.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import compact_button, dspin


_HELP_TRAPEZE = (
    "<b>Trapèze (θ)</b><br>"
    "Si les bords gauche/droit de la BM sont inclinés (le détecteur Scienta "
    "déforme le k-axe en fonction de E), redresser via deux pentes en "
    "<i>Δkpar par eV</i>. Calibrer sur une mesure d'or polycristallin pour "
    "que EF reste plat ±2 meV."
)
_HELP_PARABOLE = (
    "<b>Parabole (E)</b><br>"
    "Aplatit la courbure parabolique des iso-énergies (artefact de lentille). "
    "Warp axe E par <i>a·(kpar − k0)²</i>. Avec a&gt;0 si EF est concave "
    "vers le bas. Ne soustrait <b>pas</b> la dispersion physique."
)


def _help_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color:#9ab; font-size:10px; padding:2px 4px;"
                      " background:#2a2a3a; border-radius:3px;")
    return lbl


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#445;")
    return line


def build_bm_distortion_section(panel, lay) -> None:
    panel._distortion_widget = QGroupBox("Distorsion BM")
    panel._distortion_widget.setToolTip(
        "Correction géométrique des distorsions détecteur Scienta.\n"
        "À appliquer si l'image brute paraît trapézoïdale (bords inclinés)\n"
        "ou si les iso-EF apparaissent courbées."
    )
    outer = QVBoxLayout(panel._distortion_widget)
    outer.setSpacing(6)

    # ── bloc trapèze ────────────────────────────────────────────────────────
    outer.addWidget(_help_label(_HELP_TRAPEZE))
    trap_row1 = QHBoxLayout()
    panel.chk_distortion_trap = QCheckBox("Activer trapèze")
    panel.chk_distortion_trap.setToolTip("Active la correction trapézoïdale en kpar.")
    trap_row1.addWidget(panel.chk_distortion_trap)
    trap_row1.addStretch()
    outer.addLayout(trap_row1)

    mode_row = QHBoxLayout()
    panel.rb_distortion_trap_sym = QRadioButton("Symétrique")
    panel.rb_distortion_trap_sym.setChecked(True)
    panel.rb_distortion_trap_sym.setToolTip(
        "slope_R = -slope_L : trapèze qui s'élargit/rétrécit (artefact lentille)."
    )
    panel.rb_distortion_trap_anti = QRadioButton("Antisymétrique")
    panel.rb_distortion_trap_anti.setToolTip(
        "slope_R = +slope_L : parallélogramme (cisaillement, désalignement détecteur)."
    )
    panel.rb_distortion_trap_free = QRadioButton("Libre")
    panel.rb_distortion_trap_free.setToolTip("Bords gauche/droit indépendants.")
    panel._rb_distortion_trap_group = QButtonGroup(panel._distortion_widget)
    panel._rb_distortion_trap_group.setExclusive(True)
    for rb in (panel.rb_distortion_trap_sym, panel.rb_distortion_trap_anti,
               panel.rb_distortion_trap_free):
        panel._rb_distortion_trap_group.addButton(rb)
        mode_row.addWidget(rb)
    mode_row.addStretch()
    outer.addLayout(mode_row)
    # Compat avec ancien nom (utilisé par bm_distortion_params).
    panel.chk_distortion_trap_sym = panel.rb_distortion_trap_sym

    fl_trap = QFormLayout()
    fl_trap.setHorizontalSpacing(6)
    fl_trap.setVerticalSpacing(3)
    panel.sp_distortion_slope_l = dspin(0.0, -1.0, 1.0, 0.005, dec=4)
    panel.sp_distortion_slope_l.setToolTip(
        "Pente du bord gauche en Δkpar(π/a) par eV.\n"
        "Positif = bord gauche s'écarte vers la gauche quand E augmente."
    )
    panel.sp_distortion_slope_r = dspin(0.0, -1.0, 1.0, 0.005, dec=4)
    panel.sp_distortion_slope_r.setToolTip("Pente du bord droit en Δkpar/ΔeV.")
    panel.sp_distortion_pivot = dspin(0.0, -10.0, 10.0, 0.01, dec=3)
    panel.sp_distortion_pivot.setToolTip(
        "Énergie pivot (eV) où le redressement vaut zéro.\n"
        "Par défaut : milieu de la fenêtre énergie."
    )
    fl_trap.addRow("Pente gauche (Δk/eV):", panel.sp_distortion_slope_l)
    fl_trap.addRow("Pente droite (Δk/eV):", panel.sp_distortion_slope_r)
    fl_trap.addRow("Pivot E (eV):", panel.sp_distortion_pivot)
    outer.addLayout(fl_trap)

    def _sync_coupled(_=None):
        sl = panel.sp_distortion_slope_l.value()
        if panel.rb_distortion_trap_sym.isChecked():
            target = -sl
        elif panel.rb_distortion_trap_anti.isChecked():
            target = sl
        else:
            return
        panel.sp_distortion_slope_r.blockSignals(True)
        panel.sp_distortion_slope_r.setValue(target)
        panel.sp_distortion_slope_r.blockSignals(False)

    panel.sp_distortion_slope_l.valueChanged.connect(_sync_coupled)
    panel.rb_distortion_trap_sym.toggled.connect(_sync_coupled)
    panel.rb_distortion_trap_anti.toggled.connect(_sync_coupled)

    # Live preview : tout changement déclenche l'apparition de l'overlay
    # pointillé sur la BM (caché à nouveau après Apply / Reset).
    for w in (panel.chk_distortion_trap, panel.rb_distortion_trap_sym,
              panel.rb_distortion_trap_anti, panel.rb_distortion_trap_free,
              panel.sp_distortion_slope_l, panel.sp_distortion_slope_r,
              panel.sp_distortion_pivot):
        if hasattr(w, "valueChanged"):
            w.valueChanged.connect(panel.distortion_preview_changed)
        else:
            w.toggled.connect(panel.distortion_preview_changed)

    outer.addWidget(_hline())

    # ── bloc parabole ───────────────────────────────────────────────────────
    outer.addWidget(_help_label(_HELP_PARABOLE))
    panel.chk_distortion_para = QCheckBox("Activer parabole")
    panel.chk_distortion_para.setToolTip(
        "Active le warp parabolique de l'axe E (non-isochromaticity)."
    )
    outer.addWidget(panel.chk_distortion_para)
    fl_para = QFormLayout()
    fl_para.setHorizontalSpacing(6)
    fl_para.setVerticalSpacing(3)
    panel.sp_distortion_a = dspin(0.0, -2.0, 2.0, 0.001, dec=4)
    panel.sp_distortion_a.setToolTip(
        "Coefficient a (eV/(π/a)²) : E_corr = E + a·(kpar − k0)².\n"
        "Positif si l'iso-EF est concave vers le bas (centre plus haut en énergie)."
    )
    panel.sp_distortion_k0 = dspin(0.0, -5.0, 5.0, 0.01, dec=3)
    panel.sp_distortion_k0.setToolTip("Position k0 (π/a) du sommet de la parabole.")
    for w in (panel.chk_distortion_para, panel.sp_distortion_a, panel.sp_distortion_k0):
        if hasattr(w, "valueChanged"):
            w.valueChanged.connect(panel.distortion_preview_changed)
        else:
            w.toggled.connect(panel.distortion_preview_changed)
    fl_para.addRow("Courbure a (eV·(π/a)⁻²):", panel.sp_distortion_a)
    fl_para.addRow("Sommet k0 (π/a):", panel.sp_distortion_k0)
    outer.addLayout(fl_para)

    outer.addWidget(_hline())

    # ── option recadrage ────────────────────────────────────────────────────
    panel.chk_distortion_crop = QCheckBox("Recadrer strictement sur le signal (k//)")
    panel.chk_distortion_crop.setChecked(True)
    panel.chk_distortion_crop.setToolTip(
        "Après warp, supprime les colonnes/lignes complètement NaN aux bords\n"
        "(zones extérieures au signal). Ajuste les axes pour ne garder que la\n"
        "région utile en kpar."
    )
    outer.addWidget(panel.chk_distortion_crop)

    outer.addWidget(_hline())

    # ── propagation au volume FS (opt-in, OFF par défaut) ───────────────────
    panel.chk_distortion_fs_propagate = QCheckBox("Propager au volume FS")
    panel.chk_distortion_fs_propagate.setChecked(False)
    panel.chk_distortion_fs_propagate.setToolTip(
        "Applique la même correction (trapèze uniquement) à toutes les BM\n"
        "du volume FS, puis recalcule la carte kx/ky à E_F.\n"
        "Coûteux : O(N_ky × N_kx × N_e). Désactivé tant que la distorsion\n"
        "n'est pas calibrée. Refusé si drift ky/⟨BM⟩ > 15 % (calib non\n"
        "représentative hors centre)."
    )
    panel.chk_distortion_fs_propagate.toggled.connect(
        lambda _on: panel.propagate_distortion_fs_toggled.emit()
    )
    outer.addWidget(panel.chk_distortion_fs_propagate)

    # ── boutons d'action ────────────────────────────────────────────────────
    btn_apply = compact_button(QPushButton("Appliquer"))
    btn_apply.setToolTip("Sauve les paramètres et recalcule l'affichage BM.")
    btn_apply.clicked.connect(panel.distortion_apply_requested)
    btn_auto = compact_button(QPushButton("Auto-détecter"))
    btn_auto.setToolTip(
        "Estime automatiquement les paramètres :\n"
        "- pentes : enveloppe d'intensité (percentile 80) + fit linéaire\n"
        "- a, k0 : polyfit deg 2 sur l'argmax par ligne d'énergie\n"
        "Refusé si n_kpar < 16 ou dispersion < 10 meV."
    )
    btn_auto.clicked.connect(panel.distortion_auto_requested)
    btn_reset = compact_button(QPushButton("Réinitialiser"))
    btn_reset.setToolTip(
        "Désactive la correction pour ce fichier (toggle off réversible bit-exact)."
    )
    btn_reset.clicked.connect(panel.distortion_reset_requested)
    btn_calib = compact_button(QPushButton("Importer calib"))
    btn_calib.setToolTip(
        "Importe la calibration partagée pour cette géométrie analyseur\n"
        "(lens_mode, pass_energy, hν) depuis ~/.config/arpes/distortion_calib.json."
    )
    btn_calib.clicked.connect(panel.distortion_import_calib_requested)

    btn_grid = QWidget()
    btn_lay = QGridLayout(btn_grid)
    btn_lay.setContentsMargins(0, 0, 0, 0)
    btn_lay.setHorizontalSpacing(4)
    btn_lay.setVerticalSpacing(3)
    for i, b in enumerate((btn_apply, btn_auto, btn_reset, btn_calib)):
        b.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        b.setMaximumWidth(170)
        btn_lay.addWidget(b, i // 2, i % 2)
    for c in range(2):
        btn_lay.setColumnStretch(c, 1)
    outer.addWidget(btn_grid)

    panel.lbl_distortion = QLabel("Distorsion BM : désactivée.")
    panel.lbl_distortion.setWordWrap(True)
    panel.lbl_distortion.setStyleSheet("color:#aaa; font-size:10px;")
    outer.addWidget(panel.lbl_distortion)

    lay.addWidget(panel._distortion_widget)


def _trap_mode(panel) -> str:
    if panel.rb_distortion_trap_anti.isChecked():
        return "antisymmetric"
    if panel.rb_distortion_trap_free.isChecked():
        return "free"
    return "symmetric"


def bm_distortion_params(panel) -> dict:
    return {
        "enabled": bool(
            panel.chk_distortion_trap.isChecked() or panel.chk_distortion_para.isChecked()
        ),
        "trapezoid": {
            "enabled": bool(panel.chk_distortion_trap.isChecked()),
            "slope_left": float(panel.sp_distortion_slope_l.value()),
            "slope_right": float(panel.sp_distortion_slope_r.value()),
            "pivot_ev": float(panel.sp_distortion_pivot.value()),
            "mode": _trap_mode(panel),
            # legacy alias pour anciennes sessions
            "symmetric": bool(panel.rb_distortion_trap_sym.isChecked()),
        },
        "parabola": {
            "enabled": bool(panel.chk_distortion_para.isChecked()),
            "a": float(panel.sp_distortion_a.value()),
            "k0": float(panel.sp_distortion_k0.value()),
        },
        "crop_to_signal": bool(panel.chk_distortion_crop.isChecked()),
    }


def set_bm_distortion_state(panel, cfg: dict | None) -> None:
    cfg = cfg or {}
    trap = cfg.get("trapezoid") or {}
    para = cfg.get("parabola") or {}
    widgets = (panel.chk_distortion_trap,
               panel.rb_distortion_trap_sym, panel.rb_distortion_trap_anti,
               panel.rb_distortion_trap_free,
               panel.sp_distortion_slope_l, panel.sp_distortion_slope_r,
               panel.sp_distortion_pivot, panel.chk_distortion_para,
               panel.sp_distortion_a, panel.sp_distortion_k0,
               panel.chk_distortion_crop)
    for w in widgets:
        w.blockSignals(True)
    panel.chk_distortion_trap.setChecked(bool(trap.get("enabled", False)))
    mode = trap.get("mode")
    if mode is None:
        # legacy : `symmetric` bool seul
        mode = "symmetric" if trap.get("symmetric", True) else "free"
    panel.rb_distortion_trap_sym.setChecked(mode == "symmetric")
    panel.rb_distortion_trap_anti.setChecked(mode == "antisymmetric")
    panel.rb_distortion_trap_free.setChecked(mode == "free")
    panel.sp_distortion_slope_l.setValue(float(trap.get("slope_left", 0.0) or 0.0))
    panel.sp_distortion_slope_r.setValue(float(trap.get("slope_right", 0.0) or 0.0))
    pivot = trap.get("pivot_ev")
    panel.sp_distortion_pivot.setValue(float(pivot) if pivot is not None else 0.0)
    panel.chk_distortion_para.setChecked(bool(para.get("enabled", False)))
    panel.sp_distortion_a.setValue(float(para.get("a", 0.0) or 0.0))
    panel.sp_distortion_k0.setValue(float(para.get("k0", 0.0) or 0.0))
    panel.chk_distortion_crop.setChecked(bool(cfg.get("crop_to_signal", True)))
    for w in widgets:
        w.blockSignals(False)
