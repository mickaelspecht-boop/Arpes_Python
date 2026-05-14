"""Section 'Distorsion BM' du FitParamsPanel.

Builder externe pour le groupbox de correction trapèze + parabole
(checkboxes individuelles, spinboxes, boutons Appliquer/Auto/Réinit/Calib).
La logique vit dans `arpes.ui.controllers.distortion_controller`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import compact_button, dspin


def build_bm_distortion_section(panel, lay) -> None:
    panel._distortion_widget = QGroupBox("Distorsion BM")
    panel._distortion_widget.setToolTip(
        "Correction géométrique des distorsions détecteur Scienta :\n"
        "• Trapèze θ : redresse les bords latéraux inclinés (slopes en Δkpar/ΔeV).\n"
        "• Parabole E : warp E_corr = E + a·(kpar−k0)² (non-isochromaticity).\n"
        "Ne soustrait PAS la dispersion physique. À calibrer sur Au polycristallin."
    )
    fl = QFormLayout(panel._distortion_widget)

    # ── trapèze ─────────────────────────────────────────────────────────────
    panel.chk_distortion_trap = QCheckBox("Trapèze (θ)")
    panel.chk_distortion_trap.setToolTip("Active la correction trapézoïdale en kpar.")
    panel.chk_distortion_trap_sym = QCheckBox("Symétrique")
    panel.chk_distortion_trap_sym.setChecked(True)
    panel.chk_distortion_trap_sym.setToolTip(
        "Couple slope_left = -slope_right (trapèze symétrique). "
        "Décocher pour ajuster les deux bords indépendamment."
    )
    panel.sp_distortion_slope_l = dspin(0.0, -1.0, 1.0, 0.005, dec=4)
    panel.sp_distortion_slope_l.setToolTip("Pente bord gauche en Δkpar/ΔeV (π/a par eV).")
    panel.sp_distortion_slope_r = dspin(0.0, -1.0, 1.0, 0.005, dec=4)
    panel.sp_distortion_slope_r.setToolTip("Pente bord droit en Δkpar/ΔeV.")
    panel.sp_distortion_pivot = dspin(0.0, -10.0, 10.0, 0.01, dec=3)
    panel.sp_distortion_pivot.setToolTip("E pivot (eV). Par défaut : milieu de fenêtre.")

    def _sync_symmetric(_=None):
        if panel.chk_distortion_trap_sym.isChecked():
            panel.sp_distortion_slope_r.blockSignals(True)
            panel.sp_distortion_slope_r.setValue(-panel.sp_distortion_slope_l.value())
            panel.sp_distortion_slope_r.blockSignals(False)

    panel.sp_distortion_slope_l.valueChanged.connect(_sync_symmetric)
    panel.chk_distortion_trap_sym.toggled.connect(_sync_symmetric)

    # ── parabole ────────────────────────────────────────────────────────────
    panel.chk_distortion_para = QCheckBox("Parabole (E)")
    panel.chk_distortion_para.setToolTip(
        "Active le warp parabolique de l'axe E (non-isochromaticity)."
    )
    panel.sp_distortion_a = dspin(0.0, -2.0, 2.0, 0.001, dec=4)
    panel.sp_distortion_a.setToolTip(
        "Coefficient a : E_corr = E + a·(kpar−k0)². "
        "a > 0 si EF concave vers le bas (centre plus haut en E_kin)."
    )
    panel.sp_distortion_k0 = dspin(0.0, -5.0, 5.0, 0.01, dec=3)
    panel.sp_distortion_k0.setToolTip("k0 : sommet de la parabole (π/a).")

    # ── boutons ─────────────────────────────────────────────────────────────
    btn_apply = compact_button(QPushButton("Appliquer distorsion"))
    btn_apply.setToolTip("Sauve les paramètres et recalcule l'affichage BM.")
    btn_apply.clicked.connect(panel.distortion_apply_requested)
    btn_auto = compact_button(QPushButton("Auto-detect"))
    btn_auto.setToolTip(
        "Estime slopes (envelope p80) et a, k0 (polyfit deg 2 sur argmax).\n"
        "Refusé si n_kpar < 16 ou dispersion < 10 meV."
    )
    btn_auto.clicked.connect(panel.distortion_auto_requested)
    btn_reset = compact_button(QPushButton("Réinitialiser"))
    btn_reset.setToolTip("Désactive la correction pour ce fichier (toggle off réversible).")
    btn_reset.clicked.connect(panel.distortion_reset_requested)
    btn_calib = compact_button(QPushButton("Importer calib"))
    btn_calib.setToolTip(
        "Importe la calibration partagée (lens_mode, pass_energy, hν) depuis "
        "~/.config/arpes/distortion_calib.json si elle existe."
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

    panel.lbl_distortion = QLabel("Distorsion BM : désactivée.")
    panel.lbl_distortion.setWordWrap(True)
    panel.lbl_distortion.setStyleSheet("color:#aaa; font-size:10px;")

    fl.addRow(panel.chk_distortion_trap)
    fl.addRow(panel.chk_distortion_trap_sym)
    fl.addRow("slope L:", panel.sp_distortion_slope_l)
    fl.addRow("slope R:", panel.sp_distortion_slope_r)
    fl.addRow("pivot E:", panel.sp_distortion_pivot)
    fl.addRow(panel.chk_distortion_para)
    fl.addRow("a:", panel.sp_distortion_a)
    fl.addRow("k0:", panel.sp_distortion_k0)
    fl.addRow(btn_grid)
    fl.addRow(panel.lbl_distortion)

    lay.addWidget(panel._distortion_widget)


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
            "symmetric": bool(panel.chk_distortion_trap_sym.isChecked()),
        },
        "parabola": {
            "enabled": bool(panel.chk_distortion_para.isChecked()),
            "a": float(panel.sp_distortion_a.value()),
            "k0": float(panel.sp_distortion_k0.value()),
        },
    }


def set_bm_distortion_state(panel, cfg: dict | None) -> None:
    cfg = cfg or {}
    trap = cfg.get("trapezoid") or {}
    para = cfg.get("parabola") or {}
    for w in (panel.chk_distortion_trap, panel.chk_distortion_trap_sym,
              panel.sp_distortion_slope_l, panel.sp_distortion_slope_r,
              panel.sp_distortion_pivot, panel.chk_distortion_para,
              panel.sp_distortion_a, panel.sp_distortion_k0):
        w.blockSignals(True)
    panel.chk_distortion_trap.setChecked(bool(trap.get("enabled", False)))
    panel.chk_distortion_trap_sym.setChecked(bool(trap.get("symmetric", True)))
    panel.sp_distortion_slope_l.setValue(float(trap.get("slope_left", 0.0) or 0.0))
    panel.sp_distortion_slope_r.setValue(float(trap.get("slope_right", 0.0) or 0.0))
    pivot = trap.get("pivot_ev")
    panel.sp_distortion_pivot.setValue(float(pivot) if pivot is not None else 0.0)
    panel.chk_distortion_para.setChecked(bool(para.get("enabled", False)))
    panel.sp_distortion_a.setValue(float(para.get("a", 0.0) or 0.0))
    panel.sp_distortion_k0.setValue(float(para.get("k0", 0.0) or 0.0))
    for w in (panel.chk_distortion_trap, panel.chk_distortion_trap_sym,
              panel.sp_distortion_slope_l, panel.sp_distortion_slope_r,
              panel.sp_distortion_pivot, panel.chk_distortion_para,
              panel.sp_distortion_a, panel.sp_distortion_k0):
        w.blockSignals(False)
