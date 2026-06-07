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


_HELP_TRAPEZE = "<b>Trapezoid θ</b>: straightens tilted BM edges."
_HELP_PARABOLE = "<b>Parabola E</b>: flattens instrumental iso-energy curvature."


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
    panel._distortion_widget = QGroupBox("BM Distortion")
    panel._distortion_widget.setToolTip(
        "Geometric correction for Scienta detector distortions.\n"
        "Apply if the raw image looks trapezoidal (tilted edges)\n"
        "or if iso-EF lines appear curved."
    )
    outer = QVBoxLayout(panel._distortion_widget)
    outer.setSpacing(6)

    # ── bloc trapèze ────────────────────────────────────────────────────────
    outer.addWidget(_help_label(_HELP_TRAPEZE))
    trap_row1 = QHBoxLayout()
    panel.chk_distortion_trap = QCheckBox("Enable trapezoid")
    panel.chk_distortion_trap.setToolTip("Enables trapezoidal correction in kpar.")
    trap_row1.addWidget(panel.chk_distortion_trap)
    trap_row1.addStretch()
    outer.addLayout(trap_row1)

    mode_row = QHBoxLayout()
    panel.rb_distortion_trap_sym = QRadioButton("Symmetric")
    panel.rb_distortion_trap_sym.setChecked(True)
    panel.rb_distortion_trap_sym.setToolTip(
        "slope_R = +slope_L: trapezoid that widens/narrows (lens artifact)."
    )
    panel.rb_distortion_trap_anti = QRadioButton("Antisymmetric")
    panel.rb_distortion_trap_anti.setToolTip(
        "slope_R = -slope_L: parallelogram (shear, detector misalignment)."
    )
    panel.rb_distortion_trap_free = QRadioButton("Free")
    panel.rb_distortion_trap_free.setToolTip("Independent left/right edges.")
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
        "Left-edge slope in delta kpar(pi/a) per eV.\n"
        "Positive = left edge moves left as E increases."
    )
    panel.sp_distortion_slope_r = dspin(0.0, -1.0, 1.0, 0.005, dec=4)
    panel.sp_distortion_slope_r.setToolTip("Right-edge slope in delta kpar/delta eV.")
    panel.sp_distortion_pivot = dspin(0.0, -10.0, 10.0, 0.01, dec=3)
    panel.sp_distortion_pivot.setToolTip(
        "Pivot energy (eV) where straightening is zero.\n"
        "Default: middle of the energy window."
    )
    fl_trap.addRow("Left slope (Δk/eV):", panel.sp_distortion_slope_l)
    fl_trap.addRow("Right slope (Δk/eV):", panel.sp_distortion_slope_r)
    fl_trap.addRow("Pivot E (eV):", panel.sp_distortion_pivot)
    outer.addLayout(fl_trap)

    def _coupled_target(value: float) -> float | None:
        if panel.rb_distortion_trap_sym.isChecked():
            return float(value)
        elif panel.rb_distortion_trap_anti.isChecked():
            return -float(value)
        return None

    def _sync_right_from_left(_=None):
        target = _coupled_target(panel.sp_distortion_slope_l.value())
        if target is None:
            return
        panel.sp_distortion_slope_r.blockSignals(True)
        panel.sp_distortion_slope_r.setValue(target)
        panel.sp_distortion_slope_r.blockSignals(False)

    def _sync_left_from_right(_=None):
        target = _coupled_target(panel.sp_distortion_slope_r.value())
        if target is None:
            return
        panel.sp_distortion_slope_l.blockSignals(True)
        panel.sp_distortion_slope_l.setValue(target)
        panel.sp_distortion_slope_l.blockSignals(False)

    panel.sp_distortion_slope_l.valueChanged.connect(_sync_right_from_left)
    panel.sp_distortion_slope_r.valueChanged.connect(_sync_left_from_right)
    panel.rb_distortion_trap_sym.toggled.connect(_sync_right_from_left)
    panel.rb_distortion_trap_anti.toggled.connect(_sync_right_from_left)

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
    panel.chk_distortion_para = QCheckBox("Enable parabola")
    panel.chk_distortion_para.setToolTip(
        "Enables parabolic warping of the E axis (non-isochromaticity)."
    )
    outer.addWidget(panel.chk_distortion_para)
    fl_para = QFormLayout()
    fl_para.setHorizontalSpacing(6)
    fl_para.setVerticalSpacing(3)
    panel.sp_distortion_a = dspin(0.0, -2.0, 2.0, 0.001, dec=4)
    panel.sp_distortion_a.setToolTip(
        "Coefficient a (eV/(π/a)^2): E_corr = E + a*(kpar - k0)^2.\n"
        "Positive if the iso-EF is concave downward (center higher in energy)."
    )
    panel.sp_distortion_k0 = dspin(0.0, -5.0, 5.0, 0.01, dec=3)
    panel.sp_distortion_k0.setToolTip("k0 position (π/a) of the parabola apex.")
    for w in (panel.chk_distortion_para, panel.sp_distortion_a, panel.sp_distortion_k0):
        if hasattr(w, "valueChanged"):
            w.valueChanged.connect(panel.distortion_preview_changed)
        else:
            w.toggled.connect(panel.distortion_preview_changed)
    fl_para.addRow("Curvature a (eV·(π/a)⁻²):", panel.sp_distortion_a)
    fl_para.addRow("Apex k0 (π/a):", panel.sp_distortion_k0)
    outer.addLayout(fl_para)

    outer.addWidget(_hline())

    # ── option recadrage ────────────────────────────────────────────────────
    panel.chk_distortion_crop = QCheckBox("Strictly crop to signal (k//)")
    panel.chk_distortion_crop.setChecked(True)
    panel.chk_distortion_crop.setToolTip(
        "After warping, removes fully NaN edge columns/rows\n"
        "(regions outside the signal). Adjusts axes to keep only the\n"
        "useful kpar region."
    )
    outer.addWidget(panel.chk_distortion_crop)

    outer.addWidget(_hline())

    # ── propagation au volume FS (opt-in, OFF par défaut) ───────────────────
    panel.chk_distortion_fs_propagate = QCheckBox("Propagate to FS volume")
    panel.chk_distortion_fs_propagate.setChecked(False)
    panel.chk_distortion_fs_propagate.setToolTip(
        "Applies the same correction (trapezoid only) to every BM\n"
        "in the FS volume, then recomputes the kx/ky map at E_F.\n"
        "Expensive: O(N_ky x N_kx x N_e). Disabled until distortion\n"
        "is calibrated. Refused if ky/<BM> drift > 15% (calibration not\n"
        "representative away from the center)."
    )
    panel.chk_distortion_fs_propagate.toggled.connect(
        lambda _on: panel.propagate_distortion_fs_toggled.emit()
    )
    outer.addWidget(panel.chk_distortion_fs_propagate)

    # ── boutons d'action ────────────────────────────────────────────────────
    btn_apply = compact_button(QPushButton("Apply"))
    btn_apply.setToolTip("Saves the parameters and recomputes the BM display.")
    btn_apply.clicked.connect(panel.distortion_apply_requested)
    btn_auto = compact_button(QPushButton("Auto-detect"))
    btn_auto.setToolTip(
        "Automatically estimates parameters:\n"
        "- slopes: intensity envelope (80th percentile) + linear fit\n"
        "- a, k0: degree-2 polyfit on the argmax per energy row\n"
        "Refused if n_kpar < 16 or dispersion < 10 meV."
    )
    btn_auto.clicked.connect(panel.distortion_auto_requested)
    btn_reset = compact_button(QPushButton("Reset"))
    btn_reset.setToolTip(
        "Disables correction for this file (bit-exact reversible toggle off)."
    )
    btn_reset.clicked.connect(panel.distortion_reset_requested)
    btn_calib = compact_button(QPushButton("Import calib"))
    btn_calib.setToolTip(
        "Imports the shared calibration for this analyzer geometry\n"
        "(lens_mode, pass_energy, hν) depuis ~/.config/arpes/distortion_calib.json."
    )
    btn_calib.clicked.connect(panel.distortion_import_calib_requested)

    btn_grid = QWidget()
    btn_lay = QGridLayout(btn_grid)
    btn_lay.setContentsMargins(0, 0, 0, 0)
    btn_lay.setHorizontalSpacing(4)
    btn_lay.setVerticalSpacing(3)
    btn_reset.setStyleSheet("color:#ffd0d0;")
    for i, b in enumerate((btn_auto, btn_calib, btn_apply, btn_reset)):
        b.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        b.setMaximumWidth(170)
        btn_lay.addWidget(b, i // 2, i % 2)
    for c in range(2):
        btn_lay.setColumnStretch(c, 1)
    outer.addWidget(btn_grid)

    panel.lbl_distortion = QLabel("BM distortion: disabled.")
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
