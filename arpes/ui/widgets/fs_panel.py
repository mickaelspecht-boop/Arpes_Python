"""FS UI widgets (panel + canvas) — extracted from arpes/physics/fs.py.

Pure-physics helpers (FSParams, extract_fs_map, _robust_norm, cache helpers,
remove_detector_grid_artifact, _axis_signature, _fs_cache_key) remain in
arpes/physics/fs.py to keep the layering rule (no PyQt in physics/).
"""
from __future__ import annotations

from PyQt6.QtCore import QLocale, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSlider,
    QSpinBox, QToolButton, QVBoxLayout, QWidget,
)
from arpes.physics.bz import resolve_bz_preset
from arpes.physics.fs import FSParams
from arpes.ui.widgets._qt_helpers import compact_button
from arpes.ui.widgets.fs_canvas import FermiSurfaceCanvas
from arpes.ui.widgets.fs_panel_bz_controls import (
    build_bz_crystal_group,
    build_bz_theoretical_group,
)


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
    bz_labels_requested = pyqtSignal()  # "Label conventions..." button
    distortion_fs_toggled = pyqtSignal(bool)
    # --- Crystal BZ overlay (MP) -----------------------------------------
    mp_lattice_fetch_requested = pyqtSignal()   # "Fetch MP symmetry" button
    bz_crystal_overlay_changed = pyqtSignal()   # toggles / V0 / plan / phi_c
    dft_grid_load_requested = pyqtSignal()      # "Load 3D DFT" button
    dft_grid_clear_requested = pyqtSignal()     # "Forget DFT" button
    nesting_requested = pyqtSignal()            # "C(q) nesting" button

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        w = QWidget()
        self._lay = QVBoxLayout(w)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(w)
        # HS label convention of the current FS entry (display renames,
        # e.g. {"M": "Σ"}); set by the FS controller on entry change.
        self._bz_label_overrides: dict = {}
        self._build()

    def set_bz_label_overrides(self, overrides: dict | None, *, emit: bool = True) -> None:
        self._bz_label_overrides = dict(overrides or {})
        if emit:
            self.params_changed.emit()

    def _dspin(self, value, lo, hi, step, dec=3):
        sp = QDoubleSpinBox()
        sp.setLocale(QLocale(QLocale.Language.C))  # dot decimal regardless of system locale
        sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec); sp.setValue(value)
        sp.setKeyboardTracking(False)
        sp.valueChanged.connect(self.params_changed)
        return sp

    def _wire_rotation_slider(self) -> None:
        """Two-way sync between the rotation slider (×10 int) and the spinbox.

        The slider drives the spinbox (which emits ``params_changed`` → debounced
        redraw); the spinbox drives the slider back under ``blockSignals`` to
        avoid a feedback loop. Programmatic spinbox writes (session restore) also
        move the slider.
        """
        def _slider_to_spin(v: int) -> None:
            self.sp_fs_rotation.setValue(v / 10.0)

        def _spin_to_slider(v: float) -> None:
            self.sl_fs_rotation.blockSignals(True)
            self.sl_fs_rotation.setValue(int(round(float(v) * 10.0)))
            self.sl_fs_rotation.blockSignals(False)

        self.sl_fs_rotation.valueChanged.connect(_slider_to_spin)
        self.sp_fs_rotation.valueChanged.connect(_spin_to_slider)
        _spin_to_slider(self.sp_fs_rotation.value())

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

        grp_lat = QGroupBox("Lattice / π/a Units")
        fl = QFormLayout(grp_lat)
        self.sp_a = self._dspin(0.0, 0.0, 20.0, 0.01)
        self.sp_a.setToolTip("Lattice parameter a (Å), used for π/a units and physical results.")
        self.sp_b = self._dspin(0.0, 0.0, 20.0, 0.01)
        self.sp_b.setToolTip("Lattice parameter b (Å), used for π/a units in the FS map.")
        self.sp_kx0 = self._dspin(0.0, -5.0, 5.0, 0.01)
        self.sp_kx0.setToolTip("Γ center in kx (π/a) for recentering the FS map.")
        self.sp_ky0 = self._dspin(0.0, -5.0, 5.0, 0.01)
        self.sp_ky0.setToolTip("Γ center in ky (π/a) for recentering the FS map.")
        fl.addRow("a (Å):", self.sp_a)
        fl.addRow("b (Å):", self.sp_b)
        fl.addRow("kx center:", self.sp_kx0)
        fl.addRow("ky center:", self.sp_ky0)
        self._add_collapsible_group(lay, "Lattice / π/a Units", grp_lat, open_default=False)

        grp_fs = QGroupBox("FS Map")
        fl2 = QFormLayout(grp_fs)
        self.sp_win = self._dspin(0.030, 0.001, 0.500, 0.005)
        self.sp_win.setToolTip("Integration window around EF for building FS intensity.")
        self.sp_ref_lo = self._dspin(-0.600, -5.000, 1.000, 0.050)
        self.sp_ref_lo.setToolTip("Lower reference bound for flux normalization.")
        self.sp_ref_hi = self._dspin(-0.200, -5.000, 1.000, 0.050)
        self.sp_ref_hi.setToolTip("Upper reference bound for flux normalization.")
        self.sp_sm = self._dspin(1.0, 0.0, 8.0, 0.25, dec=2)
        self.sp_sm.setToolTip("Gaussian smoothing applied to the displayed FS map.")
        self.sp_fs_rotation = self._dspin(0.0, -180.0, 180.0, 0.5, dec=1)
        self.sp_fs_rotation.setToolTip(
            "Display rotation of the full centered FS map, pockets, BM cuts and BZ overlays."
        )
        # Slider for smooth, live rotation (couples both ways with the spinbox).
        # 0.1° resolution: int slider value = degrees × 10. Rotation only
        # re-transforms the cached FS (no re-extraction) so dragging stays fluid.
        self.sl_fs_rotation = QSlider(Qt.Orientation.Horizontal)
        self.sl_fs_rotation.setRange(-1800, 1800)
        self.sl_fs_rotation.setSingleStep(5)
        self.sl_fs_rotation.setPageStep(100)
        self.sl_fs_rotation.setToolTip("Drag to rotate the FS map and the theoretical BZ together (live).")
        self._wire_rotation_slider()
        self.btn_fs_rotation_reset = QToolButton()
        self.btn_fs_rotation_reset.setText("⟲")
        self.btn_fs_rotation_reset.setToolTip("Reset rotation to 0°.")
        self.btn_fs_rotation_reset.clicked.connect(lambda: self.sp_fs_rotation.setValue(0.0))
        self.cmb_cmap = QComboBox(); self.cmb_cmap.addItems(["inferno", "viridis", "cividis", "magma", "gray", "hot", "RdBu_r"])
        self.cmb_cmap.setToolTip("FS map color palette. cividis = color-blind safe (Nature); RdBu_r = self-energy/diff.")
        self.cmb_cmap.currentIndexChanged.connect(self.params_changed)
        self.chk_norm = QCheckBox("Flux normalization by slice"); self.chk_norm.setChecked(True)
        self.chk_norm.setToolTip(
            "Corrects flux slice by slice (ky axis) and detector profile (kx axis).\n"
            "Useful for CLS FS where intensity varies between steps and at detector edges."
        )
        self.chk_norm.stateChanged.connect(self.params_changed)
        fl2.addRow("EF window ±eV:", self.sp_win)
        fl2.addRow("Norm ref min:", self.sp_ref_lo)
        fl2.addRow("Norm ref max:", self.sp_ref_hi)
        fl2.addRow("Smoothing σ:", self.sp_sm)
        rot_row = QWidget()
        rot_lay = QHBoxLayout(rot_row)
        rot_lay.setContentsMargins(0, 0, 0, 0)
        rot_lay.setSpacing(4)
        rot_lay.addWidget(self.sl_fs_rotation, 1)
        rot_lay.addWidget(self.sp_fs_rotation)
        rot_lay.addWidget(self.btn_fs_rotation_reset)
        fl2.addRow("Rotation (°):", rot_row)
        fl2.addRow("Colormap:", self.cmb_cmap)
        fl2.addRow(self.chk_norm)
        self.btn_nesting = compact_button(QPushButton("C(q) nesting…"), max_width=200)
        self.btn_nesting.setToolTip(
            "Compute the FS autocorrelation C(q) = Σ_k A(k) A(k+q) of the current "
            "map and mark the strongest off-Γ peaks (candidate nesting / folding "
            "vectors). Geometric measure, not the susceptibility χ(q)."
        )
        self.btn_nesting.clicked.connect(self.nesting_requested)
        fl2.addRow(self.btn_nesting)
        self._add_collapsible_group(lay, "FS Map", grp_fs, open_default=False)

        build_bz_theoretical_group(self, lay)
        build_bz_crystal_group(self, lay)

        # --- Build workflow widgets (layout insertion deferred to the end) ---
        self.lbl_info = QLabel("Load a Solaris fast map or a CLS FS folder.")
        self.lbl_info.setWordWrap(True); self.lbl_info.setStyleSheet("color:#aaa; font-size:10px;")
        self.chk_distortion_fs = QCheckBox("Apply BM distortion to FS volume")
        self.chk_distortion_fs.setChecked(False)
        self.chk_distortion_fs.setToolTip(
            "Applies the shared BM distortion calibration (trapezoid) to the FS volume."
        )
        self.chk_distortion_fs.toggled.connect(self.distortion_fs_toggled)
        self._btn_redraw_fs = compact_button(QPushButton("Redraw FS"), max_width=160)
        self._btn_redraw_fs.clicked.connect(self.redraw_requested)
        grp_gamma = QGroupBox()
        gv = QVBoxLayout(grp_gamma); gv.setContentsMargins(6, 6, 6, 6); gv.setSpacing(4)
        btn_g = compact_button(QPushButton("Detect FS Γ"), max_width=200)
        btn_g.setToolTip("Detects Γ from MDC pair midpoints on the FS and recenters the map.")
        btn_g.clicked.connect(self.gamma_requested); gv.addWidget(btn_g)
        self.btn_pick_center = compact_button(QPushButton("Pick Γ Manually"), max_width=200)
        self.btn_pick_center.setCheckable(True)
        self.btn_pick_center.setToolTip("Enables cursor mode. Click = new recentered and saved Γ.")
        self.btn_pick_center.toggled.connect(self.manual_center_requested)
        gv.addWidget(self.btn_pick_center)
        btn_forget = compact_button(QPushButton("Forget Γ"), max_width=200)
        btn_forget.setToolTip("Resets the full Γ state (session reference, axis, fit_result).")
        btn_forget.clicked.connect(self.forget_gamma_requested); gv.addWidget(btn_forget)
        self._grp_gamma = grp_gamma
        self.chk_show_bm_cuts = QCheckBox("Show BM cuts")
        self.chk_show_bm_cuts.setToolTip(
            "Projects compatible BMs. Colors: cyan=exact, orange=Δazi, red=Δhv."
        )
        self.chk_show_bm_cuts.toggled.connect(self.bm_cuts_visibility_changed)
        self.sp_pairing_hv_tol = self._dspin(5.0, 0.5, 50.0, 0.5, dec=1)
        self.sp_pairing_hv_tol.setToolTip("Δhv tolerance (%) for FS↔BM pairing. 5% = same hv; 30% links kz scans.")
        self.sp_pairing_azi_tol = self._dspin(2.0, 0.0, 30.0, 0.5, dec=1)
        self.sp_pairing_azi_tol.setToolTip("Δazi tolerance (°) for pairing.")
        self.cmb_direction = QComboBox()
        self.cmb_direction.addItem("All dirs")
        for _d in ("Γ-X", "Γ-Y", "Γ-M", "Γ-K", "Γ-Σ", "Σ-X", "X-M", "M-K"):
            self.cmb_direction.addItem(_d)
        self.cmb_direction.setToolTip(
            "Filter linked BMs by cut direction (from the logbook). 'All dirs' = no filter."
        )
        self.cmb_direction.currentIndexChanged.connect(self.params_changed)
        self.bm_cuts_bar = QWidget()
        _hl = QHBoxLayout(self.bm_cuts_bar)
        _hl.setContentsMargins(6, 2, 6, 2)
        _hl.setSpacing(6)
        _hl.addWidget(self.chk_show_bm_cuts)
        _hl.addWidget(QLabel("Tol hv %:"))
        _hl.addWidget(self.sp_pairing_hv_tol)
        _hl.addWidget(QLabel("Tol azi°:"))
        _hl.addWidget(self.sp_pairing_azi_tol)
        _hl.addWidget(QLabel("Dir:"))
        _hl.addWidget(self.cmb_direction)
        _hl.addStretch(1)
        # The tolerance spinboxes already trigger params_changed (via _dspin),
        # which the controller debounces through _schedule_fs_redraw and which
        # re-collects the BM cuts with the new tolerance. Emitting an extra
        # immediate bm_cuts_visibility_changed here forced a second, un-debounced
        # full FS redraw per spin tick → lag + a backlog that kept redrawing for
        # seconds after the last click. Removed: the debounced path is enough.

        grp_pocket = QGroupBox("FS Pockets")
        fp = QFormLayout(grp_pocket)
        # Primary entry point — mirrors the canvas-toolbar "▭ Pocket" toggle,
        # which is too discreet to be discovered on its own.
        self.btn_pocket_lasso = QPushButton("▭  Select pocket (drag a box)")
        self.btn_pocket_lasso.setCheckable(True)
        self.btn_pocket_lasso.setStyleSheet("font-weight:bold;")
        self.btn_pocket_lasso.setToolTip(
            "Draw a box around ONE pocket on the map: the seed point and "
            "iso-level are derived automatically, then validate the preview."
        )
        fp.addRow(self.btn_pocket_lasso)
        self.lbl_pocket_count = QLabel("0 pockets")
        self.lbl_pocket_count.setStyleSheet("color:#aaa; font-size:10px;")
        self.cmb_pocket_quality = QComboBox()
        self.cmb_pocket_quality.addItems(["Fine", "Standard", "Stable"])
        self.cmb_pocket_quality.setCurrentText("Standard")
        self.cmb_pocket_quality.setToolTip(
            "Contour quality: Fine follows more detail, Stable resists streaks/noise better."
        )
        self.cmb_pocket_quality.currentIndexChanged.connect(self._on_pocket_quality_changed)
        self.sp_pocket_smooth_y = self._dspin(1.0, 0.0, 6.0, 0.25, dec=2); self.sp_pocket_smooth_y.setToolTip("ky smoothing before extraction.")
        self.sp_pocket_smooth_x = self._dspin(3.0, 0.0, 12.0, 0.25, dec=2); self.sp_pocket_smooth_x.setToolTip("kx smoothing before extraction (CLS anti-streaking).")
        self.sp_pocket_contour_window = self._ispin(9, 3, 25, 2); self.sp_pocket_contour_window.setToolTip("Closed-contour smoothing window (odd).")
        self.sp_pocket_simplify = self._dspin(0.015, 0.0, 0.100, 0.005, dec=3); self.sp_pocket_simplify.setToolTip("Minimum distance between stored contour points.")
        self.sp_pocket_min_area = self._dspin(0.20, 0.0, 20.0, 0.10, dec=2); self.sp_pocket_min_area.setToolTip("Minimum area in % BZ.")
        self.sp_pocket_n_bands = self._ispin(1, 1, 12, 1); self.sp_pocket_n_bands.setToolTip("Luttinger: number of bands occupying the pocket. Default: 1.")
        self.sp_pocket_spin = self._ispin(2, 1, 2, 1); self.sp_pocket_spin.setToolTip("Spin degeneracy (1 polarized, otherwise 2).")
        self.sp_pocket_hs_x_deg = self._dspin(0.0, -180.0, 180.0, 1.0, dec=1); self.sp_pocket_hs_x_deg.setToolTip("Γ-X (deg).")
        self.sp_pocket_hs_m_deg = self._dspin(45.0, -180.0, 180.0, 1.0, dec=1); self.sp_pocket_hs_m_deg.setToolTip("Γ-M (deg).")
        self.sp_pocket_hs_tol_deg = self._dspin(10.0, 1.0, 45.0, 1.0, dec=1); self.sp_pocket_hs_tol_deg.setToolTip("Sector tolerance for kF(Γ-X/M).")
        self.chk_pocket_bootstrap = QCheckBox("Bootstrap uncertainty")
        self.chk_pocket_bootstrap.setChecked(False)
        self.chk_pocket_bootstrap.setToolTip(
            "Enables bootstrap: N draws (level ±10%, smoothing ±25%) → "
            "median + standard deviation per field. Cost ≈ N× characterization time."
        )
        self.sp_pocket_bootstrap_n = self._ispin(20, 4, 100, 1)
        self.sp_pocket_bootstrap_n.setToolTip("Number of bootstrap draws. Default: 20.")
        self.sp_pocket_mdc_n = self._ispin(36, 8, 180, 4)
        self.sp_pocket_mdc_n.setToolTip("Radial MDC: number of sampled directions (deg = 360/N).")
        self.sp_pocket_mdc_r2 = self._dspin(0.5, 0.0, 1.0, 0.05, dec=2)
        self.sp_pocket_mdc_r2.setToolTip("Radial MDC: minimum R² to validate a Lorentzian fit.")
        self.chk_pocket_level_manual = QCheckBox("Manual level")
        self.chk_pocket_level_manual.setToolTip("Use the level below for the next right-click instead of the auto threshold.")
        # Level slider: dedicated signal (not params_changed) for live preview
        # without triggering a full FS redraw at each slider step.
        self.sp_pocket_level = QDoubleSpinBox()
        self.sp_pocket_level.setRange(0.0, 1.0)
        self.sp_pocket_level.setSingleStep(0.01)
        self.sp_pocket_level.setDecimals(3)
        self.sp_pocket_level.setValue(0.50)
        self.sp_pocket_level.setKeyboardTracking(False)
        self.sp_pocket_level.setToolTip(
            "Iso-intensity threshold. Live slider: drives the preview contour "
            "when a pocket is in preview mode."
        )
        self.sp_pocket_level.valueChanged.connect(
            lambda v: self.pocket_preview_level_changed.emit(float(v))
        )
        # Visible: what a physicist touches per pocket. Everything algorithmic
        # lives in the collapsed Advanced sub-group below (auto-expanded by the
        # controller when an MDC fit fails, so the relevant knobs surface
        # exactly when they are needed).
        fp.addRow(self.lbl_pocket_count)
        fp.addRow("Quality:", self.cmb_pocket_quality)
        fp.addRow(self.chk_pocket_level_manual)
        fp.addRow("Level :", self.sp_pocket_level)
        fp.addRow("n bands:", self.sp_pocket_n_bands)
        fp.addRow("Spin :", self.sp_pocket_spin)
        self._btn_pocket_adv = QPushButton("▶ Advanced settings")
        self._btn_pocket_adv.setCheckable(True)
        self._btn_pocket_adv.setStyleSheet("text-align:left; color:#aaa;")
        self._grp_pocket_adv = QGroupBox()
        fa = QFormLayout(self._grp_pocket_adv)
        fa.addRow("ky smoothing:", self.sp_pocket_smooth_y)
        fa.addRow("kx smoothing:", self.sp_pocket_smooth_x)
        fa.addRow("Contour :", self.sp_pocket_contour_window)
        fa.addRow("Simplify:", self.sp_pocket_simplify)
        fa.addRow("Min area:", self.sp_pocket_min_area)
        fa.addRow("Γ-X (°) :", self.sp_pocket_hs_x_deg)
        fa.addRow("Γ-M (°) :", self.sp_pocket_hs_m_deg)
        fa.addRow("Tol HS (°) :", self.sp_pocket_hs_tol_deg)
        fa.addRow(self.chk_pocket_bootstrap)
        fa.addRow("Bootstrap N :", self.sp_pocket_bootstrap_n)
        fa.addRow("MDC dirs :", self.sp_pocket_mdc_n)
        fa.addRow("MDC R²min :", self.sp_pocket_mdc_r2)
        self._grp_pocket_adv.setVisible(False)
        self._btn_pocket_adv.toggled.connect(self._set_pocket_advanced_visible)
        fp.addRow(self._btn_pocket_adv)
        fp.addRow(self._grp_pocket_adv)
        btn_export_pockets = compact_button(QPushButton("Export Pockets CSV"), max_width=160)
        btn_export_pockets.clicked.connect(self.pockets_export_requested)
        btn_clear_pockets = compact_button(QPushButton("Clear FS Pockets"), max_width=160)
        btn_clear_pockets.clicked.connect(self.pockets_clear_requested)
        fp.addRow(btn_export_pockets); fp.addRow(btn_clear_pockets)
        self._add_collapsible_group(lay, "FS Pockets", grp_pocket, open_default=True)
        # --- User workflow order: Γ → distortion → redraw → BM cuts ---
        lay.addWidget(self.lbl_info)
        self._add_collapsible_group(
            lay, "★  Γ Centering  ★", self._grp_gamma, open_default=True, highlight=True,
        )
        lay.addWidget(self.chk_distortion_fs)
        lay.addWidget(self._btn_redraw_fs)
        lay.addStretch(1)

    def params(self) -> FSParams:
        return FSParams(
            a_lattice=self.sp_a.value(), b_lattice=self.sp_b.value(),
            ef_window=self.sp_win.value(),
            norm_ref_lo=self.sp_ref_lo.value(), norm_ref_hi=self.sp_ref_hi.value(),
            smooth_sigma=self.sp_sm.value(),
            fs_rotation_deg=self.sp_fs_rotation.value(),
            klim=self.sp_klim.value(), kx_center=self.sp_kx0.value(), ky_center=self.sp_ky0.value(),
            bz_shape=self.cmb_bz_shape.currentText(),
            bz_half_x=self.sp_bzx.value(), bz_half_y=self.sp_bzy.value(),
            bz_angle_deg=self.sp_bz_angle.value(),
            bz_label_overrides=dict(self._bz_label_overrides),
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
        self.lbl_dft.setText(f"DFT: {label}" if label else "DFT: none")

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

    def _set_pocket_advanced_visible(self, on: bool) -> None:
        self._grp_pocket_adv.setVisible(bool(on))
        self._btn_pocket_adv.setText(
            "▼ Advanced settings" if on else "▶ Advanced settings")

    def expand_pocket_advanced(self) -> None:
        """Surface the algorithmic knobs (called when an MDC fit fails)."""
        self._btn_pocket_adv.setChecked(True)

    def set_pocket_count(self, count: int) -> None:
        n = int(count)
        self.lbl_pocket_count.setText(f"{n} pocket" + ("" if n == 1 else "s"))

    def _on_pocket_quality_changed(self, _idx: int = 0) -> None:
        presets = {
            "Fine": (0.5, 1.0, 5, 0.008),
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
