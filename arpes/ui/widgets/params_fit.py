"""MDC Fit section + Analysis range + Waterfall + Buttons + Γ tools.

External builder for the fit controls shown on the BM tab. The sub-groups
"Initials", "Constraints", "Detection / scan" and "Resolution" are
collapsible button-sections. State is persisted via the `fit_section_toggled`
signal of the parent panel.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import compact_button, dspin, hsep, ispin


# Material presets: apply a detection/scan/width parameter set.
# "Custom" = no modification. Selection is persisted in the session.
MATERIAL_PRESETS: dict[str, dict | None] = {
    "Custom": None,
    "Light metal": {
        "smooth_fit": 1.5, "smooth_detect": 2.0,
        "gamma_init": 0.05, "gamma_max": 0.20,
        "min_amplitude": 0.05, "max_jump": 0.15,
        "width_mode": "symmetric",
    },
    "Heavy metal": {
        "smooth_fit": 2.5, "smooth_detect": 3.5,
        "gamma_init": 0.12, "gamma_max": 0.40,
        "min_amplitude": 0.05, "max_jump": 0.25,
        "width_mode": "symmetric",
    },
    "Doped SC": {
        "smooth_fit": 2.0, "smooth_detect": 3.0,
        "gamma_init": 0.08, "gamma_max": 0.30,
        "min_amplitude": 0.02, "max_jump": 0.20,
        "width_mode": "independent",
    },
    "Noisy (heavy smoothing)": {
        "smooth_fit": 4.0, "smooth_detect": 5.0,
        "gamma_init": 0.10, "gamma_max": 0.30,
        "min_amplitude": 0.08, "max_jump": 0.25,
        "width_mode": "symmetric",
    },
}


class _ButtonCollapsible(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, title: str, content: QWidget, *, open_default: bool = True):
        super().__init__()
        self._title = title
        self._content = content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.btn = QPushButton()
        self.btn.setCheckable(True)
        self.btn.setStyleSheet(
            "QPushButton { background:#3a3a4a; color:#cde; padding:6px 8px;"
            " border-radius:3px; font-weight:bold; text-align:left; }"
            "QPushButton:checked { background:#4a4a6a; color:#fff; }"
            "QPushButton:hover { background:#454560; }"
        )
        self.btn.clicked.connect(self.setChecked)
        lay.addWidget(self.btn)
        lay.addWidget(content)
        self.setChecked(open_default)

    def isChecked(self) -> bool:
        return self.btn.isChecked()

    def setChecked(self, checked: bool) -> None:
        opened = bool(checked)
        self.btn.setChecked(opened)
        self._content.setVisible(opened)
        arrow = "▼" if opened else "▶"
        self.btn.setText(f"{arrow}  {self._title}")
        self.toggled.emit(opened)


def _make_collapsible(
    panel,
    title: str,
    key: str,
    *,
    open_default: bool = True,
) -> tuple[_ButtonCollapsible, QFormLayout]:
    content = QWidget()
    fl = QFormLayout(content)
    fl.setContentsMargins(2, 2, 2, 2)
    grp = _ButtonCollapsible(title, content, open_default=bool(open_default))
    grp.toggled.connect(lambda v, k=key: panel.fit_section_toggled.emit(k, bool(v)))
    panel._fit_sections[key] = grp
    return grp, fl


def build_fit_controls(panel, lay) -> None:
    panel._fit_controls_widget = QWidget()
    _fcl = QVBoxLayout(panel._fit_controls_widget)
    _fcl.setContentsMargins(0, 0, 0, 0)
    _fcl.setSpacing(4)

    panel._fit_sections = {}

    _build_zones_strip(panel, _fcl)
    _build_roi_group(panel, _fcl)
    _build_preset_combo(panel, _fcl)
    _build_init_section(panel, _fcl)
    _build_slice_inspector_section(panel, _fcl)
    _build_constraint_section(panel, _fcl)
    _build_detect_section(panel, _fcl)
    _build_resolution_section(panel, _fcl)
    _build_waterfall_group(panel, _fcl)
    _build_fit_buttons(panel, _fcl)

    lay.addWidget(panel._fit_controls_widget)


def _build_zones_strip(panel, _fcl) -> None:
    from arpes.ui.widgets.zones_strip import ZonesStrip
    panel.zones_strip = ZonesStrip()
    _fcl.addWidget(panel.zones_strip)


def _build_roi_group(panel, _fcl) -> None:
    grp_r = QGroupBox("Analysis range")
    fl2 = QFormLayout(grp_r)
    panel.sp_evs = dspin(-0.90, -5.0, 1.0, 0.05)
    panel.sp_eve = dspin(-0.005, -5.0, 1.0, 0.005)
    panel.sp_kmin = dspin(-0.80, -5.0, 5.0, 0.05)
    panel.sp_kmax = dspin(0.80, -5.0, 5.0, 0.05)
    for w in (panel.sp_evs, panel.sp_eve, panel.sp_kmin, panel.sp_kmax):
        w.valueChanged.connect(panel.params_changed)
    panel.btn_fit_roi = compact_button(QPushButton("Select on map"), max_width=180)
    panel.btn_fit_roi.setCheckable(True)
    panel.btn_fit_roi.setToolTip(
        "Enables click-drag rectangular selection on the BM/MDC Fit map.\n"
        "The selected area fills k_min/k_max and ev_start/ev_end."
    )
    panel.btn_fit_roi.toggled.connect(panel.fit_roi_requested)
    btn_fit_roi_reset = compact_button(QPushButton("Full BM"), max_width=120)
    btn_fit_roi_reset.setToolTip("Resets the analysis range to the full loaded map.")
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
        "Applies a parameter set (smoothing, γ, amplitude, jump, symmetry).\n"
        "Custom: no changes. The choice is saved in the session."
    )
    panel.cmb_fit_preset.currentTextChanged.connect(panel._on_preset_chosen)
    h.addWidget(panel.cmb_fit_preset, 1)
    _fcl.addWidget(row)


def _build_init_section(panel, _fcl) -> None:
    from arpes.ui.widgets.params import ClickablePairLabel
    grp, fl = _make_collapsible(panel, "Pair initials", "init")
    panel.sp_np = ispin(1, 1, 8)
    panel.sp_np.setToolTip("Number of Lorentzian pairs (= number of crossed bands).")
    panel.sp_np.valueChanged.connect(panel._on_n_pairs_changed)
    panel._pair_lbl = ClickablePairLabel()
    panel._pair_lbl.pair_changed.connect(panel._on_pair_changed)
    panel.sp_kfi = dspin(0.30, 0.0, 3.0, 0.01)
    panel.sp_kfi.setToolTip(
        "Initial kF position (π/a) for this pair, counted from the Γ center.\n"
        "See the colored dash-dot lines in the MDC plot."
    )
    panel.sp_gi = dspin(0.08, 0.01, 0.5, 0.01)
    panel.sp_gi.setToolTip(
        "Initial Lorentzian half-width (π/a).\n"
        "Starting value for the optimizer. See the colored curves in the MDC plot."
    )
    panel.sp_gm = dspin(0.30, 0.05, 1.0, 0.05)
    panel.sp_gm.setToolTip(
        "Maximum allowed half-width (π/a) - scipy optimizer constraint.\n"
        "See the translucent colored areas around peaks in the MDC plot."
    )
    for w in (panel.sp_kfi, panel.sp_gi, panel.sp_gm):
        w.valueChanged.connect(panel._on_pair_param_changed)
    fl.addRow("Pair count:", panel.sp_np)
    fl.addRow(panel._pair_lbl)
    fl.addRow("kF init (π/a):", panel.sp_kfi)
    fl.addRow("γ init (π/a):", panel.sp_gi)
    fl.addRow("γ max (π/a):", panel.sp_gm)
    _fcl.addWidget(grp)


def _build_constraint_section(panel, _fcl) -> None:
    grp, fl = _make_collapsible(
        panel, "Advanced - optimizer constraints", "constraints", open_default=False
    )
    panel.sp_xg = dspin(0.10, 0.0, 0.5, 0.01)
    panel.sp_xg.setToolTip(
        "Half-width of the constraint zone around the Γ center (π/a).\n"
        "The optimizer limits xg to [center - xg_range, center + xg_range].\n"
        "See the cyan rectangle in the MDC plot."
    )
    panel.sp_cx = dspin(0.0, -1.0, 1.0, 0.01)
    panel.sp_cx.setToolTip(
        "Pair symmetry center (Γ position, in π/a).\n"
        "Dashed cyan halo on the BM map in real time while editing.\n"
        "Use 'Auto Γ BM' or 'Γ FS → BM' to compute it automatically."
    )
    panel.sp_k0m = dspin(0.0, 0.0, 2.0, 0.05)
    panel.sp_k0m.setToolTip(
        "Maximum allowed kF distance from Γ (π/a).\n"
        "See the magenta lines in the MDC plot when active."
    )
    panel.chk_k0a = QCheckBox("auto")
    panel.chk_k0a.setChecked(True)
    panel.chk_k0a.setToolTip("If checked, no kF limit. Uncheck to enable kF max.")
    panel.sp_k0m.setEnabled(False)
    panel.chk_k0a.stateChanged.connect(
        lambda: panel.sp_k0m.setEnabled(not panel.chk_k0a.isChecked())
    )
    panel.cmb_wm = QComboBox()
    # Valeur = nom backend canonique. Label = explication physique.
    panel.cmb_wm.addItem("Symmetric (γL = γR)", "symmetric")
    panel.cmb_wm.addItem("Independent (γL ≠ γR)", "independent")
    panel.cmb_wm.addItem("Shared γ across pairs", "global")
    panel.cmb_wm.setFixedWidth(220)
    panel.cmb_wm.setToolTip(
        "Symmetric: γL = γR for each pair (amplitude asymmetry OK).\n"
        "Independent: γL ≠ γR per pair (width asymmetry; direction-dependent\n"
        "             mobility, mean free path).\n"
        "Shared γ: one γ for all pairs (rare; use only when the bands truly\n"
        "          have the same width)."
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
    fl.addRow("Γ window (π/a):", panel.sp_xg)
    fl.addRow("Γ center (π/a):", panel.sp_cx)
    fl.addRow("kF max (π/a):", k0w)
    fl.addRow("Pair symmetry:", panel.cmb_wm)
    panel.cmb_lineshape = QComboBox()
    panel.cmb_lineshape.addItem("Lorentzian", "lorentzian")
    panel.cmb_lineshape.addItem("Pseudo-Voigt (η_global)", "voigt")
    panel.cmb_lineshape.setFixedWidth(180)
    panel.cmb_lineshape.setToolTip(
        "Line shape used for MDC fitting.\n"
        "Lorentzian: intrinsic (lifetime; default).\n"
        "Pseudo-Voigt: (1-η)·L + η·G - instrumental resolution\n"
        "absorbed by fitted η_global ∈ [0,1]. More rigorous for correlated\n"
        "bands. Stores η per slice in fit_result.eta."
    )
    panel.cmb_lineshape.currentIndexChanged.connect(panel.fit_only_changed)
    fl.addRow("Profile:", panel.cmb_lineshape)
    _fcl.addWidget(grp)


def _build_detect_section(panel, _fcl) -> None:
    grp, fl = _make_collapsible(
        panel, "Advanced - detection / scan", "detect", open_default=False
    )
    panel.sp_sff = dspin(2.0, 0.0, 10.0, 0.5, dec=1)
    panel.sp_sff.setToolTip(
        "Gaussian smoothing sigma applied to the MDC before scipy optimization.\n"
        "Increase for noisy data. See the orange curve in the MDC plot."
    )
    panel.sp_sfd = dspin(3.0, 0.0, 10.0, 0.5, dec=1)
    panel.sp_sfd.setToolTip(
        "Gaussian smoothing sigma used to detect initial peaks.\n"
        "See the gray curve in the MDC plot."
    )
    panel.sp_ma = dspin(0.01, 0.0, 1.0, 0.01)
    panel.sp_ma.setToolTip(
        "Minimum relative peak amplitude for acceptance (0-1).\n"
        "Rejects peaks whose amplitude is < ampl_min × max(MDC)."
    )
    panel.sp_mj = dspin(0.20, 0.0, 1.0, 0.05)
    panel.sp_mj.setToolTip(
        "Maximum allowed jump between consecutive kF positions (π/a).\n"
        "Controls dispersion continuity during the full fit."
    )
    panel.sp_mdc_ewin = dspin(0.0, 0.0, 0.2, 0.005, dec=3)
    panel.sp_mdc_ewin.setToolTip(
        "Energy integration window per MDC (eV, full width). 0 = single energy "
        "row.\nAveraging ±half over E cuts noise that makes kF(E) wiggle, without "
        "biasing kF or Γ (they vary slowly with E). Keep it small vs the band "
        "dispersion."
    )
    panel.sp_chi2_threshold = dspin(5.0, 0.1, 1_000.0, 0.5, dec=1)
    panel.sp_chi2_threshold.setToolTip(
        "chi2_red threshold for marking questionable fit slices in orange.\n"
        "Only affects display when fit_result contains chi2_red."
    )
    panel.cmb_sd = QComboBox()
    panel.cmb_sd.addItems(["up", "down"])
    panel.cmb_sd.setFixedWidth(80)
    panel.cmb_sd.setToolTip(
        "up: scans the BM from ev_start (bottom) to ev_end (near EF).\n"
        "down: reverse direction."
    )
    for w in (panel.sp_sff, panel.sp_sfd, panel.sp_ma, panel.sp_mj,
              panel.sp_mdc_ewin, panel.sp_chi2_threshold):
        w.valueChanged.connect(panel.fit_only_changed)
    panel.cmb_sd.currentIndexChanged.connect(panel.fit_only_changed)
    fl.addRow("Fit smoothing σ:", panel.sp_sff)
    fl.addRow("Detect smoothing σ:", panel.sp_sfd)
    fl.addRow("Min. ampl.:", panel.sp_ma)
    fl.addRow("Max jump (π/a):", panel.sp_mj)
    fl.addRow("MDC ΔE window (eV):", panel.sp_mdc_ewin)
    fl.addRow("chi2_red threshold:", panel.sp_chi2_threshold)
    fl.addRow("Scan direction:", panel.cmb_sd)
    _fcl.addWidget(grp)


def _build_resolution_section(panel, _fcl) -> None:
    grp, fl = _make_collapsible(
        panel, "Advanced - instrumental resolution", "resolution", open_default=False
    )
    panel.sp_dE_meV = dspin(15.0, 1.0, 200.0, 1.0, dec=1)
    panel.sp_dE_meV.setToolTip(
        "Estimated or manually entered instrumental energy FWHM (meV).\n"
        "Used to compute corrected Γ after MDC fitting."
    )
    panel.sp_dk_inv_a = dspin(0.005, 0.001, 0.1, 0.001, dec=4)
    panel.sp_dk_inv_a.setToolTip(
        "Instrumental k FWHM in π/a, estimated from angle_step when available.\n"
        "Used to compute corrected Γ after MDC fitting."
    )
    panel.lbl_dE_src = QLabel("—")
    panel.lbl_dk_src = QLabel("—")
    for lbl in (panel.lbl_dE_src, panel.lbl_dk_src):
        lbl.setToolTip("Resolution provenance: Estimated, Manual, or Default")
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
    panel.sp_wf_n = ispin(24, 10, 80)
    panel.sp_wf_n.setToolTip(
        "Target number of MDCs displayed in the waterfall.\n"
        "Fewer curves = more relief and less clutter."
    )
    panel.sp_wf_relief = dspin(1.8, 0.5, 4.0, 0.1, dec=1)
    panel.sp_wf_relief.setToolTip(
        "Visual amplitude of MDCs in the waterfall.\n"
        "Increase to see peaks better; too high creates overlap."
    )
    panel.sp_wf_n.valueChanged.connect(panel.fit_only_changed)
    panel.sp_wf_relief.valueChanged.connect(panel.fit_only_changed)
    fl_wf.addRow("Curves:", panel.sp_wf_n)
    fl_wf.addRow("Relief:", panel.sp_wf_relief)
    panel._waterfall_controls_widget.setVisible(False)
    _fcl.addWidget(panel._waterfall_controls_widget)


def _build_slice_inspector_section(panel, _fcl) -> None:
    grp, fl = _make_collapsible(
        panel, "Slice inspector", "slice_inspector", open_default=False
    )
    panel.chk_fit_slice_inspector = QCheckBox("Plot overlay")
    panel.chk_fit_slice_inspector.setChecked(False)
    panel.chk_fit_slice_inspector.setToolTip(
        "Displays the parameter summary directly on the MDC plot."
    )
    panel.chk_fit_slice_inspector.stateChanged.connect(panel.fit_only_changed)
    panel.lbl_fit_slice_logic = QLabel("")
    panel.lbl_fit_slice_logic.setWordWrap(True)
    panel.lbl_fit_slice_logic.setFrameShape(QFrame.Shape.StyledPanel)
    panel.lbl_fit_slice_logic.setStyleSheet(
        "color:#d8e8ff;background:#202631;border:1px solid #3b4a60;"
        "border-radius:3px;padding:5px;font-family:monospace;font-size:10px;"
    )
    panel.lbl_fit_slice_logic.setToolTip(
        "Live summary of active parameters on the open MDC slice."
    )
    fl.addRow(panel.chk_fit_slice_inspector)
    fl.addRow(panel.lbl_fit_slice_logic)
    _fcl.addWidget(grp)


def _build_fit_buttons(panel, _fcl) -> None:
    _fcl.addWidget(hsep())
    actions_grp = QGroupBox("Fit actions")
    actions_lay_root = QVBoxLayout(actions_grp)
    actions_lay_root.setContentsMargins(6, 6, 6, 6)
    actions_lay_root.setSpacing(4)

    primary_row = QWidget()
    primary_lay = QHBoxLayout(primary_row)
    primary_lay.setContentsMargins(0, 0, 0, 0)
    primary_lay.setSpacing(4)

    btn_g = compact_button(QPushButton("Fit slice (current E)  [Ctrl+G]"), max_width=260)
    btn_g.setStyleSheet("background:#1a6b3a;color:white;font-weight:bold;padding:6px;")
    btn_g.setToolTip(
        "Fits the MDC at the current energy (selected E) with the current parameters.\n"
        "Used to calibrate initials before a full fit."
    )
    btn_g.clicked.connect(panel.guess_requested)
    primary_lay.addWidget(btn_g)

    btn_f = compact_button(QPushButton("Full fit  [Ctrl+F]"), max_width=220)
    btn_f.setStyleSheet("background:#2a6099;color:white;font-weight:bold;padding:6px;")
    btn_f.setToolTip("Runs the full MDC fit on the current analysis range.")
    btn_f.clicked.connect(panel.full_fit_requested)
    primary_lay.addWidget(btn_f)
    primary_lay.addStretch(1)
    actions_lay_root.addWidget(primary_row)

    panel._gamma_tools_widget = QWidget()
    gamma_lay = QHBoxLayout(panel._gamma_tools_widget)
    gamma_lay.setContentsMargins(0, 0, 0, 0)
    gamma_lay.setSpacing(4)
    btn_gamma = compact_button(QPushButton("Auto Γ BM"), max_width=160)
    btn_gamma.setToolTip("Estimates the Γ center from the median of MDC pair midpoints.")
    btn_gamma.clicked.connect(panel.gamma_bm_requested)
    gamma_lay.addWidget(btn_gamma)

    btn_ref = compact_button(QPushButton("Γ FS → BM"), max_width=160)
    btn_ref.setToolTip("Applies the reference Γ measured on an FS to the current BM.")
    btn_ref.clicked.connect(panel.gamma_ref_requested)
    gamma_lay.addWidget(btn_ref)
    gamma_lay.addStretch(1)
    actions_lay_root.addWidget(panel._gamma_tools_widget)

    advanced_content = QWidget()
    advanced_lay = QVBoxLayout(advanced_content)
    advanced_lay.setContentsMargins(0, 0, 0, 0)
    advanced_lay.setSpacing(4)
    advanced_actions = _ButtonCollapsible(
        "Advanced actions and display", advanced_content, open_default=False
    )

    panel.chk_smooth_kf = QCheckBox("Smooth kF(E) (σ slices)")
    panel.chk_smooth_kf.setToolTip(
        "Gaussian smoothing of kF(E) over an energy window (σ slices).\n"
        "Reduces slice-to-slice jitter without biasing the dispersion.\n"
        "Applied to display only (fit_result unchanged)."
    )
    panel.sp_smooth_kf_sigma = dspin(1.5, 0.0, 10.0, 0.5, dec=2)
    panel.sp_smooth_kf_sigma.setToolTip("σ as a number of E slices (typical 1-3).")
    _sk_row = QWidget()
    _sk_lay = QHBoxLayout(_sk_row)
    _sk_lay.setContentsMargins(0, 0, 0, 0)
    _sk_lay.addWidget(panel.chk_smooth_kf)
    _sk_lay.addWidget(panel.sp_smooth_kf_sigma)
    _sk_lay.addStretch(1)
    advanced_lay.addWidget(_sk_row)
    panel.chk_smooth_kf.stateChanged.connect(panel.fit_only_changed)
    panel.sp_smooth_kf_sigma.valueChanged.connect(panel.fit_only_changed)

    panel.chk_live_slice_fit = QCheckBox("Fit slice auto")
    panel.chk_live_slice_fit.setChecked(False)
    panel.chk_live_slice_fit.setToolTip(
        "Automatically reruns the real slice fit after edits.\n"
        "Disabled by default to keep the interface responsive; the\n"
        "Fit slice button always runs the full diagnostic on demand."
    )
    advanced_lay.addWidget(panel.chk_live_slice_fit)

    panel.chk_use_ensemble = QCheckBox("Ensemble fit (jitter initials × N)")
    panel.chk_use_ensemble.setToolTip(
        "1 pair = 1 band. Ensemble = refit N times while jittering kF_init\n"
        "and γ_init (±jitter%), then robust median (MAD-filtered).\n"
        "Gives reliable statistical σ. Slower (× N)."
    )
    panel.sp_ensemble_n = ispin(30, 5, 200)
    panel.sp_ensemble_n.setToolTip("Number of runs (5..200).")
    panel.sp_ensemble_jitter = dspin(0.10, 0.0, 0.5, 0.01, dec=3)
    panel.sp_ensemble_jitter.setToolTip(
        "Relative jitter amplitude on kF_init and γ_init (0.10 = ±10%)."
    )
    _ens_row = QWidget()
    _ens_lay = QHBoxLayout(_ens_row)
    _ens_lay.setContentsMargins(0, 0, 0, 0)
    _ens_lay.addWidget(QLabel("N:"))
    _ens_lay.addWidget(panel.sp_ensemble_n)
    _ens_lay.addWidget(QLabel("jitter:"))
    _ens_lay.addWidget(panel.sp_ensemble_jitter)
    _ens_lay.addStretch(1)
    advanced_lay.addWidget(panel.chk_use_ensemble)
    advanced_lay.addWidget(_ens_row)
    panel.btn_fit_ensemble = compact_button(QPushButton("Fit ensemble"), max_width=180)
    panel.btn_fit_ensemble.setStyleSheet(
        "background:#7c3aed;color:white;font-weight:bold;padding:6px;"
    )
    panel.btn_fit_ensemble.setToolTip(
        "Runs N full fits with jittered initials, aggregates median/MAD.\n"
        "Writes median kF/Γ + statistical σ into fit_result.ensemble."
    )
    panel.btn_fit_ensemble.clicked.connect(panel.fit_ensemble_requested)
    advanced_lay.addWidget(panel.btn_fit_ensemble)
    panel.btn_im_sigma = compact_button(QPushButton("Im Σ(E)"), max_width=140)
    panel.btn_im_sigma.setToolTip(
        "Computes Im Σ(E) = (vF/2)·Γ(E) from the corrected fit width.\n"
        "Requires a full fit and a lattice parameter a > 0."
    )
    panel.btn_im_sigma.clicked.connect(panel.im_self_energy_requested)
    advanced_lay.addWidget(panel.btn_im_sigma)
    btn_batch = compact_button(QPushButton("Batch fit folder"), max_width=200)
    btn_batch.setToolTip(
        "Runs Full fit on every folder file that does not yet have\n"
        "a fit_result. Uses the current MDC parameters.\n"
        "Cancellable progress dialog."
    )
    btn_batch.clicked.connect(panel.batch_fit_requested)
    advanced_lay.addWidget(btn_batch)

    actions_row = QWidget()
    actions_lay = QHBoxLayout(actions_row)
    actions_lay.setContentsMargins(0, 0, 0, 0)
    btn_cl = compact_button(QPushButton("Clear kF"), max_width=120)
    btn_cl.setToolTip(
        "Removes selected kF points from the MDC Fit plot.\n"
        "Reversible with 'Undo deletion' or Ctrl+Z."
    )
    btn_cl.clicked.connect(panel.clear_kf_requested)
    panel.btn_fit_undo = compact_button(QPushButton("↶ Undo deletion"), max_width=180)
    panel.btn_fit_undo.setEnabled(False)
    panel.btn_fit_undo.setToolTip(
        "Restores fit points removed by the last deletion "
        "(Delete/Backspace after selection)."
    )
    panel.btn_fit_undo.clicked.connect(panel.fit_undo_requested)
    actions_lay.addWidget(btn_cl)
    actions_lay.addWidget(panel.btn_fit_undo)
    actions_lay.addStretch(1)
    advanced_lay.addWidget(actions_row)
    actions_lay_root.addWidget(advanced_actions)
    _fcl.addWidget(actions_grp)

    panel.lbl_fit_quality = QLabel("")
    panel.lbl_fit_quality.setWordWrap(True)
    panel.lbl_fit_quality.setStyleSheet("color:#888;font-family:monospace;font-size:10px;")
    _fcl.addWidget(panel.lbl_fit_quality)

    panel.lbl_res = QLabel("")
    panel.lbl_res.setWordWrap(True)
    panel.lbl_res.setStyleSheet("color:#8fc;font-family:monospace;font-size:11px;")
    _fcl.addWidget(panel.lbl_res)
