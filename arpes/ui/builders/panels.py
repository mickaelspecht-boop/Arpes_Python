"""Builders for the ArpesExplorer main window panels."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas, FSControlPanel
from arpes.ui.widgets.help_panel import HelpPanel
from arpes.ui.widgets.kz import KzCanvas, KzControlPanel
from arpes.ui.widgets.band_analysis_panel import BandAnalysisPanel


def build_left_panel(window) -> QWidget:
    from arpes.ui.widgets.browsers import FileBrowserPanel

    window._browser = FileBrowserPanel(window._session)
    return window._browser


def build_right_panel(window) -> QWidget:
    from arpes.ui.widgets.params import FitParamsPanel

    right_split = QSplitter(Qt.Orientation.Vertical)
    right_split.setChildrenCollapsible(False)

    window._params = FitParamsPanel()
    window._params.set_context("bm")
    right_split.addWidget(window._params)
    right_split.setSizes([550])

    if FSControlPanel is not None:
        window._fs_controls = FSControlPanel()
    else:
        window._fs_controls = QWidget()

    window._right_stack = QStackedWidget()
    window._right_stack.setMinimumWidth(320)
    window._right_stack.setSizePolicy(
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Expanding,
    )
    window._right_stack.addWidget(right_split)
    window._right_stack.addWidget(window._fs_controls)
    window._kz_controls = KzControlPanel()
    window._right_stack.addWidget(window._kz_controls)
    return window._right_stack


def build_central_widget(window, left_panel: QWidget, right_panel: QWidget) -> QWidget:
    central = QWidget()
    root = QHBoxLayout(central)
    root.setContentsMargins(4, 4, 4, 4)

    window._main_split = QSplitter(Qt.Orientation.Horizontal)
    window._main_split.setChildrenCollapsible(False)
    root.addWidget(window._main_split)

    tabs = _build_tabs(window)
    tabs.setMinimumWidth(260)
    tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

    window._main_split.addWidget(left_panel)
    window._main_split.addWidget(tabs)
    window._main_split.addWidget(right_panel)
    window._main_split.setSizes([200, 930, 500])
    window._main_split.setStretchFactor(0, 0)
    window._main_split.setStretchFactor(1, 1)
    window._main_split.setStretchFactor(2, 0)
    return central


def build_ui(window) -> None:
    """Compatibility helper kept for callers that still use the old builder."""
    from arpes.ui.builders.menus import build_menubar
    from PyQt6.QtWidgets import QStatusBar

    window.setMenuBar(build_menubar(window))
    left = build_left_panel(window)
    right = build_right_panel(window)
    window.setCentralWidget(build_central_widget(window, left, right))
    wire_ui_signals(window)
    window.setStatusBar(QStatusBar())


def _build_tabs(window) -> QTabWidget:
    window._tabs = QTabWidget()
    window._tabs.setStyleSheet(
        "QTabBar::tab{background:#333;color:#ccc;padding:5px 12px;}"
        "QTabBar::tab:selected{background:#2a6099;color:white;}"
    )
    # Order = arpes/ui/tab_index.py (single source of truth — keep in sync).
    window._tabs.addTab(_build_carte_tab(window), "BM")
    window._tabs.addTab(_build_fs_tab(window), "FS")
    window._tabs.addTab(_build_fs_explorer_tab(window), "FS Explorer")
    window._tabs.addTab(_build_kz_tab(window), "KZ")
    window._tabs.addTab(_build_results_tab(window), "Results")
    window._tabs.addTab(_build_mdc_tab(window), "MDC Fit")
    window._tabs.addTab(_build_notes_tab(window), "Notes")
    window._tabs.addTab(_build_help_tab(window), "Help")
    window._tabs.addTab(_build_quickstart_tab(window), "Start")
    return window._tabs


def _build_quickstart_tab(window) -> QWidget:
    # P4.6: page d'accueil pour un·e utilisateur·rice d'un autre labo.
    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    browser.setStyleSheet("background:#1e1e1e;color:#ddd;font-size:13px;padding:10px;")
    browser.setHtml(
        "<h2>Quick start — ARPES Explorer</h2>"
        "<p>Recommended workflow:</p>"
        "<ol>"
        "<li><b>Load</b> a file (BM or FS map) from the left panel. "
        "Enter <i>φ</i> (work function) and crystal <i>a</i> if prompted.</li>"
        "<li><b>Calibrate EF</b> (BM tab → « Auto EF calibration ») — an orange "
        "banner flags uncalibrated files.</li>"
        "<li><b>FS</b>: locate Γ, set the BZ (shape + half-BZ), correct "
        "distortion if needed.</li>"
        "<li><b>BM cuts</b>: draw the cuts, fit the MDCs "
        "(MDC Fit tab) → kF, vF, Γ_k.</li>"
        "<li><b>Pockets</b>: characterize FS pockets (area, topology, "
        "Luttinger). Fine/Standard/Stable presets depending on noise.</li>"
        "<li><b>Export</b>: publication presets (PRB / npj / Nature / "
        "Science), light background + colorbar forced, vector PDF.</li>"
        "</ol>"
        "<p style='color:#9ab'>Conventions: E−E<sub>F</sub> in eV, "
        "k in π/a, Γ<sub>k</sub> = HWHM. See the <b>Help</b> tab for loader "
        "formats and shortcuts.</p>"
    )
    return browser


def _build_carte_tab(window) -> QWidget:
    from arpes.ui.widgets.canvas import MplCanvas

    carte_widget = QWidget()
    carte_lay = QVBoxLayout(carte_widget)
    carte_lay.setContentsMargins(0, 0, 0, 0)

    vbar = QHBoxLayout()
    vbar.addWidget(QLabel("View:"))
    window._cmb_view = _new_view_combo()
    vbar.addWidget(window._cmb_view)

    lbl_gamma = QLabel("  γ:")
    lbl_gamma.setStyleSheet("color:#aaa;font-size:11px;")
    lbl_gamma.setToolTip("Contrast gamma: <1 boosts weak intensities (as in Igor)")
    vbar.addWidget(lbl_gamma)

    window._sp_gamma = QDoubleSpinBox()
    window._sp_gamma.setRange(0.1, 3.0)
    window._sp_gamma.setSingleStep(0.1)
    window._sp_gamma.setDecimals(1)
    window._sp_gamma.setValue(1.0)
    window._sp_gamma.setFixedWidth(54)
    window._sp_gamma.setToolTip(
        "γ < 1  → emphasizes weak structures (useful for FS)\n"
        "γ = 1  → linear scale\n"
        "γ > 1  → emphasizes strong structures\n"
        "Same as Igor BandFinder gamma correction"
    )
    vbar.addWidget(window._sp_gamma)
    vbar.addStretch()

    lbl_hint = QLabel("Click → MDC+EDC  |  ← → navigate files")
    lbl_hint.setStyleSheet("color:#888;font-size:10px;")
    vbar.addWidget(lbl_hint)
    carte_lay.addLayout(vbar)

    # SecDev / Curvature tuning bar — only visible for those modes. σ are in
    # physical units (converted to pixels per-map); C0 α is the curvature
    # regularization (sets the noise-vs-sharpness balance — the knob that makes
    # the mode actually useful).
    carte_lay.addWidget(_build_deriv_params_bar(window))

    # P4.6: "EF not calibrated" banner, visible until an EF calibration
    # (offset or polynomial fit) is set on the current file.
    window._lbl_ef_uncal = QLabel("⚠ EF not calibrated — « Auto EF calibration » before quantitative analysis.")
    window._lbl_ef_uncal.setStyleSheet(
        "background:#5a3a1a;color:#ffcf8f;padding:3px 8px;font-size:11px;border-radius:3px;"
    )
    window._lbl_ef_uncal.setVisible(False)
    carte_lay.addWidget(window._lbl_ef_uncal)

    window._bm_canvas = MplCanvas(figsize=(7, 6), toolbar=True)
    window._bm_canvas.reset_callback = window._reset_bm_view
    carte_lay.addWidget(window._bm_canvas, stretch=1)
    return carte_widget


def _build_deriv_params_bar(window) -> QWidget:
    """Row of SecDev/Curvature tuning spinboxes (hidden unless that mode is on)."""
    bar = QWidget()
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(2, 0, 2, 0)

    def _spin(lo, hi, step, dec, val, tip, width=64):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec)
        sp.setValue(val); sp.setFixedWidth(width); sp.setKeyboardTracking(False)
        sp.setToolTip(tip)
        return sp

    lay.addWidget(QLabel("σ_E [eV]:"))
    window._sp_deriv_sigma_e = _spin(
        0.002, 0.300, 0.005, 3, 0.025,
        "Energy smoothing before differentiation (eV).\n"
        "Larger = less noise but blurs the band. Raise this if SecDev looks like noise.")
    lay.addWidget(window._sp_deriv_sigma_e)

    lay.addWidget(QLabel("σ_k [π/a]:"))
    window._sp_deriv_sigma_k = _spin(
        0.005, 0.300, 0.005, 3, 0.040,
        "Momentum smoothing before differentiation (π/a).")
    lay.addWidget(window._sp_deriv_sigma_k)

    window._lbl_deriv_c0 = QLabel("C0 α:")
    lay.addWidget(window._lbl_deriv_c0)
    window._sp_deriv_c0 = _spin(
        0.005, 0.500, 0.01, 3, 0.050,
        "Curvature regularization (Zhang C0 as a fraction of the interior\n"
        "gradient). Small = sharper but noisier; large = smoother. Curvature only.")
    lay.addWidget(window._sp_deriv_c0)
    lay.addStretch()

    bar.setVisible(False)
    window._deriv_params_bar = bar
    return bar


def _new_view_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItems(["Raw", "EDCnorm", "SecDev", "Curvature"])
    combo.setCurrentText("Raw")
    combo.setFixedWidth(120)
    combo.setToolTip(
        "Raw: raw intensity.\n"
        "EDCnorm: normalization by mean EDC.\n"
        "SecDev/Curvature: derivatives to bring out dispersions."
    )
    return combo


def _build_mdc_tab(window) -> QWidget:
    from arpes.ui.widgets.canvas import MplCanvas

    mdc_widget = QWidget()
    mdc_lay = QVBoxLayout(mdc_widget)
    mdc_lay.setContentsMargins(0, 0, 0, 0)
    window._mdc_fit_tabs = QTabWidget()
    window._mdc_fit_tabs.setStyleSheet(
        "QTabBar::tab{background:#303030;color:#bbb;padding:4px 10px;}"
        "QTabBar::tab:selected{background:#444;color:white;}"
    )

    fit_view = QWidget()
    fit_lay = QVBoxLayout(fit_view)
    fit_lay.setContentsMargins(0, 0, 0, 0)
    fit_bar = QHBoxLayout()
    fit_bar.addWidget(QLabel("View:"))
    window._cmb_view_fit = _new_view_combo()
    fit_bar.addWidget(window._cmb_view_fit)
    fit_bar.addStretch()
    window._lbl_fit_view_info = QLabel("Analysis range")
    window._lbl_fit_view_info.setStyleSheet("color:#aaa;font-size:10px;")
    window._lbl_fit_view_info.setToolTip(
        "In the Fit tab, the map is zoomed to the analysis range.\n"
        "Contrast is recomputed on this window to better see the peaks."
    )
    fit_bar.addWidget(window._lbl_fit_view_info)
    fit_lay.addLayout(fit_bar)
    mdc_split = QSplitter(Qt.Orientation.Vertical)
    window._mdc_map_canvas = MplCanvas(figsize=(7, 5), toolbar=True)
    mdc_split.addWidget(window._mdc_map_canvas)

    window._mdc_edc = MplCanvas(figsize=(7, 2.8), nrows=1)
    mdc_split.addWidget(window._mdc_edc)
    mdc_split.setSizes([620, 280])
    fit_lay.addWidget(mdc_split, stretch=1)
    window._mdc_fit_tabs.addTab(fit_view, "MDC Fit")

    window._waterfall_canvas = MplCanvas(figsize=(7, 5), toolbar=True)
    window._mdc_fit_tabs.addTab(window._waterfall_canvas, "Waterfall")

    window._edc_canvas = MplCanvas(figsize=(7, 5), toolbar=True)
    window._mdc_fit_tabs.addTab(window._edc_canvas, "EDC")

    _moved = QLabel("Band Analysis has moved to the Results tab →")
    _moved.setAlignment(Qt.AlignmentFlag.AlignCenter)
    _moved.setStyleSheet("color:#888;")
    window._mdc_fit_tabs.addTab(_moved, "Band analysis →")

    mdc_lay.addWidget(window._mdc_fit_tabs, stretch=1)
    return mdc_widget


def _build_results_tab(window) -> QWidget:
    from arpes.ui.widgets.results import ResultsPanel

    window._results = ResultsPanel(window._session, host=window)
    # Results = the analysis hub: MDC-fit physics tables + band analysis.
    tabs = QTabWidget()
    tabs.setStyleSheet(
        "QTabBar::tab{background:#303030;color:#bbb;padding:4px 10px;}"
        "QTabBar::tab:selected{background:#444;color:white;}"
    )
    tabs.addTab(window._results, "MDC Results")
    # TB fit / kink / gap are results, not fitting controls: the panel lives
    # here (the MDC tab only shows a "moved →" pointer).
    window._band_panel = BandAnalysisPanel()
    tabs.addTab(window._band_panel, "Band Analysis")
    return tabs


def _build_fs_tab(window) -> QWidget:
    fs_tabs = QTabWidget()
    fs_tabs.setStyleSheet(
        "QTabBar::tab{background:#303030;color:#bbb;padding:4px 10px;}"
        "QTabBar::tab:selected{background:#444;color:white;}"
    )
    if FermiSurfaceCanvas is not None:
        window._fs_canvas = FermiSurfaceCanvas()
    else:
        window._fs_canvas = QWidget()
    # A.5 — wrapper canvas + liste BMs reliées (O3 minimaliste).
    from arpes.ui.widgets.fs_linked_bms import FsLinkedBmsList
    fs_map_container = QWidget()
    _lay = QVBoxLayout(fs_map_container)
    _lay.setContentsMargins(0, 0, 0, 0)
    _lay.setSpacing(2)
    _lay.addWidget(window._fs_canvas, 4)
    bm_cuts_bar = getattr(getattr(window, "_fs_controls", None), "bm_cuts_bar", None)
    if bm_cuts_bar is not None:
        _lay.addWidget(bm_cuts_bar, 0)
    window._fs_linked_bms = FsLinkedBmsList()
    _lay.addWidget(window._fs_linked_bms, 1)
    fs_tabs.addTab(fs_map_container, "FS map")
    return fs_tabs


def _build_fs_explorer_tab(window) -> QWidget:
    """ARPEST-style FS browsing: iso-E map + cut line + extracted BM."""
    from PyQt6.QtWidgets import QSplitter
    from arpes.ui.widgets.fs_explorer_panel import (
        FSExplorerControlBar,
        FSExplorerCutView,
        FSExplorerMapView,
    )
    window._fs_explorer_map = FSExplorerMapView()
    window._fs_explorer_cut = FSExplorerCutView()
    window._fs_explorer_bar = FSExplorerControlBar()
    split = QSplitter(Qt.Orientation.Horizontal)
    split.addWidget(window._fs_explorer_map)
    split.addWidget(window._fs_explorer_cut)
    split.setSizes([500, 500])
    tab = QWidget()
    lay = QVBoxLayout(tab)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.addWidget(split, 1)
    lay.addWidget(window._fs_explorer_bar, 0)
    window._fs_explorer_tab = tab
    act = window._fs_explorer_action
    window._fs_explorer_map.line_changed.connect(
        lambda cx, cy, a, ln: act("line_changed",
                                  {"cx": cx, "cy": cy, "angle": a, "length": ln}))
    window._fs_explorer_map.drag_state.connect(
        lambda on: act("drag_state", {"dragging": on}))
    window._fs_explorer_bar.energy_changed.connect(
        lambda e: act("energy_changed", {"energy": e}))
    window._fs_explorer_bar.line_params_changed.connect(
        lambda a, ln: act("line_params", {"angle": a, "length": ln}))
    window._fs_explorer_bar.mode_changed.connect(
        lambda m: act("mode_changed", {"mode": m}))
    window._fs_explorer_bar.play_toggled.connect(
        lambda on: act("play_toggle", {"play": on}))
    window._fs_explorer_bar.speed_changed.connect(
        lambda v: act("speed_changed", {"speed": v}))
    window._tabs.currentChanged.connect(
        lambda _i: act("tab_activated",
                       {"active": window._tabs.currentWidget() is tab}))
    return tab


def _build_kz_tab(window) -> QWidget:
    window._kz_canvas = KzCanvas()
    return window._kz_canvas


def _build_help_tab(window) -> QWidget:
    window._help_panel = HelpPanel()
    return window._help_panel


def _build_notes_tab(window) -> QWidget:
    from arpes.ui.widgets.notes_panel import NotesPanel
    window._notes_panel = NotesPanel(window._session)
    window._notes_panel.notes_changed.connect(window._on_session_notes_changed)
    return window._notes_panel


def wire_ui_signals(window) -> None:
    """Connect all signals for widgets created by these builders."""
    window._browser.file_selected.connect(window._load_file)
    window._tabs.currentChanged.connect(window._on_tab_changed)
    window._mdc_fit_tabs.currentChanged.connect(window._on_mdc_fit_subtab_changed)
    window._cmb_view.currentIndexChanged.connect(window._on_view_changed)
    window._cmb_view_fit.currentIndexChanged.connect(window._on_view_fit_changed)
    # B: debounce gamma BM colormap (évite rafales pendant drag spinbox)
    window._sp_gamma.valueChanged.connect(window._schedule_model_redraw)
    # SecDev/Curvature tuning → recompute the display (not just recolor).
    for _sp in (window._sp_deriv_sigma_e, window._sp_deriv_sigma_k, window._sp_deriv_c0):
        _sp.valueChanged.connect(window._on_deriv_params_changed)

    _connect_map_canvas(window._bm_canvas, window, gamma_drag=True)
    _connect_map_canvas(window._mdc_map_canvas, window)
    if FermiSurfaceCanvas is not None and hasattr(window._fs_canvas, "canvas"):
        window._fs_canvas.canvas.mpl_connect("button_press_event", window._on_fs_map_click)
        window._fs_canvas.canvas.mpl_connect("scroll_event", window._on_scroll_zoom)

    wire_param_signals(window)

    if hasattr(window, "_band_panel"):
        window._band_panel.tb_fit_requested.connect(window._run_tb_fit)
        window._band_panel.kink_run_requested.connect(window._run_kink_analysis)
        window._band_panel.gap_fit_requested.connect(window._run_gap_fit)
        if hasattr(window._band_panel, "autofill_requested"):
            window._band_panel.autofill_requested.connect(window._autofill_band_analysis)
        if hasattr(window._band_panel, "csv_export_requested"):
            window._band_panel.csv_export_requested.connect(
                window._export_band_analysis_csv
            )

    from arpes.ui.controllers.fit_zone_runner import wire_zones_strip
    wire_zones_strip(window)

    if FSControlPanel is not None:
        window._fs_controls.params_changed.connect(window._schedule_fs_redraw)
        window._fs_controls.redraw_requested.connect(window._draw_fs_tab)
        if hasattr(window._fs_controls, "nesting_requested"):
            from arpes.ui.controllers.nesting_ctrl import show_cq as _show_cq
            window._fs_controls.nesting_requested.connect(lambda: _show_cq(window))
        if hasattr(window._fs_controls, "gamma_requested"):
            window._fs_controls.gamma_requested.connect(window._detect_fs_gamma)
        if hasattr(window._fs_controls, "manual_center_requested"):
            window._fs_controls.manual_center_requested.connect(window._set_fs_center_pick_mode)
        if hasattr(window._fs_controls, "forget_gamma_requested"):
            window._fs_controls.forget_gamma_requested.connect(window._forget_gamma_with_confirm)
        if hasattr(window._fs_controls, "bm_cuts_visibility_changed"):
            window._fs_controls.bm_cuts_visibility_changed.connect(
                lambda v: window._pairing_action("toggle_cuts", {"visible": bool(v)})
            )
        if hasattr(window._fs_controls, "pockets_clear_requested"):
            window._fs_controls.pockets_clear_requested.connect(
                lambda: window._pocket_action("clear", {})
            )
        if hasattr(window._fs_controls, "pockets_export_requested"):
            window._fs_controls.pockets_export_requested.connect(
                lambda: window._pocket_action("export_csv", {})
            )
        if hasattr(window._fs_controls, "dft_grid_load_requested"):
            window._fs_controls.dft_grid_load_requested.connect(
                lambda: window._pocket_action("load_dft", {})
            )
        if hasattr(window._fs_controls, "dft_grid_clear_requested"):
            window._fs_controls.dft_grid_clear_requested.connect(
                lambda: window._pocket_action("clear_dft", {})
            )
        if hasattr(window._fs_canvas, "pocket_requested"):
            window._fs_canvas.pocket_requested.connect(
                lambda kx, ky: window._pocket_action(
                    "characterize", {"kx": float(kx), "ky": float(ky)}
                )
            )
        if hasattr(window._fs_canvas, "pocket_mdc_requested"):
            window._fs_canvas.pocket_mdc_requested.connect(
                lambda kx, ky: window._pocket_action(
                    "characterize_mdc", {"kx": float(kx), "ky": float(ky)}
                )
            )
        if (hasattr(window._fs_canvas, "_act_pocket_lasso")
                and hasattr(window._fs_controls, "btn_pocket_lasso")):
            _act = window._fs_canvas._act_pocket_lasso
            _btn = window._fs_controls.btn_pocket_lasso
            # Two-way sync; setChecked with the same value emits nothing, so
            # no signal loop.
            _btn.toggled.connect(_act.setChecked)
            _act.toggled.connect(_btn.setChecked)
        if hasattr(window._fs_canvas, "pocket_preview_level_changed"):
            def _bar_level_changed(v, _w=window):
                # Inline bar drives the same source of truth as the panel spin.
                sp = _w._fs_controls.sp_pocket_level
                sp.blockSignals(True); sp.setValue(float(v)); sp.blockSignals(False)
                _w._pocket_action("preview_update", {"level": float(v)})
            window._fs_canvas.pocket_preview_level_changed.connect(_bar_level_changed)
        if hasattr(window._fs_canvas, "pocket_lasso_requested"):
            window._fs_canvas.pocket_lasso_requested.connect(
                lambda x0, x1, y0, y1: window._pocket_action(
                    "lasso", {"kx0": float(x0), "kx1": float(x1),
                              "ky0": float(y0), "ky1": float(y1)}
                )
            )
        if hasattr(window._fs_canvas, "pocket_manual_contour_requested"):
            window._fs_canvas.pocket_manual_contour_requested.connect(
                lambda payload: window._pocket_action("manual_contour", dict(payload or {}))
            )
        if hasattr(window._fs_canvas, "pairing_diagnose_requested"):
            window._fs_canvas.pairing_diagnose_requested.connect(
                lambda: window._pairing_action("diagnose", {})
            )
        if hasattr(window._fs_canvas, "pocket_preview_requested"):
            window._fs_canvas.pocket_preview_requested.connect(
                lambda kx, ky: window._pocket_action(
                    "preview_start", {"kx": float(kx), "ky": float(ky)}
                )
            )
        if hasattr(window._fs_canvas, "pocket_preview_validate_requested"):
            window._fs_canvas.pocket_preview_validate_requested.connect(
                lambda: window._pocket_action("preview_validate", {})
            )
        if hasattr(window._fs_canvas, "pocket_preview_cancel_requested"):
            window._fs_canvas.pocket_preview_cancel_requested.connect(
                lambda: window._pocket_action("preview_cancel", {})
            )
        if hasattr(window._fs_controls, "pocket_preview_level_changed"):
            window._fs_controls.pocket_preview_level_changed.connect(
                lambda lvl: window._pocket_action("preview_update", {"level": float(lvl)})
            )
        if hasattr(window._fs_canvas, "pockets_clear_requested"):
            window._fs_canvas.pockets_clear_requested.connect(
                lambda: window._pocket_action("clear", {})
            )
        if hasattr(window._fs_canvas, "pockets_export_requested"):
            window._fs_canvas.pockets_export_requested.connect(
                lambda: window._pocket_action("export_csv", {})
            )
        if hasattr(window._fs_canvas, "pocket_open_requested"):
            window._fs_canvas.pocket_open_requested.connect(
                lambda idx: window._pocket_action("show", {"index": int(idx)})
            )
    if hasattr(window, "_fs_linked_bms"):
        window._fs_linked_bms.bm_load_requested.connect(
            lambda path: window._load_ctrl.load(_resolve_session_path(window, path))
        )
        if hasattr(window._fs_controls, "bz_preset_requested"):
            window._fs_controls.bz_preset_requested.connect(window._choose_bz_preset)
        if hasattr(window._fs_controls, "bz_labels_requested"):
            window._fs_controls.bz_labels_requested.connect(window._edit_bz_labels)
        if hasattr(window._fs_controls, "bz_crystal_overlay_changed"):
            window._fs_controls.bz_crystal_overlay_changed.connect(
                window._on_bz_crystal_overlay_changed
            )
        if hasattr(window._fs_controls, "mp_lattice_fetch_requested"):
            window._fs_controls.mp_lattice_fetch_requested.connect(
                window._on_mp_lattice_fetch
            )
        if hasattr(window._fs_controls, "distortion_fs_toggled"):
            window._fs_controls.distortion_fs_toggled.connect(
                window._on_propagate_distortion_fs_toggled
            )

    window._kz_controls.folder_requested.connect(window._open_kz_folder)
    if hasattr(window._kz_controls, "kz_logbook_requested"):
        window._kz_controls.kz_logbook_requested.connect(window._open_kz_logbook)
    window._kz_controls.redraw_requested.connect(window._draw_kz_tab)
    if hasattr(window._kz_controls, "fit_v0_requested"):
        window._kz_controls.fit_v0_requested.connect(window._fit_kz_v0)
    window._kz_controls.params_changed.connect(window._on_kz_params_changed)


def _connect_map_canvas(canvas_widget, window, *, gamma_drag: bool = False) -> None:
    if gamma_drag:
        # Connect first so the press handler sets _gamma_drag_active before the
        # map-click / fit-select handlers run and they can bail on it.
        from arpes.ui.controllers.gamma_drag_handlers import wire_gamma_drag
        wire_gamma_drag(canvas_widget, window)
    canvas_widget.canvas.mpl_connect("button_press_event", window._on_map_click)
    canvas_widget.canvas.mpl_connect("button_press_event", window._on_fit_annotate_press)
    canvas_widget.canvas.mpl_connect("button_press_event", window._on_fit_roi_press)
    canvas_widget.canvas.mpl_connect("motion_notify_event", window._on_fit_roi_motion)
    canvas_widget.canvas.mpl_connect("button_release_event", window._on_fit_roi_release)
    canvas_widget.canvas.mpl_connect("button_press_event", window._on_fit_select_press)
    canvas_widget.canvas.mpl_connect("motion_notify_event", window._on_fit_select_motion)
    canvas_widget.canvas.mpl_connect("motion_notify_event", window._on_fit_annotation_motion)
    canvas_widget.canvas.mpl_connect("button_release_event", window._on_fit_select_release)
    canvas_widget.canvas.mpl_connect("scroll_event", window._on_scroll_zoom)


def _resolve_session_path(window, path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    folder = getattr(getattr(window, "_session", None), "folder", None)
    if folder:
        return str(Path(folder) / p)
    return str(p)


def wire_param_signals(window) -> None:
    p = window._params
    p.params_changed.connect(window._schedule_model_redraw)
    p.fit_only_changed.connect(window._schedule_fit_only_redraw)
    # B: live preview du fit (debounce) sur changements de params init
    p.fit_only_changed.connect(window._schedule_live_guess)
    p.sp_ev.valueChanged.connect(window._on_ev_spinbox_changed)
    p.sp_ev.valueChanged.connect(window._schedule_live_guess)
    p.guess_requested.connect(window._fit_guess)
    p.full_fit_requested.connect(window._fit_full)
    p.clear_kf_requested.connect(window._clear_kf)
    p.copy_params_requested.connect(window._copy_params)
    p.ef_calib_requested.connect(window._ef_calibrate)
    p.ef_apply_reference_requested.connect(window._apply_ef_reference_to_current)
    # Live EF offset: editing the spinbox rigidly shifts the binding-energy axis
    # (no reload — the offset is a pure axis shift). Debounced redraw/save.
    from arpes.ui.controllers.ef_offset_live import apply_live_ef_offset
    p.sp_ef.valueChanged.connect(lambda v: apply_live_ef_offset(window, v))
    p.logbook_requested.connect(window._logbook_ctrl.open_dialog)
    p.gamma_bm_requested.connect(window._estimate_gamma_bm)
    p.gamma_ref_requested.connect(window._apply_gamma_reference_to_bm)
    p.grid_requested.connect(window._apply_grid_correction)
    p.grid_reset_requested.connect(window._reset_grid_correction)
    p.distortion_apply_requested.connect(window._apply_bm_distortion)
    p.distortion_reset_requested.connect(window._reset_bm_distortion)
    p.distortion_auto_requested.connect(window._auto_bm_distortion)
    p.distortion_import_calib_requested.connect(window._import_calib_to_current)
    p.distortion_preview_changed.connect(window._on_distortion_preview_changed)
    if hasattr(p, "propagate_distortion_fs_toggled"):
        p.propagate_distortion_fs_toggled.connect(
            window._on_propagate_distortion_fs_toggled
        )
    p.fit_roi_requested.connect(window._set_fit_roi_pick_mode)
    p.fit_roi_reset_requested.connect(window._reset_fit_roi_range)
    p.fit_undo_requested.connect(window._undo_fit_delete)
    p.kf_init_drag_changed.connect(window._on_kf_init_drag)
    p.im_self_energy_requested.connect(window._calculate_im_self_energy)
    p.fit_ensemble_requested.connect(window._fit_ensemble)
    p.file_tags_changed.connect(window._on_file_tags_changed)
    # THEORY_OVERLAY: optional/removable DFT guide wiring.
    p.theory_import_requested.connect(window._import_theory_overlay)
    p.theory_refresh_requested.connect(window._refresh_theory_overlay)
    p.theory_local_import_requested.connect(window._import_local_theory_overlay)
    p.theory_clear_requested.connect(window._clear_theory_overlay)
    p.theory_overlay_changed.connect(window._on_theory_overlay_changed)
    p.theory_compare_requested.connect(window._compare_theory_overlay)
    p.self_energy_requested.connect(window._calculate_self_energy)
    from arpes.ui.controllers.polarization_ctrl import show_pkE as _show_pkE
    p.pkE_requested.connect(lambda: _show_pkE(window))
    p.theory_search_requested.connect(window._search_theory_mp)
    p.theory_band_picker_requested.connect(window._open_theory_band_picker)
    p.theory_mu_fit_requested.connect(window._fit_theory_mu_auto)
    p.theory_align_requested.connect(window._align_theory_to_arpes)
    p.theory_efalign_requested.connect(window._align_theory_efermi)
    # Manual DFT calibration from user-placed high-symmetry points (closures →
    # no PROXY_MAP entry; BM click handler wired inside).
    from arpes.ui.controllers.theory_anchor_ctrl import wire as _wire_theory_anchor
    _wire_theory_anchor(window)
    p.work_function_changed.connect(window._on_work_function_changed)
    p.crystal_a_changed.connect(window._on_crystal_a_changed)
    p.fit_section_toggled.connect(window._on_fit_section_toggled)
    p.fit_preset_changed.connect(window._on_fit_preset_changed)
    p.gamma_center_preview.connect(window._on_gamma_center_preview)
    p.batch_fit_requested.connect(window._batch_fit_folder)
    if hasattr(window, "_browser") and hasattr(window._browser, "session_reloaded"):
        window._browser.session_reloaded.connect(window._on_browser_session_reloaded)
    if hasattr(window, "_browser") and hasattr(window._browser, "folder_opened"):
        window._browser.folder_opened.connect(
            lambda: window._sample_setup_action("folder_opened"))
        window._browser.sample_setup_requested.connect(
            lambda: window._sample_setup_action("open_dialog"))
