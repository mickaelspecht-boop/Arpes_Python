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
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas, FSControlPanel
from arpes.ui.widgets.help_panel import HelpPanel
from arpes.ui.widgets.kz import KzCanvas, KzControlPanel
from arpes.ui.widgets.plots.fs_compare import FsCompareCanvas
from arpes.ui.widgets.band_analysis_panel import BandAnalysisPanel


def build_left_panel(window) -> QWidget:
    from arpes.app import FileBrowserPanel

    window._browser = FileBrowserPanel(window._session)
    return window._browser


def build_right_panel(window) -> QWidget:
    from arpes.app import FitParamsPanel

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
    window._tabs.addTab(_build_carte_tab(window), "BM")
    window._tabs.addTab(_build_mdc_tab(window), "MDC Fit")
    window._tabs.addTab(_build_results_tab(window), "Résultats")
    window._tabs.addTab(_build_fs_tab(window), "FS")
    window._tabs.addTab(_build_kz_tab(window), "KZ")
    window._tabs.addTab(_build_notes_tab(window), "Notes")
    window._tabs.addTab(_build_help_tab(window), "Aide")
    return window._tabs


def _build_carte_tab(window) -> QWidget:
    from arpes.app import MplCanvas

    carte_widget = QWidget()
    carte_lay = QVBoxLayout(carte_widget)
    carte_lay.setContentsMargins(0, 0, 0, 0)

    vbar = QHBoxLayout()
    vbar.addWidget(QLabel("Vue :"))
    window._cmb_view = _new_view_combo()
    vbar.addWidget(window._cmb_view)

    lbl_gamma = QLabel("  γ:")
    lbl_gamma.setStyleSheet("color:#aaa;font-size:11px;")
    lbl_gamma.setToolTip("Gamma de contraste : <1 booste les faibles intensités (comme dans Igor)")
    vbar.addWidget(lbl_gamma)

    window._sp_gamma = QDoubleSpinBox()
    window._sp_gamma.setRange(0.1, 3.0)
    window._sp_gamma.setSingleStep(0.1)
    window._sp_gamma.setDecimals(1)
    window._sp_gamma.setValue(1.0)
    window._sp_gamma.setFixedWidth(54)
    window._sp_gamma.setToolTip(
        "γ < 1  → accentue les structures faibles (utile pour FS)\n"
        "γ = 1  → échelle linéaire\n"
        "γ > 1  → accentue les structures fortes\n"
        "Identique à la correction gamma d'Igor BandFinder"
    )
    vbar.addWidget(window._sp_gamma)
    vbar.addStretch()

    lbl_hint = QLabel("Clic → MDC+EDC  |  ← → naviguer fichiers")
    lbl_hint.setStyleSheet("color:#888;font-size:10px;")
    vbar.addWidget(lbl_hint)
    carte_lay.addLayout(vbar)

    window._bm_canvas = MplCanvas(figsize=(7, 6), toolbar=True)
    window._bm_canvas.reset_callback = window._reset_bm_view
    carte_lay.addWidget(window._bm_canvas, stretch=1)
    return carte_widget


def _new_view_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItems(["Raw", "EDCnorm", "SecDev", "Curvature"])
    combo.setCurrentText("Raw")
    combo.setFixedWidth(120)
    combo.setToolTip(
        "Raw : intensite brute.\n"
        "EDCnorm : normalisation par EDC moyenne.\n"
        "SecDev/Curvature : derivees pour faire ressortir les dispersions."
    )
    return combo


def _build_mdc_tab(window) -> QWidget:
    from arpes.app import MplCanvas

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
    fit_bar.addWidget(QLabel("Vue :"))
    window._cmb_view_fit = _new_view_combo()
    fit_bar.addWidget(window._cmb_view_fit)
    fit_bar.addStretch()
    window._lbl_fit_view_info = QLabel("Plage d'analyse")
    window._lbl_fit_view_info.setStyleSheet("color:#aaa;font-size:10px;")
    window._lbl_fit_view_info.setToolTip(
        "Dans l'onglet Fit, la carte est zoomee sur la plage d'analyse.\n"
        "Le contraste est recalcule sur cette fenetre pour mieux voir les pics."
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
    window._mdc_fit_tabs.addTab(fit_view, "Fit MDC")

    window._waterfall_canvas = MplCanvas(figsize=(7, 5), toolbar=True)
    window._mdc_fit_tabs.addTab(window._waterfall_canvas, "Waterfall")

    window._edc_canvas = MplCanvas(figsize=(7, 5), toolbar=True)
    window._mdc_fit_tabs.addTab(window._edc_canvas, "EDC")

    window._band_panel = BandAnalysisPanel()
    window._mdc_fit_tabs.addTab(window._band_panel, "Analyse bandes")

    mdc_lay.addWidget(window._mdc_fit_tabs, stretch=1)
    return mdc_widget


def _build_results_tab(window) -> QWidget:
    from arpes.app import ResultsPanel

    window._results = ResultsPanel(window._session)
    return window._results


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
    window._fs_linked_bms = FsLinkedBmsList()
    _lay.addWidget(window._fs_linked_bms, 1)
    fs_tabs.addTab(fs_map_container, "Carte FS")
    window._fs_compare = FsCompareCanvas()
    fs_tabs.addTab(window._fs_compare, "Compare pol")
    return fs_tabs


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

    _connect_map_canvas(window._bm_canvas, window)
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

    if hasattr(window._params, "zones_strip"):
        zs = window._params.zones_strip
        zs.add_zone_requested.connect(
            lambda: (window.fit_zone_action("add", {}), window._refresh_zones_strip())
        )
        zs.remove_zone_requested.connect(
            lambda zid: (window.fit_zone_action("remove", {"zone_id": zid}),
                         window._refresh_zones_strip())
        )
        zs.active_zone_changed.connect(
            lambda zid: (window.fit_zone_action("set_active", {"zone_id": zid}),
                         window._on_zone_activated(zid))
        )
        zs.toggle_zone_active.connect(
            lambda zid, on: window.fit_zone_action(
                "toggle_active", {"zone_id": zid, "value": on},
            )
        )
        zs.run_all_zones_requested.connect(window._fit_run_all_zones)
        zs.clear_zone_results.connect(
            lambda: (window.fit_zone_action("clear_results", {}),
                     window._refresh_zones_strip())
        )

    if FSControlPanel is not None:
        window._fs_controls.params_changed.connect(window._schedule_fs_redraw)
        window._fs_controls.redraw_requested.connect(window._draw_fs_tab)
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

    if hasattr(window, "_fs_compare"):
        window._fs_compare.pair_load_requested.connect(window._on_fs_compare_pair_load)
        window._fs_compare.auto_suggest_requested.connect(window._on_fs_compare_auto_suggest)

    window._kz_controls.folder_requested.connect(window._open_kz_folder)
    if hasattr(window._kz_controls, "kz_logbook_requested"):
        window._kz_controls.kz_logbook_requested.connect(window._open_kz_logbook)
    window._kz_controls.redraw_requested.connect(window._draw_kz_tab)
    window._kz_controls.params_changed.connect(window._on_kz_params_changed)


def _connect_map_canvas(canvas_widget, window) -> None:
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
    p.theory_search_requested.connect(window._search_theory_mp)
    p.theory_band_picker_requested.connect(window._open_theory_band_picker)
    p.theory_mu_fit_requested.connect(window._fit_theory_mu_auto)
    p.theory_align_requested.connect(window._align_theory_to_arpes)
    p.theory_efalign_requested.connect(window._align_theory_efermi)
    p.crystal_a_changed.connect(window._on_crystal_a_changed)
    p.fit_section_toggled.connect(window._on_fit_section_toggled)
    p.fit_preset_changed.connect(window._on_fit_preset_changed)
    p.gamma_center_preview.connect(window._on_gamma_center_preview)
    p.batch_fit_requested.connect(window._batch_fit_folder)
    if hasattr(window, "_browser") and hasattr(window._browser, "session_reloaded"):
        window._browser.session_reloaded.connect(window._on_browser_session_reloaded)
