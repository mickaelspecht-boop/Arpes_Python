"""Builders pour la fenêtre principale ArpesExplorer.

Sort `ArpesExplorer._build_ui` (~174 LOC) de la God class. Découpé en
sous-builders par zone d'écran + un `wire_param_signals` séparé pour les
connexions.

Tous les widgets sont attachés à `window` (instance ArpesExplorer) via
`window.<attr>` ; on évite la pollution de la God class par des helpers
inline. Les classes de widgets (`MplCanvas`, `FileBrowserPanel`,
`FitParamsPanel`, `ResultsPanel`) restent dans `arpes_explorer` jusqu'à η —
import paresseux pour éviter les cycles.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from arpes.physics.fs import FermiSurfaceCanvas, FSControlPanel


def build_ui(window) -> None:
    """Orchestrateur — assemble central widget + splitters + tabs + droite."""
    central = QWidget()
    window.setCentralWidget(central)
    root = QHBoxLayout(central)
    root.setContentsMargins(4, 4, 4, 4)

    main_split = QSplitter(Qt.Orientation.Horizontal)
    root.addWidget(main_split)

    _build_browser(window, main_split)
    _build_tabs(window, main_split)
    _build_right_panel(window, main_split)

    main_split.setSizes([210, 850, 440])
    window.setStatusBar(QStatusBar())


def _build_browser(window, main_split: QSplitter) -> None:
    from arpes_explorer import FileBrowserPanel
    window._browser = FileBrowserPanel(window._session)
    window._browser.file_selected.connect(window._load_file)
    main_split.addWidget(window._browser)


def _build_tabs(window, main_split: QSplitter) -> None:
    window._tabs = QTabWidget()
    window._tabs.setStyleSheet(
        "QTabBar::tab{background:#333;color:#ccc;padding:5px 12px;}"
        "QTabBar::tab:selected{background:#2a6099;color:white;}"
    )
    window._tabs.addTab(_build_carte_tab(window), "🗺  BM")
    window._tabs.addTab(_build_mdc_tab(window), "🎯  MDC Fit")
    window._tabs.addTab(_build_results_tab(window), "📊  Résultats")
    window._tabs.addTab(_build_fs_tab(window), "🧭  FS")
    window._tabs.currentChanged.connect(window._on_tab_changed)
    main_split.addWidget(window._tabs)


def _build_carte_tab(window) -> QWidget:
    from arpes_explorer import MplCanvas
    carte_widget = QWidget()
    carte_lay = QVBoxLayout(carte_widget)
    carte_lay.setContentsMargins(0, 0, 0, 0)

    vbar = QHBoxLayout()
    vbar.addWidget(QLabel("Vue :"))
    window._cmb_view = QComboBox()
    window._cmb_view.addItems(["Raw", "EDCnorm", "SecDev", "Curvature"])
    window._cmb_view.setCurrentText("EDCnorm")
    window._cmb_view.setFixedWidth(120)
    window._cmb_view.currentIndexChanged.connect(window._on_view_changed)
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
    window._sp_gamma.valueChanged.connect(window._draw_bm)
    vbar.addWidget(window._sp_gamma)
    vbar.addStretch()
    lbl_hint = QLabel("Clic → MDC+EDC  |  ← → naviguer fichiers")
    lbl_hint.setStyleSheet("color:#888;font-size:10px;")
    vbar.addWidget(lbl_hint)
    carte_lay.addLayout(vbar)

    window._bm_canvas = MplCanvas(figsize=(7, 6), toolbar=True)
    window._bm_canvas.canvas.mpl_connect("button_press_event", window._on_map_click)
    window._bm_canvas.canvas.mpl_connect("button_press_event", window._on_fit_roi_press)
    window._bm_canvas.canvas.mpl_connect("motion_notify_event", window._on_fit_roi_motion)
    window._bm_canvas.canvas.mpl_connect("button_release_event", window._on_fit_roi_release)
    carte_lay.addWidget(window._bm_canvas, stretch=1)
    return carte_widget


def _build_mdc_tab(window) -> QWidget:
    from arpes_explorer import MplCanvas
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
    mdc_split = QSplitter(Qt.Orientation.Vertical)
    window._mdc_map_canvas = MplCanvas(figsize=(7, 5), toolbar=True)
    window._mdc_map_canvas.canvas.mpl_connect("button_press_event", window._on_map_click)
    window._mdc_map_canvas.canvas.mpl_connect("button_press_event", window._on_fit_roi_press)
    window._mdc_map_canvas.canvas.mpl_connect("motion_notify_event", window._on_fit_roi_motion)
    window._mdc_map_canvas.canvas.mpl_connect("button_release_event", window._on_fit_roi_release)
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

    window._mdc_fit_tabs.currentChanged.connect(window._on_mdc_fit_subtab_changed)
    mdc_lay.addWidget(window._mdc_fit_tabs, stretch=1)
    return mdc_widget


def _build_results_tab(window) -> QWidget:
    from arpes_explorer import ResultsPanel
    window._results = ResultsPanel(window._session)
    return window._results


def _build_fs_tab(window) -> QWidget:
    if FermiSurfaceCanvas is not None:
        window._fs_canvas = FermiSurfaceCanvas()
        if hasattr(window._fs_canvas, "canvas"):
            window._fs_canvas.canvas.mpl_connect("button_press_event", window._on_fs_map_click)
    else:
        window._fs_canvas = QWidget()
    return window._fs_canvas


def _build_right_panel(window, main_split: QSplitter) -> None:
    from arpes_explorer import FitParamsPanel
    right_split = QSplitter(Qt.Orientation.Vertical)

    window._params = FitParamsPanel()
    wire_param_signals(window)
    window._params.set_context("bm")
    right_split.addWidget(window._params)
    right_split.setSizes([550])

    if FSControlPanel is not None:
        window._fs_controls = FSControlPanel()
        window._fs_controls.params_changed.connect(window._on_fs_params_changed)
        window._fs_controls.redraw_requested.connect(window._draw_fs_tab)
        if hasattr(window._fs_controls, "gamma_requested"):
            window._fs_controls.gamma_requested.connect(window._detect_fs_gamma)
        if hasattr(window._fs_controls, "manual_center_requested"):
            window._fs_controls.manual_center_requested.connect(window._set_fs_center_pick_mode)
    else:
        window._fs_controls = QWidget()

    window._right_stack = QStackedWidget()
    window._right_stack.addWidget(right_split)
    window._right_stack.addWidget(window._fs_controls)
    main_split.addWidget(window._right_stack)


def wire_param_signals(window) -> None:
    """Connecte tous les signaux de FitParamsPanel aux slots ArpesExplorer."""
    p = window._params
    p.params_changed.connect(window._schedule_model_redraw)
    p.fit_only_changed.connect(window._schedule_fit_only_redraw)
    p.sp_ev.valueChanged.connect(window._on_ev_spinbox_changed)
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
    p.fit_roi_requested.connect(window._set_fit_roi_pick_mode)
    p.fit_roi_reset_requested.connect(window._reset_fit_roi_range)
