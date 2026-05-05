#!/usr/bin/env python3
"""
arpes_explorer.py — Interface interactive ARPES (BaNi₂As₂) v3
═══════════════════════════════════════════════════════════════
Features :
  • Panneau fichiers (browse dossier, statut ○/◑/●)
  • Session JSON (.arpes_session.json) — sauvegarde auto à chaque fit
  • Band map avec modes Raw / EDCnorm / SecDev / Curvature
  • MDC (en énergie) + EDC (en k) live sur clic
  • Modèle Lorentzien par paire, temps réel
  • Bouton Guess (fit MDC à l'énergie courante)
  • Bouton Fit complet → kF superposés sur la carte
  • Calibration EF sample-based intégrée
  • Onglet Résultats : dispersions superposées + table + export CSV/PDF

Lancement :
    /Users/alexandrespecht/.local/share/mamba/envs/peaks/bin/python3 arpes_explorer.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
import traceback
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from arpes.physics.cls_geometry import (
    geometry_for_path as _cls_geometry_for_path_pure,
    manipulator_from_param as _cls_manipulator_from_param_pure,
)
from arpes.io.export import result_rows, write_results_csv
from arpes.physics.ef_calibration import (
    ReferenceError as EFReferenceError,
    already_applied as ef_reference_already_applied,
    apply_reference_to_target as apply_ef_reference_to_target,
    compute_calibration_update as compute_ef_calibration_update,
)
from arpes.physics.fit import FitController
from arpes.physics.display import apply_edcnorm
from arpes.physics.gamma import (
    angle_offset_candidates_for_load as _gamma_angle_offset_candidates,
    score_bm_gamma_residual as _gamma_score_bm_residual,
)
from arpes.io.logbook import (
    _cell_float,
    _cell_text,
    _format_direction_label,
    _record_matches_path,
)
from arpes.ui.controllers.logbook_controller import LogbookIngestController
from arpes.ui.controllers.load_controller import LoadController
from arpes.ui.controllers.plot_controller import PlotController
from arpes.ui.controllers.gamma_controller import GammaController
from arpes.ui.controllers.norm_controller import NormController
from arpes.ui.controllers.fs_controller import FSController
from arpes.core.session import FileEntry, FitParams, Session

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QFileDialog, QScrollArea, QGroupBox,
    QSizePolicy, QFrame, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QDialog,
)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement arpes_plots
# ─────────────────────────────────────────────────────────────────────────────

def _load_ap():
    try:
        import arpes.ui.widgets.plots as plots
        return plots
    except Exception:
        pass
    code_dir = Path(__file__).resolve().parent
    for name in ["arpes_plots.py", "arpes_plots(1).py"]:
        p = code_dir / name
        if p.exists():
            spec = importlib.util.spec_from_file_location("arpes_plots", p)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("arpes_plots.py introuvable")

try:
    from arpes.io.loaders import (
        load_arpes,
        load_arpes_file,
        loader_label as _loader_label,
        detect_format,
        detect_scan_kind,
        ARPESData,
    )
    from arpes.physics.fs import FermiSurfaceCanvas, FSControlPanel
    ERLAB_OK = True
except Exception:
    load_arpes = None
    load_arpes_file = None
    _loader_label = lambda *a, **k: ""  # noqa: E731
    detect_format = None
    detect_scan_kind = None
    ARPESData = None
    FermiSurfaceCanvas = None
    FSControlPanel = None
    ERLAB_OK = False

from arpes.physics.display import apply_ef_correction_to_dict
from arpes.ui.widgets.canvas import MplCanvas
from arpes.ui.widgets.browsers import FileBrowserPanel
from arpes.ui.widgets.params import ClickablePairLabel, FitParamsPanel, PAIR_COLORS
from arpes.ui.widgets.results import ResultsPanel
from arpes.ui.widgets.dialogs import EFCalibrationDialog

AP = None



# ─────────────────────────────────────────────────────────────────────────────
# Fenêtre principale
# ─────────────────────────────────────────────────────────────────────────────

class ArpesExplorer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARPES Explorer — BaNi₂As₂")
        self.resize(1500, 900)

        self._session     = Session()
        self._current_path: str | None = None
        self._raw_data:   dict | None  = None   # chargé depuis fichier
        self._data_disp:  np.ndarray | None = None  # données affichées (mode)
        self._grid_display_info: dict = {}
        self._fit_res:    dict | None  = None

        self._sel_ev = -0.30
        self._sel_k  = 0.0
        self._fs_pick_center_active = False
        self._fit_roi_active = False
        self._fit_roi_start: tuple[float, float] | None = None
        self._fit_roi_ax = None
        self._fit_roi_rect = None

        # Debouncers : évitent N redraws quand l'utilisateur clique-clique
        # rapidement sur un spinbox ou tape une valeur.
        self._redraw_timer = QTimer(self); self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._on_model_changed)
        self._fit_redraw_timer = QTimer(self); self._fit_redraw_timer.setSingleShot(True)
        self._fit_redraw_timer.timeout.connect(self._on_fit_only_changed)

        # Cache de _update_display_data : recompute uniquement si une des clés
        # influence le résultat affiché.
        self._disp_cache_key: tuple | None = None

        self._logbook_ctrl = LogbookIngestController(self)
        self._load_ctrl = LoadController(self)
        self._plot_ctrl = PlotController(self)
        self._gamma_ctrl = GammaController(self)
        self._norm_ctrl = NormController(self)
        self._fs_ctrl = FSController(self)
        self._build_ui()
        self._install_shortcuts()
        self._status("Prêt — ouvrir un dossier ou un fichier")

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        from PyQt6.QtWidgets import QStatusBar
        from arpes.ui.builders.menus import build_menubar
        from arpes.ui.builders.panels import (
            build_central_widget,
            build_left_panel,
            build_right_panel,
        )

        self.setMenuBar(build_menubar(self))
        self._left = build_left_panel(self)
        self._right = build_right_panel(self)
        central = build_central_widget(self, self._left, self._right)
        self.setCentralWidget(central)
        self._wire_signals()
        self.setStatusBar(QStatusBar())

    def _wire_signals(self):
        from arpes.ui.builders.panels import wire_ui_signals
        wire_ui_signals(self)

    def _on_tab_changed(self, index: int):
        # 0=BM, 1=MDC Fit, 2=Résultats, 3=FS
        if hasattr(self, "_right_stack"):
            self._right_stack.setCurrentIndex(1 if index == 3 else 0)
        if index == 0:
            self._params.set_context("bm")
        elif index == 1:
            self._params.set_context("mdc")
            if hasattr(self, "_mdc_fit_tabs"):
                self._params.set_waterfall_controls_visible(self._mdc_fit_tabs.currentIndex() == 1)
        else:
            self._params.set_context("other")
            self._set_fit_roi_pick_mode(False)
        if index == 2:
            self._set_fs_center_pick_mode(False)
            self._results.refresh()
        elif index == 3:
            self._draw_fs_tab()
        elif index == 1:
            self._set_fs_center_pick_mode(False)
            self._draw_bm()
            self._draw_mdc_edc()
        else:
            self._set_fs_center_pick_mode(False)

    def _on_mdc_fit_subtab_changed(self, index: int):
        if hasattr(self, "_params"):
            self._params.set_waterfall_controls_visible(index == 1 and self._tabs.currentIndex() == 1)
        if index == 0:
            self._draw_bm()
            self._draw_mdc_edc()
        elif index == 1:
            self._draw_mdc_waterfall()
        elif index == 2:
            self._draw_mdc_edc()

    def _current_entry(self) -> FileEntry | None:
        if not self._current_path:
            return None
        return self._session.get_or_create(self._session.key_for_path(self._current_path))

    def _current_is_fs(self) -> bool:
        return self._fs_ctrl._current_is_fs()
    def _on_fs_params_changed(self):
        return self._fs_ctrl._on_fs_params_changed()
    def _save_current_fs_center(self):
        return self._fs_ctrl._save_current_fs_center()
    def _same_path(self, a, b) -> bool:
        if not a or not b:
            return False
        try:
            return Path(a).resolve() == Path(b).resolve()
        except Exception:
            return str(a) == str(b)

    def _on_scroll_zoom(self, event):
        return self._plot_ctrl._on_scroll_zoom(event)
    def _draw_fs_tab(self):
        return self._fs_ctrl._draw_fs_tab()
    def _store_fs_center_reference(self, kx: float, ky: float, *, source: str):
        return self._gamma_ctrl._store_fs_center_reference(kx, ky, source=source)
    def _k_to_angle_offset_deg(self, k_pi_a: float, *, hv: float | None = None) -> float | None:
        return self._gamma_ctrl._k_to_angle_offset_deg(k_pi_a, hv=hv)
    def _angle_offsets_from_k_center(
        self,
        kx: float,
        ky: float = 0.0,
        *,
        hv: float | None = None,
        source: str = "",
        ref_path: str | None = None,
        azi: float | None = None,
    ) -> dict:
        return self._gamma_ctrl._angle_offsets_from_k_center(
            kx, ky, hv=hv, source=source, ref_path=ref_path, azi=azi
        )

    def _project_gamma_by_azi(
        self,
        ref: dict,
        azi_target: float | None,
        *,
        warn_label: str = "Γ",
    ) -> tuple[float, float]:
        return self._gamma_ctrl._project_gamma_by_azi(
            ref, azi_target, warn_label=warn_label
        )
    def _cls_manipulator_from_param(self, path: str | Path) -> dict:
        """Wrapper UI : délègue à `arpes_cls_geometry.manipulator_from_param`."""
        return _cls_manipulator_from_param_pure(path)

    def _cls_geometry_for_path(self, path: str | Path, entry: FileEntry | None = None) -> dict:
        """Wrapper UI : délègue à `arpes_cls_geometry.geometry_for_path`."""
        return _cls_geometry_for_path_pure(
            path,
            entry_meta=(entry.meta if entry is not None else None),
            logbook_record=self._logbook_ctrl.find_record_for_path(path),
            logbook_mapping=self._session.logbook_mapping,
            cell_float=_cell_float,
        )

    def _angle_offsets_for_load(self, path: str | Path, entry: FileEntry | None, hv: float | None) -> dict:
        """Retourne les offsets angulaires a injecter dans le loader CLS."""
        ref = self._stored_gamma_reference()
        if not ref:
            return self._session.angle_offsets or {}

        # Pour les BM, on projette le Γ FS dans la direction de fente du fichier
        # courant avec azi_ref/azi_bm, puis on convertit ce k en theta0.
        p = Path(path)
        is_cls_bm_file = p.is_file() and (p.parent / f"{p.name}_param.txt").exists()
        is_cls_fs_dir = p.is_dir()
        geom = self._cls_geometry_for_path(p, entry)
        if is_cls_bm_file:
            azi_bm = geom.get("azi", entry.meta.azi if (entry and entry.meta.azi is not None) else None)
            gamma_bm, _ = self._project_gamma_by_azi(
                ref, azi_bm, warn_label="Γ référence → BM"
            )
            if not np.isfinite(gamma_bm):
                return {}
            offsets = self._angle_offsets_from_k_center(
                float(gamma_bm), 0.0,
                hv=hv,
                source="gamma_reference_projected_to_bm",
                ref_path=ref.get("path"),
                azi=azi_bm,
            )
            if offsets:
                offsets["gamma_bm_pi_over_a"] = float(gamma_bm)
                offsets["gamma_ref_source"] = ref.get("source", "")
                offsets["target_polar"] = geom.get("polar")
                offsets["target_tilt"] = geom.get("tilt")
                return offsets

        # Pour une autre FS CLS, on recentre les deux axes via la même rotation
        # azimutale. Le dessin affichera ensuite le centre à (0, 0), car le
        # loader aura déjà appliqué theta0/tilt0.
        if is_cls_fs_dir:
            azi_fs = geom.get("azi", entry.meta.azi if (entry and entry.meta.azi is not None) else None)
            gamma_kx, gamma_ky = self._project_gamma_by_azi(
                ref, azi_fs, warn_label="Γ référence → FS"
            )
            if not np.isfinite(gamma_kx) or not np.isfinite(gamma_ky):
                return {}
            offsets = self._angle_offsets_from_k_center(
                float(gamma_kx), float(gamma_ky),
                hv=hv,
                source="gamma_reference_projected_to_fs",
                ref_path=ref.get("path"),
                azi=azi_fs,
            )
            if offsets:
                offsets["gamma_fs_kx_pi_over_a"] = float(gamma_kx)
                offsets["gamma_fs_ky_pi_over_a"] = float(gamma_ky)
                offsets["gamma_ref_source"] = ref.get("source", "")
                offsets["target_polar"] = geom.get("polar")
                offsets["target_tilt"] = geom.get("tilt")
                return offsets

        return {}

    def _angle_offset_candidates_for_load(
        self,
        path: str | Path,
        entry: FileEntry | None,
        hv: float | None,
        primary: dict,
    ) -> list[dict]:
        """Wrapper UI : délègue à `arpes_gamma.angle_offset_candidates_for_load`."""
        target_geom = (
            self._cls_geometry_for_path(path, entry)
            if (entry is not None and Path(path).is_file()) else None
        )
        target_azi_fallback = (
            entry.meta.azi if (entry is not None and entry.meta.azi is not None) else None
        )
        return _gamma_angle_offset_candidates(
            primary=primary,
            is_file=Path(path).is_file(),
            ref=self._stored_gamma_reference() or None,
            target_geom=target_geom,
            target_azi_fallback=target_azi_fallback,
            hv=hv,
            work_func=float(self._params.sp_phi.value()),
        )

    def _score_bm_gamma_residual(self, d: dict) -> float:
        """Wrapper UI : délègue à `arpes_gamma.score_bm_gamma_residual`."""
        if AP is None:
            return float("inf")
        return _gamma_score_bm_residual(
            d,
            ev_range=(self._params.sp_evs.value(), self._params.sp_eve.value()),
            k_range=(self._params.sp_kmin.value(), self._params.sp_kmax.value()),
            center_window=self._params.sp_xg.value() * 2.0,
            smooth_sigma=self._params.sp_sfd.value(),
            estimate_fn=AP.estimate_gamma_bm_mdc,
        )

    def _load_with_best_angle_offsets(
        self,
        path: str,
        entry: FileEntry,
        hv_for_load: float,
        angle_offsets: dict,
    ) -> tuple[dict | None, dict]:
        """Charge une BM CLS avec la convention d'offset qui centre le mieux Γ."""
        candidates = self._angle_offset_candidates_for_load(path, entry, hv_for_load, angle_offsets)
        if len(candidates) <= 1:
            d = load_arpes_file(
                path, self._params.sp_phi.value(), self._params.sp_ef.value(),
                hv=hv_for_load,
                temperature=entry.meta.temperature if entry.meta.temperature > 0 else None,
                azi=entry.meta.azi,
                pol=entry.meta.polarization,
                angle_offsets=angle_offsets,
                bessy_energy_reference=self._bessy_energy_reference_mode(),
            )
            return d, angle_offsets

        best_d = None
        best_cfg = candidates[0]
        best_score = float("inf")
        for cfg in candidates:
            d_try = load_arpes_file(
                path, self._params.sp_phi.value(), self._params.sp_ef.value(),
                hv=hv_for_load,
                temperature=entry.meta.temperature if entry.meta.temperature > 0 else None,
                azi=entry.meta.azi,
                pol=entry.meta.polarization,
                angle_offsets=cfg,
                bessy_energy_reference=self._bessy_energy_reference_mode(),
            )
            if d_try is None:
                continue
            score = self._score_bm_gamma_residual(d_try)
            if score < best_score:
                best_score = score
                best_d = d_try
                best_cfg = cfg

        if best_d is not None and np.isfinite(best_score):
            try:
                md = best_d.get("metadata", {}) or {}
                md["angle_offset_candidate_score"] = float(best_score)
                md["angle_offset_candidate"] = best_cfg.get("candidate", "")
                best_d["metadata"] = md
            except Exception:
                pass
            return best_d, best_cfg

        d = load_arpes_file(
            path, self._params.sp_phi.value(), self._params.sp_ef.value(),
            hv=hv_for_load,
            temperature=entry.meta.temperature if entry.meta.temperature > 0 else None,
            azi=entry.meta.azi,
            pol=entry.meta.polarization,
            angle_offsets=angle_offsets,
            bessy_energy_reference=self._bessy_energy_reference_mode(),
        )
        return d, angle_offsets

    def _bessy_energy_reference_mode(self) -> str:
        """Mode BESSY exposé à l'app principale.

        L'UI quotidienne reste simple : BESSY utilise le mode auto, qui se
        résout côté loader en `ses_center_energy`. Le mode hν−φ reste disponible
        dans `arpes_io.load_bessy_ses_ibw(...)` pour tests/diagnostic explicites.
        """
        return "auto"

    def _set_fs_center_pick_mode(self, active: bool):
        return self._gamma_ctrl._set_fs_center_pick_mode(active)
    def _on_fs_map_click(self, event):
        return self._gamma_ctrl._on_fs_map_click(event)
    def _detect_fs_gamma(self):
        return self._gamma_ctrl._detect_fs_gamma()
    def _stored_gamma_reference(self) -> dict:
        return self._gamma_ctrl._stored_gamma_reference()
    def _gamma_reference_to_bm_center(self, ref: dict) -> tuple[float, float]:
        return self._gamma_ctrl._gamma_reference_to_bm_center(ref)
    def _center_current_bm_axis_on_gamma(self, gamma_bm: float, ref: dict | None = None) -> bool:
        return self._gamma_ctrl._center_current_bm_axis_on_gamma(gamma_bm, ref)
    def _apply_stored_gamma_to_current_file(self, *, save_entry: bool = False):
        return self._gamma_ctrl._apply_stored_gamma_to_current_file(save_entry=save_entry)
    def _load_grid_controls(self, cfg: dict | None):
        return self._norm_ctrl._load_grid_controls(cfg)
    def _display_grid_config(self, cfg: dict | None) -> dict:
        return self._norm_ctrl._display_grid_config(cfg)
    def _grid_status_text(self, info: dict, target: str) -> str:
        return self._norm_ctrl._grid_status_text(info, target)
    def _apply_grid_correction(self):
        return self._norm_ctrl._apply_grid_correction()
    def _reset_grid_correction(self):
        return self._norm_ctrl._reset_grid_correction()
    def _install_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+G"), self).activated.connect(self._fit_guess)
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(self._fit_full)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(
            lambda: self._session.save() or self._status("Session sauvegardée"))
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self).activated.connect(
            lambda: self._browser.navigate(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(
            lambda: self._browser.navigate(+1))

    # ─────────────────────────────────────────────────────────────────────────
    # Chargement fichier
    # ─────────────────────────────────────────────────────────────────────────

    def _load_file(self, path: str):
        self._load_ctrl.load(path)

    def _update_display_data(self):
        return self._plot_ctrl._update_display_data()
    def _on_view_changed(self):
        if self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.view_mode = self._cmb_view.currentText()
        self._update_display_data()
        self._draw_bm()

    def _on_ev_spinbox_changed(self, val: float):
        """L'utilisateur saisit directement une énergie → met à jour sel_ev + carte."""
        if self._raw_data is None:
            return
        ev_arr = self._raw_data["ev_arr"]
        self._sel_ev = float(np.clip(val, ev_arr.min(), ev_arr.max()))
        self._draw_bm()
        self._draw_mdc_edc()
        if hasattr(self, "_mdc_fit_tabs") and self._tabs.currentIndex() == 1 and self._mdc_fit_tabs.currentIndex() == 1:
            self._draw_mdc_waterfall()

    def _schedule_model_redraw(self, _=None):
        self._redraw_timer.start(120)

    def _schedule_fit_only_redraw(self, _=None):
        self._fit_redraw_timer.start(120)

    def _on_model_changed(self, _=None):
        self._update_display_data()
        self._draw_bm()
        if self._tabs.currentIndex() == 1:
            self._draw_mdc_edc()
            if hasattr(self, "_mdc_fit_tabs") and self._mdc_fit_tabs.currentIndex() == 1:
                self._draw_mdc_waterfall()

    def _on_fit_only_changed(self, _=None):
        # Paramètres qui n'affectent ni la BM affichée ni la donnée raw :
        # uniquement les overlays MDC/EDC et le waterfall.
        if self._tabs.currentIndex() != 1:
            return
        self._draw_mdc_edc()
        if hasattr(self, "_mdc_fit_tabs") and self._mdc_fit_tabs.currentIndex() == 1:
            self._draw_mdc_waterfall()

    def _fit_roi_bounds(self) -> tuple[float, float, float, float] | None:
        return self._plot_ctrl._fit_roi_bounds()
    def _fit_roi_data(self, disp: np.ndarray, kpar: np.ndarray, ev: np.ndarray) -> np.ndarray:
        return self._plot_ctrl._fit_roi_data(disp, kpar, ev)
    def _map_color_kwargs(self, disp: np.ndarray, mode: str, *, roi_scale: bool = False) -> tuple[str, dict]:
        return self._plot_ctrl._map_color_kwargs(disp, mode, roi_scale=roi_scale)
    def _draw_fit_roi_overlay(self, ax):
        return self._plot_ctrl._draw_fit_roi_overlay(ax)
    def _ef_offset_text(self) -> str:
        return self._plot_ctrl._ef_offset_text()
    def _draw_ef_label(self, ax, *, horizontal: bool = True):
        return self._plot_ctrl._draw_ef_label(ax, horizontal=horizontal)
    def _draw_bm(self):
        return self._plot_ctrl._draw_bm()
    def _draw_mdc_energy_map(self):
        return self._plot_ctrl._draw_mdc_energy_map()
    def _draw_mdc_waterfall(self):
        return self._plot_ctrl._draw_mdc_waterfall()
    def _draw_kf_overlay(self, ax):
        return self._plot_ctrl._draw_kf_overlay(ax)
    def _get_mdc(self):
        return self._plot_ctrl._get_mdc()
    def _get_edc(self):
        return self._plot_ctrl._get_edc()
    def _draw_mdc_edc(self):
        return self._plot_ctrl._draw_mdc_edc()
    def _set_fit_roi_pick_mode(self, active: bool):
        active = bool(active)
        if not active and self._fit_roi_rect is not None:
            try:
                canvas = self._fit_roi_rect.figure.canvas
                self._fit_roi_rect.remove()
                canvas.draw_idle()
            except Exception:
                pass
        self._fit_roi_active = active
        self._fit_roi_start = None
        self._fit_roi_ax = None
        self._fit_roi_rect = None
        self._params.set_fit_roi_active(active)
        for canv in (getattr(self, "_bm_canvas", None), getattr(self, "_mdc_map_canvas", None)):
            if canv is None or not hasattr(canv, "canvas"):
                continue
            if active:
                canv.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else:
                canv.canvas.unsetCursor()
        if active:
            if self._tabs.currentIndex() not in (0, 1):
                self._tabs.setCurrentIndex(1)
            self._status("Sélection zone fit : cliquer-glisser un rectangle sur la carte.")

    def _on_fit_roi_press(self, event):
        if not self._fit_roi_active:
            return
        if event.inaxes not in (self._bm_canvas.ax, self._mdc_map_canvas.ax):
            return
        button = getattr(event.button, "value", event.button)
        if button != 1 or event.xdata is None or event.ydata is None:
            return
        self._fit_roi_start = (float(event.xdata), float(event.ydata))
        self._fit_roi_ax = event.inaxes
        if self._fit_roi_rect is not None:
            try:
                self._fit_roi_rect.remove()
            except Exception:
                pass
        self._fit_roi_rect = Rectangle(
            self._fit_roi_start, 0.0, 0.0,
            fill=False, edgecolor="#38bdf8", linewidth=1.4,
            linestyle="-", alpha=0.95, zorder=20,
        )
        event.inaxes.add_patch(self._fit_roi_rect)
        event.canvas.draw_idle()

    def _on_fit_roi_motion(self, event):
        if not self._fit_roi_active or self._fit_roi_start is None or self._fit_roi_rect is None:
            return
        if event.inaxes is not self._fit_roi_ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._fit_roi_start
        x1, y1 = float(event.xdata), float(event.ydata)
        self._fit_roi_rect.set_x(min(x0, x1))
        self._fit_roi_rect.set_y(min(y0, y1))
        self._fit_roi_rect.set_width(abs(x1 - x0))
        self._fit_roi_rect.set_height(abs(y1 - y0))
        event.canvas.draw_idle()

    def _on_fit_roi_release(self, event):
        if not self._fit_roi_active or self._fit_roi_start is None:
            return
        if event.inaxes is not self._fit_roi_ax or event.xdata is None or event.ydata is None:
            self._set_fit_roi_pick_mode(False)
            return
        x0, y0 = self._fit_roi_start
        x1, y1 = float(event.xdata), float(event.ydata)
        if abs(x1 - x0) < 1e-4 or abs(y1 - y0) < 1e-4:
            self._set_fit_roi_pick_mode(False)
            return
        self._apply_fit_roi_from_bounds(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        self._set_fit_roi_pick_mode(False)

    def _apply_fit_roi_from_bounds(self, k0: float, k1: float, e0: float, e1: float):
        if self._raw_data is None:
            return
        d = self._raw_data
        k0 = float(np.clip(k0, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        k1 = float(np.clip(k1, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        e0 = float(np.clip(e0, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        e1 = float(np.clip(e1, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        if k1 <= k0 or e1 <= e0:
            return
        for sp, val in (
            (self._params.sp_kmin, k0), (self._params.sp_kmax, k1),
            (self._params.sp_evs, e0), (self._params.sp_eve, e1),
        ):
            sp.blockSignals(True)
            sp.setValue(float(val))
            sp.blockSignals(False)
        self._sel_k = float((k0 + k1) * 0.5)
        self._sel_ev = float((e0 + e1) * 0.5)
        self._sync_ev_spinbox()
        self._params.params_changed.emit()
        self._draw_bm()
        self._draw_mdc_edc()
        self._status(
            f"Zone fit : k={k0:+.3f}→{k1:+.3f} π/a, "
            f"E={e0:+.3f}→{e1:+.3f} eV"
        )

    def _reset_fit_roi_range(self):
        if self._raw_data is None:
            return
        d = self._raw_data
        self._apply_fit_roi_from_bounds(
            float(np.nanmin(d["kpar"])), float(np.nanmax(d["kpar"])),
            float(np.nanmin(d["ev_arr"])), float(np.nanmax(d["ev_arr"])),
        )

    def _on_map_click(self, event):
        if self._fit_roi_active:
            return
        if event.inaxes not in (self._bm_canvas.ax, self._mdc_map_canvas.ax): return
        if event.xdata is None or event.ydata is None: return
        d = self._raw_data
        self._sel_ev = float(np.clip(event.ydata,
                                     d["ev_arr"].min(), d["ev_arr"].max()))
        self._sel_k  = float(np.clip(event.xdata,
                                     d["kpar"].min(), d["kpar"].max()))
        self._sync_ev_spinbox()
        self._draw_bm()
        self._draw_mdc_edc()

    def _sync_ev_spinbox(self):
        self._params.sp_ev.blockSignals(True)
        self._params.sp_ev.setValue(self._sel_ev)
        self._params.sp_ev.blockSignals(False)

    # ─────────────────────────────────────────────────────────────────────────
    # Fit
    # ─────────────────────────────────────────────────────────────────────────

    def _get_work_data(self):
        """Données normalisées (pour le fit)."""
        if self._raw_data is None: return None, None, None
        d    = self._raw_data
        norm = apply_edcnorm(d["data"]) if self._params.chk_norm.isChecked() else d["data"]
        return norm, d["kpar"], d["ev_arr"]

    def _fit_guess(self):
        if AP is None: self._status("⚠ arpes_plots non chargé"); return
        data, kpar, ev = self._get_work_data()
        if data is None: return
        fp = self._params.get_fit_params()

        ax = self._mdc_edc.axes[0]
        ax.cla(); ax.set_facecolor("#1a1a1a")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kF_init_list = [p.get("kF_init", 0.30) for p in (fp.pairs or [])]
                r = AP.debug_mdc_fit(
                    data, kpar, ev,
                    energy=self._sel_ev, n_pairs=fp.n_pairs,
                    smooth_fit=fp.smooth_fit, smooth_detect=fp.smooth_detect,
                    gamma_init=fp.gamma_init, gamma_max=fp.gamma_max,
                    kF_init=kF_init_list or None, center_init=fp.center_init,
                    xg_range=fp.xg_range, k_min=fp.k_min, k_max=fp.k_max,
                    k0_max=fp.k0_max, width_mode=fp.width_mode, ax=ax,
                )
            ax.set_title(f"Guess  E={self._sel_ev:.3f} eV", fontsize=8, color="w")
            ax.tick_params(colors="w", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#555")
            try:
                leg = ax.get_legend()
                if leg:
                    leg.get_frame().set_facecolor("#333")
                    for t in leg.get_texts(): t.set_color("w")
            except Exception: pass

            if r["success"]:
                k0s = "  ".join(f"{v:.3f}" for v in r["k0"])
                gamma_vals = r["gamma"] if isinstance(r["gamma"], (list, tuple, np.ndarray)) else [r["gamma"]]
                gammas = "  ".join(f"{float(v):.4f}" for v in gamma_vals)
                self._params.lbl_res.setText(
                    f"✓  E={self._sel_ev:.3f} eV\n"
                    f"kF=[{k0s}] π/a\n"
                    f"γ=[{gammas}]  rms={r['residual']:.4f}\n"
                    f"xg={r['xg']:.4f} π/a")
                self._status(f"Guess OK  kF={k0s}  γ=[{gammas}]")
            else:
                self._params.lbl_res.setText("✗  Fit échoué")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                    ha="center", va="center", color="tomato", fontsize=8)
            traceback.print_exc()
        self._mdc_edc.fig.tight_layout(pad=0.5)
        self._mdc_edc.redraw()

    def _estimate_gamma_bm(self):
        return self._gamma_ctrl._estimate_gamma_bm()
    def _apply_gamma_reference_to_bm(self):
        return self._gamma_ctrl._apply_gamma_reference_to_bm()
    def _fit_full(self):
        if AP is None: self._status("⚠ arpes_plots non chargé"); return
        data, kpar, ev = self._get_work_data()
        if data is None: return
        fp = self._params.get_fit_params()

        self._status("Fit complet en cours …")
        QApplication.processEvents()
        try:
            controller = FitController(AP)
            fr = controller.run_full_fit(
                data,
                kpar,
                ev,
                fp,
                resolution_source=getattr(self._params, "_resolution_source_detail", ""),
            )
            self._fit_res = fr

            # Sauvegarder dans la session
            if self._current_path:
                name  = self._session.key_for_path(self._current_path)
                entry = self._session.get_or_create(name)
                controller.update_entry_after_fit(
                    entry,
                    fp,
                    ef_offset=self._params.sp_ef.value(),
                    edcnorm=self._params.chk_norm.isChecked(),
                    view_mode=self._cmb_view.currentText(),
                    hv=self._raw_data["hv"],
                )
                self._session.set_fit_result(name, fr)
                self._browser.refresh_item(name)
                self._refresh_helper_buttons()

            summary = controller.summarize(fr)
            self._params.lbl_res.setText(summary.label_text)
            self._params.lbl_res.setToolTip(
                "Résolution instrumentale domine, fit non fiable"
                if summary.resolution_dominates else ""
            )
            self._draw_bm()
            self._status(summary.status_text)
        except Exception as e:
            self._status(f"⚠ Fit complet : {e}"); traceback.print_exc()

    def _clear_kf(self):
        self._fit_res = None
        self._draw_bm()
        self._params.lbl_res.setText("kF effacé")

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration EF
    # ─────────────────────────────────────────────────────────────────────────

    def _ef_calibrate(self):
        if self._raw_data is None:
            self._status("⚠ Aucune donnée chargée"); return
        d = self._raw_data
        # Température : metadata > entry > défaut
        entry_now = self._current_entry()
        T_md = (d.get("metadata", {}) or {}).get("temperature")
        try:
            T_md = float(T_md) if T_md is not None else None
        except (TypeError, ValueError):
            T_md = None
        if T_md and np.isfinite(T_md) and T_md > 0:
            T_init = T_md
        elif entry_now and entry_now.meta.temperature and entry_now.meta.temperature > 0:
            T_init = float(entry_now.meta.temperature)
        else:
            T_init = 28.0

        try:
            dlg = EFCalibrationDialog(
                self,
                data=d["data"], kpar=d["kpar"], ev_arr=d["ev_arr"],
                T_init=T_init, half_width_init=0.15,
                source_name=Path(self._current_path).name if self._current_path else "",
                current_offset=self._params.sp_ef.value(),
                metadata=d.get("metadata", {}) or {},
            )
            if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.result_payload:
                return
            payload = dlg.result_payload
            self._apply_ef_calibration_result(payload)
        except Exception as e:
            self._status(f"⚠ Calibration EF : {e}"); traceback.print_exc()

    def _apply_ef_calibration_result(self, payload: dict):
        """Sauvegarde la correction sur le fichier courant et recharge."""
        if not self._current_path:
            return
        key = self._session.key_for_path(self._current_path)
        entry = self._session.get_or_create(key)

        update = compute_ef_calibration_update(
            payload,
            current_ef_offset=float(self._params.sp_ef.value()),
            source_meta=(self._raw_data or {}).get("metadata") or {},
            source_path=str(self._current_path),
        )
        entry.ef_offset = update.new_ef_offset
        entry.ef_correction = update.ef_correction
        self._params.sp_ef.blockSignals(True)
        self._params.sp_ef.setValue(update.new_ef_offset)
        self._params.sp_ef.blockSignals(False)
        msg = update.msg

        if payload.get("save_as_reference"):
            self._session.ef_reference = update.ref_payload
            msg += "  |  référence dossier sauvegardée"

        self._session.save()
        self._load_file(self._current_path)
        self._refresh_helper_buttons()
        self._status(msg)

    def _apply_ef_reference_to_current(self):
        """Copie session.ef_reference vers FileEntry courant."""
        ref = self._session.ef_reference or {}
        if not ref or not self._current_path:
            self._status("⚠ Aucune référence EF en session — calibrer un Au d'abord")
            return
        key = self._session.key_for_path(self._current_path)
        entry = self._session.get_or_create(key)

        # Garde-fou : éviter une double application en mode scalaire (qui
        # soustrait cumulativement ef_shift à l'offset courant).
        if ef_reference_already_applied(entry.ef_correction):
            ans = QMessageBox.question(
                self,
                "Référence EF déjà appliquée",
                "Une référence EF est déjà appliquée à ce fichier. "
                "L'appliquer à nouveau cumulerait les décalages et serait probablement faux.\n\n"
                "Continuer quand même ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                self._status("Application de la référence EF annulée")
                return

        ref_path = ref.get("source_file", "")
        ref_name = Path(ref_path).name if ref_path else "?"

        try:
            app = apply_ef_reference_to_target(
                ref,
                current_ef_offset=float(self._params.sp_ef.value()),
                target_meta=(self._raw_data or {}).get("metadata") or {},
                ref_path_str=ref_name,
            )
        except EFReferenceError:
            self._status("⚠ Référence EF mal formée")
            return
        entry.ef_offset = app.new_ef_offset
        entry.ef_correction = app.ef_correction
        self._params.sp_ef.blockSignals(True)
        self._params.sp_ef.setValue(app.new_ef_offset)
        self._params.sp_ef.blockSignals(False)
        self._session.save()
        self._load_file(self._current_path)
        self._status(app.msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Copy params
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_helper_buttons(self):
        """Met à jour l'état/label des boutons EF réf et Propager params."""
        self._params.update_ef_reference_button(self._session.ef_reference or None)
        if not self._current_path:
            self._params.update_copy_params_button(0)
            return
        cur_key = self._session.key_for_path(self._current_path)
        n = sum(
            1 for name, entry in self._session.files.items()
            if entry.fit_result is None and name != cur_key
        )
        self._params.update_copy_params_button(n)

    def _copy_params(self):
        """Sauvegarde les params courants dans tous les fichiers non-fittés."""
        if not self._current_path:
            return
        fp = self._params.get_fit_params()
        cur_key = self._session.key_for_path(self._current_path)
        targets: list[str] = []
        for name, entry in self._session.files.items():
            if entry.fit_result is None and name != cur_key:
                entry.fit_params = fp
                targets.append(name)
        self._session.save()
        n = len(targets)
        if n == 0:
            self._status("Aucun fichier cible — tous les autres sont déjà fittés")
        elif n <= 3:
            self._status(f"Params copiés vers {n} fichier(s) : {', '.join(targets)}")
        else:
            preview = ", ".join(targets[:2])
            self._status(f"Params copiés vers {n} fichiers : {preview}, … (+{n-2})")
        self._refresh_helper_buttons()

    # ─────────────────────────────────────────────────────────────────────────
    def _status(self, msg: str): self.statusBar().showMessage(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in {
        QPalette.ColorRole.Window:          QColor(43,  43,  43),
        QPalette.ColorRole.WindowText:      QColor(220, 220, 220),
        QPalette.ColorRole.Base:            QColor(30,  30,  30),
        QPalette.ColorRole.AlternateBase:   QColor(50,  50,  50),
        QPalette.ColorRole.Text:            QColor(220, 220, 220),
        QPalette.ColorRole.Button:          QColor(60,  60,  60),
        QPalette.ColorRole.ButtonText:      QColor(220, 220, 220),
        QPalette.ColorRole.Highlight:       QColor(42,  130, 218),
        QPalette.ColorRole.HighlightedText: QColor(255, 255, 255),
    }.items():
        pal.setColor(role, color)
    app.setPalette(pal)

    win = ArpesExplorer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
