#!/usr/bin/env python3
"""
arpes_explorer.py — Interface interactive ARPES (BaNi₂As₂) v3
═══════════════════════════════════════════════════════════════
Features :
  • Panneau fichiers (browse dossier, statut unloaded/loaded/fitted)
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
from collections import OrderedDict
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
from arpes.physics.fit import MdcFitter
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
from arpes.ui.controllers.distortion_controller import DistortionController
from arpes.ui.controllers.gamma_controller import GammaController
from arpes.ui.controllers.norm_controller import NormController
from arpes.ui.controllers.fs_controller import FSController
from arpes.ui.controllers.fs_compare_controller import FsCompareController
from arpes.ui.controllers.interaction_controller import InteractionController
from arpes.ui.controllers.fit_runner_controller import FitRunnerController
from arpes.ui.controllers.kz_controller import KzController
from arpes.ui.controllers.proxy_map import PROXY_MAP
from arpes.ui.controllers.theory_overlay_controller import TheoryOverlayController
from arpes.ui.controllers.session_io_controller import SessionIOController
from arpes.core.session import FileEntry, FitParams, Session
from arpes.core.undo import UndoStack

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
    """Charge le module arpes_plots — retourne None si introuvable."""
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
    return None

try:
    from arpes.io.loaders import (
        load_arpes,
        load_arpes_file,
        detect_format,
        detect_scan_kind,
        ARPESData,
    )
    from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas, FSControlPanel
    ERLAB_OK = True
except Exception:
    load_arpes = None
    load_arpes_file = None
    detect_format = None
    detect_scan_kind = None
    ARPESData = None
    FermiSurfaceCanvas = None
    FSControlPanel = None
    ERLAB_OK = False

from arpes.physics.plot_compute import apply_ef_correction_to_dict
from arpes.ui.widgets.canvas import MplCanvas
from arpes.ui.widgets.browsers import FileBrowserPanel
from arpes.ui.widgets.params import ClickablePairLabel, FitParamsPanel, PAIR_COLORS
from arpes.ui.widgets.results import ResultsPanel
from arpes.ui.widgets.dialogs import EFCalibrationDialog

# ─────────────────────────────────────────────────────────────────────────────
# Fenêtre principale
# ─────────────────────────────────────────────────────────────────────────────

class ArpesExplorer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARPES Explorer — BaNi₂As₂")
        self.resize(1650, 950)

        self.ap = _load_ap()
        self._session     = Session()
        self._current_path: str | None = None
        # A.4 — FS épinglée pour overlay BM cuts (cf BM_FS_ORGANIZATION_PLAN.md).
        # Quand on charge une BM, le pin peut pointer vers la FS « contexte »
        # de cette BM (auto ou manual via parent_fs_path) → utilisé par
        # l'overlay Phase B pour savoir sur quelle FS dessiner les lignes.
        self._pinned_fs_path: str | None = None
        # B.4 — toggle overlay BM cuts (False par défaut, off-screen safe).
        self._show_bm_cuts: bool = False
        self._raw_data:   dict | None  = None   # chargé depuis fichier
        self._data_disp:  np.ndarray | None = None  # données affichées (mode)
        self._grid_display_info: dict = {}
        self._fit_res:    dict | None  = None
        self._theory_overlay: dict = {}

        self._sel_ev = -0.30
        self._sel_k  = 0.0
        self._fs_pick_center_active = False
        self._fit_roi_active = False
        self._fit_roi_start: tuple[float, float] | None = None
        self._fit_roi_ax = None
        self._fit_roi_rect = None
        self._fit_selected: list[tuple[str, int, int]] = []
        self._undo_stack = UndoStack(max_size=50)
        self._fit_select_press_xy: tuple[float, float] | None = None
        self._fit_select_press_ax = None
        self._fit_select_rect = None

        # Cache de _update_display_data : recompute uniquement si une des clés
        # influence le résultat affiché.
        self._disp_cache_key: tuple | None = None
        self._display_cache: OrderedDict[tuple, tuple[np.ndarray, dict]] = OrderedDict()
        self._display_cache_max = 24
        self._current_raw_load_cache_key: tuple | None = None
        self._raw_load_cache: OrderedDict[tuple, tuple[dict, dict]] = OrderedDict()
        self._raw_load_cache_max = 16
        self._raw_disk_cache_enabled = True
        self._raw_disk_cache_quota_mb = 250.0
        self._last_load_cache_source = ""
        self._path_signature_cache: OrderedDict[str, tuple[tuple, tuple]] = OrderedDict()
        self._path_signature_cache_max = 128

        self._logbook_ctrl = LogbookIngestController(self)
        self._load_ctrl = LoadController(self)
        self._plot_ctrl = PlotController(self)
        self._gamma_ctrl = GammaController(self)
        self._norm_ctrl = NormController(self)
        self._distortion_ctrl = DistortionController(self)
        self._fs_ctrl = FSController(self)
        self._fs_compare_ctrl = FsCompareController(self)
        self._interaction_ctrl = InteractionController(self)
        self._fit_runner_ctrl = FitRunnerController(self)
        self._kz_ctrl = KzController(self)
        self._theory_overlay_ctrl = TheoryOverlayController(self)
        self._session_io_ctrl = SessionIOController(self)
        from arpes.ui.controllers.batch_controller import BatchController
        self._batch_ctrl = BatchController(self)
        from arpes.ui.controllers.band_analysis_controller import BandAnalysisController
        self._band_analysis_ctrl = BandAnalysisController(self)
        from arpes.ui.controllers.fit_zones_controller import FitZonesController
        self._fit_zones_ctrl = FitZonesController(self)
        from arpes.ui.controllers.pairing_controller import PairingController
        self._pairing_ctrl = PairingController(self)

        # Debouncers : évitent N redraws quand l'utilisateur clique-clique
        # rapidement sur un spinbox ou tape une valeur.
        self._redraw_timer = QTimer(self); self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._on_model_changed)
        self._fit_redraw_timer = QTimer(self); self._fit_redraw_timer.setSingleShot(True)
        self._fit_redraw_timer.timeout.connect(self._on_fit_only_changed)
        # P2-B : preview de fit live (debounce) — lance _fit_guess (non
        # persistant) quand l'utilisateur ajuste un kF init / γ init /
        # E sélectionnée / etc.
        self._live_fit_timer = QTimer(self); self._live_fit_timer.setSingleShot(True)
        self._live_fit_timer.timeout.connect(self._on_live_fit_guess)
        self._distortion_preview_timer = QTimer(self); self._distortion_preview_timer.setSingleShot(True)
        self._distortion_preview_timer.timeout.connect(self._redraw_distortion_preview)
        self._fs_redraw_timer = QTimer(self); self._fs_redraw_timer.setSingleShot(True)
        self._fs_redraw_timer.timeout.connect(self._on_fs_params_changed)

        self._build_ui()
        self._install_shortcuts()
        self._status("Prêt - ouvrir un dossier ou un fichier")

    # ─────────────────────────────────────────────────────────────────────────
    # Proxy dispatch — délègue les méthodes legacy aux controllers.
    #
    # Permet de garder l'API publique de ArpesExplorer (utilisée par les
    # signaux Qt + appels internes) sans dupliquer ~40 stubs `def _x(self):
    # return self._ctrl._x()`. Au connect-time Qt résout `getattr(window,
    # "_method")` → dispatch ici → bound method du controller.
    # ─────────────────────────────────────────────────────────────────────────

    _PROXY_MAP = PROXY_MAP

    def __getattr__(self, name: str):
        if name in self._PROXY_MAP:
            ctrl = object.__getattribute__(self, self._PROXY_MAP[name])
            return getattr(ctrl, name)
        raise AttributeError(name)

    def _on_fit_section_toggled(self, key: str, expanded: bool) -> None:
        self._session.fit_panel_sections[str(key)] = bool(expanded)
        self._session.save()

    def _on_fit_preset_changed(self, name: str) -> None:
        self._session.fit_panel_preset = str(name or "Custom")
        self._session.save()

    def _on_gamma_center_preview(self, value: float) -> None:
        self._update_gamma_preview(float(value))

    def _on_browser_session_reloaded(self) -> None:
        params = getattr(self, "_params", None)
        if params is not None:
            try:
                params.apply_fit_section_states(self._session.fit_panel_sections)
                params.set_fit_preset_silent(self._session.fit_panel_preset)
            except Exception:
                pass
        notes = getattr(self, "_notes_panel", None)
        if notes is not None:
            try:
                notes.refresh_from_session()
            except Exception:
                pass

    def _on_session_notes_changed(self, text: str) -> None:
        self._session.session_notes = str(text or "")
        self._session.save()

    def _clear_disk_cache(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        from arpes.io.artifact_cache import clear_cache_folder, cache_size_mb
        folder = self._session.folder
        if folder is None:
            QMessageBox.information(self, "Cache disque",
                                    "Aucun dossier de session ouvert.")
            return
        size_before = cache_size_mb(folder)
        confirm = QMessageBox.question(
            self, "Vider cache disque",
            f"Supprimer {size_before:.1f} MB d'artefacts dans\n"
            f"{folder}/.arpes_cache/ ?\n\n"
            "Les fichiers seront rechargés depuis la source au prochain accès.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        n, total = clear_cache_folder(folder)
        self._raw_load_cache.clear()
        self._display_cache.clear()
        self._status(f"Cache vidé : {n} fichier(s), {total / 1024 / 1024:.1f} MB libérés.")

    def _reload_current_no_cache(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        path = getattr(self, "_current_path", None)
        if not path:
            QMessageBox.information(self, "Recharger",
                                    "Aucun fichier courant chargé.")
            return
        self._load_ctrl.load(path, force_reload=True)

    def _toggle_disk_cache(self, enabled: bool) -> None:
        self._raw_disk_cache_enabled = bool(enabled)
        state = "activé" if enabled else "désactivé"
        self._status(f"Cache disque {state}.")

    def _refresh_recent_sessions_menu(self) -> None:
        menu = getattr(self, "_recent_sessions_menu", None)
        if menu is None:
            return
        from arpes.ui.builders.menus import _populate_recent_menu
        _populate_recent_menu(self, menu)

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
        # P3 — badge état Γ permanent à droite de la statusbar
        from PyQt6.QtWidgets import QLabel as _QLabel
        self._gamma_status_label = _QLabel("Γ ∅")
        self._gamma_status_label.setToolTip("État Γ courant (référence + axe).")
        self.statusBar().addPermanentWidget(self._gamma_status_label)

    def _wire_signals(self):
        from arpes.ui.builders.panels import wire_ui_signals
        wire_ui_signals(self)

    def _on_tab_changed(self, index: int):
        # 0=BM, 1=MDC Fit, 2=Résultats, 3=FS, 4=KZ, 5=Notes, 6=Aide
        if hasattr(self, "_right_stack"):
            self._right_stack.setCurrentIndex(2 if index == 4 else (1 if index == 3 else 0))
        if index == 0:
            self._params.set_context("bm")
        elif index == 1:
            self._params.set_context("mdc")
            if hasattr(self, "_mdc_fit_tabs"):
                self._params.set_waterfall_controls_visible(self._mdc_fit_tabs.currentIndex() == 1)
        elif index == 4:
            self._params.set_context("other")
            self._set_fit_roi_pick_mode(False)
            self._set_fs_center_pick_mode(False)
            self._draw_kz_tab()
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
            self._draw_mdc_energy_map()
            self._draw_mdc_edc()
        elif index == 0:
            self._draw_bm()
        else:
            self._set_fs_center_pick_mode(False)

    def _on_mdc_fit_subtab_changed(self, index: int):
        if hasattr(self, "_params"):
            self._params.set_waterfall_controls_visible(index == 1 and self._tabs.currentIndex() == 1)
        if index == 0:
            self._draw_mdc_energy_map()
            self._draw_mdc_edc()
        elif index == 1:
            self._draw_mdc_waterfall()
        elif index == 2:
            self._draw_mdc_edc()

    def _current_entry(self) -> FileEntry | None:
        if not self._current_path:
            return None
        return self._session.get_or_create(self._session.key_for_path(self._current_path))

    def _same_path(self, a, b) -> bool:
        if not a or not b:
            return False
        try:
            return Path(a).resolve() == Path(b).resolve()
        except Exception:
            return str(a) == str(b)


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
        from arpes.app_angle_offsets import angle_offsets_for_load
        return angle_offsets_for_load(self, path, entry, hv)

    def _angle_offset_candidates_for_load(
        self,
        path: str | Path,
        entry: FileEntry | None,
        hv: float | None,
        primary: dict,
    ) -> list[dict]:
        from arpes.app_angle_offsets import angle_offset_candidates_for_load
        return angle_offset_candidates_for_load(self, path, entry, hv, primary)

    def _score_bm_gamma_residual(self, d: dict) -> float:
        from arpes.app_angle_offsets import score_bm_gamma_residual
        return score_bm_gamma_residual(self, d)

    def _load_with_best_angle_offsets(
        self,
        path: str,
        entry: FileEntry,
        hv_for_load: float,
        angle_offsets: dict,
    ) -> tuple[dict | None, dict]:
        from arpes.app_angle_offsets import load_with_best_angle_offsets
        return load_with_best_angle_offsets(self, path, entry, hv_for_load, angle_offsets)

    def _bessy_energy_reference_mode(self) -> str:
        """Mode BESSY exposé à l'app principale.

        L'UI quotidienne reste simple : BESSY utilise le mode auto, qui se
        résout côté loader en `ses_center_energy`. Le mode hν−φ reste disponible
        dans `arpes_io.load_bessy_ses_ibw(...)` pour tests/diagnostic explicites.
        """
        return "auto"

    def _install_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+G"), self).activated.connect(self._fit_guess)
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(self._fit_full)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(
            lambda: self._session.save() or self._status("Session sauvegardée"))
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self).activated.connect(
            lambda: self._browser.navigate(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(
            lambda: self._browser.navigate(+1))
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self).activated.connect(
            self._delete_selected_fit_points)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self).activated.connect(
            self._delete_selected_fit_points)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo_fit_delete)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo_fit_delete)

    # ─────────────────────────────────────────────────────────────────────────
    # Chargement fichier
    # ─────────────────────────────────────────────────────────────────────────

    def _load_file(self, path: str):
        self._load_ctrl.load(path)















    # ─────────────────────────────────────────────────────────────────────────
    # Fit
    # ─────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────
    # Calibration EF
    # ─────────────────────────────────────────────────────────────────────────




    # ─────────────────────────────────────────────────────────────────────────
    # Copy params
    # ─────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────
    def _status(self, msg: str):
        text = str(msg or "").strip()
        if text.startswith(("✓", "⚠", "✗")):
            self.statusBar().showMessage(text)
            return
        if text.startswith("Attention:"):
            text = "⚠ " + text.removeprefix("Attention:").strip()
        elif text.startswith("OK "):
            text = "✓ " + text
        self.statusBar().showMessage(text)


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
