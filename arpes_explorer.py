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
import json
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
from arpes_ef_controller import (
    ReferenceError as EFReferenceError,
    already_applied as ef_reference_already_applied,
    apply_reference_to_target as apply_ef_reference_to_target,
    compute_calibration_update as compute_ef_calibration_update,
)
from arpes_fit_controller import FitController
from arpes.physics.gamma import (
    angle_offset_candidates_for_load as _gamma_angle_offset_candidates,
    angle_offsets_from_k_center as _gamma_angle_offsets_from_k_center,
    apply_bm_gamma_axis_shift as _gamma_apply_bm_axis_shift,
    build_gamma_reference as _gamma_build_reference,
    gamma_reference_to_bm_center as _gamma_ref_to_bm_center,
    k_to_angle_offset_deg as _gamma_k_to_angle_offset_deg,
    project_gamma_by_azi as _gamma_project_by_azi,
    score_bm_gamma_residual as _gamma_score_bm_residual,
    stored_gamma_reference as _gamma_stored_reference,
)
from arpes.io.logbook import (
    _cell_float,
    _cell_text,
    _format_direction_label,
    _record_matches_path,
)
from arpes.ui.controllers.logbook_controller import LogbookIngestController
from arpes.ui.controllers.load_controller import LoadController
from arpes.physics.norm import remove_grid_artifact as remove_detector_grid_artifact
from arpes_plot_controller import (
    apply_edcnorm,
    compute_bandmap_display,
    draw_bandmap_axes as _plot_draw_bandmap_axes,
    draw_ef_label as _plot_draw_ef_label,
    draw_fit_roi_overlay as _plot_draw_fit_roi_overlay,
    draw_waterfall_axes as _plot_draw_waterfall_axes,
    display_grid_config as _plot_display_grid_config,
    edc_curve as _plot_edc_curve,
    fit_roi_bounds as _plot_fit_roi_bounds,
    fit_roi_data as _plot_fit_roi_data,
    map_color_kwargs as _plot_map_color_kwargs,
    mdc_curve as _plot_mdc_curve,
    scroll_zoom_limits as _plot_scroll_zoom_limits,
)
from arpes.core.session import FileEntry, FitParams, Session

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QFileDialog, QScrollArea, QGroupBox, QStatusBar,
    QSizePolicy, QFrame, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QDialog, QDialogButtonBox, QStackedWidget,
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
    from arpes.io.loaders import load_arpes, detect_format, detect_scan_kind, ARPESData
    from arpes.physics.fs import FermiSurfaceCanvas, FSControlPanel
    ERLAB_OK = True
except Exception:
    load_arpes = None
    detect_format = None
    detect_scan_kind = None
    ARPESData = None
    FermiSurfaceCanvas = None
    FSControlPanel = None
    ERLAB_OK = False

AP = None


# ─────────────────────────────────────────────────────────────────────────────
# Chargement données ARPES
# ─────────────────────────────────────────────────────────────────────────────

def load_arpes_file(path: str, work_func: float, ef_offset: float,
                    a_lattice: float = 3.96, hv: float | None = None,
                    temperature: float | None = None,
                    azi: float | None = None,
                    pol: str = "",
                    angle_offsets: dict | None = None,
                    bessy_energy_reference: str = "auto") -> dict | None:
    if not ERLAB_OK or load_arpes is None:
        return None
    ds = load_arpes(path, work_func=work_func, ef_offset=ef_offset,
                    a_lattice=a_lattice, hv=hv,
                    temperature=temperature,
                    azi=float(azi) if azi is not None else 0.0,
                    pol=pol,
                    angle_offsets=angle_offsets,
                    bessy_energy_reference=bessy_energy_reference)
    return ds.as_legacy_bandmap_dict()


def _loader_label(source_format: str | None, metadata: dict | None = None) -> str:
    """Label court et stable pour l'affichage utilisateur."""
    fmt = (source_format or "").strip()
    md = metadata or {}
    explicit = str(md.get("loader_label") or md.get("lab_label") or "").strip()
    if explicit:
        return explicit
    lab = str(md.get("lab") or "").strip().lower()
    if "cls" in lab or "lnls" in lab:
        return "CLS"
    labels = {
        "cls_txt": "CLS",
        "solaris_da30": "Solaris",
        "bessy_ses_ibw": "BESSY",
    }
    if fmt in labels:
        return labels[fmt]
    if not fmt:
        return ""
    return fmt.replace("_", " ").replace("-", " ").title()


def apply_ef_correction_to_dict(d: dict, cfg: dict) -> tuple[dict, dict]:
    """Applique une correction EF par colonne (poly) au dict legacy.

    Renvoie (dict_corrigé, info) où info contient ef_smooth, ef_at_center, etc.
    Modifie une copie : ne touche pas l'objet d'origine.
    """
    if not cfg or cfg.get("mode") != "poly":
        return d, {}
    coefs = np.asarray(cfg.get("poly_coefs", []), dtype=float)
    if coefs.size == 0:
        return d, {}
    kpar = np.asarray(d["kpar"], dtype=float)
    ev   = np.asarray(d["ev_arr"], dtype=float)
    data = np.asarray(d["data"], dtype=float)
    ef_smooth = np.polyval(coefs, kpar)
    try:
        from arpes.ui.widgets.plots import apply_ef_correction_per_column as _apply
    except Exception:
        return d, {}
    data_corr = _apply(data, kpar, ev, ef_smooth)
    out = dict(d)
    out["data"] = data_corr
    info = {"ef_smooth": ef_smooth, "ef_center": float(np.interp(0.0, kpar, ef_smooth))}
    return out, info


# ─────────────────────────────────────────────────────────────────────────────
# Modèle Lorentzien (visualisation temps réel)
# ─────────────────────────────────────────────────────────────────────────────

def _lorentzian(k, k0, gamma, A):
    return A * gamma**2 / ((k - k0)**2 + gamma**2)


def build_model_pairs(k_arr, mdc, n_pairs, gamma_init,
                      k_min, k_max, center_init, smooth_sigma,
                      spacing=0.25):
    """Retourne (pairs, mdc_smooth_norm).

    pairs : list de (curve_total, km, kp, curve_left, curve_right)
      • curve_total  : somme des deux Lorentziennes de la paire
      • km / kp      : centres détectés (−x0+xg  /  +x0+xg)
      • curve_left/right : contribution individuelle de chaque pic
    mdc_smooth_norm : MDC lissée pour la détection, normalisée [0-1]
    """
    mask = (k_arr >= k_min) & (k_arr <= k_max)
    k_w  = k_arr[mask]
    m_w  = mdc[mask]

    # courbe lissée pleine résolution (pour overlay visuel)
    s_full = max(0.5, float(smooth_sigma))
    m_sm_full = gaussian_filter1d(np.nan_to_num(mdc.copy()), sigma=s_full)
    lo_f, hi_f = m_sm_full.min(), m_sm_full.max()
    mdc_smooth_norm = (m_sm_full - lo_f) / (hi_f - lo_f + 1e-12)

    if k_w.size < 10:
        return [], mdc_smooth_norm

    s  = max(1, int(smooth_sigma))
    m_sm = gaussian_filter1d(np.nan_to_num(m_w), sigma=s)
    lo, hi = m_sm.min(), m_sm.max()
    if hi - lo < 1e-10:
        return [], mdc_smooth_norm
    m_n  = (m_sm - lo) / (hi - lo)
    bg   = float(np.nanpercentile(m_sm, 10))
    A0   = float(hi - lo)

    pks, _ = find_peaks(m_n, height=0.10, distance=max(3, s))
    if len(pks):
        pks = pks[np.argsort(m_n[pks])[::-1]]

    params = []
    if len(pks) >= 2:
        k_pks = k_w[pks]; A_pks = m_sm[pks] - bg
        pos = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp >= center_init]
        neg = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp <  center_init]
        for i in range(min(n_pairs, max(len(pos), len(neg)))):
            km = neg[i][0] if i < len(neg) else center_init - spacing * (i + 1)
            kp = pos[i][0] if i < len(pos) else center_init + spacing * (i + 1)
            params.append((km, kp, A0))
    elif len(pks) == 1:
        k0 = float(k_w[pks[0]])
        params.append((2 * center_init - k0, k0, float(m_sm[pks[0]] - bg)))
    else:
        params.append((center_init - spacing, center_init + spacing, A0))

    while len(params) < n_pairs:
        km0, kp0, A0p = params[0]
        gap = abs(kp0 - km0); i = len(params)
        params.append((km0 - i * gap * 0.5, kp0 + i * gap * 0.5, A0p * 0.7))

    result = []
    for km, kp, A in params[:n_pairs]:
        cl = _lorentzian(k_arr, km, gamma_init, A)
        cr = _lorentzian(k_arr, kp, gamma_init, A)
        c  = cl + cr
        c[~mask] = np.nan; cl[~mask] = np.nan; cr[~mask] = np.nan
        result.append((c, km, kp, cl, cr))

    total = np.nansum([c for c, *_ in result], axis=0)
    tm = float(np.nanmax(total[mask])) if np.any(np.isfinite(total[mask])) else 1.0
    dm = float(np.nanmax(m_w))
    scale = dm / tm if tm > 0 else 1.0
    return [(c*scale, km, kp, cl*scale, cr*scale)
            for c, km, kp, cl, cr in result], mdc_smooth_norm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers UI
# ─────────────────────────────────────────────────────────────────────────────

def _dspin(val, lo, hi, step, dec=3):
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setSingleStep(step)
    w.setDecimals(dec); w.setValue(val); w.setFixedWidth(82)
    return w

def _ispin(val, lo, hi):
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); w.setFixedWidth(60)
    return w

def _sep():
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken); return f

PAIR_COLORS = ["#ff8c00", "#00e5ff", "#7fff00", "#ff44cc"]


# ─────────────────────────────────────────────────────────────────────────────
# MplCanvas
# ─────────────────────────────────────────────────────────────────────────────

class MplCanvas(QWidget):
    def __init__(self, figsize=(5, 4), toolbar=False, nrows=1):
        super().__init__()
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            lay.addWidget(NavToolbar(self.canvas, self))
        lay.addWidget(self.canvas)
        if nrows == 1:
            self.ax  = self.fig.add_subplot(111)
            self.axes = [self.ax]
        else:
            self.axes = list(self.fig.subplots(nrows, 1))
            self.ax   = self.axes[0]
        self._dark()

    def _dark(self):
        self.fig.set_facecolor("#2b2b2b")
        for ax in self.axes:
            ax.set_facecolor("#1a1a1a")

    def redraw(self): self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Panneau fichiers
# ─────────────────────────────────────────────────────────────────────────────

class FileBrowserPanel(QWidget):
    file_selected = pyqtSignal(str)   # émet le chemin complet

    STATUS_ICONS = {"unloaded": "○", "loaded": "◑", "fitted": "●"}
    STATUS_COLORS = {"unloaded": "#888", "loaded": "#f0c040", "fitted": "#60e080"}

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._folder: Path | None = None
        self._collapsed_groups: set[str] = set()
        self._group_mode = "Dossier"
        self._group_fields: list[str] = ["Dossier"]
        self._items_cache: list[Path] | None = None
        self._loader_label_cache: dict[str, tuple[tuple[int, int] | None, str]] = {}
        self._scan_kind_cache: dict[str, tuple[tuple[int, int] | None, str]] = {}
        self._logbook_record_cache: dict[str, dict | None] = {}
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        btn = QPushButton("📂 Dossier")
        btn.clicked.connect(self._open_folder)
        top.addWidget(btn)
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(32)
        btn_refresh.setToolTip("Rafraîchir la liste des fichiers")
        btn_refresh.clicked.connect(self.refresh)
        top.addWidget(btn_refresh)
        self._lbl_folder = QLabel("—")
        self._lbl_folder.setWordWrap(True)
        self._lbl_folder.setStyleSheet("font-size:10px; color:#aaa;")
        lay.addLayout(top)
        lay.addWidget(self._lbl_folder)

        self._lbl_summary = QLabel("Aucun dossier chargé")
        self._lbl_summary.setWordWrap(True)
        self._lbl_summary.setStyleSheet("font-size:10px; color:#aaa;")
        lay.addWidget(self._lbl_summary)

        mode_row = QVBoxLayout()
        mode_title = QLabel("Organiser par:")
        mode_title.setStyleSheet("font-size:10px; color:#aaa;")
        mode_row.addWidget(mode_title)
        checks_row_1 = QHBoxLayout()
        checks_row_2 = QHBoxLayout()
        self._group_checks: dict[str, QCheckBox] = {}
        group_defs = [
            ("Dossier", "Dossier"),
            ("Type", "Type"),
            ("hν", "hν"),
            ("Température", "T"),
            ("Chemin", "Chemin"),
            ("Polarisation", "Pol"),
            ("Labo", "Labo"),
        ]
        for i, (field, label) in enumerate(group_defs):
            chk = QCheckBox(label)
            chk.setChecked(field == "Dossier")
            chk.setToolTip(
                "Critère cumulable d'organisation visuelle.\n"
                "N'applique aucune correction EF/Γ et ne prouve pas que les "
                "fichiers sont directement comparables."
            )
            chk.stateChanged.connect(self._on_group_checks_changed)
            self._group_checks[field] = chk
            (checks_row_1 if i < 4 else checks_row_2).addWidget(chk)
        checks_row_1.addStretch(1)
        checks_row_2.addStretch(1)
        mode_row.addLayout(checks_row_1)
        mode_row.addLayout(checks_row_2)
        lay.addLayout(mode_row)

        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget { background:#222; color:#ddd; font-size:11px; }
            QListWidget::item:selected { background:#2a6099; }
        """)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_selection_change)
        lay.addWidget(self._list, stretch=1)

        self._lbl_selection = QLabel("Sélectionne un fichier à charger")
        self._lbl_selection.setWordWrap(True)
        self._lbl_selection.setStyleSheet(
            "font-size:10px; color:#c8c8c8; background:#1c1c1c; "
            "border:1px solid #333; padding:5px; border-radius:3px;"
        )
        lay.addWidget(self._lbl_selection)

        self._btn_load = QPushButton("↵ Charger la sélection")
        self._btn_load.clicked.connect(self._load_selected)
        self._btn_load.setEnabled(False)
        lay.addWidget(self._btn_load)

        self.setMinimumWidth(250)
        self.setMaximumWidth(340)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier données ARPES",
                                                   str(Path.home()))
        if folder:
            self.set_folder(Path(folder))

    def set_folder(self, folder: Path):
        self._folder = folder
        self._session.folder = folder
        self._lbl_folder.setText(folder.name)
        self._items_cache = None
        self._loader_label_cache.clear()
        self._scan_kind_cache.clear()
        self._logbook_record_cache.clear()
        if self._session.json_path and self._session.json_path.exists():
            try:
                self._session.load(self._session.json_path)
            except Exception:
                pass
        self._populate()

    def refresh(self):
        self._items_cache = None
        self._loader_label_cache.clear()
        self._scan_kind_cache.clear()
        self._logbook_record_cache.clear()
        self._populate()

    def _is_cls_dataset_dir(self, p: Path) -> bool:
        if not p.is_dir():
            return False
        for param_file in p.glob("*_param.txt"):
            prefix = param_file.name.removesuffix("_param.txt")
            if any(p.glob(f"{prefix}_Cycle_*_Step_*.txt")):
                return True
        return False

    def _is_data_file(self, p: Path) -> bool:
        if not p.is_file():
            return False
        if p.name.endswith("_param.txt"):
            return False
        if p.suffix.lower() in {".pxt", ".ibw", ".zip"}:
            return True
        # CLS BM : fichier sans extension avec un fichier voisin <nom>_param.txt
        return p.suffix == "" and (p.parent / f"{p.name}_param.txt").exists()

    def _discover_items(self) -> list[Path]:
        if self._items_cache is not None:
            return list(self._items_cache)
        if not self._folder:
            return []
        out: list[Path] = []
        for p in sorted(self._folder.rglob("*")):
            if p.name.startswith("."):
                continue
            if self._is_cls_dataset_dir(p):
                out.append(p)
                # ne pas lister aussi tous les Cycle/Step comme fichiers séparés
                continue
            if self._is_data_file(p):
                # Si le fichier est à l'intérieur d'un dataset CLS FS, on l'ignore
                if any(parent != self._folder and self._is_cls_dataset_dir(parent)
                       for parent in p.parents if self._folder in parent.parents or parent == self._folder):
                    continue
                out.append(p)
        self._items_cache = sorted(set(out), key=lambda x: str(x.relative_to(self._folder)).lower())
        return list(self._items_cache)

    def _group_label(self, group: str) -> str:
        if group == ".":
            return self._folder.name if self._folder else "."
        return group

    def _on_group_checks_changed(self):
        fields = [name for name, chk in self._group_checks.items() if chk.isChecked()]
        if not fields:
            fields = ["Dossier"]
            self._group_checks["Dossier"].blockSignals(True)
            self._group_checks["Dossier"].setChecked(True)
            self._group_checks["Dossier"].blockSignals(False)
        self._group_fields = fields
        self._group_mode = fields[0] if len(fields) == 1 else " + ".join(fields)
        self._collapsed_groups.clear()
        self._populate()

    def _loader_suffix_for_path(self, path: str | Path, key: str | None = None) -> str:
        label = self._loader_label_for_path(path, key)
        return f" ({label})" if label else ""

    def _path_signature(self, path: Path) -> tuple[int, int] | None:
        try:
            st = path.stat()
            return (int(st.st_mtime_ns), int(st.st_size if path.is_file() else -1))
        except OSError:
            return None

    def _loader_label_for_path(self, path: str | Path, key: str | None = None) -> str:
        p = Path(path)
        key = key or self._session.key_for_path(p)
        entry = self._session.files.get(key)
        label = ""
        if entry is not None:
            label = entry.meta.loader_label or _loader_label(entry.meta.source_format)
        if label:
            return label
        cache_key = str(p)
        sig = self._path_signature(p)
        cached = self._loader_label_cache.get(cache_key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        if not label and detect_format is not None:
            try:
                label = _loader_label(detect_format(p))
            except Exception:
                label = ""
        self._loader_label_cache[cache_key] = (sig, label)
        return label

    def _fs_suffix_for_path(self, path: str | Path) -> str:
        return "  [FS]" if self._file_kind_for_path(path) == "FS" else ""

    def _item_label(self, path: str | Path, status: str, key: str | None = None) -> str:
        p = Path(path)
        icon = self.STATUS_ICONS[status]
        extra = self._item_context_suffix(p, key)
        return f"  {icon}  {p.name}{self._loader_suffix_for_path(p, key)}{self._fs_suffix_for_path(p)}{extra}"

    def _logbook_record_for_path(self, path: str | Path) -> dict | None:
        mapping = self._session.logbook_mapping or {}
        records = self._session.logbook_records or []
        file_col = mapping.get("file", "")
        if not file_col or not records:
            return None
        p = Path(path)
        cache_key = str(p)
        if cache_key in self._logbook_record_cache:
            return self._logbook_record_cache[cache_key]
        rec_out = None
        for rec in records:
            if _record_matches_path(rec.get(file_col), p, self._session.folder):
                rec_out = rec
                break
        self._logbook_record_cache[cache_key] = rec_out
        return rec_out

    def _meta_value_for_path(self, path: str | Path, field: str):
        p = Path(path)
        key = self._session.key_for_path(p)
        entry = self._session.files.get(key)
        if entry is not None:
            meta = entry.meta
            if field == "hv" and meta.hv and meta.hv > 0:
                return float(meta.hv), "session"
            if field == "temperature" and meta.temperature and meta.temperature > 0:
                return float(meta.temperature), "session"
            if field == "polarization" and meta.polarization:
                return meta.polarization, "session"
            if field == "direction" and meta.direction:
                return _format_direction_label(meta.direction), "session"
            if field == "azi" and meta.azi is not None:
                return float(meta.azi), "session"
            if field == "polar" and meta.polar is not None:
                return float(meta.polar), "session"
            if field == "tilt" and meta.tilt is not None:
                return float(meta.tilt), "session"

        rec = self._logbook_record_for_path(p)
        mapping = self._session.logbook_mapping or {}
        if rec is None:
            return None, ""
        col = mapping.get(field, "")
        if not col:
            return None, ""
        if field in {"hv", "temperature", "azi", "polar", "tilt"}:
            val = _cell_float(rec.get(col))
            if val is not None and np.isfinite(val):
                return float(val), "logbook"
            return None, ""
        val = _format_direction_label(rec.get(col)) if field == "direction" else _cell_text(rec.get(col))
        return (val, "logbook") if val else (None, "")

    def _fmt_float_group(self, label: str, value, unit: str = "", step: float = 0.1) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "Métadonnées inconnues"
        if not np.isfinite(v):
            return "Métadonnées inconnues"
        if step > 0:
            v = round(v / step) * step
        suffix = f" {unit}" if unit else ""
        return f"{label} {v:.1f}{suffix}"

    def _group_part_for_field(self, path: Path, field: str) -> str:
        if field == "Dossier":
            if not self._folder:
                return "."
            rel = path.relative_to(self._folder)
            group = str(rel.parent) if str(rel.parent) != "." else "."
            return self._group_label(group)
        if field == "Labo":
            return self._loader_label_for_path(path) or "Labo inconnu"
        if field == "Type":
            return self._file_kind_for_path(path)
        if field == "hν":
            hv, source = self._meta_value_for_path(path, "hv")
            group = self._fmt_float_group("hν", hv, "eV", step=0.1)
            return f"{group} ({source})" if source and group != "Métadonnées inconnues" else group
        if field == "Température":
            temp, source = self._meta_value_for_path(path, "temperature")
            group = self._fmt_float_group("T", temp, "K", step=0.1)
            return f"{group} ({source})" if source and group != "Métadonnées inconnues" else group
        if field in {"Chemin", "Géométrie"}:
            direction, source = self._meta_value_for_path(path, "direction")
            if not direction:
                return "Chemin inconnu"
            return f"{direction} ({source})" if source else str(direction)
        if field == "Polarisation":
            pol, source = self._meta_value_for_path(path, "polarization")
            if not pol:
                return "Polarisation inconnue"
            return f"Pol {pol} ({source})" if source else f"Pol {pol}"
        return "."

    def _file_kind_for_path(self, path: str | Path) -> str:
        p = Path(path)
        cache_key = str(p)
        sig = self._path_signature(p)
        cached = self._scan_kind_cache.get(cache_key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        kind = "unknown"
        entry = self._session.files.get(self._session.key_for_path(p))
        if entry is not None and entry.meta.source_format == "cls_txt":
            kind = "FS" if self._is_cls_dataset_dir(p) else "BM"
        if kind == "unknown" and detect_scan_kind is not None:
            try:
                kind = detect_scan_kind(p, format_hint=None)
            except Exception:
                kind = "unknown"
        if kind == "unknown":
            if self._is_cls_dataset_dir(p) or p.suffix.lower() == ".zip":
                kind = "FS"
            else:
                kind = "BM"
        self._scan_kind_cache[cache_key] = (sig, kind)
        return kind

    def _group_key_for_path(self, path: Path) -> str:
        fields = list(getattr(self, "_group_fields", None) or [self._group_mode or "Dossier"])
        parts = [self._group_part_for_field(path, field) for field in fields]
        return " / ".join(parts) if parts else "."

    def _group_sort_key(self, group: str):
        if group in {self._folder.name if self._folder else ".", ".", "BM", "FS"}:
            priority = {"BM": 0, "FS": 1, ".": 0, self._folder.name if self._folder else ".": 0}.get(group, 5)
            return (priority, -1.0, group.lower())
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", group)
        if m:
            try:
                return (2, float(m.group(1)), group.lower())
            except ValueError:
                pass
        unknown = "inconn" in group.lower() or "métadonnées" in group.lower()
        return (9 if unknown else 3, -1.0, group.lower())

    def _item_context_suffix(self, path: Path, key: str | None = None) -> str:
        fields = set(getattr(self, "_group_fields", None) or [self._group_mode or "Dossier"])
        if fields == {"Dossier"}:
            return ""
        bits: list[str] = []
        if "hν" not in fields:
            hv, _ = self._meta_value_for_path(path, "hv")
            if hv is not None:
                bits.append(f"hν={float(hv):.1f}")
        if "Température" not in fields:
            temp, _ = self._meta_value_for_path(path, "temperature")
            if temp is not None:
                bits.append(f"T={float(temp):.1f}")
        if "Chemin" not in fields and "Géométrie" not in fields:
            direction, _ = self._meta_value_for_path(path, "direction")
            if direction:
                bits.append(str(direction))
        if "Polarisation" not in fields:
            pol, _ = self._meta_value_for_path(path, "polarization")
            if pol:
                bits.append(f"Pol={pol}")
        if not bits:
            return ""
        return "  " + "  ".join(bits[:2])

    def _update_summary(self, paths: list[Path]):
        total = len(paths)
        counts = {"unloaded": 0, "loaded": 0, "fitted": 0}
        loaders: dict[str, int] = {}
        for p in paths:
            key = self._session.key_for_path(p)
            counts[self._file_status(key)] += 1
            label = self._loader_label_for_path(p, key) or "?"
            loaders[label] = loaders.get(label, 0) + 1
        loader_txt = ", ".join(f"{k}:{v}" for k, v in sorted(loaders.items())) if loaders else "—"
        self._lbl_summary.setText(
            f"{total} éléments  •  "
            f"{counts['loaded']} chargés  •  {counts['fitted']} fittés  •  {loader_txt}"
        )

    def _describe_item(self, item: QListWidgetItem | None) -> str:
        if item is None:
            return "Sélectionne un fichier à charger"
        group = item.data(Qt.ItemDataRole.UserRole + 2)
        if group is not None:
            return "Dossier de groupe : double-clic ou Charger pour ouvrir/réduire"
        path_txt = item.data(Qt.ItemDataRole.UserRole)
        if not path_txt:
            return "Sélectionne un fichier à charger"
        p = Path(path_txt)
        key = item.data(Qt.ItemDataRole.UserRole + 1) or self._session.key_for_path(p)
        status = self._file_status(key)
        entry = self._session.files.get(key)
        loader = self._loader_label_for_path(p, key) or "inconnu"
        kind = "FS" if self._fs_suffix_for_path(p) else "BM"
        try:
            rel = str(p.relative_to(self._folder)) if self._folder else str(p)
        except Exception:
            rel = str(p)
        bits = [f"{p.name}", rel, f"{kind} {loader}", f"état: {status}"]
        hv, hv_src = self._meta_value_for_path(p, "hv")
        temp, temp_src = self._meta_value_for_path(p, "temperature")
        pol, pol_src = self._meta_value_for_path(p, "polarization")
        direction, dir_src = self._meta_value_for_path(p, "direction")
        azi, azi_src = self._meta_value_for_path(p, "azi")
        polar, p_src = self._meta_value_for_path(p, "polar")
        tilt, t_src = self._meta_value_for_path(p, "tilt")
        if hv is not None:
            bits.append(f"hν={float(hv):.1f} eV ({hv_src})")
        if temp is not None:
            bits.append(f"T={float(temp):.1f} K ({temp_src})")
        if pol:
            bits.append(f"pol={pol} ({pol_src})")
        if direction:
            bits.append(f"direction={direction} ({dir_src})")
        geom = []
        if azi is not None:
            geom.append(f"azi={float(azi):.1f}°")
        if polar is not None:
            geom.append(f"P={float(polar):.1f}°")
        if tilt is not None:
            geom.append(f"T={float(tilt):.1f}°")
        if geom:
            sources = sorted({s for s in (azi_src, p_src, t_src) if s})
            src_txt = f" ({'+'.join(sources)})" if sources else ""
            bits.append("géom: " + ", ".join(geom) + src_txt)
        if entry is not None:
            if entry.fit_result:
                bits.append("fit enregistré")
        return "\n".join(bits)

    def _refresh_selection_state(self):
        item = self._list.currentItem()
        has_path = bool(item and item.data(Qt.ItemDataRole.UserRole))
        has_group = bool(item and item.data(Qt.ItemDataRole.UserRole + 2) is not None)
        if has_path:
            self._btn_load.setText("↵ Charger ce fichier")
            self._btn_load.setEnabled(True)
        elif has_group:
            self._btn_load.setText("↵ Ouvrir/réduire le groupe")
            self._btn_load.setEnabled(True)
        else:
            self._btn_load.setText("↵ Charger la sélection")
            self._btn_load.setEnabled(False)
        self._lbl_selection.setText(self._describe_item(item))

    def _add_header(self, group: str, n_items: int):
        label = self._group_label(group)
        collapsed = group in self._collapsed_groups
        arrow = "▶" if collapsed else "▼"
        item = QListWidgetItem(f"{arrow}  📁  {label}  ({n_items})")
        item.setData(Qt.ItemDataRole.UserRole, None)
        item.setData(Qt.ItemDataRole.UserRole + 2, group)
        item.setToolTip("Double-cliquer pour ouvrir/réduire ce dossier")
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setForeground(QColor("#9ab"))
        self._list.addItem(item)

    def _populate(self):
        selected_path = None
        cur = self._list.currentItem()
        if cur is not None:
            selected_path = cur.data(Qt.ItemDataRole.UserRole)

        self._list.clear()
        if not self._folder:
            self._update_summary([])
            self._refresh_selection_state()
            return

        all_paths = self._discover_items()
        groups: dict[str, list[Path]] = {}
        for p in all_paths:
            group = self._group_key_for_path(p)
            groups.setdefault(group, []).append(p)

        self._update_summary(all_paths)

        for group in sorted(groups, key=self._group_sort_key):
            paths = groups[group]
            self._add_header(group, len(paths))
            if group in self._collapsed_groups:
                continue

            for p in paths:
                rel = p.relative_to(self._folder)
                key = self._session.key_for_path(p)
                status = self._file_status(key)
                icon   = self.STATUS_ICONS[status]
                color  = self.STATUS_COLORS[status]
                item   = QListWidgetItem(self._item_label(p, status, key))
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                item.setData(Qt.ItemDataRole.UserRole + 1, key)
                item.setToolTip(str(rel))
                item.setForeground(QColor(color))
                self._list.addItem(item)

        if selected_path:
            self.select_file(selected_path)
        self._refresh_selection_state()

    def _file_status(self, key: str) -> str:
        if key not in self._session.files:
            return "unloaded"
        return self._session.files[key].status

    def refresh_item(self, filename_or_key: str):
        """Met à jour l'icône d'un fichier dans la liste."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if not path:
                continue
            key = item.data(Qt.ItemDataRole.UserRole + 1) or self._session.key_for_path(path)
            if key == filename_or_key or Path(path).name == filename_or_key:
                status = self._file_status(key)
                icon   = self.STATUS_ICONS[status]
                color  = self.STATUS_COLORS[status]
                item.setText(self._item_label(path, status, key))
                item.setForeground(QColor(color))
                all_paths = self._discover_items()
                self._update_summary(all_paths)
                self._refresh_selection_state()
                break

    def _toggle_group(self, group: str):
        if group in self._collapsed_groups:
            self._collapsed_groups.remove(group)
        else:
            self._collapsed_groups.add(group)
        self._populate()

    def _on_double_click(self, item):
        group = item.data(Qt.ItemDataRole.UserRole + 2)
        if group is not None:
            self._toggle_group(group)
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.file_selected.emit(path)

    def _on_selection_change(self, current, _):
        self._refresh_selection_state()

    def _load_selected(self):
        item = self._list.currentItem()
        if item:
            group = item.data(Qt.ItemDataRole.UserRole + 2)
            if group is not None:
                self._toggle_group(group)
                return
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self.file_selected.emit(path)

    def navigate(self, delta: int):
        if self._list.count() == 0:
            return
        row = self._list.currentRow()
        if row < 0:
            row = 0 if delta >= 0 else self._list.count() - 1
        step = 1 if delta >= 0 else -1
        for new in range(row + step, self._list.count() if step > 0 else -1, step):
            item = self._list.item(new)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self._list.setCurrentRow(new)
                self.file_selected.emit(path)
                return

    def select_file(self, path: str) -> bool:
        """Sélectionne visuellement le fichier dans la liste, en ouvrant son dossier si besoin."""
        if self._folder:
            try:
                rel = Path(path).relative_to(self._folder)
                group = str(rel.parent) if str(rel.parent) != "." else "."
                if group in self._collapsed_groups:
                    self._collapsed_groups.remove(group)
                    self._populate()
            except Exception:
                pass
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                self._list.setCurrentRow(i)
                return True
        return False
class ClickablePairLabel(QLabel):
    """Label cliquable pour naviguer entre les paires de Lorentziennes.
    Clic gauche → paire suivante.  Clic droit → paire précédente."""
    pair_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._current = 0
        self._n = 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "background:#3a3a4a; color:#cde; font-weight:bold;"
            " padding:4px 8px; border-radius:3px; border:1px solid #556;"
        )
        self._update()

    def setup(self, n: int, current: int = 0):
        self._n = max(1, n)
        self._current = max(0, min(current, self._n - 1))
        self._update()

    @property
    def current(self) -> int:
        return self._current

    def _update(self):
        if self._n == 1:
            self.setText(f"Paire 1 / 1")
        else:
            self.setText(f"◀  Paire {self._current + 1} / {self._n}  ▶")

    def mousePressEvent(self, event):
        if self._n < 2:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._current = (self._current + 1) % self._n
        elif event.button() == Qt.MouseButton.RightButton:
            self._current = (self._current - 1) % self._n
        else:
            super().mousePressEvent(event)
            return
        self._update()
        self.pair_changed.emit(self._current)


class FitParamsPanel(QScrollArea):
    params_changed = pyqtSignal()
    fit_only_changed = pyqtSignal()
    guess_requested = pyqtSignal()
    full_fit_requested = pyqtSignal()
    clear_kf_requested = pyqtSignal()
    copy_params_requested = pyqtSignal()
    ef_calib_requested = pyqtSignal()
    ef_apply_reference_requested = pyqtSignal()
    logbook_requested = pyqtSignal()
    gamma_bm_requested = pyqtSignal()
    gamma_ref_requested = pyqtSignal()
    grid_requested = pyqtSignal()
    grid_reset_requested = pyqtSignal()
    fit_roi_requested = pyqtSignal(bool)
    fit_roi_reset_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        w = QWidget()
        self._lay = QVBoxLayout(w)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(w)
        self._pair_params: list[dict] = [{"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}]
        self._current_pair: int = 0
        self._resolution_source_lock = False
        self._resolution_source = "default"
        self._resolution_source_detail = "defaut"
        self._build()

    def _build(self):
        lay = self._lay

        # ── énergie ──────────────────────────────────────────────────────────
        self._energy_widget = QGroupBox("Énergie sélectionnée")
        fl = QFormLayout(self._energy_widget)
        self.sp_ev = _dspin(-0.30, -3.0, 0.2, 0.01)
        # sp_ev est connecté dans ArpesExplorer._build_ui (→ _on_ev_spinbox_changed)
        self.sp_int_win = _dspin(0.010, 0.001, 0.200, 0.005, dec=3)
        self.sp_int_win.setToolTip(
            "Fenêtre d'intégration ±eV pour la MDC\n"
            "Élargir = moins de bruit, moins de résolution en énergie\n"
            "Correspond au 'range' d'extraction d'une coupe dans Igor")
        self.sp_int_win.valueChanged.connect(self.fit_only_changed)
        fl.addRow("E (eV):", self.sp_ev)
        fl.addRow("± intég. (eV):", self.sp_int_win)
        fl.addRow(QLabel("💡 Clic sur la carte ou ici"))
        lay.addWidget(self._energy_widget)

        # ── calibration EF ────────────────────────────────────────────────────
        self._ef_widget = QGroupBox("EF / Chargement")
        fl_ef = QFormLayout(self._ef_widget)
        self.sp_phi = _dspin(4.031, 3.0, 6.0, 0.01)
        self.sp_phi.setToolTip("Fonction de travail φ (eV). Utilisée pour calculer E_kin → E−EF.")
        self.sp_hv  = _dspin(0.0, 0.0, 500.0, 1.0)
        self.sp_hv.setToolTip(
            "Énergie du photon incident (eV).\n"
            "→ CLS/LNLS : entrer manuellement AVANT de charger (obligatoire).\n"
            "→ Solaris/DA30 : lu automatiquement depuis le fichier.\n"
            "→ BESSY/SES : gardé pour diagnostic/kz; E−EF utilise automatiquement Center Energy."
        )
        self.sp_ef  = _dspin(0.052, -0.3, 0.3, 0.005)
        self.sp_ef.setToolTip(
            "Décalage EF en eV. Ajuste le zéro d'énergie.\n"
            "Utiliser 'Calibrer EF auto' pour le calculer par fit Fermi-Dirac."
        )
        self.chk_norm = QCheckBox("EDCnorm"); self.chk_norm.setChecked(True)
        self.chk_norm.stateChanged.connect(self.params_changed)
        btn_ef = QPushButton("🎛  Calibrer EF auto")
        btn_ef.clicked.connect(self.ef_calib_requested)
        self.btn_ef_ref = QPushButton("⇩  Aucune réf EF (calibrer un Au d'abord)")
        self.btn_ef_ref.clicked.connect(self.ef_apply_reference_requested)
        self.btn_ef_ref.setEnabled(False)
        btn_log = QPushButton("📒  Charger logbook")
        btn_log.clicked.connect(self.logbook_requested)
        self.btn_copy = QPushButton("📋  Propager fit params (0 cible)")
        self.btn_copy.clicked.connect(self.copy_params_requested)
        self.btn_copy.setEnabled(False)
        self.update_ef_reference_button(None)
        self.update_copy_params_button(0)
        # Indicateur de provenance pour hν (📁 fichier / 📋 logbook / ✏️ manuel / — inconnu)
        self.lbl_hv_src = QLabel("—")
        self.lbl_hv_src.setToolTip(
            "Provenance de hν :\n"
            "📁 = lue depuis le fichier\n"
            "📋 = lue depuis le logbook\n"
            "✏️ = saisie manuelle\n"
            "— = inconnu"
        )
        hv_row = QWidget()
        hv_lay = QHBoxLayout(hv_row); hv_lay.setContentsMargins(0, 0, 0, 0)
        hv_lay.addWidget(self.sp_hv, 1)
        hv_lay.addWidget(self.lbl_hv_src)
        # éditer la spinbox manuellement marque la source comme manuelle
        self.sp_hv.valueChanged.connect(lambda _v: self._mark_hv_manual_if_user_edit())
        self._hv_source_lock = False  # True quand on set par code (file/logbook), pour ne pas marquer "manual"
        fl_ef.addRow("φ (eV):",       self.sp_phi)
        fl_ef.addRow("hν (eV):", hv_row)
        fl_ef.addRow("EF offset:",    self.sp_ef)
        fl_ef.addRow(self.chk_norm)
        fl_ef.addRow(btn_log)
        fl_ef.addRow(btn_ef)
        fl_ef.addRow(self.btn_ef_ref)
        fl_ef.addRow(self.btn_copy)
        lay.addWidget(self._ef_widget)

        # ── utilitaires BM ────────────────────────────────────────────────────
        self._utils_widget = QGroupBox("Utilitaires")
        fl_ut = QFormLayout(self._utils_widget)
        self.sp_grid_strength = _dspin(0.85, 0.0, 1.0, 0.05, dec=2)
        self.sp_grid_strength.setToolTip(
            "Force de suppression de la trame affichée.\n"
            "0 = aucun effet, 1 = correction complète. Valeur conseillée : 0.8-0.9."
        )
        btn_grid = QPushButton("Retirer effet grille")
        btn_grid.setToolTip(
            "Active un masque Fourier 2D automatique sur la carte BM affichée.\n"
            "La donnée brute reste inchangée."
        )
        btn_grid.clicked.connect(self.grid_requested)
        btn_grid_reset = QPushButton("Recharger brut")
        btn_grid_reset.setToolTip("Désactive la correction grille sauvegardée pour ce fichier.")
        btn_grid_reset.clicked.connect(self.grid_reset_requested)
        self.lbl_grid = QLabel("Correction BM : masque Fourier 2D automatique sur l'affichage.")
        self.lbl_grid.setWordWrap(True)
        self.lbl_grid.setStyleSheet("color:#aaa; font-size:10px;")
        fl_ut.addRow("Force:", self.sp_grid_strength)
        fl_ut.addRow(btn_grid)
        fl_ut.addRow(btn_grid_reset)
        fl_ut.addRow(self.lbl_grid)
        lay.addWidget(self._utils_widget)

        # ── contrôles fit (cachés sur l'onglet BM) ────────────────────────────
        self._fit_controls_widget = QWidget()
        _fcl = QVBoxLayout(self._fit_controls_widget)
        _fcl.setContentsMargins(0, 0, 0, 0)
        _fcl.setSpacing(4)

        # ── plage d'analyse ───────────────────────────────────────────────────
        grp_r = QGroupBox("Plage d'analyse")
        fl2 = QFormLayout(grp_r)
        self.sp_evs  = _dspin(-0.90, -5.0, 1.0, 0.05)
        self.sp_eve  = _dspin(-0.005, -5.0, 1.0, 0.005)
        self.sp_kmin = _dspin(-0.80, -5.0, 5.0, 0.05)
        self.sp_kmax = _dspin( 0.80, -5.0, 5.0, 0.05)
        for w in (self.sp_evs, self.sp_eve, self.sp_kmin, self.sp_kmax):
            w.valueChanged.connect(self.params_changed)
        self.btn_fit_roi = QPushButton("▢  Sélectionner sur carte")
        self.btn_fit_roi.setCheckable(True)
        self.btn_fit_roi.setToolTip(
            "Active une sélection rectangulaire par cliquer-glisser sur la carte BM/MDC Fit.\n"
            "La zone choisie remplit k_min/k_max et ev_start/ev_end."
        )
        self.btn_fit_roi.toggled.connect(self.fit_roi_requested)
        btn_fit_roi_reset = QPushButton("Pleine BM")
        btn_fit_roi_reset.setToolTip("Remet la plage d'analyse sur toute la carte chargée.")
        btn_fit_roi_reset.clicked.connect(self.fit_roi_reset_requested)
        roi_row = QWidget()
        roi_lay = QHBoxLayout(roi_row)
        roi_lay.setContentsMargins(0, 0, 0, 0)
        roi_lay.setSpacing(4)
        roi_lay.addWidget(self.btn_fit_roi)
        roi_lay.addWidget(btn_fit_roi_reset)
        fl2.addRow("ev_start:", self.sp_evs)
        fl2.addRow("ev_end:",   self.sp_eve)
        fl2.addRow("k_min:",    self.sp_kmin)
        fl2.addRow("k_max:",    self.sp_kmax)
        fl2.addRow(roi_row)
        _fcl.addWidget(grp_r)

        # ── fit MDC ───────────────────────────────────────────────────────────
        grp_f = QGroupBox("Fit MDC (Lorentzien)")
        fl3 = QFormLayout(grp_f)
        self.sp_np   = _ispin(1,   1, 8)
        self.sp_np.setToolTip("Nombre de paires de Lorentziennes (= nombre de bandes croisées).")
        self.sp_np.valueChanged.connect(self._on_n_pairs_changed)
        self.sp_sff  = _dspin(2.0,  0.0, 10.0, 0.5, dec=1)
        self.sp_sff.setToolTip(
            "Sigma du lissage gaussien appliqué à la MDC avant l'optimisation scipy.\n"
            "Augmenter pour données bruitées. Voir la courbe orange dans le graphique MDC."
        )
        self.sp_sfd  = _dspin(3.0,  0.0, 10.0, 0.5, dec=1)
        self.sp_sfd.setToolTip(
            "Sigma du lissage gaussien utilisé pour détecter les pics initiaux.\n"
            "Voir la courbe grise dans le graphique MDC."
        )

        # ── paramètres par paire (navigables) ────────────────────────────────
        self._pair_lbl = ClickablePairLabel()
        self._pair_lbl.pair_changed.connect(self._on_pair_changed)
        self.sp_kfi  = _dspin(0.30,  0.0,  3.0, 0.01)
        self.sp_kfi.setToolTip(
            "Position initiale kF (π/a) pour cette paire, comptée depuis centre Γ.\n"
            "Voir les lignes tiret-point colorées dans le graphique MDC."
        )
        self.sp_gi   = _dspin(0.08, 0.01,  0.5, 0.01)
        self.sp_gi.setToolTip(
            "Demi-largeur initiale de la Lorentzienne (π/a).\n"
            "Valeur de départ pour l'optimiseur. Voir les courbes colorées dans le graphique MDC."
        )
        self.sp_gm   = _dspin(0.30, 0.05,  1.0, 0.05)
        self.sp_gm.setToolTip(
            "Demi-largeur maximale autorisée (π/a) — contrainte de l'optimiseur scipy.\n"
            "Voir les zones colorées translucides autour des pics dans le graphique MDC."
        )

        # ── paramètres globaux ────────────────────────────────────────────────
        self.sp_xg   = _dspin(0.10, 0.0,  0.5,  0.01)
        self.sp_xg.setToolTip(
            "Demi-largeur de la zone de contrainte autour du centre Γ (π/a).\n"
            "L'optimiseur limite xg dans [centre − xg_range, centre + xg_range].\n"
            "Voir le rectangle cyan dans le graphique MDC."
        )
        self.sp_cx   = _dspin(0.0, -1.0,  1.0,  0.01)
        self.sp_cx.setToolTip(
            "Centre de symétrie des paires (position Γ, en π/a).\n"
            "Voir la ligne cyan pointillée dans le graphique MDC.\n"
            "Utiliser 'Auto Γ BM' ou 'Γ FS → BM' pour le calculer automatiquement."
        )
        self.sp_k0m  = _dspin(0.0,  0.0,  2.0,  0.05)
        self.sp_k0m.setToolTip(
            "Distance maximale autorisée de kF par rapport à Γ (π/a).\n"
            "Voir les lignes magenta dans le graphique MDC si actif."
        )
        self.chk_k0a = QCheckBox("auto"); self.chk_k0a.setChecked(True)
        self.chk_k0a.setToolTip("Si coché, pas de limite sur kF. Décocher pour activer kF max.")
        self.sp_k0m.setEnabled(False)
        self.chk_k0a.stateChanged.connect(
            lambda: self.sp_k0m.setEnabled(not self.chk_k0a.isChecked()))
        self.cmb_wm  = QComboBox(); self.cmb_wm.addItems(["symmetric","asymmetric"])
        self.cmb_wm.setFixedWidth(110)
        self.cmb_wm.setToolTip(
            "symmetric : les deux pics de la paire ont le même γ.\n"
            "asymmetric : γ gauche et droit peuvent différer (pics asymétriques)."
        )
        self.sp_ma   = _dspin(0.01, 0.0, 1.0, 0.01)
        self.sp_ma.setToolTip(
            "Amplitude minimale relative d'un pic pour être accepté (0–1).\n"
            "Rejette les pics dont l'amplitude est < ampl_min × max(MDC).\n"
            "Augmenter pour éliminer les faux pics dus au bruit."
        )
        self.sp_mj   = _dspin(0.20, 0.0, 1.0, 0.05)
        self.sp_mj.setToolTip(
            "Saut maximal autorisé entre positions kF consécutives (π/a).\n"
            "Contrôle la continuité de la dispersion lors du fit complet.\n"
            "Réduire si la dispersion saute d'un point à l'autre."
        )
        self.cmb_sd  = QComboBox(); self.cmb_sd.addItems(["up","down"])
        self.cmb_sd.setFixedWidth(80)
        self.cmb_sd.setToolTip(
            "up : parcourt la BM de ev_start (bas) vers ev_end (proche EF).\n"
            "down : sens inverse. Choisir le sens où les pics sont les plus nets en départ."
        )

        for w in (self.sp_sff, self.sp_sfd, self.sp_kfi, self.sp_gi, self.sp_gm,
                  self.sp_xg, self.sp_cx, self.sp_k0m, self.sp_ma, self.sp_mj):
            w.valueChanged.connect(self.fit_only_changed)
        self.cmb_wm.currentIndexChanged.connect(self.fit_only_changed)

        self.sp_dE_meV = _dspin(15.0, 1.0, 200.0, 1.0, dec=1)
        self.sp_dE_meV.setToolTip(
            "FWHM énergie instrumentale estimée ou saisie manuellement (meV).\n"
            "Utilisée pour calculer Γ corrigé après fit MDC."
        )
        self.sp_dk_inv_a = _dspin(0.005, 0.001, 0.1, 0.001, dec=4)
        self.sp_dk_inv_a.setToolTip(
            "FWHM k instrumentale en π/a, estimée depuis angle_step si disponible.\n"
            "Utilisée pour calculer Γ corrigé après fit MDC."
        )
        self.lbl_dE_src = QLabel("—")
        self.lbl_dk_src = QLabel("—")
        for lbl in (self.lbl_dE_src, self.lbl_dk_src):
            lbl.setToolTip("Provenance résolution : 🔬 = estimée, ✏️ = manuelle, — = défaut")
        self.sp_dE_meV.valueChanged.connect(self._mark_resolution_manual_if_user_edit)
        self.sp_dk_inv_a.valueChanged.connect(self._mark_resolution_manual_if_user_edit)
        self.sp_dE_meV.valueChanged.connect(self.fit_only_changed)
        self.sp_dk_inv_a.valueChanged.connect(self.fit_only_changed)

        k0w = QWidget(); k0l = QHBoxLayout(k0w); k0l.setContentsMargins(0,0,0,0)
        k0l.addWidget(self.sp_k0m); k0l.addWidget(self.chk_k0a)

        fl3.addRow("Nb paires:",        self.sp_np)
        fl3.addRow("Lissage fit σ:",    self.sp_sff)
        fl3.addRow("Lissage détect σ:", self.sp_sfd)
        fl3.addRow(_sep())
        fl3.addRow(self._pair_lbl)
        fl3.addRow("kF init (π/a):",    self.sp_kfi)
        fl3.addRow("γ init (π/a):",     self.sp_gi)
        fl3.addRow("γ max (π/a):",      self.sp_gm)
        fl3.addRow(_sep())
        fl3.addRow("Fenêtre Γ (π/a):",  self.sp_xg)
        fl3.addRow("Centre Γ (π/a):",   self.sp_cx)
        fl3.addRow("kF max (π/a):",     k0w)
        fl3.addRow("Symétrie paire:",   self.cmb_wm)
        fl3.addRow(_sep())
        fl3.addRow("Ampl. min:",        self.sp_ma)
        fl3.addRow("Saut max (π/a):",   self.sp_mj)
        fl3.addRow("Sens scan:",        self.cmb_sd)
        fl3.addRow(_sep())
        de_row = QWidget(); de_lay = QHBoxLayout(de_row); de_lay.setContentsMargins(0,0,0,0)
        de_lay.addWidget(self.sp_dE_meV, 1); de_lay.addWidget(self.lbl_dE_src)
        dk_row = QWidget(); dk_lay = QHBoxLayout(dk_row); dk_lay.setContentsMargins(0,0,0,0)
        dk_lay.addWidget(self.sp_dk_inv_a, 1); dk_lay.addWidget(self.lbl_dk_src)
        fl3.addRow("ΔE FWHM (meV):", de_row)
        fl3.addRow("Δk FWHM (π/a):", dk_row)
        _fcl.addWidget(grp_f)

        # ── waterfall MDC (visible seulement dans le sous-onglet Waterfall) ────
        self._waterfall_controls_widget = QGroupBox("Waterfall MDC")
        fl_wf = QFormLayout(self._waterfall_controls_widget)
        self.sp_wf_n = _ispin(32, 10, 80)
        self.sp_wf_n.setToolTip(
            "Nombre cible de MDCs affichées dans le waterfall.\n"
            "Moins de courbes = plus de relief et moins de surcharge."
        )
        self.sp_wf_relief = _dspin(1.8, 0.5, 4.0, 0.1, dec=1)
        self.sp_wf_relief.setToolTip(
            "Amplitude visuelle des MDCs dans le waterfall.\n"
            "Augmenter pour mieux voir les pics ; trop haut crée du chevauchement."
        )
        self.sp_wf_n.valueChanged.connect(self.fit_only_changed)
        self.sp_wf_relief.valueChanged.connect(self.fit_only_changed)
        fl_wf.addRow("Courbes:", self.sp_wf_n)
        fl_wf.addRow("Relief:", self.sp_wf_relief)
        self._waterfall_controls_widget.setVisible(False)
        _fcl.addWidget(self._waterfall_controls_widget)

        # ── boutons ───────────────────────────────────────────────────────────
        _fcl.addWidget(_sep())
        btn_g = QPushButton("🎯  Guess  (fit MDC ici)  [Ctrl+G]")
        btn_g.setStyleSheet("background:#1a6b3a;color:white;font-weight:bold;padding:6px;")
        btn_g.clicked.connect(self.guess_requested)
        _fcl.addWidget(btn_g)

        self._gamma_tools_widget = QWidget()
        gamma_lay = QVBoxLayout(self._gamma_tools_widget)
        gamma_lay.setContentsMargins(0, 0, 0, 0)
        gamma_lay.setSpacing(4)
        btn_gamma = QPushButton("◎  Auto Γ BM")
        btn_gamma.setToolTip("Estime le centre Γ par la médiane des milieux de paires MDC.")
        btn_gamma.clicked.connect(self.gamma_bm_requested)
        gamma_lay.addWidget(btn_gamma)

        btn_ref = QPushButton("↳  Γ FS → BM")
        btn_ref.setToolTip("Applique le Γ de référence mesuré sur une FS à la BM courante.")
        btn_ref.clicked.connect(self.gamma_ref_requested)
        gamma_lay.addWidget(btn_ref)
        _fcl.addWidget(self._gamma_tools_widget)

        btn_f = QPushButton("▶  Fit complet  [Ctrl+F]")
        btn_f.setStyleSheet("background:#2a6099;color:white;font-weight:bold;padding:6px;")
        btn_f.clicked.connect(self.full_fit_requested)
        _fcl.addWidget(btn_f)

        btn_cl = QPushButton("✕  Effacer kF")
        btn_cl.clicked.connect(self.clear_kf_requested)
        _fcl.addWidget(btn_cl)

        self.lbl_res = QLabel("—")
        self.lbl_res.setWordWrap(True)
        self.lbl_res.setStyleSheet("color:#8fc;font-family:monospace;font-size:11px;")
        _fcl.addWidget(self.lbl_res)

        lay.addWidget(self._fit_controls_widget)
        lay.addStretch()

    # ── accès params ──────────────────────────────────────────────────────────
    def update_ef_reference_button(self, ref: dict | None):
        """Met à jour le label/état du bouton EF réf selon la session."""
        if not ref:
            self.btn_ef_ref.setText("⇩  Aucune réf EF (calibrer un Au d'abord)")
            self.btn_ef_ref.setEnabled(False)
            self.btn_ef_ref.setToolTip(
                "Aucune référence EF enregistrée dans cette session.\n"
                "Pour en créer une : 'Calibrer EF auto' sur un scan Au, "
                "puis cocher 'Enregistrer comme référence' dans le dialog."
            )
            return
        mode = ref.get("mode", "?")
        src_path = ref.get("source_file", "")
        src_name = Path(src_path).name if src_path else "(source inconnue)"
        if mode == "scalar":
            shift_meV = float(ref.get("ef_shift", 0.0)) * 1000.0
            label = f"⇩  Appliquer EF réf : {src_name} (Δ={shift_meV:+.1f} meV)"
        elif mode == "poly":
            n_valid = int(ref.get("n_valid", 0))
            fwhm = float(ref.get("fwhm_res", 0.0)) * 1000.0
            label = f"⇩  Appliquer EF réf poly : {src_name} (n={n_valid}, FWHM≈{fwhm:.0f} meV)"
        else:
            label = f"⇩  Appliquer EF réf : {src_name}"
        self.btn_ef_ref.setText(label)
        self.btn_ef_ref.setEnabled(True)
        self.btn_ef_ref.setToolTip(
            f"Référence EF enregistrée :\n"
            f"  mode = {mode}\n"
            f"  source = {src_path or '?'}\n"
            f"Applique cette correction au fichier courant."
        )

    def update_hv_source(self, source: str | None):
        """Affiche la provenance de hν : 'file', 'logbook', 'manual', None."""
        icons = {"file": "📁", "logbook": "📋", "manual": "✏️"}
        self.lbl_hv_src.setText(icons.get(source or "", "—"))

    def _mark_hv_manual_if_user_edit(self):
        if not getattr(self, "_hv_source_lock", False):
            self.update_hv_source("manual")

    def set_hv_value_with_source(self, value: float, source: str):
        """Set la spinbox hν sans déclencher le marquage 'manuel'."""
        self._hv_source_lock = True
        try:
            self.sp_hv.blockSignals(True)
            self.sp_hv.setValue(float(value))
            self.sp_hv.blockSignals(False)
            self.update_hv_source(source)
        finally:
            self._hv_source_lock = False

    def update_resolution_source(self, source: str | None):
        """Affiche la provenance de la resolution : 'estimated', 'manual', 'default'."""
        self._resolution_source = source or "default"
        self._resolution_source_detail = self._resolution_source
        icon = {"estimated": "🔬", "manual": "✏️", "default": "—"}.get(self._resolution_source, "—")
        self.lbl_dE_src.setText(icon)
        self.lbl_dk_src.setText(icon)

    def _mark_resolution_manual_if_user_edit(self):
        if not getattr(self, "_resolution_source_lock", False):
            self.update_resolution_source("manual")
            self._resolution_source_detail = "manual"

    def set_resolution_with_source(self, dE_meV: float, dk_inv_a: float, source: str, detail: str | None = None):
        """Set les spinboxes resolution sans déclencher le marquage manuel."""
        self._resolution_source_lock = True
        try:
            for sp, value in ((self.sp_dE_meV, dE_meV), (self.sp_dk_inv_a, dk_inv_a)):
                sp.blockSignals(True)
                sp.setValue(float(value))
                sp.blockSignals(False)
            self.update_resolution_source(source)
            self._resolution_source_detail = detail or source
        finally:
            self._resolution_source_lock = False

    def update_copy_params_button(self, n_targets: int):
        """Met à jour le label/état du bouton 'Propager fit params'."""
        if n_targets <= 0:
            self.btn_copy.setText("📋  Propager fit params (0 cible)")
            self.btn_copy.setEnabled(False)
            self.btn_copy.setToolTip(
                "Aucun fichier non-fitté dans le dossier (hors fichier courant).\n"
                "Tous les autres ont déjà un fit_result enregistré : ils ne seront pas écrasés."
            )
        else:
            self.btn_copy.setText(f"📋  Propager fit params ({n_targets} cible{'s' if n_targets > 1 else ''})")
            self.btn_copy.setEnabled(True)
            self.btn_copy.setToolTip(
                f"Copie les paramètres de fit MDC actuels vers les {n_targets} "
                f"fichier(s) du dossier qui n'ont pas encore été fittés.\n"
                f"Les fichiers déjà fittés ne sont jamais écrasés."
            )

    def get_fit_params(self) -> FitParams:
        self._save_pair()
        p0 = self._pair_params[0] if self._pair_params else {}
        return FitParams(
            n_pairs       = self.sp_np.value(),
            ev_start      = self.sp_evs.value(),
            ev_end        = self.sp_eve.value(),
            k_min         = self.sp_kmin.value(),
            k_max         = self.sp_kmax.value(),
            smooth_fit    = self.sp_sff.value(),
            smooth_detect = self.sp_sfd.value(),
            gamma_init    = p0.get("gamma_init", 0.08),
            gamma_max     = p0.get("gamma_max",  0.30),
            xg_range      = self.sp_xg.value(),
            center_init   = self.sp_cx.value(),
            k0_max        = None if self.chk_k0a.isChecked() else self.sp_k0m.value(),
            width_mode    = self.cmb_wm.currentText(),
            min_amplitude = self.sp_ma.value(),
            max_jump      = self.sp_mj.value(),
            scan_direction= self.cmb_sd.currentText(),
            dE_meV        = self.sp_dE_meV.value(),
            dk_inv_a      = self.sp_dk_inv_a.value(),
            pairs         = [dict(p) for p in self._pair_params],
        )

    def set_fit_controls_visible(self, visible: bool):
        self._fit_controls_widget.setVisible(visible)

    def set_utilities_visible(self, visible: bool):
        self._utils_widget.setVisible(visible)

    def set_fit_roi_active(self, active: bool):
        self.btn_fit_roi.blockSignals(True)
        self.btn_fit_roi.setChecked(bool(active))
        self.btn_fit_roi.blockSignals(False)

    def set_context(self, context: str):
        """Adapte le panneau droit à l'onglet actif."""
        is_bm = context == "bm"
        is_mdc = context == "mdc"
        self._energy_widget.setVisible(is_bm)
        self._ef_widget.setVisible(is_bm)
        self._utils_widget.setVisible(is_bm)
        self._fit_controls_widget.setVisible(is_mdc)
        self._gamma_tools_widget.setVisible(False)
        if not is_mdc:
            self.set_waterfall_controls_visible(False)

    def set_waterfall_controls_visible(self, visible: bool):
        self._waterfall_controls_widget.setVisible(bool(visible))

    def grid_params(self) -> dict:
        return {
            "enabled": True,
            "method": "display_fft2mask",
            "grid_period_px": None,
            "grid_freq": None,
            "notch_width": 2,
            "notch_sigma": 0.8,
            "strength": float(self.sp_grid_strength.value()),
            "fft2_center_radius": 18.0,
            "fft2_peak_sensitivity": 2.5,
            "fft2_plane": "display",
        }

    def load_fit_params(self, fp: FitParams):
        for sp, val in [
            (self.sp_evs,  fp.ev_start),  (self.sp_eve,  fp.ev_end),
            (self.sp_kmin, fp.k_min),     (self.sp_kmax, fp.k_max),
            (self.sp_sff,  fp.smooth_fit),(self.sp_sfd,  fp.smooth_detect),
            (self.sp_xg,   fp.xg_range),  (self.sp_cx,   fp.center_init),
            (self.sp_ma,   fp.min_amplitude),(self.sp_mj, fp.max_jump),
            (self.sp_dE_meV, getattr(fp, "dE_meV", 15.0)),
            (self.sp_dk_inv_a, getattr(fp, "dk_inv_a", 0.005)),
        ]:
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        if fp.k0_max is not None:
            self.chk_k0a.setChecked(False); self.sp_k0m.setValue(fp.k0_max)
        else:
            self.chk_k0a.setChecked(True)
        self.cmb_wm.setCurrentText(fp.width_mode)
        self.cmb_sd.setCurrentText(fp.scan_direction)

        # ── paires ────────────────────────────────────────────────────────────
        n = fp.n_pairs
        raw = list(getattr(fp, "pairs", None) or [])
        if not raw:
            raw = [{"kF_init": 0.30, "gamma_init": fp.gamma_init, "gamma_max": fp.gamma_max}]
        while len(raw) < n:
            raw.append(dict(raw[-1]))
        self._pair_params = raw[:max(n, 1)]
        self._current_pair = 0
        self.sp_np.blockSignals(True); self.sp_np.setValue(n); self.sp_np.blockSignals(False)
        self._pair_lbl.setup(n, 0)
        self._load_pair(0)

    # ── gestion par paire ─────────────────────────────────────────────────────
    def _save_pair(self):
        i = self._current_pair
        if i < len(self._pair_params):
            self._pair_params[i] = {
                "kF_init":    self.sp_kfi.value(),
                "gamma_init": self.sp_gi.value(),
                "gamma_max":  self.sp_gm.value(),
            }

    def _load_pair(self, i: int):
        i = max(0, min(i, len(self._pair_params) - 1))
        p = self._pair_params[i]
        for sp, key, default in [
            (self.sp_kfi, "kF_init",    0.30),
            (self.sp_gi,  "gamma_init", 0.08),
            (self.sp_gm,  "gamma_max",  0.30),
        ]:
            sp.blockSignals(True); sp.setValue(p.get(key, default)); sp.blockSignals(False)

    def _on_n_pairs_changed(self, n: int):
        self._save_pair()
        default = dict(self._pair_params[-1]) if self._pair_params else \
                  {"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}
        while len(self._pair_params) < n:
            self._pair_params.append(dict(default))
        self._pair_params = self._pair_params[:max(n, 1)]
        self._current_pair = min(self._current_pair, n - 1)
        self._pair_lbl.setup(n, self._current_pair)
        self._load_pair(self._current_pair)
        self.params_changed.emit()

    def _on_pair_changed(self, i: int):
        self._save_pair()
        self._current_pair = i
        self._load_pair(i)


# ─────────────────────────────────────────────────────────────────────────────
# Onglet Résultats
# ─────────────────────────────────────────────────────────────────────────────

class ResultsPanel(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)

        # canvas dispersion
        self._canvas = MplCanvas(figsize=(6, 5))
        lay.addWidget(self._canvas, stretch=2)

        # droite : table + boutons
        right = QVBoxLayout()
        right.addWidget(QLabel("Résultats fittés"))

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["Fichier", "hν", "T (K)", "Dir.", "kF+ (π/a)", "xg (π/a)", "Γ brut", "Γ corr."])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setStyleSheet(
            "QTableWidget{background:#222;color:#ddd;font-size:10px;}"
            "QHeaderView::section{background:#333;color:#ddd;}")
        right.addWidget(self._table, stretch=1)

        btn_ref = QPushButton("🔄  Actualiser")
        btn_ref.clicked.connect(self.refresh)
        btn_csv = QPushButton("💾  Export CSV")
        btn_csv.clicked.connect(self._export_csv)
        btn_pdf = QPushButton("🖼  Export figure")
        btn_pdf.clicked.connect(self._export_fig)
        for b in (btn_ref, btn_csv, btn_pdf):
            right.addWidget(b)

        rw = QWidget(); rw.setLayout(right)
        rw.setMaximumWidth(350)
        lay.addWidget(rw, stretch=1)

    def refresh(self):
        self._table.setRowCount(0)
        ax = self._canvas.ax
        ax.cla(); ax.set_facecolor("#1a1a1a")
        self._canvas.fig.set_facecolor("#2b2b2b")

        colors = plt.cm.plasma(np.linspace(0.1, 0.9,
                                           max(1, len(self._session.files))))
        row = 0
        for ci, (name, entry) in enumerate(self._session.files.items()):
            if entry.fit_result is None:
                continue
            fr   = entry.fit_result
            meta = entry.meta
            ev_f = np.asarray(fr["e_fitted"])
            n    = entry.fit_params.n_pairs
            label = f"{name}  T={meta.temperature:.0f}K  {meta.direction}"
            c = colors[ci]

            for i in range(n):
                km = np.asarray(fr["kF_minus"][i]) if i < len(fr["kF_minus"]) else []
                kp = np.asarray(fr["kF_plus"][i])  if i < len(fr["kF_plus"])  else []
                ax.scatter(km, ev_f, s=8, color=c, marker="o", alpha=0.8,
                           label=label if i == 0 else "_")
                ax.scatter(kp, ev_f, s=8, color=c, marker="^", alpha=0.8)

            # Table row
            kf_ef = np.nan
            if len(fr["kF_plus"]) > 0:
                idx_ef = np.argmin(np.abs(ev_f))
                kf_arr = np.asarray(fr["kF_plus"][0])
                if len(kf_arr) > idx_ef:
                    kf_ef = kf_arr[idx_ef]
            xg_m = float(np.nanmean(fr.get("xg", [np.nan])))
            gamma_b = np.nan
            gamma_c = np.nan
            if fr.get("gamma_brut"):
                gamma_b = float(np.nanmedian(np.asarray(fr["gamma_brut"][0], dtype=float)))
            if fr.get("gamma_corrige"):
                gamma_c = float(np.nanmedian(np.asarray(fr["gamma_corrige"][0], dtype=float)))

            self._table.insertRow(row)
            for col, val in enumerate([
                name, f"{meta.hv:.0f}", f"{meta.temperature:.0f}",
                meta.direction, f"{kf_ef:.4f}", f"{xg_m:.4f}",
                f"{gamma_b:.4f}", f"{gamma_c:.4f}",
            ]):
                self._table.setItem(row, col, QTableWidgetItem(val))
            row += 1

        ax.axhline(0, color="cyan", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="w",    lw=0.5, ls="--", alpha=0.3)
        ax.set_xlabel("k// (π/a)", fontsize=10, color="w")
        ax.set_ylabel("E − EF (eV)", fontsize=10, color="w")
        ax.set_title("Dispersions kF — tous fichiers fittés", fontsize=10, color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        if row > 0:
            ax.legend(fontsize=8, facecolor="#333", labelcolor="w",
                      loc="upper right", markerscale=2)
        self._canvas.redraw()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(self._session.folder or Path.home()),
            "CSV (*.csv)")
        if not path:
            return
        rows = result_rows(self._session)
        if not rows:
            return
        write_results_csv(path, rows)

    def _export_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", str(self._session.folder or Path.home()),
            "PDF (*.pdf);;PNG (*.png)")
        if path:
            self._canvas.fig.savefig(path, dpi=200, bbox_inches="tight",
                                     facecolor=self._canvas.fig.get_facecolor())


import matplotlib.pyplot as plt   # pour plt.cm dans ResultsPanel


# ─────────────────────────────────────────────────────────────────────────────
# Dialogue de calibration EF
# ─────────────────────────────────────────────────────────────────────────────

class EFCalibrationDialog(QDialog):
    """Calibration EF interactive : scalaire ou par colonne (poly).

    Inputs : data (n_k, n_E), kpar, ev_arr, T_init, half_width_init, source_name.
    Outputs (via .result_payload après accept) :
        {"mode": "scalar"|"poly", "ef_offset": float | None,
         "poly_coefs": [...] | None, "T": float, "fwhm_res": float,
         "rms": float, "n_valid": int, "k_min": float, "k_max": float,
         "save_as_reference": bool}
    """

    def __init__(self, parent, data, kpar, ev_arr, T_init=28.0,
                 half_width_init=0.15, source_name="", current_offset=0.0,
                 metadata: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Calibration EF")
        self.resize(900, 620)
        self._data  = np.asarray(data, dtype=float)
        self._kpar  = np.asarray(kpar, dtype=float)
        self._ev    = np.asarray(ev_arr, dtype=float)
        self._fit   = None
        self.result_payload = None
        self._current_offset = float(current_offset)
        self._metadata = metadata or {}
        self._ef_search = self._default_ef_search_range()

        # ── widgets ────────────────────────────────────────────────────────────
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        lay = QHBoxLayout(self)

        # Panneau gauche : contrôles
        left = QWidget(); fl = QFormLayout(left); left.setMaximumWidth(320)
        info = QLabel(f"Source : {source_name or '—'}\nDimensions : {self._data.shape[0]} k × {self._data.shape[1]} E")
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        fl.addRow(info)

        self.rb_scalar = QRadioButton("Scalaire (un EF moyen)")
        self.rb_poly   = QRadioButton("Par colonne (polynôme)")
        self.rb_scalar.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_scalar); grp.addButton(self.rb_poly)
        fl.addRow(self.rb_scalar)
        fl.addRow(self.rb_poly)

        self.sp_T = QDoubleSpinBox(); self.sp_T.setRange(1.0, 400.0); self.sp_T.setDecimals(1)
        self.sp_T.setValue(float(T_init)); self.sp_T.setSuffix(" K")
        self.sp_T.setToolTip("Température utilisée pour fixer kBT dans la FD.")
        fl.addRow("Température :", self.sp_T)

        self.sp_hw = QDoubleSpinBox(); self.sp_hw.setRange(0.03, 0.50); self.sp_hw.setDecimals(3)
        self.sp_hw.setSingleStep(0.01); self.sp_hw.setValue(float(half_width_init)); self.sp_hw.setSuffix(" eV")
        self.sp_hw.setToolTip("Demi-largeur de la fenêtre de fit autour de EF estimé.")
        fl.addRow("Demi-fenêtre :", self.sp_hw)

        self.chk_auto = QCheckBox("Auto-fenêtre (gradient max)")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "Centre la fenêtre sur le gradient max de l'EDC moyenne.\n"
            f"Recherche actuelle : {self._ef_search[0]:+.2f} à {self._ef_search[1]:+.2f} eV."
        )
        fl.addRow(self.chk_auto)

        self.sp_deg = QSpinBox(); self.sp_deg.setRange(0, 4); self.sp_deg.setValue(2)
        self.sp_deg.setToolTip("Degré du polynôme EF(k). 0=constant, 2=parabole (défaut).")
        fl.addRow("Degré poly :", self.sp_deg)

        self.sp_sigma = QDoubleSpinBox(); self.sp_sigma.setRange(0.005, 0.10); self.sp_sigma.setDecimals(3)
        self.sp_sigma.setSingleStep(0.005); self.sp_sigma.setValue(0.025); self.sp_sigma.setSuffix(" eV")
        self.sp_sigma.setToolTip("Sigma initiale de résolution gaussienne pour le fit (FWHM=2.355σ).")
        fl.addRow("σ init :", self.sp_sigma)

        self.btn_fit = QPushButton("▶ Fitter")
        self.btn_fit.clicked.connect(self._do_fit)
        fl.addRow(self.btn_fit)

        self.lbl_result = QLabel("—")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setStyleSheet("background:#222; padding:6px; border-radius:4px;")
        fl.addRow(self.lbl_result)

        self.chk_save_ref = QCheckBox("Sauvegarder comme référence dossier (Au)")
        self.chk_save_ref.setToolTip(
            "La correction sera proposée comme référence appliquable aux autres\n"
            "fichiers du dossier via 'Appliquer Au de référence'."
        )
        fl.addRow(self.chk_save_ref)

        self.btn_apply  = QPushButton("✓ Appliquer à ce fichier")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.clicked.connect(self.reject)
        fl.addRow(self.btn_apply)
        fl.addRow(self.btn_cancel)

        lay.addWidget(left)

        # Panneau droit : preview matplotlib
        from matplotlib.figure import Figure as _Fig
        self._fig = _Fig(figsize=(6, 5))
        self._canvas = FigureCanvas(self._fig)
        self._ax_edc  = self._fig.add_subplot(2, 1, 1)
        self._ax_poly = self._fig.add_subplot(2, 1, 2)
        self._fig.tight_layout()
        right = QWidget(); rl = QVBoxLayout(right); rl.addWidget(self._canvas)
        lay.addWidget(right, 1)

        self._draw_initial_preview()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _default_ef_search_range(self) -> tuple[float, float]:
        fmt = str(self._metadata.get("fs_source") or self._metadata.get("source_format") or "").lower()
        lab = str(self._metadata.get("lab") or "").lower()
        ref = str(self._metadata.get("energy_reference") or "").lower()
        # BESSY Center Energy mode: l'expérimentateur peut avoir centré l'analyseur
        # à n'importe quel offset d'EF. On vise donc d'abord le bord détecté dans
        # l'EDC (max drop d'intensité), puis on élargit autour. Si la détection
        # échoue, fallback large couvrant tout l'axe.
        if "bessy" in fmt or "bessy" in lab or ref == "ses_center_energy":
            try:
                edc = np.nanmean(self._data, axis=0)
                e = np.asarray(self._ev, dtype=float)
                grad = np.gradient(edc, e)
                drop_idx = int(np.nanargmin(grad))
                ef_hint = float(e[drop_idx])
                e_min, e_max = float(e.min()), float(e.max())
                lo = max(e_min, ef_hint - 0.30)
                hi = min(e_max, ef_hint + 0.20)
                if hi - lo < 0.15:
                    return (e_min, e_max)
                return (lo, hi)
            except Exception:
                return (-0.5, 0.5)
        return (-0.5, 0.2)

    def _draw_initial_preview(self):
        edc = np.nanmean(self._data, axis=0)
        self._ax_edc.clear()
        self._ax_edc.plot(self._ev, edc, "k-", lw=1.2, label="EDC moyenne")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_edc.axvspan(self._ef_search[0], self._ef_search[1], color="orange", alpha=0.08,
                             label="recherche EF")
        self._ax_edc.set_xlabel("E − EF (eV)"); self._ax_edc.set_ylabel("Intensité")
        self._ax_edc.set_title("EDC moyennée sur k")
        self._ax_edc.legend(fontsize=8)
        self._ax_poly.clear()
        self._ax_poly.text(0.5, 0.5, "Cliquer 'Fitter' pour lancer la calibration",
                           ha="center", va="center", transform=self._ax_poly.transAxes,
                           fontsize=10, color="gray")
        self._ax_poly.set_axis_off()
        self._canvas.draw_idle()

    def _do_fit(self):
        try:
            ap = _load_ap()
        except Exception as e:
            self.lbl_result.setText(f"⚠ arpes_plots indisponible : {e}")
            return
        T  = self.sp_T.value()
        hw = self.sp_hw.value()
        sig = self.sp_sigma.value()
        auto = self.chk_auto.isChecked()
        edc = np.nanmean(self._data, axis=0)

        if self.rb_scalar.isChecked():
            win = ap.auto_ef_window(self._ev, edc, half_width=hw, search=self._ef_search) if auto else (-hw, hw)
            try:
                from matplotlib.figure import Figure as _Fig
                _ax = _Fig().add_subplot(111)
                r = ap.fit_fermi_edge(
                    self._ev, edc,
                    temperature_K=T, fit_range=win,
                    sigma_resolution_init=sig, fix_kBT=True,
                    units="binding", ax=_ax, verbose=False,
                )
            except Exception as e:
                self.lbl_result.setText(f"⚠ fit échoué : {e}")
                return
            ef     = float(r["EF"])
            efe    = float(r.get("EF_err", np.nan))
            fwhm   = float(r["fwhm_res"])
            resid  = float(r["residual"])
            self._fit = {
                "mode": "scalar",
                "ef_shift": ef,
                "ef_err":   efe,
                "fwhm_res": fwhm,
                "rms":      resid,
                "n_valid":  int(self._data.shape[0]),
                "k_min":    float(self._kpar.min()),
                "k_max":    float(self._kpar.max()),
                "T":        T,
                "window":   win,
            }
            self._draw_scalar_preview(r, win)
            new_offset = self._current_offset - ef
            self.lbl_result.setText(
                f"<b>Mode scalaire</b><br>"
                f"EF fit : {ef*1000:+.1f} meV (±{efe*1000:.1f} meV)<br>"
                f"FWHM résolution : {fwhm*1000:.0f} meV<br>"
                f"Résidu rms : {resid:.4f}<br>"
                f"Fenêtre : [{win[0]*1000:+.0f}, {win[1]*1000:+.0f}] meV<br>"
                f"→ nouvel offset proposé : {new_offset:.4f} eV"
            )
        else:
            try:
                r = ap.fit_fermi_edge_per_column(
                    self._data, self._kpar, self._ev,
                    temperature_K=T, half_width=hw,
                    sigma_resolution_init=sig,
                    poly_deg=self.sp_deg.value(),
                    auto_window=auto,
                    ef_search=self._ef_search,
                    verbose=False,
                )
            except Exception as e:
                self.lbl_result.setText(f"⚠ fit par colonne échoué : {e}")
                return
            self._fit = {
                "mode": "poly",
                "poly_coefs": r["poly_coefs"].tolist(),
                "ef_per_col": r["ef_per_col"],
                "ef_smooth":  r["ef_smooth"],
                "fwhm_res":   r["mean_fwhm"],
                "rms":        r["rms"],
                "n_valid":    r["n_valid"],
                "k_min":      float(self._kpar.min()),
                "k_max":      float(self._kpar.max()),
                "T":          T,
                "window":     r["window"],
                "mean_ef":    r["mean_ef"],
            }
            self._draw_poly_preview(r)
            self.lbl_result.setText(
                f"<b>Mode par colonne (poly deg {self.sp_deg.value()})</b><br>"
                f"Colonnes valides : {r['n_valid']}/{self._data.shape[0]}<br>"
                f"&lt;EF&gt; : {r['mean_ef']*1000:+.1f} meV<br>"
                f"FWHM médian : {r['mean_fwhm']*1000:.0f} meV<br>"
                f"RMS résidu poly : {r['rms']*1000:.1f} meV<br>"
                f"Fenêtre : [{r['window'][0]*1000:+.0f}, {r['window'][1]*1000:+.0f}] meV"
            )
        self.btn_apply.setEnabled(True)

    def _draw_scalar_preview(self, fit_result, win):
        self._ax_edc.clear()
        edc = np.nanmean(self._data, axis=0)
        self._ax_edc.plot(self._ev, edc / max(np.nanmax(edc), 1e-9), "k-", lw=1.0,
                          label="EDC normée")
        self._ax_edc.plot(fit_result["model_ev"], fit_result["model_I"], "r-", lw=2.0,
                          label=f"FD fit  EF={fit_result['EF']*1000:+.0f} meV")
        self._ax_edc.axvline(fit_result["EF"], color="red", lw=1.0)
        self._ax_edc.axvspan(win[0], win[1], color="orange", alpha=0.10, label="fenêtre")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_edc.set_xlim(min(win[0]-0.05, -0.4), max(win[1]+0.05, 0.1))
        self._ax_edc.set_xlabel("E − EF (eV)"); self._ax_edc.set_ylabel("I norm.")
        self._ax_edc.legend(fontsize=8)
        self._ax_poly.clear()
        self._ax_poly.text(0.5, 0.5,
                           "Mode scalaire : pas de courbe EF(k).\n"
                           "Passer en 'Par colonne' pour voir la dispersion.",
                           ha="center", va="center", transform=self._ax_poly.transAxes,
                           fontsize=9, color="gray")
        self._ax_poly.set_axis_off()
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _draw_poly_preview(self, r):
        self._ax_edc.clear()
        edc = np.nanmean(self._data, axis=0)
        self._ax_edc.plot(self._ev, edc / max(np.nanmax(edc), 1e-9), "k-", lw=1.0,
                          label="EDC moyenne")
        win = r["window"]
        self._ax_edc.axvspan(win[0], win[1], color="orange", alpha=0.10, label="fenêtre")
        self._ax_edc.axvline(r["mean_ef"], color="red", lw=1.0,
                             label=f"<EF>={r['mean_ef']*1000:+.0f} meV")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_edc.set_xlim(min(win[0]-0.05, -0.4), max(win[1]+0.05, 0.1))
        self._ax_edc.set_xlabel("E − EF (eV)"); self._ax_edc.set_ylabel("I norm.")
        self._ax_edc.legend(fontsize=8)

        self._ax_poly.clear()
        kp = self._kpar
        ef_raw = r["ef_per_col"]
        ef_sm  = r["ef_smooth"]
        valid = np.isfinite(ef_raw)
        self._ax_poly.plot(kp[valid], ef_raw[valid] * 1000, ".", color="#888", ms=3,
                           label="fits par colonne")
        self._ax_poly.plot(kp, ef_sm * 1000, "r-", lw=2.0,
                           label=f"poly deg {self.sp_deg.value()}")
        self._ax_poly.axhline(0.0, color="gray", ls="--", lw=0.7)
        self._ax_poly.set_xlabel("k (π/a)"); self._ax_poly.set_ylabel("EF (meV)")
        self._ax_poly.set_title("EF(k) — courbure du détecteur")
        self._ax_poly.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _on_apply(self):
        if not self._fit:
            return
        save_ref = bool(self.chk_save_ref.isChecked())
        if self._fit["mode"] == "scalar":
            self.result_payload = {
                "mode": "scalar",
                "ef_shift": self._fit["ef_shift"],
                "T": self._fit["T"],
                "fwhm_res": self._fit["fwhm_res"],
                "rms": self._fit["rms"],
                "k_min": self._fit["k_min"],
                "k_max": self._fit["k_max"],
                "save_as_reference": save_ref,
            }
        else:
            self.result_payload = {
                "mode": "poly",
                "poly_coefs": list(self._fit["poly_coefs"]),
                "T": self._fit["T"],
                "fwhm_res": self._fit["fwhm_res"],
                "rms": self._fit["rms"],
                "n_valid": self._fit["n_valid"],
                "k_min": self._fit["k_min"],
                "k_max": self._fit["k_max"],
                "save_as_reference": save_ref,
            }
        self.accept()


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
        meta = (self._raw_data or {}).get("metadata", {}) or {}
        return meta.get("fs_data") is not None

    def _on_fs_params_changed(self):
        self._save_current_fs_center()
        self._draw_fs_tab()

    def _save_current_fs_center(self):
        if self._raw_data is None or not self._current_path or not self._current_is_fs():
            return
        if FSControlPanel is None or not hasattr(self, "_fs_controls"):
            return
        entry = self._current_entry()
        if entry is None:
            return
        try:
            p = self._fs_controls.params()
            entry.fs_center_kx = float(p.kx_center)
            entry.fs_center_ky = float(p.ky_center)
            self._session.save()
        except Exception:
            pass

    def _same_path(self, a, b) -> bool:
        if not a or not b:
            return False
        try:
            return Path(a).resolve() == Path(b).resolve()
        except Exception:
            return str(a) == str(b)

    def _on_scroll_zoom(self, event):
        """Zoom molette centré sur la position du curseur dans un axe matplotlib."""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        try:
            xlim, ylim = _plot_scroll_zoom_limits(
                ax.get_xlim(),
                ax.get_ylim(),
                xdata=float(event.xdata),
                ydata=float(event.ydata),
                step=float(getattr(event, "step", 0.0) or 0.0),
                button=getattr(event, "button", ""),
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            event.canvas.draw_idle()
        except Exception:
            return

    def _draw_fs_tab(self):
        if not hasattr(self, "_fs_canvas") or FermiSurfaceCanvas is None:
            return
        if not hasattr(self, "_fs_controls") or FSControlPanel is None:
            return
        info = self._fs_canvas.draw_fs(self._raw_data, self._fs_controls.params())
        try:
            self._fs_controls.lbl_info.setText(info)
        except Exception:
            pass

    def _store_fs_center_reference(self, kx: float, ky: float, *, source: str):
        if self._raw_data is None:
            return
        meta = (self._raw_data or {}).get("metadata", {}) or {}
        entry_now = self._current_entry()
        azi_ref = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
        self._session.gamma_reference = _gamma_build_reference(
            kx=kx, ky=ky,
            metadata=meta,
            hv=self._raw_data.get("hv"),
            path=self._raw_data.get("path"),
            azi=azi_ref,
            source=source,
            direction=(entry_now.meta.direction if entry_now else None),
        )
        offsets = self._angle_offsets_from_k_center(
            float(kx), float(ky),
            hv=self._raw_data.get("hv"),
            source=source,
            ref_path=self._raw_data.get("path"),
            azi=azi_ref,
        )
        if offsets:
            self._session.angle_offsets = offsets
        if self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fs_center_kx = float(kx)
            entry.fs_center_ky = float(ky)
        self._session.save()

    def _k_to_angle_offset_deg(self, k_pi_a: float, *, hv: float | None = None) -> float | None:
        """Convertit un decalage k (pi/a) en offset angulaire CLS (wrapper UI)."""
        try:
            hv_val = float(hv if hv is not None else self._params.sp_hv.value())
            work_func = float(self._params.sp_phi.value())
        except Exception:
            return None
        return _gamma_k_to_angle_offset_deg(k_pi_a, hv=hv_val, work_func=work_func)

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
        try:
            hv_val = float(hv if hv is not None else self._params.sp_hv.value())
            work_func = float(self._params.sp_phi.value())
        except Exception:
            return {}
        return _gamma_angle_offsets_from_k_center(
            kx, ky,
            hv=hv_val, work_func=work_func,
            source=source, ref_path=ref_path, azi=azi,
        )

    def _project_gamma_by_azi(
        self,
        ref: dict,
        azi_target: float | None,
        *,
        warn_label: str = "Γ",
    ) -> tuple[float, float]:
        """Projette le Γ de référence dans le repère du fichier courant (wrapper UI)."""
        return _gamma_project_by_azi(
            ref, azi_target,
            on_warn=self._status,
            warn_label=warn_label,
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
        active = bool(active)
        self._fs_pick_center_active = active
        if hasattr(self, "_fs_controls") and hasattr(self._fs_controls, "set_manual_center_active"):
            self._fs_controls.set_manual_center_active(active)
        if hasattr(self, "_fs_canvas") and hasattr(self._fs_canvas, "canvas"):
            if active:
                self._fs_canvas.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._fs_canvas.canvas.unsetCursor()
        if active:
            if self._tabs.currentIndex() != 3:
                self._tabs.setCurrentIndex(3)
            self._status("Centrage manuel Γ : clique sur le centre à viser dans la carte FS.")

    def _on_fs_map_click(self, event):
        if not getattr(self, "_fs_pick_center_active", False):
            return
        if self._raw_data is None or not self._current_is_fs():
            self._set_fs_center_pick_mode(False)
            QMessageBox.warning(self, "Centrage manuel Γ", "Charge d'abord une FS.")
            return
        if FSControlPanel is None or not hasattr(self, "_fs_controls"):
            return
        if not hasattr(self, "_fs_canvas") or event.inaxes is not getattr(self._fs_canvas, "ax", None):
            return
        if event.xdata is None or event.ydata is None:
            return
        params = self._fs_controls.params()
        kx = float(params.kx_center + event.xdata)
        ky = float(params.ky_center + event.ydata)
        self._fs_controls.set_center(kx, ky)
        self._store_fs_center_reference(kx, ky, source="fs_manual")
        self._set_fs_center_pick_mode(False)
        self._draw_fs_tab()
        msg = f"Gamma FS manuel : kx={kx:+.4f}, ky={ky:+.4f} π/a"
        self._status(msg)
        try:
            self._fs_controls.lbl_info.setText(msg)
        except Exception:
            pass

    def _detect_fs_gamma(self):
        if FermiSurfaceCanvas is None or FSControlPanel is None:
            return
        try:
            params = self._fs_controls.params()
            res = self._fs_canvas.detect_gamma(self._raw_data, params)
            self._fs_controls.set_center(res["kx"], res["ky"])
            self._store_fs_center_reference(res["kx"], res["ky"], source="fs_auto")
            self._draw_fs_tab()
            msg = (f"Gamma FS détecté : kx={res['kx']:+.4f}, ky={res['ky']:+.4f} π/a "
                   f"| {len(res.get('gamma_kx_list', []))} coupes kx, "
                   f"{len(res.get('gamma_ky_list', []))} coupes ky")
            self._status(msg)
            try:
                self._fs_controls.lbl_info.setText(msg)
            except Exception:
                pass
        except Exception as exc:
            QMessageBox.warning(self, "Détection Gamma", str(exc))

    def _stored_gamma_reference(self) -> dict:
        return _gamma_stored_reference(self._session.gamma_reference)

    def _gamma_reference_to_bm_center(self, ref: dict) -> tuple[float, float]:
        """Wrapper UI : délègue à `arpes_gamma.gamma_reference_to_bm_center`."""
        if self._raw_data is None:
            return np.nan, 0.0
        meta = self._raw_data.get("metadata", {}) or {}
        entry_now = self._current_entry()
        azi_bm = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
        return _gamma_ref_to_bm_center(
            ref,
            bm_metadata=meta,
            bm_hv=self._raw_data.get("hv"),
            work_func=float(self._params.sp_phi.value()),
            bm_azi=azi_bm,
            on_warn=self._status,
        )

    def _center_current_bm_axis_on_gamma(self, gamma_bm: float, ref: dict | None = None) -> bool:
        """Wrapper UI : délègue à `arpes_gamma.apply_bm_gamma_axis_shift` puis
        synchronise la sélection MDC `_sel_k`."""
        if self._raw_data is None:
            return False
        applied = _gamma_apply_bm_axis_shift(self._raw_data, gamma_bm, ref=ref)
        if applied and hasattr(self, "_sel_k"):
            self._sel_k = float(self._sel_k - float(gamma_bm))
        return applied

    def _apply_stored_gamma_to_current_file(self, *, save_entry: bool = False):
        if self._raw_data is None:
            return

        meta = self._raw_data.get("metadata", {}) or {}
        is_fs = meta.get("fs_data") is not None

        if is_fs and FSControlPanel is not None and hasattr(self, "_fs_controls"):
            entry = self._current_entry()
            if meta.get("angle_offsets_applied"):
                self._fs_controls.set_center(0.0, 0.0)
                if save_entry and entry is not None:
                    entry.fs_center_kx = 0.0
                    entry.fs_center_ky = 0.0
                    self._session.save()
                return
            if entry is not None and entry.fs_center_kx is not None and entry.fs_center_ky is not None:
                self._fs_controls.set_center(float(entry.fs_center_kx), float(entry.fs_center_ky))
                return

        ref = self._stored_gamma_reference()
        if not ref:
            return

        if is_fs and FSControlPanel is not None and hasattr(self, "_fs_controls"):
            entry = self._current_entry()
            if self._same_path(ref.get("path"), self._raw_data.get("path")):
                kx_fs = float(ref["kx"])
                ky_fs = float(ref.get("ky", 0.0) or 0.0)
            else:
                azi_fs = entry.meta.azi if (entry and entry.meta.azi is not None) else None
                kx_fs, ky_fs = self._project_gamma_by_azi(
                    ref, azi_fs, warn_label="Γ référence → FS"
                )
                if not np.isfinite(kx_fs) or not np.isfinite(ky_fs):
                    return
            self._fs_controls.set_center(float(kx_fs), float(ky_fs))
            if save_entry and entry is not None:
                entry.fs_center_kx = float(kx_fs)
                entry.fs_center_ky = float(ky_fs)
                self._session.save()
            if not self._same_path(ref.get("path"), self._raw_data.get("path")):
                self._status(f"Γ FS propagé par azimut : kx={kx_fs:+.4f}, ky={ky_fs:+.4f} π/a")
            return

        if meta.get("angle_offsets_applied"):
            self._params.sp_cx.setValue(0.0)
            if save_entry and self._current_path:
                entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
                entry.fit_params.center_init = 0.0
                self._session.save()
            self._status("Γ mémorisé appliqué par offset angulaire : centre fit=0")
            return

        gamma_bm, correction = self._gamma_reference_to_bm_center(ref)
        if not np.isfinite(gamma_bm):
            return

        axis_centered = self._center_current_bm_axis_on_gamma(float(gamma_bm), ref)
        self._params.sp_cx.setValue(0.0 if axis_centered else float(gamma_bm))
        if save_entry and self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fit_params.center_init = 0.0 if axis_centered else float(gamma_bm)
            self._session.save()
        axis_msg = "axe k recentré" if axis_centered else "centre fit seul"
        self._status(
            f"Γ mémorisé appliqué : {gamma_bm:+.4f} π/a  correction={correction:+.4f}  |  {axis_msg}"
        )

    def _load_grid_controls(self, cfg: dict | None):
        cfg = cfg or {}
        self._params.sp_grid_strength.setValue(self._display_grid_config(cfg)["strength"])
        if cfg.get("enabled"):
            self._params.lbl_grid.setText("Correction BM active : masque Fourier 2D automatique.")
        else:
            self._params.lbl_grid.setText("Correction BM : masque Fourier 2D automatique sur l'affichage.")

    def _display_grid_config(self, cfg: dict | None) -> dict:
        return _plot_display_grid_config(cfg)

    def _grid_status_text(self, info: dict, target: str) -> str:
        info = info or {}
        method = info.get("method", "none")
        if info.get("error"):
            return f"Correction grille ({target}) impossible : {info.get('error')}"
        if method in {"fft2mask", "display_fft2mask"}:
            removed = int(info.get("removed_peak_count", 0) or 0)
            delta = float(info.get("rms_delta_percent", 0.0) or 0.0)
            view = info.get("view_mode")
            view_txt = f" {view}" if view else ""
            return (
                f"Correction grille ({target}{view_txt}) : masque Fourier 2D auto, "
                f"{removed} pics FFT, Δ≈{delta:.1f}%, force={float(info.get('strength', 1.0)):.2f}"
            )
        return f"Correction grille ({target}) active."

    def _apply_grid_correction(self):
        if self._raw_data is None or not self._current_path:
            QMessageBox.warning(self, "Effet grille", "Charge d'abord une BM ou une FS.")
            return
        cfg = self._params.grid_params()
        try:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.grid_correction = dict(cfg)
            self._session.save()
            self._update_display_data()
            self._draw_bm()
            if self._tabs.currentIndex() == 1:
                self._draw_mdc_edc()
            msg = self._grid_status_text(self._grid_display_info, "affichage BM")
            self._params.lbl_grid.setText(msg)
            self._status(msg)
        except Exception as exc:
            QMessageBox.warning(self, "Effet grille", str(exc))
            self._status(f"⚠ Effet grille : {exc}")

    def _reset_grid_correction(self):
        if not self._current_path:
            return
        entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
        entry.grid_correction = {}
        self._session.save()
        self._grid_display_info = {}
        self._update_display_data()
        self._draw_bm()
        if self._tabs.currentIndex() == 1:
            self._draw_mdc_edc()
        self._params.lbl_grid.setText("Correction grille désactivée pour ce fichier.")
        self._status("Correction grille désactivée pour ce fichier.")

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
        if self._raw_data is None:
            return
        d    = self._raw_data
        raw  = d["data"]
        mode = self._cmb_view.currentText()

        entry = self._current_entry()
        grid_cfg_active = entry.grid_correction if entry and entry.grid_correction.get("enabled") else None
        grid_key = (
            grid_cfg_active.get("strength"),
            grid_cfg_active.get("center_radius"),
            grid_cfg_active.get("peak_sensitivity"),
            grid_cfg_active.get("notch_width"),
        ) if grid_cfg_active else None
        cache_key = (id(raw), mode, bool(self._params.chk_norm.isChecked()), grid_key)
        if cache_key == self._disp_cache_key and self._data_disp is not None:
            return  # rien n'a changé qui affecte l'affichage BM

        result = compute_bandmap_display(
            d,
            mode=mode,
            edc_norm_enabled=bool(self._params.chk_norm.isChecked()),
            grid_correction=grid_cfg_active,
            grid_artifact_fn=remove_detector_grid_artifact,
        )
        self._data_disp = result.data
        self._grid_display_info = result.grid_info
        self._disp_cache_key = cache_key

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
        if self._raw_data is None:
            return None
        d = self._raw_data
        return _plot_fit_roi_bounds(
            d["kpar"], d["ev_arr"],
            k_min=self._params.sp_kmin.value(),
            k_max=self._params.sp_kmax.value(),
            ev_start=self._params.sp_evs.value(),
            ev_end=self._params.sp_eve.value(),
        )

    def _fit_roi_data(self, disp: np.ndarray, kpar: np.ndarray, ev: np.ndarray) -> np.ndarray:
        return _plot_fit_roi_data(disp, kpar, ev, self._fit_roi_bounds())

    def _map_color_kwargs(self, disp: np.ndarray, mode: str, *, roi_scale: bool = False) -> tuple[str, dict]:
        d = self._raw_data
        ref = self._fit_roi_data(disp, d["kpar"], d["ev_arr"]) if roi_scale and d is not None else disp
        return _plot_map_color_kwargs(disp, mode=mode, roi_ref=ref)

    def _draw_fit_roi_overlay(self, ax):
        _plot_draw_fit_roi_overlay(ax, self._fit_roi_bounds())

    def _ef_offset_text(self) -> str:
        return f"EF offset={self._params.sp_ef.value()*1000:+.0f} meV"

    def _draw_ef_label(self, ax, *, horizontal: bool = True):
        txt = f"EF  {self._ef_offset_text()}"
        _plot_draw_ef_label(ax, txt, horizontal=horizontal)

    # ─────────────────────────────────────────────────────────────────────────
    # Band map
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_bm(self):
        if self._data_disp is None:
            return
        d    = self._raw_data
        disp = self._data_disp
        mode = self._cmb_view.currentText()
        kpar = d["kpar"]; ev = d["ev_arr"]

        ax = self._bm_canvas.ax
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=False)
        int_win = self._params.sp_int_win.value()
        fname = Path(d["path"]).name
        _plot_draw_bandmap_axes(
            ax,
            kpar=kpar, ev=ev, disp=disp,
            cmap=cmap, color_kwargs=ckw,
            gamma=self._sp_gamma.value(),
            sel_ev=self._sel_ev, sel_k=self._sel_k, int_win=int_win,
            title=f"{fname}  [{mode}]  {self._ef_offset_text()}",
            title_size=9, label_size=10,
            show_k_zero=True,
        )

        self._draw_fit_roi_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)
        self._bm_canvas.redraw()
        self._draw_mdc_energy_map()

    def _draw_mdc_energy_map(self):
        """Mini BM visible dans l'onglet MDC Fit pour choisir E,k sans revenir à BM."""
        if not hasattr(self, "_mdc_map_canvas") or self._data_disp is None:
            return
        d = self._raw_data
        disp = self._data_disp
        mode = self._cmb_view.currentText()
        kpar = d["kpar"]
        ev = d["ev_arr"]
        ax = self._mdc_map_canvas.ax
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=True)
        int_win = self._params.sp_int_win.value()
        _plot_draw_bandmap_axes(
            ax,
            kpar=kpar, ev=ev, disp=disp,
            cmap=cmap, color_kwargs=ckw,
            gamma=1.0,
            sel_ev=self._sel_ev, sel_k=self._sel_k, int_win=int_win,
            title=f"BM [{mode}]  {self._ef_offset_text()}",
            title_size=8, label_size=8, tick_label_size=8,
            show_k_zero=False,
        )
        bounds = self._fit_roi_bounds()
        if bounds is not None:
            k0, k1, e0, e1 = bounds
            ax.set_xlim(k0, k1)
            ax.set_ylim(e0, e1)
        self._draw_fit_roi_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)
        self._mdc_map_canvas.redraw()

    def _draw_mdc_waterfall(self):
        if not hasattr(self, "_waterfall_canvas") or self._raw_data is None:
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return

        ax = self._waterfall_canvas.ax
        self._waterfall_canvas.fig.set_facecolor("#2b2b2b")

        bounds = self._fit_roi_bounds() or (
            float(self._params.sp_kmin.value()),
            float(self._params.sp_kmax.value()),
            float(self._params.sp_evs.value()),
            float(self._params.sp_eve.value()),
        )
        n_target = int(self._params.sp_wf_n.value()) if hasattr(self._params, "sp_wf_n") else 32
        amp_scale = float(self._params.sp_wf_relief.value()) if hasattr(self._params, "sp_wf_relief") else 1.8
        _plot_draw_waterfall_axes(
            ax, data, kpar, ev,
            bounds=bounds,
            n_target=n_target,
            amp_scale=amp_scale,
            smooth_sigma=self._params.sp_sff.value(),
            fit_result=self._fit_res,
            n_pairs=self._params.sp_np.value(),
            pair_colors=PAIR_COLORS,
            gamma_center=self._params.sp_cx.value(),
        )
        self._waterfall_canvas.fig.tight_layout(pad=0.6)
        self._waterfall_canvas.redraw()

    def _draw_kf_overlay(self, ax):
        if self._fit_res is None:
            return
        fr = self._fit_res
        n  = self._params.sp_np.value()
        for i in range(n):
            c = PAIR_COLORS[i % len(PAIR_COLORS)]
            ev_f = np.asarray(fr["e_fitted"])
            if i < len(fr.get("kF_minus", [])):
                ax.scatter(np.asarray(fr["kF_minus"][i]), ev_f,
                           s=7, color=c, marker="o", zorder=5, alpha=0.85)
            if i < len(fr.get("kF_plus", [])):
                ax.scatter(np.asarray(fr["kF_plus"][i]), ev_f,
                           s=7, color=c, marker="^", zorder=5, alpha=0.85)

    # ─────────────────────────────────────────────────────────────────────────
    # MDC + EDC
    # ─────────────────────────────────────────────────────────────────────────

    def _get_mdc(self):
        if self._raw_data is None: return None
        return _plot_mdc_curve(
            self._raw_data,
            selected_ev=self._sel_ev,
            int_window=self._params.sp_int_win.value(),
            edc_norm_enabled=bool(self._params.chk_norm.isChecked()),
        )

    def _get_edc(self):
        if self._raw_data is None: return None
        return _plot_edc_curve(
            self._raw_data,
            selected_k=self._sel_k,
            edc_norm_enabled=bool(self._params.chk_norm.isChecked()),
        )

    def _draw_mdc_edc(self):
        ax_mdc = self._mdc_edc.axes[0]
        ax_edc = self._edc_canvas.axes[0] if hasattr(self, "_edc_canvas") else None
        ax_mdc.cla()
        if ax_edc is not None:
            ax_edc.cla()
        self._mdc_edc._dark()
        if hasattr(self, "_edc_canvas"):
            self._edc_canvas._dark()

        # ── MDC ──────────────────────────────────────────────────────────────
        res = self._get_mdc()
        if res is not None:
            kpar, mdc = res
            lo, hi = np.nanpercentile(mdc, [1, 99])
            mdc_n = np.clip((mdc - lo) / (hi - lo + 1e-12), 0, 1)

            ax_mdc.plot(kpar, mdc_n, color="white", lw=1.2, label="MDC", zorder=3)
            ax_mdc.fill_between(kpar, 0, mdc_n, alpha=0.08, color="white", zorder=1)

            kmin = self._params.sp_kmin.value()
            kmax = self._params.sp_kmax.value()
            ax_mdc.axvspan(kpar.min(), kmin, alpha=0.15, color="gray", zorder=0)
            ax_mdc.axvspan(kmax, kpar.max(), alpha=0.15, color="gray", zorder=0)

            pairs, mdc_smooth = build_model_pairs(
                kpar, mdc_n,
                n_pairs      = self._params.sp_np.value(),
                gamma_init   = self._params.sp_gi.value(),
                k_min        = kmin, k_max = kmax,
                center_init  = self._params.sp_cx.value(),
                smooth_sigma = self._params.sp_sfd.value(),
            )

            # ── courbe lissée détection (comme Igor "smooth before detect") ──
            ax_mdc.plot(kpar, mdc_smooth, color="#aaa", lw=0.8, ls="-",
                        alpha=0.55, label=f"lissé-det (σ={self._params.sp_sfd.value():.1f})", zorder=2)

            # ── courbe lissée ajustement (utilisée par l'optimiseur scipy) ────
            sff = self._params.sp_sff.value()
            sfd = self._params.sp_sfd.value()
            if sff > 0.5 and abs(sff - sfd) > 0.3:
                _mdc_fit_sm = gaussian_filter1d(np.nan_to_num(mdc_n.copy()), sigma=max(0.5, sff))
                ax_mdc.plot(kpar, _mdc_fit_sm, color="#ffa040", lw=0.8, ls="-",
                            alpha=0.55, zorder=2, label=f"lissé-fit (σ={sff:.1f})")

            # ── zone de contrainte xg (center_init ± xg_range) ───────────────
            cx  = self._params.sp_cx.value()
            xgr = self._params.sp_xg.value()
            ax_mdc.axvspan(cx - xgr, cx + xgr, alpha=0.08, color="cyan",
                           zorder=0, label=f"Fenêtre Γ ±{xgr:.2f}")
            ax_mdc.axvline(cx, color="cyan", lw=0.6, ls=":", alpha=0.45, zorder=1)

            # ── contrainte kF max (si active) ─────────────────────────────────
            if not self._params.chk_k0a.isChecked():
                k0m = self._params.sp_k0m.value()
                ax_mdc.axvline(cx + k0m, color="plum", lw=0.9, ls=":", alpha=0.7, zorder=1,
                               label=f"|kF|<{k0m:.2f}")
                ax_mdc.axvline(cx - k0m, color="plum", lw=0.9, ls=":", alpha=0.7, zorder=1)

            # ── marqueurs kF_init par paire ───────────────────────────────────
            n_p = self._params.sp_np.value()
            for pi, pp in enumerate(self._params._pair_params[:n_p]):
                kf = pp.get("kF_init", 0.30)
                pc = PAIR_COLORS[pi % len(PAIR_COLORS)]
                ax_mdc.axvline(cx + kf, color=pc, lw=0.8, ls="-.", alpha=0.7, zorder=2)
                ax_mdc.axvline(cx - kf, color=pc, lw=0.8, ls="-.", alpha=0.7, zorder=2)

            # ── modèle Lorentzien décomposé ───────────────────────────────────
            gmax = self._params.sp_gm.value()
            total = np.zeros_like(mdc_n)
            for i, (curve, km, kp, cl, cr) in enumerate(pairs):
                c = PAIR_COLORS[i % len(PAIR_COLORS)]
                # zones γ_max autour des pics détectés (largeur maximale autorisée)
                for k0 in (km, kp):
                    ax_mdc.axvspan(k0 - gmax, k0 + gmax, alpha=0.05, color=c, zorder=0)
                valid = np.isfinite(curve)
                if valid.any():
                    # courbe totale de la paire
                    ax_mdc.plot(kpar, np.where(valid, curve, np.nan),
                                color=c, lw=1.3, ls="--", zorder=4,
                                label=f"P{i+1}  kF≈{abs(kp-km)/2:.3f}")
                    # pics individuels (gauche / droite) — modèle Igor lor_pair
                    for comp in (cl, cr):
                        vc = np.isfinite(comp)
                        if vc.any():
                            ax_mdc.plot(kpar, np.where(vc, comp, np.nan),
                                        color=c, lw=0.7, ls=":", alpha=0.55, zorder=3)
                    total += np.where(valid, curve, 0.)
            if n_p > 1:
                ax_mdc.plot(kpar, total, color="white", lw=0.8, ls=":",
                            alpha=0.45, label="Σ", zorder=4)

            ax_mdc.axvline(0, color="w", lw=0.5, ls="--", alpha=0.3)
            int_win = self._params.sp_int_win.value()
            ax_mdc.set_xlabel("k// (π/a)", fontsize=8, color="w")
            ax_mdc.set_ylabel("I (norm.)", fontsize=8, color="w")
            ax_mdc.set_title(
                f"MDC  E={self._sel_ev:.3f} eV  ±{int_win*1000:.0f} meV  |  {self._ef_offset_text()}",
                fontsize=8, color="w")
            ax_mdc.tick_params(colors="w", labelsize=7)
            ax_mdc.legend(fontsize=7, facecolor="#333", labelcolor="w",
                          loc="upper right", framealpha=0.7, ncol=2)
            for sp in ax_mdc.spines.values(): sp.set_edgecolor("#555")

        # ── EDC ──────────────────────────────────────────────────────────────
        res2 = self._get_edc()
        if ax_edc is not None and res2 is not None:
            ev_arr, edc = res2
            lo, hi = np.nanpercentile(edc, [1, 99])
            edc_n = np.clip((edc - lo) / (hi - lo + 1e-12), 0, 1)

            ax_edc.plot(ev_arr, edc_n, color="#7dd3fc", lw=1.2)
            ax_edc.fill_between(ev_arr, 0, edc_n, alpha=0.15, color="#7dd3fc")
            ax_edc.axvline(0, color="cyan", lw=0.8, ls="--", alpha=0.7)
            ax_edc.axvline(self._sel_ev, color="lime", lw=1.0, ls=":")
            self._draw_ef_label(ax_edc, horizontal=False)
            ax_edc.set_xlabel("E − EF (eV)", fontsize=8, color="w")
            ax_edc.set_ylabel("I (norm.)", fontsize=8, color="w")
            ax_edc.set_title(f"EDC  k={self._sel_k:.3f} π/a  |  {self._ef_offset_text()}",
                             fontsize=8, color="w")
            ax_edc.tick_params(colors="w", labelsize=7)
            for sp in ax_edc.spines.values(): sp.set_edgecolor("#555")

        self._mdc_edc.fig.tight_layout(pad=0.5)
        self._mdc_edc.redraw()
        if hasattr(self, "_edc_canvas"):
            self._edc_canvas.fig.tight_layout(pad=0.5)
            self._edc_canvas.redraw()

    # ─────────────────────────────────────────────────────────────────────────
    # Interactions carte
    # ─────────────────────────────────────────────────────────────────────────

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
        if AP is None:
            self._status("⚠ arpes_plots non chargé")
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return
        try:
            res = AP.estimate_gamma_bm_mdc(
                data, kpar, ev,
                ev_range=(self._params.sp_evs.value(), self._params.sp_eve.value()),
                k_range=(self._params.sp_kmin.value(), self._params.sp_kmax.value()),
                center_guess=self._params.sp_cx.value(),
                center_window=max(self._params.sp_xg.value() * 2.0, 0.25),
                smooth_sigma=self._params.sp_sfd.value(),
                verbose=False,
            )
            gamma = float(res["gamma"])
            if not np.isfinite(gamma):
                QMessageBox.warning(
                    self, "Auto Γ BM",
                    "Impossible d'estimer Γ : pas assez de paires MDC valides. "
                    "Ajuste la plage d'énergie, k_min/k_max ou centre_init."
                )
                return
            self._params.sp_cx.setValue(gamma)
            entry_now = self._current_entry()
            azi_ref = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
            meta_now = self._raw_data.get("metadata", {}) or {}
            self._session.gamma_reference = _gamma_build_reference(
                kx=gamma, ky=0.0,
                metadata=meta_now,
                hv=self._raw_data.get("hv"),
                path=self._raw_data.get("path"),
                azi=azi_ref,
                source="bm",
                direction=(entry_now.meta.direction if entry_now else None),
            )
            if not meta_now.get("angle_offsets_applied") and not meta_now.get("bm_gamma_axis_centered"):
                offsets = self._angle_offsets_from_k_center(
                    float(gamma), 0.0,
                    hv=self._raw_data.get("hv"),
                    source="bm_auto",
                    ref_path=self._raw_data.get("path"),
                    azi=azi_ref,
                )
                if offsets:
                    self._session.angle_offsets = offsets
            if self._current_path:
                entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
                entry.fit_params.center_init = float(gamma)
            self._session.save()
            self._params.lbl_res.setText(
                f"Γ BM = {gamma:+.4f} π/a\n"
                f"n={res['n']}  MAD={res['mad']:.4f}"
            )
            self._draw_bm()
            self._draw_mdc_edc()
            if hasattr(self, "_mdc_fit_tabs") and self._tabs.currentIndex() == 1:
                self._draw_mdc_waterfall()
            self._status(f"Γ BM estimé : {gamma:+.4f} π/a  n={res['n']}  MAD={res['mad']:.4f}")
        except Exception as exc:
            QMessageBox.warning(self, "Auto Γ BM", str(exc))
            self._status(f"⚠ Auto Γ BM : {exc}")

    def _apply_gamma_reference_to_bm(self):
        ref = self._stored_gamma_reference()
        if not ref:
            QMessageBox.warning(self, "Γ FS → BM", "Aucun Γ de référence. Va sur l'onglet FS et clique d'abord sur Détecter Γ FS.")
            return
        if self._raw_data is None:
            return
        meta = self._raw_data.get("metadata", {}) or {}
        if meta.get("angle_offsets_applied"):
            self._params.sp_cx.setValue(0.0)
            if self._current_path:
                entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
                entry.fit_params.center_init = 0.0
                self._session.save()
            self._params.lbl_res.setText("Γ déjà appliqué par offset angulaire loader")
            self._draw_bm()
            self._draw_mdc_edc()
            self._status("Γ FS appliqué : offset angulaire loader déjà actif")
            return
        gamma_bm, correction = self._gamma_reference_to_bm_center(ref)
        if not np.isfinite(gamma_bm):
            QMessageBox.warning(self, "Γ FS → BM", "La référence Γ stockée est invalide.")
            return
        angular_applied = bool(meta.get("angle_offsets_applied"))
        axis_centered = False if angular_applied else self._center_current_bm_axis_on_gamma(float(gamma_bm), ref)
        self._params.sp_cx.setValue(0.0 if (axis_centered or angular_applied) else gamma_bm)
        if self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fit_params.center_init = 0.0 if (axis_centered or angular_applied) else float(gamma_bm)
            self._session.save()
        mode_msg = "offset angulaire loader" if angular_applied else ("axe k recentré" if axis_centered else "centre fit seul")
        self._params.lbl_res.setText(
            f"Γ FS→BM = {gamma_bm:+.4f} π/a\n"
            f"corr polar={correction:+.4f}\n"
            f"{mode_msg}"
        )
        self._update_display_data()
        self._draw_bm()
        self._draw_mdc_edc()
        self._status(
            f"Γ FS appliqué à la BM : {gamma_bm:+.4f} π/a  correction={correction:+.4f}"
            f"  |  {mode_msg}"
        )

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
