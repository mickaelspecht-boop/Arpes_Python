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
import sys
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import find_peaks

from PyQt6.QtCore import Qt, pyqtSignal
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
# Session — dataclasses + JSON
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FitParams:
    n_pairs: int       = 1
    ev_start: float    = -0.90
    ev_end: float      = -0.005
    k_min: float       = -0.80
    k_max: float       =  0.80
    smooth_fit: float  = 2.0
    smooth_detect: float = 3.0
    gamma_init: float  = 0.08
    gamma_max: float   = 0.30
    xg_range: float    = 0.10
    center_init: float = 0.0
    k0_max: Optional[float] = None
    width_mode: str    = "symmetric"
    min_amplitude: float = 0.01
    max_jump: float    = 0.20
    scan_direction: str = "up"
    pairs: list = field(default_factory=lambda: [
        {"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}
    ])


@dataclass
class FileMeta:
    hv: float          = 0.0
    temperature: float = 0.0
    direction: str     = ""
    polarization: str  = ""
    meas_no: int       = 0


@dataclass
class FileEntry:
    ef_offset: float   = 0.052
    edcnorm: bool      = True
    view_mode: str     = "EDCnorm"          # Raw / EDCnorm / SecDev / Curvature
    fit_params: FitParams   = field(default_factory=FitParams)
    fit_result: Optional[dict] = None        # sérialisé (listes, pas ndarray)
    meta: FileMeta     = field(default_factory=FileMeta)

    @property
    def status(self) -> str:
        if self.fit_result:
            return "fitted"
        return "loaded"


def _to_serial(obj):
    """Convertit récursivement np.ndarray / np.floating en types JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serial(v) for v in obj]
    return obj


class Session:
    VERSION = 1

    def __init__(self, folder: Path | None = None, work_func: float = 4.031):
        self.folder:    Path | None = folder
        self.work_func: float       = work_func
        self.files:     dict[str, FileEntry] = {}

    # ── persistance ───────────────────────────────────────────────────────────
    @property
    def json_path(self) -> Path | None:
        return self.folder / ".arpes_session.json" if self.folder else None

    def save(self):
        if not self.json_path:
            return
        data = {
            "version":   self.VERSION,
            "folder":    str(self.folder),
            "work_func": self.work_func,
            "files": {
                name: _to_serial(asdict(entry))
                for name, entry in self.files.items()
            },
        }
        self.json_path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path):
        raw = json.loads(path.read_text())
        self.work_func = raw.get("work_func", 4.031)
        for name, edict in raw.get("files", {}).items():
            fp = FitParams(**edict.get("fit_params", {}))
            mt = FileMeta(**edict.get("meta", {}))
            entry = FileEntry(
                ef_offset  = edict.get("ef_offset", 0.052),
                edcnorm    = edict.get("edcnorm", True),
                view_mode  = edict.get("view_mode", "EDCnorm"),
                fit_params = fp,
                fit_result = edict.get("fit_result"),
                meta       = mt,
            )
            self.files[name] = entry

    # ── helpers ───────────────────────────────────────────────────────────────
    def get_or_create(self, filename: str) -> FileEntry:
        if filename not in self.files:
            self.files[filename] = FileEntry()
        return self.files[filename]

    def set_fit_result(self, filename: str, fr: dict):
        entry = self.get_or_create(filename)
        entry.fit_result = _to_serial(fr)
        self.save()

    def key_for_path(self, path: str | Path) -> str:
        """Clé stable de session : chemin relatif au dossier racine si possible."""
        p = Path(path)
        if self.folder is not None:
            try:
                return str(p.resolve().relative_to(self.folder.resolve()))
            except Exception:
                pass
        return p.name


# ─────────────────────────────────────────────────────────────────────────────
# Chargement arpes_plots
# ─────────────────────────────────────────────────────────────────────────────

def _load_ap():
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
    from arpes_io import load_arpes, ARPESData
    from arpes_fs import FermiSurfaceCanvas, FSControlPanel
    ERLAB_OK = True
except Exception:
    load_arpes = None
    ARPESData = None
    FermiSurfaceCanvas = None
    FSControlPanel = None
    ERLAB_OK = False

AP = None


# ─────────────────────────────────────────────────────────────────────────────
# Chargement données ARPES
# ─────────────────────────────────────────────────────────────────────────────

def load_arpes_file(path: str, work_func: float, ef_offset: float,
                    a_lattice: float = 3.96, hv: float | None = None) -> dict | None:
    if not ERLAB_OK or load_arpes is None:
        return None
    ds = load_arpes(path, work_func=work_func, ef_offset=ef_offset,
                    a_lattice=a_lattice, hv=hv)
    return ds.as_legacy_bandmap_dict()


def apply_edcnorm(data: np.ndarray) -> np.ndarray:
    edc  = np.nanmean(data, axis=0, keepdims=True)
    safe = np.where((np.abs(edc) > 1e-12) & np.isfinite(edc), edc, 1.0)
    return data / safe


def compute_secdev(data: np.ndarray, kpar, ev_arr,
                   sigma_k=2.0, sigma_e=2.0) -> np.ndarray:
    """−d²I/dE² lissée."""
    d = gaussian_filter(data.astype(float), sigma=[sigma_k, sigma_e])
    de = np.gradient(np.gradient(d, ev_arr, axis=1), ev_arr, axis=1)
    return -de


def compute_curvature(data: np.ndarray, kpar, ev_arr,
                      sigma_k=2.0, sigma_e=2.0) -> np.ndarray:
    """Courbure 2D −∇²I / (1+|∇I|²)^(3/2)."""
    d = gaussian_filter(data.astype(float), sigma=[sigma_k, sigma_e])
    gk = np.gradient(d, kpar, axis=0)
    ge = np.gradient(d, ev_arr, axis=1)
    denom = (1.0 + gk**2 + ge**2) ** 1.5
    lap = (np.gradient(np.gradient(d, kpar, axis=0), kpar, axis=0) +
           np.gradient(np.gradient(d, ev_arr, axis=1), ev_arr, axis=1))
    return -lap / (denom + 1e-30)


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
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        btn = QPushButton("📂 Dossier")
        btn.clicked.connect(self._open_folder)
        top.addWidget(btn)
        self._lbl_folder = QLabel("—")
        self._lbl_folder.setWordWrap(True)
        self._lbl_folder.setStyleSheet("font-size:10px; color:#aaa;")
        lay.addLayout(top)
        lay.addWidget(self._lbl_folder)

        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget { background:#222; color:#ddd; font-size:11px; }
            QListWidget::item:selected { background:#2a6099; }
        """)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_selection_change)
        lay.addWidget(self._list, stretch=1)

        self._btn_load = QPushButton("↵ Charger")
        self._btn_load.clicked.connect(self._load_selected)
        lay.addWidget(self._btn_load)

        self.setMinimumWidth(210)
        self.setMaximumWidth(280)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier données ARPES",
                                                   str(Path.home()))
        if folder:
            self.set_folder(Path(folder))

    def set_folder(self, folder: Path):
        self._folder = folder
        self._session.folder = folder
        self._lbl_folder.setText(folder.name)
        if self._session.json_path and self._session.json_path.exists():
            try:
                self._session.load(self._session.json_path)
            except Exception:
                pass
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
        return sorted(set(out), key=lambda x: str(x.relative_to(self._folder)).lower())

    def _group_label(self, group: str) -> str:
        if group == ".":
            return self._folder.name if self._folder else "."
        return group

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
            return

        groups: dict[str, list[Path]] = {}
        for p in self._discover_items():
            rel = p.relative_to(self._folder)
            group = str(rel.parent) if str(rel.parent) != "." else "."
            groups.setdefault(group, []).append(p)

        for group in sorted(groups, key=lambda g: (g != ".", g.lower())):
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
                suffix = "  [FS]" if self._is_cls_dataset_dir(p) or p.suffix.lower() == ".zip" else ""
                item   = QListWidgetItem(f"  {icon}  {p.name}{suffix}")
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                item.setData(Qt.ItemDataRole.UserRole + 1, key)
                item.setToolTip(str(rel))
                item.setForeground(QColor(color))
                self._list.addItem(item)

        if selected_path:
            self.select_file(selected_path)

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
                suffix = "  [FS]" if Path(path).is_dir() or Path(path).suffix.lower() == ".zip" else ""
                item.setText(f"  {icon}  {Path(path).name}{suffix}")
                item.setForeground(QColor(color))
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
        pass

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

    def select_file(self, path: str):
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
                break
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
    guess_requested = pyqtSignal()
    full_fit_requested = pyqtSignal()
    clear_kf_requested = pyqtSignal()
    copy_params_requested = pyqtSignal()
    ef_calib_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        w = QWidget()
        self._lay = QVBoxLayout(w)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(w)
        self._pair_params: list[dict] = [{"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}]
        self._current_pair: int = 0
        self._build()

    def _build(self):
        lay = self._lay

        # ── énergie ──────────────────────────────────────────────────────────
        grp_e = QGroupBox("Énergie sélectionnée")
        fl = QFormLayout(grp_e)
        self.sp_ev = _dspin(-0.30, -3.0, 0.2, 0.01)
        # sp_ev est connecté dans ArpesExplorer._build_ui (→ _on_ev_spinbox_changed)
        self.sp_int_win = _dspin(0.010, 0.001, 0.200, 0.005, dec=3)
        self.sp_int_win.setToolTip(
            "Fenêtre d'intégration ±eV pour la MDC\n"
            "Élargir = moins de bruit, moins de résolution en énergie\n"
            "Correspond au 'range' d'extraction d'une coupe dans Igor")
        self.sp_int_win.valueChanged.connect(self.params_changed)
        fl.addRow("E (eV):", self.sp_ev)
        fl.addRow("± intég. (eV):", self.sp_int_win)
        fl.addRow(QLabel("💡 Clic sur la carte ou ici"))
        lay.addWidget(grp_e)

        # ── calibration EF ────────────────────────────────────────────────────
        grp_ef = QGroupBox("EF / Chargement")
        fl_ef = QFormLayout(grp_ef)
        self.sp_phi = _dspin(4.031, 3.0, 6.0, 0.01)
        self.sp_hv  = _dspin(0.0, 0.0, 500.0, 1.0)
        self.sp_ef  = _dspin(0.052, -0.3, 0.3, 0.005)
        self.chk_norm = QCheckBox("EDCnorm"); self.chk_norm.setChecked(True)
        self.chk_norm.stateChanged.connect(self.params_changed)
        btn_ef = QPushButton("🎛  Calibrer EF auto")
        btn_ef.clicked.connect(self.ef_calib_requested)
        btn_copy = QPushButton("📋  Copier params → fichier suivant")
        btn_copy.clicked.connect(self.copy_params_requested)
        fl_ef.addRow("φ (eV):",       self.sp_phi)
        fl_ef.addRow("hν (eV):", self.sp_hv)
        fl_ef.addRow("EF offset:",    self.sp_ef)
        fl_ef.addRow(self.chk_norm)
        fl_ef.addRow(btn_ef)
        fl_ef.addRow(btn_copy)
        lay.addWidget(grp_ef)

        # ── contrôles fit (cachés sur l'onglet BM) ────────────────────────────
        self._fit_controls_widget = QWidget()
        _fcl = QVBoxLayout(self._fit_controls_widget)
        _fcl.setContentsMargins(0, 0, 0, 0)
        _fcl.setSpacing(4)

        # ── plage d'analyse ───────────────────────────────────────────────────
        grp_r = QGroupBox("Plage d'analyse")
        fl2 = QFormLayout(grp_r)
        self.sp_evs  = _dspin(-0.90, -3.0,  0.0,  0.05)
        self.sp_eve  = _dspin(-0.005, -1.0, 0.10, 0.005)
        self.sp_kmin = _dspin(-0.80, -3.0,  0.0,  0.05)
        self.sp_kmax = _dspin( 0.80,  0.0,  3.0,  0.05)
        for w in (self.sp_evs, self.sp_eve, self.sp_kmin, self.sp_kmax):
            w.valueChanged.connect(self.params_changed)
        fl2.addRow("ev_start:", self.sp_evs)
        fl2.addRow("ev_end:",   self.sp_eve)
        fl2.addRow("k_min:",    self.sp_kmin)
        fl2.addRow("k_max:",    self.sp_kmax)
        _fcl.addWidget(grp_r)

        # ── fit MDC ───────────────────────────────────────────────────────────
        grp_f = QGroupBox("Fit MDC (Lorentzien)")
        fl3 = QFormLayout(grp_f)
        self.sp_np   = _ispin(1,   1, 8)
        self.sp_np.valueChanged.connect(self._on_n_pairs_changed)
        self.sp_sff  = _dspin(2.0,  0.0, 10.0, 0.5, dec=1)
        self.sp_sfd  = _dspin(3.0,  0.0, 10.0, 0.5, dec=1)

        # ── paramètres par paire (navigables) ────────────────────────────────
        self._pair_lbl = ClickablePairLabel()
        self._pair_lbl.pair_changed.connect(self._on_pair_changed)
        self.sp_kfi  = _dspin(0.30,  0.0,  3.0, 0.01)
        self.sp_gi   = _dspin(0.08, 0.01,  0.5, 0.01)
        self.sp_gm   = _dspin(0.30, 0.05,  1.0, 0.05)

        # ── paramètres globaux ────────────────────────────────────────────────
        self.sp_xg   = _dspin(0.10, 0.0,  0.5,  0.01)
        self.sp_cx   = _dspin(0.0, -1.0,  1.0,  0.01)
        self.sp_k0m  = _dspin(0.0,  0.0,  2.0,  0.05)
        self.chk_k0a = QCheckBox("auto"); self.chk_k0a.setChecked(True)
        self.sp_k0m.setEnabled(False)
        self.chk_k0a.stateChanged.connect(
            lambda: self.sp_k0m.setEnabled(not self.chk_k0a.isChecked()))
        self.cmb_wm  = QComboBox(); self.cmb_wm.addItems(["symmetric","asymmetric"])
        self.cmb_wm.setFixedWidth(110)
        self.sp_ma   = _dspin(0.01, 0.0, 1.0, 0.01)
        self.sp_mj   = _dspin(0.20, 0.0, 1.0, 0.05)
        self.cmb_sd  = QComboBox(); self.cmb_sd.addItems(["up","down"])
        self.cmb_sd.setFixedWidth(80)

        for w in (self.sp_sff, self.sp_sfd, self.sp_kfi, self.sp_gi, self.sp_gm,
                  self.sp_xg, self.sp_cx, self.sp_k0m, self.sp_ma, self.sp_mj):
            w.valueChanged.connect(self.params_changed)
        self.cmb_wm.currentIndexChanged.connect(self.params_changed)

        k0w = QWidget(); k0l = QHBoxLayout(k0w); k0l.setContentsMargins(0,0,0,0)
        k0l.addWidget(self.sp_k0m); k0l.addWidget(self.chk_k0a)

        fl3.addRow("n_pairs:",      self.sp_np)
        fl3.addRow("smooth_fit:",   self.sp_sff)
        fl3.addRow("smooth_det:",   self.sp_sfd)
        fl3.addRow(_sep())
        fl3.addRow(self._pair_lbl)
        fl3.addRow("kF_init:",      self.sp_kfi)
        fl3.addRow("γ_init:",       self.sp_gi)
        fl3.addRow("γ_max:",        self.sp_gm)
        fl3.addRow(_sep())
        fl3.addRow("xg_range:",     self.sp_xg)
        fl3.addRow("center_init:",  self.sp_cx)
        fl3.addRow("k0_max:",       k0w)
        fl3.addRow("width_mode:",   self.cmb_wm)
        fl3.addRow(_sep())
        fl3.addRow("min_ampl:",     self.sp_ma)
        fl3.addRow("max_jump:",     self.sp_mj)
        fl3.addRow("scan_dir:",     self.cmb_sd)
        _fcl.addWidget(grp_f)

        # ── boutons ───────────────────────────────────────────────────────────
        _fcl.addWidget(_sep())
        btn_g = QPushButton("🎯  Guess  (fit MDC ici)  [Ctrl+G]")
        btn_g.setStyleSheet("background:#1a6b3a;color:white;font-weight:bold;padding:6px;")
        btn_g.clicked.connect(self.guess_requested)
        _fcl.addWidget(btn_g)

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
            pairs         = [dict(p) for p in self._pair_params],
        )

    def set_fit_controls_visible(self, visible: bool):
        self._fit_controls_widget.setVisible(visible)

    def load_fit_params(self, fp: FitParams):
        for sp, val in [
            (self.sp_evs,  fp.ev_start),  (self.sp_eve,  fp.ev_end),
            (self.sp_kmin, fp.k_min),     (self.sp_kmax, fp.k_max),
            (self.sp_sff,  fp.smooth_fit),(self.sp_sfd,  fp.smooth_detect),
            (self.sp_xg,   fp.xg_range),  (self.sp_cx,   fp.center_init),
            (self.sp_ma,   fp.min_amplitude),(self.sp_mj, fp.max_jump),
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

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Fichier", "hν", "T (K)", "Dir.", "kF+ (π/a)", "xg (π/a)"])
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

            self._table.insertRow(row)
            for col, val in enumerate([
                name, f"{meta.hv:.0f}", f"{meta.temperature:.0f}",
                meta.direction, f"{kf_ef:.4f}", f"{xg_m:.4f}",
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
        rows = []
        for name, entry in self._session.files.items():
            if entry.fit_result is None:
                continue
            fr   = entry.fit_result
            meta = entry.meta
            ev_f = np.asarray(fr["e_fitted"])
            n    = entry.fit_params.n_pairs
            for ie, ev in enumerate(ev_f):
                d = {"file": name, "hv": meta.hv, "T_K": meta.temperature,
                     "direction": meta.direction, "E_eV": ev}
                for i in range(n):
                    km_arr = fr["kF_minus"][i] if i < len(fr["kF_minus"]) else []
                    kp_arr = fr["kF_plus"][i]  if i < len(fr["kF_plus"])  else []
                    d[f"kF_minus_{i+1}"] = km_arr[ie] if ie < len(km_arr) else ""
                    d[f"kF_plus_{i+1}"]  = kp_arr[ie] if ie < len(kp_arr) else ""
                rows.append(d)
        if not rows:
            return
        import csv
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    def _export_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", str(self._session.folder or Path.home()),
            "PDF (*.pdf);;PNG (*.png)")
        if path:
            self._canvas.fig.savefig(path, dpi=200, bbox_inches="tight",
                                     facecolor=self._canvas.fig.get_facecolor())


import matplotlib.pyplot as plt   # pour plt.cm dans ResultsPanel


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
        self._fit_res:    dict | None  = None

        self._sel_ev = -0.30
        self._sel_k  = 0.0

        self._build_ui()
        self._install_shortcuts()
        self._status("Prêt — ouvrir un dossier ou un fichier")

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(main_split)

        # ── file browser ──────────────────────────────────────────────────────
        self._browser = FileBrowserPanel(self._session)
        self._browser.file_selected.connect(self._load_file)
        main_split.addWidget(self._browser)

        # ── centre : tabs ─────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab{background:#333;color:#ccc;padding:5px 12px;}"
            "QTabBar::tab:selected{background:#2a6099;color:white;}")

        # Tab Carte
        carte_widget = QWidget()
        carte_lay = QVBoxLayout(carte_widget)
        carte_lay.setContentsMargins(0, 0, 0, 0)

        # barre view
        vbar = QHBoxLayout()
        vbar.addWidget(QLabel("Vue :"))
        self._cmb_view = QComboBox()
        self._cmb_view.addItems(["Raw", "EDCnorm", "SecDev", "Curvature"])
        self._cmb_view.setCurrentText("EDCnorm")
        self._cmb_view.setFixedWidth(120)
        self._cmb_view.currentIndexChanged.connect(self._on_view_changed)
        vbar.addWidget(self._cmb_view)
        lbl_gamma = QLabel("  γ:")
        lbl_gamma.setStyleSheet("color:#aaa;font-size:11px;")
        lbl_gamma.setToolTip("Gamma de contraste : <1 booste les faibles intensités (comme dans Igor)")
        vbar.addWidget(lbl_gamma)
        from PyQt6.QtWidgets import QDoubleSpinBox as _DSB
        self._sp_gamma = _DSB()
        self._sp_gamma.setRange(0.1, 3.0); self._sp_gamma.setSingleStep(0.1)
        self._sp_gamma.setDecimals(1); self._sp_gamma.setValue(1.0)
        self._sp_gamma.setFixedWidth(54)
        self._sp_gamma.setToolTip(
            "γ < 1  → accentue les structures faibles (utile pour FS)\n"
            "γ = 1  → échelle linéaire\n"
            "γ > 1  → accentue les structures fortes\n"
            "Identique à la correction gamma d'Igor BandFinder")
        self._sp_gamma.valueChanged.connect(self._draw_bm)
        vbar.addWidget(self._sp_gamma)
        vbar.addStretch()
        lbl_hint = QLabel("Clic → MDC+EDC  |  ← → naviguer fichiers")
        lbl_hint.setStyleSheet("color:#888;font-size:10px;")
        vbar.addWidget(lbl_hint)
        carte_lay.addLayout(vbar)

        self._bm_canvas = MplCanvas(figsize=(7, 6), toolbar=True)
        self._bm_canvas.canvas.mpl_connect(
            "button_press_event", self._on_map_click)
        carte_lay.addWidget(self._bm_canvas, stretch=1)
        self._tabs.addTab(carte_widget, "🗺  BM")

        # Tab MDC Fit : mini-BM + diagrammes MDC/EDC, pour garder le fit séparé
        mdc_widget = QWidget()
        mdc_lay = QVBoxLayout(mdc_widget)
        mdc_lay.setContentsMargins(0, 0, 0, 0)
        self._mdc_map_canvas = MplCanvas(figsize=(6, 3), toolbar=True)
        self._mdc_map_canvas.canvas.mpl_connect(
            "button_press_event", self._on_map_click)
        mdc_lay.addWidget(self._mdc_map_canvas, stretch=2)
        self._mdc_edc = MplCanvas(figsize=(6, 4), nrows=2)
        mdc_lay.addWidget(self._mdc_edc, stretch=3)
        self._tabs.addTab(mdc_widget, "🎯  MDC Fit")

        # Tab Résultats
        self._results = ResultsPanel(self._session)
        self._tabs.addTab(self._results, "📊  Résultats")

        # Tab FS : carte de Fermi dédiée, contrôles dans le panneau droit FS
        self._fs_canvas = FermiSurfaceCanvas() if FermiSurfaceCanvas is not None else QWidget()
        self._tabs.addTab(self._fs_canvas, "🧭  FS")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        main_split.addWidget(self._tabs)

        # ── droite : MDC/EDC + params ─────────────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Vertical)

        self._params = FitParamsPanel()
        self._params.params_changed.connect(self._on_model_changed)
        self._params.sp_ev.valueChanged.connect(self._on_ev_spinbox_changed)
        self._params.guess_requested.connect(self._fit_guess)
        self._params.full_fit_requested.connect(self._fit_full)
        self._params.clear_kf_requested.connect(self._clear_kf)
        self._params.copy_params_requested.connect(self._copy_params)
        self._params.ef_calib_requested.connect(self._ef_calibrate)
        self._params.set_fit_controls_visible(False)  # caché sur BM (tab 0 par défaut)
        right_split.addWidget(self._params)
        right_split.setSizes([550])

        self._fs_controls = FSControlPanel() if FSControlPanel is not None else QWidget()
        if FSControlPanel is not None:
            self._fs_controls.params_changed.connect(self._draw_fs_tab)
            self._fs_controls.redraw_requested.connect(self._draw_fs_tab)
            if hasattr(self._fs_controls, "gamma_requested"):
                self._fs_controls.gamma_requested.connect(self._detect_fs_gamma)

        self._right_stack = QStackedWidget()
        self._right_stack.addWidget(right_split)
        self._right_stack.addWidget(self._fs_controls)
        main_split.addWidget(self._right_stack)
        main_split.setSizes([210, 850, 440])

        self.setStatusBar(QStatusBar())

    def _on_tab_changed(self, index: int):
        # 0=BM, 1=MDC Fit, 2=Résultats, 3=FS
        if hasattr(self, "_right_stack"):
            self._right_stack.setCurrentIndex(1 if index == 3 else 0)
        self._params.set_fit_controls_visible(index == 1)
        if index == 2:
            self._results.refresh()
        elif index == 3:
            self._draw_fs_tab()
        elif index == 1:
            self._draw_bm()
            self._draw_mdc_edc()

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

    def _detect_fs_gamma(self):
        if FermiSurfaceCanvas is None or FSControlPanel is None:
            return
        try:
            params = self._fs_controls.params()
            res = self._fs_canvas.detect_gamma(self._raw_data, params)
            self._fs_controls.set_center(res["kx"], res["ky"])
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
        global AP
        if AP is None:
            try:
                AP = _load_ap()
            except Exception as e:
                self._status(f"⚠ arpes_plots : {e}")

        self._status(f"Chargement {Path(path).name} …")
        QApplication.processEvents()
        try:
            entry = self._session.get_or_create(self._session.key_for_path(path))
            self._params.sp_ef.blockSignals(True)
            self._params.sp_ef.setValue(entry.ef_offset)
            self._params.sp_ef.blockSignals(False)
            if entry.meta.hv and entry.meta.hv > 0 and self._params.sp_hv.value() <= 0:
                self._params.sp_hv.blockSignals(True)
                self._params.sp_hv.setValue(float(entry.meta.hv))
                self._params.sp_hv.blockSignals(False)
            hv_for_load = self._params.sp_hv.value()
            d = load_arpes_file(path,
                                self._params.sp_phi.value(),
                                self._params.sp_ef.value(),
                                hv=hv_for_load)
            if d is None:
                self._status("⚠ erlab non disponible")
                return
            self._raw_data    = d
            self._current_path = path
            self._fit_res = None

            # Remplir hν depuis les données si disponible
            hv_in_data = d.get("hv")
            if hv_in_data is not None and np.isfinite(float(hv_in_data)) and float(hv_in_data) > 0:
                self._params.sp_hv.blockSignals(True)
                self._params.sp_hv.setValue(float(hv_in_data))
                self._params.sp_hv.blockSignals(False)
                entry.meta.hv = float(hv_in_data)
            elif hv_for_load and hv_for_load > 0:
                entry.meta.hv = float(hv_for_load)

            # Restaurer params depuis session
            self._params.sp_ef.blockSignals(True)
            self._params.sp_ef.setValue(entry.ef_offset)
            self._params.sp_ef.blockSignals(False)
            self._params.chk_norm.blockSignals(True)
            self._params.chk_norm.setChecked(entry.edcnorm)
            self._params.chk_norm.blockSignals(False)
            self._params.load_fit_params(entry.fit_params)
            self._cmb_view.blockSignals(True)
            self._cmb_view.setCurrentText(entry.view_mode)
            self._cmb_view.blockSignals(False)

            # Restaurer fit_result si disponible
            if entry.fit_result:
                self._fit_res = entry.fit_result

            self._update_display_data()
            self._sel_ev = float(np.clip(-0.30, d["ev_arr"].min(), d["ev_arr"].max()))
            self._sel_k  = 0.0
            self._sync_ev_spinbox()

            self._draw_bm()
            self._draw_mdc_edc()
            if self._tabs.currentIndex() == 3:
                self._draw_fs_tab()

            self._browser.select_file(path)
            hv_txt = f"{d['hv']:.0f} eV" if d.get("hv") is not None else "—"
            self._status(
                f"Chargé : {Path(path).name}  hν={hv_txt}  |  "
                f"k {d['kpar'].min():.2f}→{d['kpar'].max():.2f} π/a  |  "
                f"E {d['ev_arr'].min():.3f}→{d['ev_arr'].max():.3f} eV"
            )
        except Exception as e:
            self._status(f"⚠ {e}")
            traceback.print_exc()

    def _update_display_data(self):
        if self._raw_data is None:
            return
        d    = self._raw_data
        raw  = d["data"]
        mode = self._cmb_view.currentText()

        if mode == "Raw":
            self._data_disp = raw
        elif mode == "EDCnorm":
            self._data_disp = apply_edcnorm(raw) if self._params.chk_norm.isChecked() else raw
        elif mode == "SecDev":
            norm = apply_edcnorm(raw) if self._params.chk_norm.isChecked() else raw
            self._data_disp = compute_secdev(norm, d["kpar"], d["ev_arr"])
        elif mode == "Curvature":
            norm = apply_edcnorm(raw) if self._params.chk_norm.isChecked() else raw
            self._data_disp = compute_curvature(norm, d["kpar"], d["ev_arr"])

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

    def _on_model_changed(self, _=None):
        self._update_display_data()
        self._draw_bm()
        self._draw_mdc_edc()

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
        ax.cla(); ax.set_facecolor("#1a1a1a")
        self._bm_canvas.fig.set_facecolor("#2b2b2b")

        gamma = self._sp_gamma.value()
        _norm = lambda vmin, vmax: (PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
                                    if gamma != 1.0 else None)
        if mode in ("Raw", "EDCnorm"):
            vmax = float(np.nanpercentile(disp, 99))
            n = _norm(0, vmax)
            kw = dict(norm=n) if n else dict(vmin=0, vmax=vmax)
            ax.pcolormesh(kpar, ev, disp.T, cmap="inferno", shading="auto", **kw)
        elif mode == "SecDev":
            pos = disp[np.isfinite(disp) & (disp > 0)]
            vmax = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
            n = _norm(0, vmax)
            kw = dict(norm=n) if n else dict(vmin=0, vmax=vmax)
            ax.pcolormesh(kpar, ev, disp.T, cmap="hot_r", shading="auto", **kw)
        elif mode == "Curvature":
            pos = disp[np.isfinite(disp) & (disp > 0)]
            vmax = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
            n = _norm(0, vmax)
            kw = dict(norm=n) if n else dict(vmin=0, vmax=vmax)
            ax.pcolormesh(kpar, ev, disp.T, cmap="hot_r", shading="auto", **kw)

        int_win = self._params.sp_int_win.value()
        ax.axhline(0,          color="cyan", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(0,          color="w",    lw=0.5, ls="--", alpha=0.4)
        ax.axhspan(self._sel_ev - int_win, self._sel_ev + int_win,
                   alpha=0.14, color="lime", zorder=2, lw=0)
        ax.axhline(self._sel_ev, color="lime", lw=0.8, ls="--", zorder=3)
        ax.axvline(self._sel_k,  color="lime", lw=1.0, ls=":",  zorder=3)

        self._draw_kf_overlay(ax)

        fname = Path(d["path"]).name
        ax.set_xlabel("k// (π/a)", fontsize=10, color="w")
        ax.set_ylabel("E − EF (eV)", fontsize=10, color="w")
        ax.set_title(f"{fname}  [{mode}]", fontsize=9, color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
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
        ax.cla(); ax.set_facecolor("#1a1a1a")
        self._mdc_map_canvas.fig.set_facecolor("#2b2b2b")
        if mode in ("Raw", "EDCnorm"):
            vmax = float(np.nanpercentile(disp, 99))
            ax.pcolormesh(kpar, ev, disp.T, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        else:
            pos = disp[np.isfinite(disp) & (disp > 0)]
            vmax = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
            ax.pcolormesh(kpar, ev, disp.T, cmap="hot_r", vmin=0, vmax=vmax, shading="auto")
        int_win = self._params.sp_int_win.value()
        ax.axhline(0, color="cyan", lw=0.7, ls="--", alpha=0.6)
        ax.axhspan(self._sel_ev - int_win, self._sel_ev + int_win,
                   alpha=0.14, color="lime", zorder=2, lw=0)
        ax.axhline(self._sel_ev, color="lime", lw=0.7, ls="--", zorder=3)
        ax.axvline(self._sel_k,  color="lime", lw=0.9, ls=":", zorder=3)
        self._draw_kf_overlay(ax)
        ax.set_xlabel("k// (π/a)", fontsize=8, color="w")
        ax.set_ylabel("E − EF (eV)", fontsize=8, color="w")
        ax.set_title(f"BM [{mode}] — clic pour choisir E,k", fontsize=8, color="w")
        ax.tick_params(colors="w", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        self._mdc_map_canvas.redraw()

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
        d = self._raw_data
        norm = apply_edcnorm(d["data"]) if self._params.chk_norm.isChecked() else d["data"]
        int_win = self._params.sp_int_win.value()
        mask_e = np.abs(d["ev_arr"] - self._sel_ev) <= int_win
        if not mask_e.any():
            mask_e[np.argmin(np.abs(d["ev_arr"] - self._sel_ev))] = True
        mdc = np.nanmean(norm[:, mask_e], axis=1).astype(float)
        return d["kpar"], mdc

    def _get_edc(self):
        if self._raw_data is None: return None
        d = self._raw_data
        norm = apply_edcnorm(d["data"]) if self._params.chk_norm.isChecked() else d["data"]
        idx = int(np.argmin(np.abs(d["kpar"] - self._sel_k)))
        return d["ev_arr"], norm[idx, :].astype(float)

    def _draw_mdc_edc(self):
        ax_mdc = self._mdc_edc.axes[0]
        ax_edc = self._mdc_edc.axes[1]
        ax_mdc.cla(); ax_edc.cla()
        self._mdc_edc._dark()

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
                        alpha=0.55, label="lissé-det", zorder=2)

            # ── zone de contrainte xg (center_init ± xg_range) ───────────────
            cx  = self._params.sp_cx.value()
            xgr = self._params.sp_xg.value()
            ax_mdc.axvspan(cx - xgr, cx + xgr, alpha=0.08, color="cyan",
                           zorder=0, label=f"xg±{xgr:.2f}")
            ax_mdc.axvline(cx, color="cyan", lw=0.6, ls=":", alpha=0.45, zorder=1)

            # ── marqueurs kF_init par paire ───────────────────────────────────
            n_p = self._params.sp_np.value()
            for pi, pp in enumerate(self._params._pair_params[:n_p]):
                kf = pp.get("kF_init", 0.30)
                pc = PAIR_COLORS[pi % len(PAIR_COLORS)]
                ax_mdc.axvline(cx + kf, color=pc, lw=0.8, ls="-.", alpha=0.7, zorder=2)
                ax_mdc.axvline(cx - kf, color=pc, lw=0.8, ls="-.", alpha=0.7, zorder=2)

            # ── modèle Lorentzien décomposé ───────────────────────────────────
            total = np.zeros_like(mdc_n)
            for i, (curve, km, kp, cl, cr) in enumerate(pairs):
                c = PAIR_COLORS[i % len(PAIR_COLORS)]
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
                f"MDC  E={self._sel_ev:.3f} eV  ±{int_win*1000:.0f} meV",
                fontsize=8, color="w")
            ax_mdc.tick_params(colors="w", labelsize=7)
            ax_mdc.legend(fontsize=7, facecolor="#333", labelcolor="w",
                          loc="upper right", framealpha=0.7, ncol=2)
            for sp in ax_mdc.spines.values(): sp.set_edgecolor("#555")

        # ── EDC ──────────────────────────────────────────────────────────────
        res2 = self._get_edc()
        if res2 is not None:
            ev_arr, edc = res2
            lo, hi = np.nanpercentile(edc, [1, 99])
            edc_n = np.clip((edc - lo) / (hi - lo + 1e-12), 0, 1)

            ax_edc.plot(ev_arr, edc_n, color="#7dd3fc", lw=1.2)
            ax_edc.fill_between(ev_arr, 0, edc_n, alpha=0.15, color="#7dd3fc")
            ax_edc.axvline(0, color="cyan", lw=0.8, ls="--", alpha=0.7)
            ax_edc.axvline(self._sel_ev, color="lime", lw=1.0, ls=":")
            ax_edc.set_xlabel("E − EF (eV)", fontsize=8, color="w")
            ax_edc.set_ylabel("I (norm.)", fontsize=8, color="w")
            ax_edc.set_title(f"EDC  k={self._sel_k:.3f} π/a",
                             fontsize=8, color="w")
            ax_edc.tick_params(colors="w", labelsize=7)
            for sp in ax_edc.spines.values(): sp.set_edgecolor("#555")

        self._mdc_edc.fig.tight_layout(pad=0.5)
        self._mdc_edc.redraw()

    # ─────────────────────────────────────────────────────────────────────────
    # Interactions carte
    # ─────────────────────────────────────────────────────────────────────────

    def _on_map_click(self, event):
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
                self._params.lbl_res.setText(
                    f"✓  E={self._sel_ev:.3f} eV\n"
                    f"kF=[{k0s}] π/a\n"
                    f"γ={r['gamma']:.4f}  rms={r['residual']:.4f}\n"
                    f"xg={r['xg']:.4f} π/a")
                self._status(f"Guess OK  kF={k0s}  γ={r['gamma']:.4f}")
            else:
                self._params.lbl_res.setText("✗  Fit échoué")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                    ha="center", va="center", color="tomato", fontsize=8)
            traceback.print_exc()
        self._mdc_edc.fig.tight_layout(pad=0.5)
        self._mdc_edc.redraw()

    def _fit_full(self):
        if AP is None: self._status("⚠ arpes_plots non chargé"); return
        data, kpar, ev = self._get_work_data()
        if data is None: return
        fp = self._params.get_fit_params()

        self._status("Fit complet en cours …")
        QApplication.processEvents()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kF_init_list = [p.get("kF_init", 0.30) for p in (fp.pairs or [])]
                fr = AP.fit_mdc_peak_pairs(
                    data, kpar, ev,
                    n_pairs=fp.n_pairs, ev_start=fp.ev_start, ev_end=fp.ev_end,
                    smooth_fit=fp.smooth_fit, smooth_detect=fp.smooth_detect,
                    gamma_init=fp.gamma_init, gamma_max=fp.gamma_max,
                    kF_init=kF_init_list or None, center_init=fp.center_init,
                    xg_range=fp.xg_range, min_amplitude=fp.min_amplitude,
                    max_jump=fp.max_jump, scan_direction=fp.scan_direction,
                    width_mode=fp.width_mode, k_min=fp.k_min, k_max=fp.k_max,
                    k0_max=fp.k0_max, verbose=False,
                )
            self._fit_res = fr

            # Sauvegarder dans la session
            if self._current_path:
                name  = self._session.key_for_path(self._current_path)
                entry = self._session.get_or_create(name)
                entry.fit_params  = fp
                entry.ef_offset   = self._params.sp_ef.value()
                entry.edcnorm     = self._params.chk_norm.isChecked()
                entry.view_mode   = self._cmb_view.currentText()
                entry.meta.hv     = self._raw_data["hv"]
                self._session.set_fit_result(name, fr)
                self._browser.refresh_item(name)

            n_e  = len(fr["e_fitted"])
            n_ok = int(np.isfinite(np.asarray(fr["kF_minus"][0])).sum())
            self._params.lbl_res.setText(
                f"✓  Fit complet  {n_ok}/{n_e} points\n"
                f"xg = {float(np.nanmean(fr['xg'])):.4f} π/a")
            self._draw_bm()
            self._status(f"Fit OK — {n_ok}/{n_e}  xg={float(np.nanmean(fr['xg'])):.4f}")
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
        if self._raw_data is None or AP is None:
            self._status("⚠ Données ou arpes_plots manquants"); return
        d  = self._raw_data
        T  = 28.0   # valeur par défaut — à améliorer via logbook
        edc_avg = np.nanmean(d["data"], axis=0).astype(float)

        # Fenêtre de recherche
        search = (-0.35, 0.05)
        mask = (d["ev_arr"] >= search[0]) & (d["ev_arr"] <= search[1])
        if mask.sum() < 20:
            self._status("⚠ Plage EF trop étroite"); return

        try:
            import matplotlib.pyplot as _plt
            fig_tmp, ax_tmp = _plt.subplots(figsize=(5, 3))
            r = AP.fit_fermi_edge(
                d["ev_arr"], edc_avg,
                temperature_K=T, fit_range=(-0.15, 0.10),
                sigma_resolution_init=0.025, fix_kBT=True,
                units="binding", ax=ax_tmp, verbose=False,
            )
            _plt.close(fig_tmp)
            ef_shift = float(r["EF"])
            msg = (f"EF fit : {ef_shift*1000:+.1f} meV\n"
                   f"FWHM_res : {r['fwhm_res']*1000:.0f} meV\n"
                   f"Appliquer comme correction EF ?")
            reply = QMessageBox.question(self, "Calibration EF", msg,
                                         QMessageBox.StandardButton.Yes |
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                new_off = self._params.sp_ef.value() - ef_shift
                self._params.sp_ef.setValue(new_off)
                self._load_file(self._current_path)
                self._status(f"EF corrigé : {ef_shift*1000:+.1f} meV → offset={new_off:.4f} eV")
        except Exception as e:
            self._status(f"⚠ Calibration EF : {e}"); traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────────────
    # Copy params
    # ─────────────────────────────────────────────────────────────────────────

    def _copy_params(self):
        """Sauvegarde les params courants dans tous les fichiers non-fittés."""
        if not self._current_path: return
        fp = self._params.get_fit_params()
        n  = 0
        for name, entry in self._session.files.items():
            if entry.fit_result is None and name != self._session.key_for_path(self._current_path):
                entry.fit_params = fp; n += 1
        self._session.save()
        self._status(f"Params copiés vers {n} fichier(s) non-fittés")

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
