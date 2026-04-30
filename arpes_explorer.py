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
import unicodedata
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
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import find_peaks
from arpes_norm import remove_grid_artifact as remove_detector_grid_artifact

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
    azi: Optional[float] = None        # azimut échantillon (deg) — None = inconnu
    polar: Optional[float] = None      # manipulateur P / theta (deg)
    tilt: Optional[float] = None       # manipulateur T / phi (deg)


@dataclass
class FileEntry:
    ef_offset: float   = 0.052
    edcnorm: bool      = True
    view_mode: str     = "EDCnorm"          # Raw / EDCnorm / SecDev / Curvature
    fit_params: FitParams   = field(default_factory=FitParams)
    fit_result: Optional[dict] = None        # sérialisé (listes, pas ndarray)
    meta: FileMeta     = field(default_factory=FileMeta)
    fs_center_kx: Optional[float] = None
    fs_center_ky: Optional[float] = None
    grid_correction: dict = field(default_factory=dict)
    ef_correction: dict = field(default_factory=dict)
    # ef_correction = {} ou {"mode": "poly", "poly_coefs": [...], "k_min": ..,
    # "k_max": .., "T": .., "fwhm_res": .., "rms": .., "n_valid": ..,
    # "source": "self"|"reference", "ref_file": "..."}

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
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
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
        self.logbook_path: str = ""
        self.logbook_sheet: str = ""
        self.logbook_mapping: dict[str, str] = {}
        self.logbook_records: list[dict] = []
        self.gamma_reference: dict = {}
        self.angle_offsets: dict = {}
        self.ef_reference: dict    = {}    # correction EF(theta) de référence (Au)

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
            "logbook_path": self.logbook_path,
            "logbook_sheet": self.logbook_sheet,
            "logbook_mapping": _to_serial(self.logbook_mapping),
            "logbook_records": _to_serial(self.logbook_records),
            "gamma_reference": _to_serial(self.gamma_reference),
            "angle_offsets":   _to_serial(self.angle_offsets),
            "ef_reference":    _to_serial(self.ef_reference),
            "files": {
                name: _to_serial(asdict(entry))
                for name, entry in self.files.items()
            },
        }
        self.json_path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path):
        raw = json.loads(path.read_text())
        self.work_func = raw.get("work_func", 4.031)
        self.logbook_path = raw.get("logbook_path", "")
        self.logbook_sheet = raw.get("logbook_sheet", "")
        self.logbook_mapping = raw.get("logbook_mapping", {})
        self.logbook_records = raw.get("logbook_records", [])
        self.gamma_reference = raw.get("gamma_reference", {})
        self.angle_offsets   = raw.get("angle_offsets", {}) or {}
        self.ef_reference    = raw.get("ef_reference", {}) or {}
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
                fs_center_kx = edict.get("fs_center_kx"),
                fs_center_ky = edict.get("fs_center_ky"),
                grid_correction = edict.get("grid_correction", {}) or {},
                ef_correction   = edict.get("ef_correction", {}) or {},
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


def _norm_text(value) -> str:
    s = "" if value is None else str(value)
    s = s.replace("ν", "nu").replace("Ν", "nu")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _cell_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", ".")
        m = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if not m:
            return None
        value = m.group(0)
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _pick_column(columns: list[str], groups: list[list[str]]) -> str:
    normalized = {c: _norm_text(c) for c in columns}
    for group in groups:
        keys = [_norm_text(k) for k in group]
        for col, name in normalized.items():
            if all(k in name for k in keys):
                return col
    return ""


def _infer_logbook_mapping(columns: list[str]) -> dict[str, str]:
    return {
        "file": _pick_column(columns, [
            ["file"], ["filename"], ["fichier"], ["scan"], ["measurement"],
            ["measure"], ["name"], ["nom"], ["run"], ["sample"],
        ]),
        "hv": _pick_column(columns, [
            ["hv"], ["hnu"], ["photon", "energy"], ["photon", "energie"],
            ["energy", "ev"], ["energie", "ev"], ["hn"],
        ]),
        "temperature": _pick_column(columns, [
            ["temperature"], ["temp"], ["t", "k"],
        ]),
        "polarization": _pick_column(columns, [
            ["polarization"], ["polarisation"], ["pol"],
        ]),
        "azi": _pick_column(columns, [
            ["azi"], ["azimuth"], ["azimut"], ["phi", "azimuth"],
        ]),
        "polar": _pick_column(columns, [
            ["theta"], ["polar", "angle"], ["polar", "deg"],
            ["manip", "p"], ["sample", "p"],
        ]),
        "tilt": _pick_column(columns, [
            ["phi"], ["tilt"], ["manip", "t"], ["sample", "t"],
        ]),
        "direction": _pick_column(columns, [
            ["direction"], ["direct"], ["cut"], ["coupe"], ["chemin"],
        ]),
    }


def _path_match_tokens(path: str | Path, session_folder: Path | None) -> list[str]:
    p = Path(path)
    tokens = [p.name, p.stem]
    if session_folder is not None:
        try:
            rel = p.resolve().relative_to(session_folder.resolve())
            tokens.extend([str(rel), rel.name, rel.stem])
        except Exception:
            pass
    return sorted({t for t in tokens if t}, key=len, reverse=True)


def _record_matches_path(record_value, path: str | Path, session_folder: Path | None) -> bool:
    value = _cell_text(record_value)
    if not value:
        return False
    value_norm = value.lower()
    for token in _path_match_tokens(path, session_folder):
        token_norm = token.lower()
        if value_norm == token_norm:
            return True
        pat = r"(?<![A-Za-z0-9])" + re.escape(token_norm) + r"(?![A-Za-z0-9])"
        if re.search(pat, value_norm):
            return True
    return False


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
                    a_lattice: float = 3.96, hv: float | None = None,
                    temperature: float | None = None,
                    azi: float | None = None,
                    pol: str = "",
                    angle_offsets: dict | None = None) -> dict | None:
    if not ERLAB_OK or load_arpes is None:
        return None
    ds = load_arpes(path, work_func=work_func, ef_offset=ef_offset,
                    a_lattice=a_lattice, hv=hv,
                    temperature=temperature,
                    azi=float(azi) if azi is not None else 0.0,
                    pol=pol,
                    angle_offsets=angle_offsets)
    return ds.as_legacy_bandmap_dict()


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
        from arpes_plots import apply_ef_correction_per_column as _apply
    except Exception:
        return d, {}
    data_corr = _apply(data, kpar, ev, ef_smooth)
    out = dict(d)
    out["data"] = data_corr
    info = {"ef_smooth": ef_smooth, "ef_center": float(np.interp(0.0, kpar, ef_smooth))}
    return out, info


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
        self.sp_int_win.valueChanged.connect(self.params_changed)
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
            "→ Solaris/DA30 : lu automatiquement depuis le fichier."
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
        btn_ef_ref = QPushButton("⇩  Appliquer Au de référence")
        btn_ef_ref.setToolTip(
            "Applique au fichier courant la correction EF de référence sauvegardée\n"
            "dans la session (mesurée sur un échantillon Au, par exemple)."
        )
        btn_ef_ref.clicked.connect(self.ef_apply_reference_requested)
        btn_log = QPushButton("📒  Charger logbook")
        btn_log.clicked.connect(self.logbook_requested)
        btn_copy = QPushButton("📋  Copier params → fichier suivant")
        btn_copy.clicked.connect(self.copy_params_requested)
        fl_ef.addRow("φ (eV):",       self.sp_phi)
        fl_ef.addRow("hν (eV):", self.sp_hv)
        fl_ef.addRow("EF offset:",    self.sp_ef)
        fl_ef.addRow(self.chk_norm)
        fl_ef.addRow(btn_log)
        fl_ef.addRow(btn_ef)
        fl_ef.addRow(btn_ef_ref)
        fl_ef.addRow(btn_copy)
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
            w.valueChanged.connect(self.params_changed)
        self.cmb_wm.currentIndexChanged.connect(self.params_changed)

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
        _fcl.addWidget(grp_f)

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
                 half_width_init=0.15, source_name="", current_offset=0.0):
        super().__init__(parent)
        self.setWindowTitle("Calibration EF")
        self.resize(900, 620)
        self._data  = np.asarray(data, dtype=float)
        self._kpar  = np.asarray(kpar, dtype=float)
        self._ev    = np.asarray(ev_arr, dtype=float)
        self._fit   = None
        self.result_payload = None
        self._current_offset = float(current_offset)

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
        self.chk_auto.setToolTip("Centre la fenêtre sur le gradient max de l'EDC moyenne.")
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
    def _draw_initial_preview(self):
        edc = np.nanmean(self._data, axis=0)
        self._ax_edc.clear()
        self._ax_edc.plot(self._ev, edc, "k-", lw=1.2, label="EDC moyenne")
        self._ax_edc.axvline(0.0, color="gray", ls="--", lw=0.7)
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
            win = ap.auto_ef_window(self._ev, edc, half_width=hw) if auto else (-hw, hw)
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
        self._bm_canvas.canvas.mpl_connect(
            "button_press_event", self._on_fit_roi_press)
        self._bm_canvas.canvas.mpl_connect(
            "motion_notify_event", self._on_fit_roi_motion)
        self._bm_canvas.canvas.mpl_connect(
            "button_release_event", self._on_fit_roi_release)
        carte_lay.addWidget(self._bm_canvas, stretch=1)
        self._tabs.addTab(carte_widget, "🗺  BM")

        # Tab MDC Fit : mini-BM + diagrammes MDC/EDC, pour garder le fit séparé
        mdc_widget = QWidget()
        mdc_lay = QVBoxLayout(mdc_widget)
        mdc_lay.setContentsMargins(0, 0, 0, 0)
        self._mdc_map_canvas = MplCanvas(figsize=(7, 5), toolbar=True)
        self._mdc_map_canvas.canvas.mpl_connect(
            "button_press_event", self._on_map_click)
        self._mdc_map_canvas.canvas.mpl_connect(
            "button_press_event", self._on_fit_roi_press)
        self._mdc_map_canvas.canvas.mpl_connect(
            "motion_notify_event", self._on_fit_roi_motion)
        self._mdc_map_canvas.canvas.mpl_connect(
            "button_release_event", self._on_fit_roi_release)
        mdc_lay.addWidget(self._mdc_map_canvas, stretch=5)
        self._mdc_edc = MplCanvas(figsize=(7, 2.5), nrows=2)
        mdc_lay.addWidget(self._mdc_edc, stretch=2)
        self._tabs.addTab(mdc_widget, "🎯  MDC Fit")

        # Tab Résultats
        self._results = ResultsPanel(self._session)
        self._tabs.addTab(self._results, "📊  Résultats")

        # Tab FS : carte de Fermi dédiée, contrôles dans le panneau droit FS
        self._fs_canvas = FermiSurfaceCanvas() if FermiSurfaceCanvas is not None else QWidget()
        if FermiSurfaceCanvas is not None and hasattr(self._fs_canvas, "canvas"):
            self._fs_canvas.canvas.mpl_connect(
                "button_press_event", self._on_fs_map_click)
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
        self._params.ef_apply_reference_requested.connect(self._apply_ef_reference_to_current)
        self._params.logbook_requested.connect(self._load_logbook_dialog)
        self._params.gamma_bm_requested.connect(self._estimate_gamma_bm)
        self._params.gamma_ref_requested.connect(self._apply_gamma_reference_to_bm)
        self._params.grid_requested.connect(self._apply_grid_correction)
        self._params.grid_reset_requested.connect(self._reset_grid_correction)
        self._params.fit_roi_requested.connect(self._set_fit_roi_pick_mode)
        self._params.fit_roi_reset_requested.connect(self._reset_fit_roi_range)
        self._params.set_context("bm")
        right_split.addWidget(self._params)
        right_split.setSizes([550])

        self._fs_controls = FSControlPanel() if FSControlPanel is not None else QWidget()
        if FSControlPanel is not None:
            self._fs_controls.params_changed.connect(self._on_fs_params_changed)
            self._fs_controls.redraw_requested.connect(self._draw_fs_tab)
            if hasattr(self._fs_controls, "gamma_requested"):
                self._fs_controls.gamma_requested.connect(self._detect_fs_gamma)
            if hasattr(self._fs_controls, "manual_center_requested"):
                self._fs_controls.manual_center_requested.connect(self._set_fs_center_pick_mode)

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
        if index == 0:
            self._params.set_context("bm")
        elif index == 1:
            self._params.set_context("mdc")
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
        self._session.gamma_reference = {
            "kx": float(kx),
            "ky": float(ky),
            "polar": float(meta.get("polar", 0.0) or 0.0),
            "tilt": float(meta.get("tilt_ref", 0.0) or 0.0),
            "azi": float(azi_ref) if azi_ref is not None else None,
            "hv": self._raw_data.get("hv"),
            "path": self._raw_data.get("path"),
            "polar_already_applied_to_kx": bool(meta.get("polar_already_applied_to_kx", False)),
            "source": source,
        }
        if entry_now and entry_now.meta.direction:
            self._session.gamma_reference["direction"] = entry_now.meta.direction
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
        """Convertit un decalage k (pi/a) en offset angulaire CLS."""
        try:
            hv_val = float(hv if hv is not None else self._params.sp_hv.value())
            work_func = float(self._params.sp_phi.value())
        except Exception:
            return None
        ek = hv_val - work_func
        if not np.isfinite(ek) or ek <= 0:
            return None
        c_arpes = 0.51233
        a_lattice = 3.96
        scale = c_arpes * np.sqrt(ek) * a_lattice / np.pi
        if not np.isfinite(scale) or scale <= 0:
            return None
        arg = float(k_pi_a) / scale
        if abs(arg) > 1.0:
            arg = float(np.clip(arg, -1.0, 1.0))
        return float(np.degrees(np.arcsin(arg)))

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
        theta0 = self._k_to_angle_offset_deg(kx, hv=hv)
        tilt0 = self._k_to_angle_offset_deg(ky, hv=hv)
        if theta0 is None or tilt0 is None:
            return {}
        out = {
            "mode": "cls_angle_offsets",
            "theta0_deg": float(theta0),
            "tilt0_deg": float(tilt0),
            "source": source,
            "ref_path": ref_path or "",
            "hv": float(hv) if hv is not None and np.isfinite(float(hv)) else None,
            "work_func": float(self._params.sp_phi.value()),
            "a_lattice": 3.96,
        }
        if azi is not None:
            out["azi"] = float(azi)
        return out

    def _project_gamma_by_azi(
        self,
        ref: dict,
        azi_target: float | None,
        *,
        warn_label: str = "Γ",
    ) -> tuple[float, float]:
        """Projette le Γ de référence dans le repère du fichier courant.

        La direction ZDB n'est pas utilisée ici : seule la différence d'azimut
        définit la rotation entre la FS de référence et la donnée cible.
        """
        kx_ref = float(ref.get("kx", np.nan))
        ky_ref = float(ref.get("ky", 0.0) or 0.0)
        if not np.isfinite(kx_ref) or not np.isfinite(ky_ref):
            return np.nan, np.nan

        azi_ref = ref.get("azi")
        if azi_ref is None or azi_target is None:
            if abs(ky_ref) > 1e-3:
                self._status(f"⚠ {warn_label} : azi inconnu — projection non corrigée")
            return kx_ref, ky_ref

        d_azi = np.radians(float(azi_target) - float(azi_ref))
        k_parallel = kx_ref * np.cos(d_azi) + ky_ref * np.sin(d_azi)
        k_perp = -kx_ref * np.sin(d_azi) + ky_ref * np.cos(d_azi)
        return float(k_parallel), float(k_perp)

    def _cls_manipulator_from_param(self, path: str | Path) -> dict:
        """Lit rapidement P/T depuis le *_param.txt sans charger les données."""
        p = Path(path)
        if p.is_file():
            param_files = [p.parent / f"{p.name}_param.txt"]
        elif p.is_dir():
            param_files = sorted(p.glob("*_param.txt"))
        else:
            return {}
        for param_file in param_files:
            if not param_file.exists():
                continue
            try:
                for line in param_file.read_text(errors="replace").splitlines():
                    if not line.strip().startswith("{"):
                        continue
                    motors = json.loads(line).get("d", {})
                    out = {}
                    for motor, key in (("P", "polar"), ("T", "tilt")):
                        value = motors.get(motor, {}).get("position")
                        if value is not None:
                            out[key] = float(value)
                    if out:
                        return out
            except Exception:
                continue
        return {}

    def _cls_geometry_for_path(self, path: str | Path, entry: FileEntry | None = None) -> dict:
        """Retourne la meilleure géométrie CLS connue.

        Priorité : `_param.txt` pour P/T, puis métadonnées de session/logbook.
        Les Excel CLS changent beaucoup de forme ; le logbook n'est donc qu'un
        secours pour les champs absents des fichiers bruts, surtout `azi`.
        """
        geom = self._cls_manipulator_from_param(path)
        if entry is not None:
            if geom.get("polar") is None and entry.meta.polar is not None:
                geom["polar"] = float(entry.meta.polar)
            if geom.get("tilt") is None and entry.meta.tilt is not None:
                geom["tilt"] = float(entry.meta.tilt)
            if entry.meta.azi is not None:
                geom["azi"] = float(entry.meta.azi)
            if entry.meta.hv:
                geom["hv"] = float(entry.meta.hv)

        rec = self._find_logbook_record(path)
        mapping = self._session.logbook_mapping or {}
        if rec is not None:
            for key in ("polar", "tilt", "azi", "hv"):
                col = mapping.get(key, "")
                value = _cell_float(rec.get(col)) if col else None
                if value is not None and np.isfinite(value) and geom.get(key) is None:
                    geom[key] = float(value)

        return geom

    def _cls_polar_from_param(self, path: str | Path) -> float | None:
        return self._cls_manipulator_from_param(path).get("polar")

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
        """Candidats de convention pour une BM CLS.

        Les conventions exactes peuvent changer selon la definition de l'azi
        dans le logbook. On teste donc le signe de theta0 et, si possible, les
        deux projections `+ky sin(d_azi)` / `-ky sin(d_azi)`.
        """
        if not primary or not Path(path).is_file():
            return [primary] if primary else []
        candidates: list[dict] = []

        def add(cfg: dict, label: str):
            if not cfg:
                return
            c = dict(cfg)
            c["candidate"] = label
            key = (round(float(c.get("theta0_deg", 0.0)), 8),
                   round(float(c.get("tilt0_deg", 0.0)), 8),
                   c.get("candidate", ""))
            for old in candidates:
                old_key = (round(float(old.get("theta0_deg", 0.0)), 8),
                           round(float(old.get("tilt0_deg", 0.0)), 8),
                           old.get("candidate", ""))
                if old_key == key:
                    return
            candidates.append(c)

        add(primary, "theta0")
        neg = dict(primary)
        neg["theta0_deg"] = -float(neg.get("theta0_deg", 0.0) or 0.0)
        neg["gamma_bm_pi_over_a"] = -float(neg.get("gamma_bm_pi_over_a", 0.0) or 0.0)
        add(neg, "-theta0")

        ref = self._stored_gamma_reference()
        if ref and entry is not None:
            geom = self._cls_geometry_for_path(path, entry)
            p_ref = ref.get("polar")
            p_target = geom.get("polar")
            theta_ref = self._k_to_angle_offset_deg(float(ref.get("kx", 0.0) or 0.0), hv=hv)
            if p_ref is not None and p_target is not None and theta_ref is not None:
                # Convention "angle brut analyseur": le Γ repéré sur la FS
                # correspond à theta_raw ~= P_ref + theta_ref. Pour une BM à
                # polar différent, l'offset à appliquer devient
                # theta_raw - P_target. Cette correction est indispensable pour
                # des cas comme BM9/BM10 où P change fortement.
                raw_theta0 = float(theta_ref) + float(p_ref) - float(p_target)
                cfg = dict(primary)
                cfg["theta0_deg"] = raw_theta0
                cfg["tilt0_deg"] = 0.0
                cfg["source"] = "gamma_reference_projected_to_bm_raw_polar"
                cfg["target_polar"] = float(p_target)
                cfg["ref_polar"] = float(p_ref)
                add(cfg, "raw_polar")
                cfg_neg = dict(cfg)
                cfg_neg["theta0_deg"] = -raw_theta0
                add(cfg_neg, "raw_polar_neg")

            azi_ref = ref.get("azi")
            azi_bm = geom.get("azi", entry.meta.azi if entry.meta.azi is not None else None)
            if azi_ref is not None and azi_bm is not None:
                kx_ref = float(ref.get("kx", np.nan))
                ky_ref = float(ref.get("ky", 0.0) or 0.0)
                if np.isfinite(kx_ref) and np.isfinite(ky_ref):
                    d_azi = np.radians(float(azi_bm) - float(azi_ref))
                    for label, gamma_bm in (
                        ("azi_plus", kx_ref * np.cos(d_azi) + ky_ref * np.sin(d_azi)),
                        ("azi_minus", kx_ref * np.cos(d_azi) - ky_ref * np.sin(d_azi)),
                    ):
                        cfg = self._angle_offsets_from_k_center(
                            float(gamma_bm), 0.0,
                            hv=hv,
                            source=f"gamma_reference_projected_to_bm_{label}",
                            ref_path=ref.get("path"),
                            azi=azi_bm,
                        )
                        if cfg:
                            cfg["gamma_bm_pi_over_a"] = float(gamma_bm)
                            cfg["gamma_ref_source"] = ref.get("source", "")
                            add(cfg, label)
                            if p_ref is not None and p_target is not None:
                                theta_proj = self._k_to_angle_offset_deg(float(gamma_bm), hv=hv)
                                if theta_proj is not None:
                                    cfg_raw = dict(cfg)
                                    cfg_raw["theta0_deg"] = float(theta_proj) + float(p_ref) - float(p_target)
                                    cfg_raw["source"] = f"gamma_reference_projected_to_bm_{label}_raw_polar"
                                    cfg_raw["target_polar"] = float(p_target)
                                    cfg_raw["ref_polar"] = float(p_ref)
                                    add(cfg_raw, f"{label}_raw_polar")
                            cfg_neg = dict(cfg)
                            cfg_neg["theta0_deg"] = -float(cfg_neg.get("theta0_deg", 0.0) or 0.0)
                            cfg_neg["gamma_bm_pi_over_a"] = -float(cfg_neg.get("gamma_bm_pi_over_a", 0.0) or 0.0)
                            add(cfg_neg, f"{label}_neg")

        return candidates

    def _score_bm_gamma_residual(self, d: dict) -> float:
        """Score petit si la BM chargee est centree autour de Γ=0."""
        if AP is None:
            return float("inf")
        try:
            res = AP.estimate_gamma_bm_mdc(
                np.asarray(d["data"], dtype=float),
                np.asarray(d["kpar"], dtype=float),
                np.asarray(d["ev_arr"], dtype=float),
                ev_range=(self._params.sp_evs.value(), self._params.sp_eve.value()),
                k_range=(self._params.sp_kmin.value(), self._params.sp_kmax.value()),
                center_guess=0.0,
                center_window=max(self._params.sp_xg.value() * 2.0, 0.25),
                smooth_sigma=self._params.sp_sfd.value(),
                verbose=False,
            )
            gamma = float(res.get("gamma", np.nan))
            mad = float(res.get("mad", 0.0) or 0.0)
            n = int(res.get("n", 0) or 0)
            if not np.isfinite(gamma) or n < 2:
                return float("inf")
            kpar = np.asarray(d["kpar"], dtype=float)
            k_mid = 0.5 * (float(np.nanmin(kpar)) + float(np.nanmax(kpar)))
            return abs(gamma) + 0.25 * mad + 0.10 * abs(k_mid)
        except Exception:
            return float("inf")

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
        )
        return d, angle_offsets

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
        ref = self._session.gamma_reference or {}
        try:
            kx = float(ref.get("kx", np.nan))
            ky = float(ref.get("ky", 0.0) or 0.0)
        except Exception:
            return {}
        if not np.isfinite(kx) or not np.isfinite(ky):
            return {}
        return ref

    def _gamma_reference_to_bm_center(self, ref: dict) -> tuple[float, float]:
        """Projette le Γ mesuré sur la FS vers l'axe k de la BM courante.

        Le transfert n'est physiquement valide que si :
          - le polar du fichier courant est proche de celui de la référence
            (sinon le `kx` produit par le loader CLS est complètement
            différent — voir point 6 dans la doc) ;
          - l'azi est connu des deux côtés pour corriger la rotation entre
            la référence FS et la nouvelle direction de fente.
        Si le polar diffère trop, on retourne NaN pour signaler à l'appelant
        de NE PAS écrire `sp_cx` automatiquement. Si l'azimut manque, on garde
        les coordonnées de référence non tournées et on affiche un avertissement.
        """
        POLAR_TOLERANCE_DEG = 2.0   # au-delà, transfert non fiable

        if self._raw_data is None:
            return np.nan, 0.0
        meta = self._raw_data.get("metadata", {}) or {}
        entry_now = self._current_entry()

        # Vérification polar : refus net si écart > tolérance
        p_ref = float(ref.get("polar", 0.0) or 0.0)
        p_bm = float(meta.get("polar", 0.0) or 0.0)
        if abs(p_bm - p_ref) > POLAR_TOLERANCE_DEG:
            self._status(
                f"⚠ Γ FS→BM ignoré : polar diffère de {p_bm - p_ref:+.1f}° "
                f"(>±{POLAR_TOLERANCE_DEG:.0f}°). Utilise 'Auto Γ BM'."
            )
            return np.nan, 0.0

        # Étape 1 : rotation azi. La direction ZDB saisie dans le tableau ne
        # bloque plus la propagation : l'azimut est la source de vérité.
        azi_bm = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
        gamma, _ = self._project_gamma_by_azi(ref, azi_bm, warn_label="Γ FS→BM")
        if not np.isfinite(gamma):
            return np.nan, 0.0

        # Étape 2 : correction polar (résidu, pour les loaders qui ne
        # soustraient pas déjà polar à la conversion)
        correction = 0.0
        ref_polar_applied = bool(ref.get("polar_already_applied_to_kx", False))
        bm_polar_applied = bool(meta.get("polar_already_applied_to_kx", False))
        if not (ref_polar_applied and bm_polar_applied):
            hv = self._raw_data.get("hv") or ref.get("hv")
            work_func = self._params.sp_phi.value()
            if hv is not None and float(hv) > work_func:
                c_arpes = 0.51233
                a = 3.96
                ek = float(hv) - float(work_func)
                correction = c_arpes * np.sqrt(ek) * (
                    np.sin(np.radians(p_bm)) - np.sin(np.radians(p_ref))
                ) * a / np.pi
        return gamma + correction, correction

    def _center_current_bm_axis_on_gamma(self, gamma_bm: float, ref: dict | None = None) -> bool:
        """Recentre l'axe k// d'une BM pour que Γ soit affiche a k//=0.

        Les FS sont deja recentrees au dessin via FSParams.kx_center/ky_center.
        Pour les BM, `center_init` seul ne suffit pas : l'axe kpar utilise par
        les graphes/MDC reste brut. On applique donc une translation locale au
        dict charge, sans modifier les fichiers bruts.
        """
        if self._raw_data is None:
            return False
        meta = self._raw_data.get("metadata", {}) or {}
        if meta.get("fs_data") is not None:
            return False
        if meta.get("angle_offsets_applied"):
            return False
        if bool(meta.get("bm_gamma_axis_centered", False)):
            return False
        if not np.isfinite(gamma_bm):
            return False

        kpar = np.asarray(self._raw_data.get("kpar"), dtype=float)
        if kpar.size == 0 or not np.isfinite(kpar).any():
            return False

        shift = float(gamma_bm)
        self._raw_data["kpar"] = kpar - shift
        meta["bm_gamma_axis_centered"] = True
        meta["bm_gamma_axis_shift"] = shift
        meta["bm_gamma_axis_note"] = "kpar_display = kpar_raw - gamma_bm"
        if ref:
            meta["bm_gamma_reference_source"] = ref.get("source", "")
            meta["bm_gamma_reference_path"] = ref.get("path", "")
            meta["bm_gamma_reference_azi"] = ref.get("azi")
        self._raw_data["metadata"] = meta

        if hasattr(self, "_sel_k"):
            self._sel_k = float(self._sel_k - shift)
        return True

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
        cfg = cfg or {}
        try:
            strength = float(cfg.get("strength", 0.85))
        except Exception:
            strength = 0.85
        return {
            "method": "fft2mask",
            "grid_freq": None,
            "grid_period_px": None,
            "notch_width": 2,
            "notch_sigma": 0.8,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "fft2_center_radius": 18.0,
            "fft2_peak_sensitivity": 2.5,
            "fft2_plane": "display",
        }

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
    # Logbook souple CLS/SOLARIS
    # ─────────────────────────────────────────────────────────────────────────

    def _load_logbook_dialog(self):
        start = str(self._session.folder or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Logbook ARPES", start,
            "Logbook (*.xlsx *.xls *.csv *.tsv);;Tous les fichiers (*)")
        if not path:
            return
        try:
            records, mapping, sheet_name = self._read_logbook(Path(path))
            self._session.logbook_path = path
            self._session.logbook_sheet = sheet_name
            self._session.logbook_mapping = mapping
            self._session.logbook_records = records
            self._session.save()
            used = ", ".join(f"{k}={v or '—'}" for k, v in mapping.items())
            sheet_txt = f" [{sheet_name}]" if sheet_name else ""
            self._status(f"Logbook chargé : {Path(path).name}{sheet_txt} | {len(records)} lignes | {used}")
            QMessageBox.information(
                self, "Logbook chargé",
                f"{Path(path).name}{sheet_txt}\n{len(records)} lignes lues.\n\nColonnes détectées :\n{used}"
            )
            if self._current_path:
                self._apply_logbook_to_controls(self._current_path)
        except Exception as exc:
            QMessageBox.warning(self, "Logbook", str(exc))
            self._status(f"⚠ Logbook : {exc}")

    def _read_logbook(self, path: Path) -> tuple[list[dict], dict[str, str], str]:
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("pandas est nécessaire pour lire les logbooks Excel/CSV.") from exc

        suffix = path.suffix.lower()
        sheet_name = ""
        if suffix in {".xlsx", ".xls"}:
            book = pd.ExcelFile(path)
            sheet_name = self._choose_excel_sheet(book.sheet_names)
            if not sheet_name:
                raise ValueError("Aucune feuille Excel sélectionnée.")
            raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
            if raw.dropna(how="all").empty:
                raise ValueError("Le logbook ne contient aucune ligne exploitable.")
            candidates = self._excel_header_candidates(raw)
            guessed = self._best_excel_table(raw, candidates)
            if guessed is None:
                guessed = self._choose_excel_table(raw, candidates)
            if guessed is None:
                raise ValueError("Aucune ligne d'en-tête sélectionnée pour le logbook.")
            df, mapping = guessed
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
            df = df.dropna(how="all")
            if df.empty:
                raise ValueError("Le logbook ne contient aucune ligne exploitable.")
            df.columns = [str(c).strip() for c in df.columns]
            mapping = _infer_logbook_mapping(list(df.columns))
        else:
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path)
            df = df.dropna(how="all")
            if df.empty:
                raise ValueError("Le logbook ne contient aucune ligne exploitable.")
            df.columns = [str(c).strip() for c in df.columns]
            mapping = _infer_logbook_mapping(list(df.columns))

        if not mapping.get("file") or not mapping.get("hv"):
            mapping = self._choose_logbook_mapping(list(df.columns), mapping)
        if not mapping.get("file") or not mapping.get("hv"):
            raise ValueError("Les colonnes fichier et hν sont obligatoires pour appliquer un logbook.")
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        records = self._inherit_logbook_context(records, mapping)
        return records, mapping, sheet_name

    def _inherit_logbook_context(self, records: list[dict], mapping: dict[str, str]) -> list[dict]:
        """Propage les champs de contexte du logbook quand les cellules sont vides.

        Dans les logbooks CLS, `azi` est souvent indique sur une ligne
        d'alignement (FS) puis laisse vide sur les BM suivantes. Ces BM doivent
        pourtant heriter de cet azimut pour que la projection FS->BM soit juste.
        """
        azi_col = mapping.get("azi", "")
        dir_col = mapping.get("direction", "")
        pol_col = mapping.get("polarization", "")
        last_azi = None
        last_dir = ""
        last_pol = ""
        out = []
        for rec in records:
            rec = dict(rec)

            if dir_col:
                direct = _cell_text(rec.get(dir_col))
                if direct:
                    last_dir = direct
                elif last_dir:
                    rec[dir_col] = last_dir

            if pol_col:
                pol = _cell_text(rec.get(pol_col))
                if pol:
                    last_pol = pol
                elif last_pol:
                    rec[pol_col] = last_pol

            if azi_col:
                azi = _cell_float(rec.get(azi_col))
                if azi is not None and np.isfinite(azi):
                    last_azi = float(azi)
                elif last_azi is not None:
                    rec[azi_col] = last_azi

            out.append(rec)
        return out

    def _choose_excel_sheet(self, sheet_names: list[str]) -> str:
        if not sheet_names:
            return ""
        if len(sheet_names) == 1:
            return sheet_names[0]
        dlg = QDialog(self)
        dlg.setWindowTitle("Feuille du logbook")
        lay = QVBoxLayout(dlg)
        label = QLabel("Choisis la feuille qui correspond au compound / dataset.")
        label.setWordWrap(True)
        lay.addWidget(label)
        cmb = QComboBox()
        cmb.addItems(sheet_names)
        preferred = ""
        if self._session.folder is not None:
            preferred = self._session.folder.name
        preferred_norm = _norm_text(preferred)
        if self._session.logbook_sheet in sheet_names:
            cmb.setCurrentText(self._session.logbook_sheet)
        elif preferred in sheet_names:
            cmb.setCurrentText(preferred)
        else:
            for sheet in sheet_names:
                sheet_norm = _norm_text(sheet)
                if sheet_norm and sheet_norm in preferred_norm:
                    cmb.setCurrentText(sheet)
                    break
        lay.addWidget(cmb)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return ""
        return cmb.currentText()

    def _excel_header_candidates(self, raw) -> list[int]:
        candidates: list[int] = []
        for row_idx in range(min(len(raw), 120)):
            values = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
            nonempty = [v for v in values if v]
            if len(nonempty) >= 2:
                candidates.append(row_idx)
        return candidates

    def _excel_table_from_header(self, raw, row_idx: int):
        headers = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
        cols = [h if h else f"column_{i}" for i, h in enumerate(headers)]
        seen: dict[str, int] = {}
        unique_cols = []
        for col in cols:
            n = seen.get(col, 0)
            seen[col] = n + 1
            unique_cols.append(col if n == 0 else f"{col}_{n+1}")
        df = raw.iloc[row_idx + 1:].copy()
        df.columns = unique_cols
        df = df.dropna(how="all")
        mapping = _infer_logbook_mapping(list(df.columns))
        return df, mapping

    def _best_excel_table(self, raw, candidates: list[int]):
        best = None
        best_score = -1
        for row_idx in candidates:
            df, mapping = self._excel_table_from_header(raw, row_idx)
            score = int(bool(mapping.get("file"))) * 3 + int(bool(mapping.get("hv"))) * 3
            score += int(bool(mapping.get("temperature"))) + int(bool(mapping.get("polarization")))
            score += int(bool(mapping.get("direction"))) + int(bool(mapping.get("azi")))
            score += int(bool(mapping.get("polar"))) + int(bool(mapping.get("tilt")))
            score += min(len(df), 20) / 1000
            if score > best_score:
                best = (df, mapping, row_idx)
                best_score = score
        if best is None or best_score < 6:
            return None
        return best[0], best[1]

    def _choose_excel_table(self, raw, candidates: list[int]):
        if not candidates:
            return None
        dlg = QDialog(self)
        dlg.setWindowTitle("Ligne d'en-tête du logbook")
        lay = QVBoxLayout(dlg)
        label = QLabel("Choisis la ligne qui contient les vrais noms de colonnes.")
        label.setWordWrap(True)
        lay.addWidget(label)
        cmb = QComboBox()
        for row_idx in candidates:
            values = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
            preview = " | ".join(v for v in values if v)
            cmb.addItem(f"Ligne {row_idx + 1}: {preview[:140]}", row_idx)
        lay.addWidget(cmb)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return self._excel_table_from_header(raw, int(cmb.currentData()))

    def _choose_logbook_mapping(self, columns: list[str], mapping: dict[str, str]) -> dict[str, str]:
        dlg = QDialog(self)
        dlg.setWindowTitle("Colonnes du logbook")
        lay = QFormLayout(dlg)
        combos: dict[str, QComboBox] = {}
        labels = {
            "file": "Fichier / scan:",
            "hv": "hν:",
            "temperature": "Température:",
            "polarization": "Polarisation:",
            "direction": "Direction / chemin:",
            "azi": "Azimut:",
            "polar": "Polar / theta manip:",
            "tilt": "Tilt / phi manip:",
        }
        choices = [""] + columns
        for key, label in labels.items():
            cmb = QComboBox()
            cmb.addItems(choices)
            current = mapping.get(key, "")
            if current in choices:
                cmb.setCurrentText(current)
            lay.addRow(label, cmb)
            combos[key] = cmb
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return mapping
        return {key: cmb.currentText() for key, cmb in combos.items()}

    def _find_logbook_record(self, path: str | Path) -> dict | None:
        mapping = self._session.logbook_mapping or {}
        file_col = mapping.get("file", "")
        if not file_col:
            return None
        for rec in self._session.logbook_records:
            if _record_matches_path(rec.get(file_col), path, self._session.folder):
                return rec
        return None

    def _apply_logbook_to_controls(self, path: str | Path) -> bool:
        rec = self._find_logbook_record(path)
        if rec is None:
            return False
        mapping = self._session.logbook_mapping or {}
        changed = False

        hv = _cell_float(rec.get(mapping.get("hv", "")))
        if hv is not None and hv > 0:
            self._params.sp_hv.blockSignals(True)
            self._params.sp_hv.setValue(hv)
            self._params.sp_hv.blockSignals(False)
            entry = self._session.get_or_create(self._session.key_for_path(path))
            entry.meta.hv = hv
            changed = True

        temp_col = mapping.get("temperature", "")
        temp = _cell_float(rec.get(temp_col)) if temp_col else None
        if temp is not None:
            entry = self._session.get_or_create(self._session.key_for_path(path))
            entry.meta.temperature = temp
            changed = True

        pol_col = mapping.get("polarization", "")
        pol = _cell_text(rec.get(pol_col)) if pol_col else ""
        if pol:
            entry = self._session.get_or_create(self._session.key_for_path(path))
            entry.meta.polarization = pol
            changed = True

        azi_col = mapping.get("azi", "")
        azi_val = _cell_float(rec.get(azi_col)) if azi_col else None
        if azi_val is not None and np.isfinite(azi_val):
            entry = self._session.get_or_create(self._session.key_for_path(path))
            entry.meta.azi = float(azi_val)
            changed = True

        dir_col = mapping.get("direction", "")
        dir_val = _cell_text(rec.get(dir_col)) if dir_col else ""
        if dir_val:
            entry = self._session.get_or_create(self._session.key_for_path(path))
            entry.meta.direction = dir_val
            changed = True

        polar_col = mapping.get("polar", "")
        polar_val = _cell_float(rec.get(polar_col)) if polar_col else None
        if polar_val is not None and np.isfinite(polar_val):
            entry = self._session.get_or_create(self._session.key_for_path(path))
            entry.meta.polar = float(polar_val)
            changed = True

        tilt_col = mapping.get("tilt", "")
        tilt_val = _cell_float(rec.get(tilt_col)) if tilt_col else None
        if tilt_val is not None and np.isfinite(tilt_val):
            entry = self._session.get_or_create(self._session.key_for_path(path))
            entry.meta.tilt = float(tilt_val)
            changed = True

        if changed:
            self._session.save()
        return changed

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
            logbook_hit = self._apply_logbook_to_controls(path)
            if entry.meta.hv and entry.meta.hv > 0 and self._params.sp_hv.value() <= 0:
                self._params.sp_hv.blockSignals(True)
                self._params.sp_hv.setValue(float(entry.meta.hv))
                self._params.sp_hv.blockSignals(False)
            hv_for_load = self._params.sp_hv.value()
            angle_offsets = self._angle_offsets_for_load(path, entry, hv_for_load)
            if angle_offsets and Path(path).is_file():
                d, angle_offsets = self._load_with_best_angle_offsets(path, entry, hv_for_load, angle_offsets)
            else:
                d = load_arpes_file(path,
                                    self._params.sp_phi.value(),
                                    self._params.sp_ef.value(),
                                    hv=hv_for_load,
                                    temperature=entry.meta.temperature if entry.meta.temperature > 0 else None,
                                    azi=entry.meta.azi,
                                    pol=entry.meta.polarization,
                                    angle_offsets=angle_offsets)
            if d is None:
                self._status("⚠ erlab non disponible")
                return

            # Température lue depuis les métadonnées du loader (Solaris/CLS)
            md = d.get("metadata", {}) or {}
            t_md = md.get("temperature")
            try:
                t_md = float(t_md) if t_md is not None else None
            except (TypeError, ValueError):
                t_md = None
            if t_md is not None and np.isfinite(t_md) and t_md > 0:
                entry.meta.temperature = t_md

            # Correction EF par colonne si calibrée pour ce fichier
            if entry.ef_correction.get("mode") == "poly":
                d, ef_info = apply_ef_correction_to_dict(d, entry.ef_correction)
                self._ef_correction_info = ef_info
            else:
                self._ef_correction_info = {}

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
            self._load_grid_controls(entry.grid_correction)

            # Restaurer fit_result si disponible
            if entry.fit_result:
                self._fit_res = entry.fit_result

            self._apply_stored_gamma_to_current_file(save_entry=True)

            self._update_display_data()
            grid_note = ""
            if entry.grid_correction.get("enabled"):
                grid_msg = self._grid_status_text(self._grid_display_info, "affichage BM")
                grid_note = "  |  " + grid_msg
                self._params.lbl_grid.setText(grid_msg)
            gamma_note = ""
            md_now = d.get("metadata", {}) or {}
            if md_now.get("angle_offsets_applied"):
                ao = md_now.get("angle_offsets_applied") or {}
                cand = md_now.get("angle_offset_candidate", ao.get("candidate", ""))
                cand_txt = f" {cand}" if cand else ""
                gamma_note = (
                    f"  |  Γ offset angulaire{cand_txt} θ0={float(ao.get('theta0_deg', 0.0)):+.3f}°"
                )
            elif md_now.get("bm_gamma_axis_centered"):
                gamma_note = f"  |  Γ axe shift={float(md_now.get('bm_gamma_axis_shift', 0.0)):+.4f}"
            self._sel_ev = float(np.clip(-0.30, d["ev_arr"].min(), d["ev_arr"].max()))
            self._sel_k  = 0.0
            self._sync_ev_spinbox()

            self._draw_bm()
            self._draw_mdc_edc()
            if self._tabs.currentIndex() == 3:
                self._draw_fs_tab()

            self._browser.select_file(path)
            hv_txt = f"{d['hv']:.0f} eV" if d.get("hv") is not None else "—"
            lb_txt = "  |  logbook" if logbook_hit else ""
            self._status(
                f"Chargé : {Path(path).name}  hν={hv_txt}  |  "
                f"k {d['kpar'].min():.2f}→{d['kpar'].max():.2f} π/a  |  "
                f"E {d['ev_arr'].min():.3f}→{d['ev_arr'].max():.3f} eV{lb_txt}{grid_note}{gamma_note}"
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
        self._grid_display_info = {}

        if mode == "Raw":
            disp = raw
        elif mode == "EDCnorm":
            disp = apply_edcnorm(raw) if self._params.chk_norm.isChecked() else raw
        elif mode == "SecDev":
            norm = apply_edcnorm(raw) if self._params.chk_norm.isChecked() else raw
            disp = compute_secdev(norm, d["kpar"], d["ev_arr"])
        elif mode == "Curvature":
            norm = apply_edcnorm(raw) if self._params.chk_norm.isChecked() else raw
            disp = compute_curvature(norm, d["kpar"], d["ev_arr"])
        else:
            disp = raw

        entry = self._current_entry()
        cfg = entry.grid_correction if entry and entry.grid_correction.get("enabled") else None
        if cfg:
            grid_cfg = self._display_grid_config(cfg)
            try:
                disp, info = remove_detector_grid_artifact(np.asarray(disp, dtype=float), axis=0, **grid_cfg)
                info.update({
                    "method": "display_fft2mask",
                    "view_mode": mode,
                    "target": "display",
                    "shape": tuple(np.asarray(disp).shape),
                    "strength": grid_cfg["strength"],
                })
                self._grid_display_info = info
            except Exception as exc:
                self._grid_display_info = {
                    "method": "display_fft2mask",
                    "error": str(exc),
                    "view_mode": mode,
                    "strength": grid_cfg["strength"],
                }

        self._data_disp = disp

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
        if self._tabs.currentIndex() == 1:
            self._draw_mdc_edc()

    def _fit_roi_bounds(self) -> tuple[float, float, float, float] | None:
        if self._raw_data is None:
            return None
        d = self._raw_data
        k0, k1 = sorted((float(self._params.sp_kmin.value()), float(self._params.sp_kmax.value())))
        e0, e1 = sorted((float(self._params.sp_evs.value()), float(self._params.sp_eve.value())))
        k0 = float(np.clip(k0, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        k1 = float(np.clip(k1, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        e0 = float(np.clip(e0, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        e1 = float(np.clip(e1, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        if k1 <= k0 or e1 <= e0:
            return None
        return k0, k1, e0, e1

    def _fit_roi_data(self, disp: np.ndarray, kpar: np.ndarray, ev: np.ndarray) -> np.ndarray:
        bounds = self._fit_roi_bounds()
        if bounds is None:
            return np.asarray(disp)
        k0, k1, e0, e1 = bounds
        mk = (kpar >= k0) & (kpar <= k1)
        me = (ev >= e0) & (ev <= e1)
        if not mk.any() or not me.any():
            return np.asarray(disp)
        return np.asarray(disp)[np.ix_(mk, me)]

    def _map_color_kwargs(self, disp: np.ndarray, mode: str, *, roi_scale: bool = False) -> tuple[str, dict]:
        d = self._raw_data
        ref = self._fit_roi_data(disp, d["kpar"], d["ev_arr"]) if roi_scale and d is not None else disp
        if mode in ("Raw", "EDCnorm"):
            finite = ref[np.isfinite(ref)]
            vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
            return "inferno", {"vmin": 0, "vmax": max(vmax, 1e-12)}
        pos = ref[np.isfinite(ref) & (ref > 0)]
        vmax = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
        return "hot_r", {"vmin": 0, "vmax": max(vmax, 1e-12)}

    def _draw_fit_roi_overlay(self, ax):
        bounds = self._fit_roi_bounds()
        if bounds is None:
            return
        k0, k1, e0, e1 = bounds
        rect = Rectangle(
            (k0, e0), k1 - k0, e1 - e0,
            fill=False, edgecolor="#7dd3fc", linewidth=1.1,
            linestyle="--", alpha=0.95, zorder=8,
        )
        ax.add_patch(rect)

    def _ef_offset_text(self) -> str:
        return f"EF offset={self._params.sp_ef.value()*1000:+.0f} meV"

    def _draw_ef_label(self, ax, *, horizontal: bool = True):
        txt = f"EF  {self._ef_offset_text()}"
        if horizontal:
            x0, x1 = ax.get_xlim()
            x = x0 + 0.012 * (x1 - x0)
            ax.text(
                x, 0.0, txt,
                color="cyan", fontsize=8, va="bottom", ha="left",
                bbox=dict(facecolor="#1a1a1a", edgecolor="none", alpha=0.65, pad=1.5),
                zorder=9,
            )
        else:
            y0, y1 = ax.get_ylim()
            y = y0 + 0.88 * (y1 - y0)
            ax.text(
                0.0, y, txt,
                color="cyan", fontsize=7, va="top", ha="left", rotation=90,
                bbox=dict(facecolor="#1a1a1a", edgecolor="none", alpha=0.65, pad=1.2),
                zorder=9,
            )

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
            cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=False)
            n = _norm(ckw["vmin"], ckw["vmax"])
            kw = dict(norm=n) if n else ckw
            ax.pcolormesh(kpar, ev, disp.T, cmap=cmap, shading="auto", **kw)
        elif mode == "SecDev":
            cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=False)
            n = _norm(ckw["vmin"], ckw["vmax"])
            kw = dict(norm=n) if n else ckw
            ax.pcolormesh(kpar, ev, disp.T, cmap=cmap, shading="auto", **kw)
        elif mode == "Curvature":
            cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=False)
            n = _norm(ckw["vmin"], ckw["vmax"])
            kw = dict(norm=n) if n else ckw
            ax.pcolormesh(kpar, ev, disp.T, cmap=cmap, shading="auto", **kw)

        int_win = self._params.sp_int_win.value()
        ax.axhline(0,          color="cyan", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(0,          color="w",    lw=0.5, ls="--", alpha=0.4)
        ax.axhspan(self._sel_ev - int_win, self._sel_ev + int_win,
                   alpha=0.14, color="lime", zorder=2, lw=0)
        ax.axhline(self._sel_ev, color="lime", lw=0.8, ls="--", zorder=3)
        ax.axvline(self._sel_k,  color="lime", lw=1.0, ls=":",  zorder=3)

        self._draw_fit_roi_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)

        fname = Path(d["path"]).name
        ax.set_xlabel("k// (π/a)", fontsize=10, color="w")
        ax.set_ylabel("E − EF (eV)", fontsize=10, color="w")
        ax.set_title(f"{fname}  [{mode}]  {self._ef_offset_text()}", fontsize=9, color="w")
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
        cmap, ckw = self._map_color_kwargs(disp, mode, roi_scale=True)
        ax.pcolormesh(kpar, ev, disp.T, cmap=cmap, shading="auto", **ckw)
        int_win = self._params.sp_int_win.value()
        ax.axhline(0, color="cyan", lw=0.7, ls="--", alpha=0.6)
        ax.axhspan(self._sel_ev - int_win, self._sel_ev + int_win,
                   alpha=0.14, color="lime", zorder=2, lw=0)
        ax.axhline(self._sel_ev, color="lime", lw=0.7, ls="--", zorder=3)
        ax.axvline(self._sel_k,  color="lime", lw=0.9, ls=":", zorder=3)
        bounds = self._fit_roi_bounds()
        if bounds is not None:
            k0, k1, e0, e1 = bounds
            ax.set_xlim(k0, k1)
            ax.set_ylim(e0, e1)
            self._draw_fit_roi_overlay(ax)
        self._draw_kf_overlay(ax)
        self._draw_ef_label(ax, horizontal=True)
        ax.set_xlabel("k// (π/a)", fontsize=8, color="w")
        ax.set_ylabel("E − EF (eV)", fontsize=8, color="w")
        ax.set_title(f"BM [{mode}]  {self._ef_offset_text()}", fontsize=8, color="w")
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
        if res2 is not None:
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
            self._session.gamma_reference = {
                "kx": float(gamma),
                "ky": 0.0,
                "polar": float((self._raw_data.get("metadata", {}) or {}).get("polar", 0.0) or 0.0),
                "tilt": float((self._raw_data.get("metadata", {}) or {}).get("tilt_ref", 0.0) or 0.0),
                "azi": float(azi_ref) if azi_ref is not None else None,
                "hv": self._raw_data.get("hv"),
                "path": self._raw_data.get("path"),
                "polar_already_applied_to_kx": bool(
                    (self._raw_data.get("metadata", {}) or {}).get("polar_already_applied_to_kx", False)
                ),
                "source": "bm",
            }
            if entry_now and entry_now.meta.direction:
                self._session.gamma_reference["direction"] = entry_now.meta.direction
            meta_now = self._raw_data.get("metadata", {}) or {}
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

        if payload["mode"] == "scalar":
            new_off = float(self._params.sp_ef.value()) - float(payload["ef_shift"])
            entry.ef_offset = new_off
            entry.ef_correction = {}   # supprimer une éventuelle correction poly
            self._params.sp_ef.blockSignals(True)
            self._params.sp_ef.setValue(new_off)
            self._params.sp_ef.blockSignals(False)
            msg = f"EF scalaire : Δ={payload['ef_shift']*1000:+.1f} meV → offset={new_off:.4f} eV"
            ref_payload = {
                "mode": "scalar",
                "ef_shift": float(payload["ef_shift"]),
                "T": float(payload["T"]),
                "fwhm_res": float(payload["fwhm_res"]),
                "source_file": str(self._current_path),
            }
        else:
            # En mode poly, on remet l'offset scalaire à 0 : la correction
            # par colonne porte tout le décalage EF.
            entry.ef_offset = 0.0
            entry.ef_correction = {
                "mode": "poly",
                "poly_coefs": [float(c) for c in payload["poly_coefs"]],
                "k_min": float(payload["k_min"]),
                "k_max": float(payload["k_max"]),
                "T": float(payload["T"]),
                "fwhm_res": float(payload["fwhm_res"]),
                "rms": float(payload["rms"]),
                "n_valid": int(payload["n_valid"]),
                "source": "self",
                "source_file": str(self._current_path),
            }
            self._params.sp_ef.blockSignals(True)
            self._params.sp_ef.setValue(0.0)
            self._params.sp_ef.blockSignals(False)
            msg = (f"EF par colonne : {payload['n_valid']} k valides, "
                   f"FWHM≈{payload['fwhm_res']*1000:.0f} meV, "
                   f"rms={payload['rms']*1000:.1f} meV")
            ref_payload = dict(entry.ef_correction)

        if payload.get("save_as_reference"):
            self._session.ef_reference = ref_payload
            msg += "  |  référence dossier sauvegardée"

        self._session.save()
        self._load_file(self._current_path)
        self._status(msg)

    def _apply_ef_reference_to_current(self):
        """Copie session.ef_reference vers FileEntry courant."""
        ref = self._session.ef_reference or {}
        if not ref or not self._current_path:
            self._status("⚠ Aucune référence EF en session")
            return
        key = self._session.key_for_path(self._current_path)
        entry = self._session.get_or_create(key)
        if ref.get("mode") == "poly":
            entry.ef_offset = 0.0
            entry.ef_correction = dict(ref)
            entry.ef_correction["source"] = "reference"
            self._params.sp_ef.blockSignals(True)
            self._params.sp_ef.setValue(0.0)
            self._params.sp_ef.blockSignals(False)
        elif ref.get("mode") == "scalar":
            cur_off = float(self._params.sp_ef.value())
            new_off = cur_off - float(ref.get("ef_shift", 0.0))
            entry.ef_offset = new_off
            entry.ef_correction = {}
            self._params.sp_ef.blockSignals(True)
            self._params.sp_ef.setValue(new_off)
            self._params.sp_ef.blockSignals(False)
        else:
            self._status("⚠ Référence EF mal formée"); return
        self._session.save()
        self._load_file(self._current_path)
        self._status(f"Référence EF appliquée ({ref.get('mode')})")

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
