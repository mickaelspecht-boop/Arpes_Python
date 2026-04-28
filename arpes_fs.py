#!/usr/bin/env python3
"""arpes_fs.py — widgets et utilitaires FS pour ARPES Explorer.

Séparé de l'explorer pour garder la logique Surface de Fermi indépendante du
fit MDC des band maps. Fonctionne avec les métadonnées standardisées produites
par arpes_io.py pour Solaris/DA30 et CLS/LNLS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QDoubleSpinBox, QCheckBox, QLabel, QScrollArea, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # scipy absent: fallback sans lissage
    gaussian_filter = None


@dataclass
class FSParams:
    a_lattice: float = 3.96
    b_lattice: float = 3.96
    ef_window: float = 0.030
    smooth_sigma: float = 1.0
    klim: float = 1.3
    kx_center: float = 0.0
    ky_center: float = 0.0
    bz_half_x: float = 1.0
    bz_half_y: float = 1.0
    normalize_profile: bool = True
    overlay_bz: bool = True
    show_hsym: bool = True
    cmap: str = "inferno"


class FSControlPanel(QScrollArea):
    params_changed = pyqtSignal()
    redraw_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        w = QWidget()
        self._lay = QVBoxLayout(w)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(w)
        self._build()

    def _dspin(self, value, lo, hi, step, dec=3):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec); sp.setValue(value)
        sp.valueChanged.connect(self.params_changed)
        return sp

    def _build(self):
        lay = self._lay

        grp_lat = QGroupBox("Réseau / unités π/a")
        fl = QFormLayout(grp_lat)
        self.sp_a = self._dspin(3.960, 1.0, 20.0, 0.01)
        self.sp_b = self._dspin(3.960, 1.0, 20.0, 0.01)
        self.sp_kx0 = self._dspin(0.0, -5.0, 5.0, 0.01)
        self.sp_ky0 = self._dspin(0.0, -5.0, 5.0, 0.01)
        fl.addRow("a (Å):", self.sp_a)
        fl.addRow("b (Å):", self.sp_b)
        fl.addRow("centre kx:", self.sp_kx0)
        fl.addRow("centre ky:", self.sp_ky0)
        lay.addWidget(grp_lat)

        grp_fs = QGroupBox("Carte FS")
        fl2 = QFormLayout(grp_fs)
        self.sp_win = self._dspin(0.030, 0.001, 0.500, 0.005)
        self.sp_sm = self._dspin(1.0, 0.0, 8.0, 0.25, dec=2)
        self.cmb_cmap = QComboBox(); self.cmb_cmap.addItems(["inferno", "viridis", "magma", "gray", "hot"])
        self.cmb_cmap.currentIndexChanged.connect(self.params_changed)
        self.chk_norm = QCheckBox("Normalisation profil y/ky"); self.chk_norm.setChecked(True)
        self.chk_norm.stateChanged.connect(self.params_changed)
        fl2.addRow("Fenêtre EF ±eV:", self.sp_win)
        fl2.addRow("Lissage σ:", self.sp_sm)
        fl2.addRow("Colormap:", self.cmb_cmap)
        fl2.addRow(self.chk_norm)
        lay.addWidget(grp_fs)

        grp_bz = QGroupBox("ZDB théorique")
        fl3 = QFormLayout(grp_bz)
        self.chk_bz = QCheckBox("Afficher ZDB"); self.chk_bz.setChecked(True)
        self.chk_hsym = QCheckBox("Points Γ/X/M"); self.chk_hsym.setChecked(True)
        self.sp_bzx = self._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
        self.sp_bzy = self._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
        self.sp_klim = self._dspin(1.3, 0.1, 10.0, 0.05, dec=2)
        self.chk_bz.stateChanged.connect(self.params_changed)
        self.chk_hsym.stateChanged.connect(self.params_changed)
        fl3.addRow(self.chk_bz)
        fl3.addRow(self.chk_hsym)
        fl3.addRow("demi-ZDB x:", self.sp_bzx)
        fl3.addRow("demi-ZDB y:", self.sp_bzy)
        fl3.addRow("limite affichage:", self.sp_klim)
        lay.addWidget(grp_bz)

        self.lbl_info = QLabel("Charge un fast map Solaris ou un dossier FS CLS.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#aaa; font-size:10px;")
        lay.addWidget(self.lbl_info)
        btn = QPushButton("↻ Redessiner FS")
        btn.clicked.connect(self.redraw_requested)
        lay.addWidget(btn)
        lay.addStretch(1)

    def params(self) -> FSParams:
        return FSParams(
            a_lattice=self.sp_a.value(), b_lattice=self.sp_b.value(),
            ef_window=self.sp_win.value(), smooth_sigma=self.sp_sm.value(),
            klim=self.sp_klim.value(), kx_center=self.sp_kx0.value(), ky_center=self.sp_ky0.value(),
            bz_half_x=self.sp_bzx.value(), bz_half_y=self.sp_bzy.value(),
            normalize_profile=self.chk_norm.isChecked(), overlay_bz=self.chk_bz.isChecked(),
            show_hsym=self.chk_hsym.isChecked(), cmap=self.cmb_cmap.currentText())


def _robust_norm(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=float)
    if not np.isfinite(arr).any(): return arr
    lo, hi = np.nanpercentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)


def extract_fs_map(raw_data: dict[str, Any], params: FSParams):
    """Retourne kx, ky, fs_norm, titre à partir du dict legacy de l'explorer."""
    if raw_data is None:
        raise ValueError("Aucune donnée chargée")
    meta = raw_data.get("metadata", {}) or {}
    fs_data = meta.get("fs_data")
    if fs_data is None:
        # Fallback BM 2D: montre une MDC intégrée autour EF comme image 1 ligne.
        data = np.asarray(raw_data["data"], dtype=float)
        ev = np.asarray(raw_data["ev_arr"], dtype=float)
        kx = np.asarray(raw_data["kpar"], dtype=float)
        mask = np.abs(ev) <= params.ef_window
        if mask.sum() == 0: mask[np.argmin(np.abs(ev))] = True
        mdc = np.nanmean(data[:, mask], axis=1)
        return kx, np.array([0.0]), _robust_norm(mdc[None, :]), "Pas de volume FS: MDC à EF seulement"

    fs_data = np.asarray(fs_data, dtype=float)  # attendu (ny, nx, ne)
    kx = np.asarray(meta.get("fs_kx"), dtype=float)
    ky = np.asarray(meta.get("fs_ky"), dtype=float)
    ev = np.asarray(meta.get("fs_energy"), dtype=float)
    if fs_data.ndim != 3:
        raise ValueError(f"Volume FS invalide: shape={fs_data.shape}")
    if fs_data.shape[-1] != len(ev):
        raise ValueError("Volume FS: dernier axe ≠ énergie")

    mask = np.abs(ev) <= params.ef_window
    if mask.sum() == 0: mask[np.argmin(np.abs(ev))] = True
    fs = np.nanmean(fs_data[:, :, mask], axis=2)

    if params.normalize_profile and fs.shape[0] > 1:
        prof = np.nanmean(fs, axis=1, keepdims=True)
        fs = fs / (prof + 1e-12)
    if params.smooth_sigma > 0 and gaussian_filter is not None:
        nan = ~np.isfinite(fs)
        tmp = np.where(nan, np.nanmedian(fs[np.isfinite(fs)]) if np.isfinite(fs).any() else 0, fs)
        fs = gaussian_filter(tmp, sigma=params.smooth_sigma)
        fs[nan] = np.nan
    fs_n = _robust_norm(fs)

    # CLS: ky est souvent tilt en degrés. On l'affiche tel quel sauf si l'utilisateur recentre.
    source = meta.get("fs_source", raw_data.get("source_format", ""))
    return kx, ky, fs_n, f"FS {source} — intégration ±{params.ef_window*1000:.0f} meV"


class FermiSurfaceCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(NavToolbar(self.canvas, self)); lay.addWidget(self.canvas)
        self._dark()

    def _dark(self):
        self.fig.set_facecolor("#2b2b2b"); self.ax.set_facecolor("#1a1a1a")

    def draw_fs(self, raw_data: dict[str, Any] | None, params: FSParams):
        self.ax.cla(); self._dark()
        if raw_data is None:
            self.ax.text(0.5, 0.5, "Charge une FS", transform=self.ax.transAxes,
                         ha="center", va="center", color="w")
            self.canvas.draw_idle(); return "Aucune donnée"
        try:
            kx, ky, fs, title = extract_fs_map(raw_data, params)
            x = kx - params.kx_center
            y = ky - params.ky_center
            self.ax.pcolormesh(x, y, fs, cmap=params.cmap, shading="auto", vmin=0, vmax=1)
            self.ax.set_aspect("equal" if len(ky) > 1 and np.nanmax(np.abs(ky)) < 5 else "auto")
            self.ax.set_xlabel("kx (π/a)", color="w")
            ylabel = "ky (π/a)" if (raw_data.get("metadata", {}) or {}).get("fs_source") == "solaris_da30" else "tilt / ky (deg ou π/a)"
            self.ax.set_ylabel(ylabel, color="w")
            self.ax.set_title(title, color="w", fontsize=10)
            self._overlay_bz(params)
            self.ax.tick_params(colors="w")
            for sp in self.ax.spines.values(): sp.set_edgecolor("#555")
            self.canvas.draw_idle()
            return f"{title} | shape={fs.shape}"
        except Exception as exc:
            self.ax.text(0.5, 0.5, str(exc), transform=self.ax.transAxes,
                         ha="center", va="center", color="tomato", wrap=True)
            self.canvas.draw_idle(); return f"Erreur FS: {exc}"

    def _overlay_bz(self, p: FSParams):
        if not p.overlay_bz: return
        bx, by = p.bz_half_x, p.bz_half_y
        corners = np.array([[-bx,-by],[bx,-by],[bx,by],[-bx,by],[-bx,-by]])
        self.ax.plot(corners[:,0], corners[:,1], color="white", lw=1.2, ls="--", alpha=0.85)
        self.ax.axhline(0, color="white", lw=0.5, ls=":", alpha=0.5)
        self.ax.axvline(0, color="white", lw=0.5, ls=":", alpha=0.5)
        if p.show_hsym:
            def dot(x,y,name,color):
                self.ax.scatter([x],[y], c=color, s=35, zorder=5, linewidths=0)
                self.ax.annotate(name, (x,y), xytext=(4,4), textcoords="offset points", color=color, fontsize=9, fontweight="bold")
            dot(0,0,"Γ","white")
            for x,y in [(bx,0),(-bx,0),(0,by),(0,-by)]: dot(x,y,"X","cyan")
            for x,y in [(bx,by),(bx,-by),(-bx,by),(-bx,-by)]: dot(x,y,"M","lime")
        self.ax.set_xlim(-p.klim, p.klim)
        if p.klim > 0: self.ax.set_ylim(-p.klim, p.klim)
