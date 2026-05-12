#!/usr/bin/env python3
"""arpes_fs.py — widgets et utilitaires FS pour ARPES Explorer.

Séparé de l'explorer pour garder la logique Surface de Fermi indépendante du
fit MDC des band maps. Fonctionne avec les métadonnées standardisées produites
par arpes_io.py pour Solaris/DA30 et CLS/LNLS.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Any
import hashlib

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

from arpes.physics.norm import apply_fs_flux_factors_to_map, fs_flux_profile_factors
from arpes.physics.bz import bz_high_symmetry_points, bz_polygon, resolve_bz_preset
from arpes.ui.widgets._qt_helpers import compact_button


@dataclass
class FSParams:
    a_lattice: float = 3.96
    b_lattice: float = 3.96
    ef_window: float = 0.030
    norm_ref_lo: float = -0.60
    norm_ref_hi: float = -0.20
    smooth_sigma: float = 1.0
    klim: float = 1.3
    kx_center: float = 0.0
    ky_center: float = 0.0
    bz_shape: str = "rectangle"
    bz_half_x: float = 1.0
    bz_half_y: float = 1.0
    bz_angle_deg: float = 90.0
    normalize_profile: bool = True
    overlay_bz: bool = True
    show_hsym: bool = True
    cmap: str = "inferno"


class FSControlPanel(QScrollArea):
    params_changed = pyqtSignal()
    redraw_requested = pyqtSignal()
    gamma_requested = pyqtSignal()
    manual_center_requested = pyqtSignal(bool)
    bz_preset_requested = pyqtSignal()

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
        self.sp_ref_lo = self._dspin(-0.600, -5.000, 1.000, 0.050)
        self.sp_ref_hi = self._dspin(-0.200, -5.000, 1.000, 0.050)
        self.sp_sm = self._dspin(1.0, 0.0, 8.0, 0.25, dec=2)
        self.cmb_cmap = QComboBox(); self.cmb_cmap.addItems(["inferno", "viridis", "magma", "gray", "hot"])
        self.cmb_cmap.currentIndexChanged.connect(self.params_changed)
        self.chk_norm = QCheckBox("Normalisation flux par slice"); self.chk_norm.setChecked(True)
        self.chk_norm.setToolTip(
            "Corrige le flux slice par slice (axe ky) et le profil détecteur (axe kx).\n"
            "Utile pour les FS CLS où l'intensité varie entre les steps et aux bords du détecteur."
        )
        self.chk_norm.stateChanged.connect(self.params_changed)
        fl2.addRow("Fenêtre EF ±eV:", self.sp_win)
        fl2.addRow("Norm ref min:", self.sp_ref_lo)
        fl2.addRow("Norm ref max:", self.sp_ref_hi)
        fl2.addRow("Lissage σ:", self.sp_sm)
        fl2.addRow("Colormap:", self.cmb_cmap)
        fl2.addRow(self.chk_norm)
        lay.addWidget(grp_fs)

        grp_bz = QGroupBox("ZDB théorique")
        fl3 = QFormLayout(grp_bz)
        self.chk_bz = QCheckBox("Afficher ZDB"); self.chk_bz.setChecked(True)
        self.chk_hsym = QCheckBox("Points Γ/X/M"); self.chk_hsym.setChecked(True)
        self.cmb_bz_shape = QComboBox(); self.cmb_bz_shape.addItems(["square", "rectangle", "hexagon", "centered_rect", "oblique"])
        self.sp_bzx = self._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
        self.sp_bzy = self._dspin(1.0, 0.05, 5.0, 0.05, dec=3)
        self.sp_bz_angle = self._dspin(90.0, 20.0, 160.0, 1.0, dec=1)
        self.sp_klim = self._dspin(1.3, 0.1, 10.0, 0.05, dec=2)
        self.cmb_bz_shape.currentIndexChanged.connect(self.params_changed)
        self.cmb_bz_shape.currentIndexChanged.connect(self._update_bz_angle_visibility)
        self.chk_bz.stateChanged.connect(self.params_changed)
        self.chk_hsym.stateChanged.connect(self.params_changed)
        btn_bz = compact_button(QPushButton("Choisir ZDB..."), max_width=160)
        btn_bz.setToolTip("Ouvre un sélecteur avec schéma pour choisir une ZDB 2D Bravais.")
        btn_bz.clicked.connect(self.bz_preset_requested)
        fl3.addRow(self.chk_bz)
        fl3.addRow(self.chk_hsym)
        fl3.addRow("Forme:", self.cmb_bz_shape)
        fl3.addRow("demi-ZDB x:", self.sp_bzx)
        fl3.addRow("demi-ZDB y:", self.sp_bzy)
        fl3.addRow("angle réseau:", self.sp_bz_angle)
        fl3.addRow("limite affichage:", self.sp_klim)
        fl3.addRow(btn_bz)
        lay.addWidget(grp_bz)
        self._update_bz_angle_visibility()

        self.lbl_info = QLabel("Charge un fast map Solaris ou un dossier FS CLS.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#aaa; font-size:10px;")
        lay.addWidget(self.lbl_info)
        btn = compact_button(QPushButton("Redessiner FS"), max_width=160)
        btn.clicked.connect(self.redraw_requested)
        lay.addWidget(btn)
        btn_g = compact_button(QPushButton("Détecter Γ FS"), max_width=160)
        btn_g.setToolTip("Détecte Γ par milieux de paires MDC sur la FS et recentre la carte.")
        btn_g.clicked.connect(self.gamma_requested)
        lay.addWidget(btn_g)
        self.btn_pick_center = compact_button(QPushButton("Viser Γ manuel"), max_width=160)
        self.btn_pick_center.setCheckable(True)
        self.btn_pick_center.setToolTip(
            "Active un curseur sur la carte FS.\n"
            "Clique sur le point qui doit devenir Γ : la carte est recentrée sur ce point "
            "et le centre est sauvegardé pour ce fichier."
        )
        self.btn_pick_center.toggled.connect(self.manual_center_requested)
        lay.addWidget(self.btn_pick_center)
        lay.addStretch(1)

    def params(self) -> FSParams:
        return FSParams(
            a_lattice=self.sp_a.value(), b_lattice=self.sp_b.value(),
            ef_window=self.sp_win.value(),
            norm_ref_lo=self.sp_ref_lo.value(), norm_ref_hi=self.sp_ref_hi.value(),
            smooth_sigma=self.sp_sm.value(),
            klim=self.sp_klim.value(), kx_center=self.sp_kx0.value(), ky_center=self.sp_ky0.value(),
            bz_shape=self.cmb_bz_shape.currentText(),
            bz_half_x=self.sp_bzx.value(), bz_half_y=self.sp_bzy.value(),
            bz_angle_deg=self.sp_bz_angle.value(),
            normalize_profile=self.chk_norm.isChecked(), overlay_bz=self.chk_bz.isChecked(),
            show_hsym=self.chk_hsym.isChecked(), cmap=self.cmb_cmap.currentText())

    def set_center(self, kx: float, ky: float):
        self.sp_kx0.blockSignals(True); self.sp_ky0.blockSignals(True)
        self.sp_kx0.setValue(float(kx)); self.sp_ky0.setValue(float(ky))
        self.sp_kx0.blockSignals(False); self.sp_ky0.blockSignals(False)
        self.params_changed.emit()

    def set_manual_center_active(self, active: bool):
        self.btn_pick_center.blockSignals(True)
        self.btn_pick_center.setChecked(bool(active))
        self.btn_pick_center.blockSignals(False)

    def apply_bz_preset(self, key: str) -> None:
        preset = resolve_bz_preset(key)
        self.cmb_bz_shape.blockSignals(True)
        self.sp_bzx.blockSignals(True)
        self.sp_bzy.blockSignals(True)
        self.sp_bz_angle.blockSignals(True)
        self.cmb_bz_shape.setCurrentText(preset.shape)
        self.sp_bzx.setValue(preset.half_x)
        self.sp_bzy.setValue(preset.half_y)
        self.sp_bz_angle.setValue(preset.angle_deg)
        self.cmb_bz_shape.blockSignals(False)
        self.sp_bzx.blockSignals(False)
        self.sp_bzy.blockSignals(False)
        self.sp_bz_angle.blockSignals(False)
        self._update_bz_angle_visibility()
        self.chk_bz.setChecked(True)
        self.params_changed.emit()

    def _update_bz_angle_visibility(self) -> None:
        show = self.cmb_bz_shape.currentText() == "oblique"
        label = self.sp_bz_angle.parentWidget().layout().labelForField(self.sp_bz_angle)
        if label is not None:
            label.setVisible(show)
        self.sp_bz_angle.setVisible(show)


def _robust_norm(img: np.ndarray) -> np.ndarray:
    """Normalise une image 2D vers [0,1].

    L'échelle est calculée préférentiellement sur la région centrale (80 % des
    colonnes kx) pour éviter que les bords du détecteur dominent le contraste.
    Si le centre manque de variation (données vides ou fond plat), repli sur
    l'image complète avec percentiles 1-99.
    """
    arr = np.asarray(img, dtype=float)
    if not np.isfinite(arr).any():
        return arr
    lo, hi = None, None
    if arr.ndim == 2 and arr.shape[1] >= 10:
        margin = max(1, arr.shape[1] // 10)
        ref = arr[:, margin: arr.shape[1] - margin]
        valid_c = ref[np.isfinite(ref)]
        if valid_c.size >= 4:
            lo_c, hi_c = np.percentile(valid_c, [2, 98])
            if hi_c - lo_c > 1e-6:
                lo, hi = lo_c, hi_c
    if lo is None:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return arr
        lo, hi = np.percentile(valid, [1, 99])
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

    norm_note = "sans norm"
    if params.normalize_profile:
        safe_y, safe_x, norm_note = fs_flux_profile_factors(
            fs_data,
            ev,
            ref_range=(params.norm_ref_lo, params.norm_ref_hi),
            normalize_y=True,
            normalize_x=True,
        )
        fs = apply_fs_flux_factors_to_map(fs, safe_y, safe_x)

    if params.smooth_sigma > 0 and gaussian_filter is not None:
        nan = ~np.isfinite(fs)
        tmp = np.where(nan, np.nanmedian(fs[np.isfinite(fs)]) if np.isfinite(fs).any() else 0, fs)
        fs = gaussian_filter(tmp, sigma=params.smooth_sigma)
        fs[nan] = np.nan
    fs_n = _robust_norm(fs)

    # CLS: ky est souvent tilt en degrés. On l'affiche tel quel sauf si l'utilisateur recentre.
    source = meta.get("fs_source", raw_data.get("source_format", ""))
    return kx, ky, fs_n, f"FS {source} — ±{params.ef_window*1000:.0f} meV | {norm_note}"


def _axis_signature(axis: Any) -> tuple:
    arr = np.asarray(axis, dtype=float)
    if arr.size == 0:
        return (0,)
    payload = np.ascontiguousarray(arr, dtype=np.float64)
    digest = hashlib.sha256(payload.tobytes()).hexdigest()
    return (tuple(payload.shape), digest)


def _fs_cache_key(raw_data: dict[str, Any], params: FSParams) -> tuple:
    meta = raw_data.get("metadata", {}) or {}
    fs_data = meta.get("fs_data")
    if fs_data is None:
        data = np.asarray(raw_data.get("data"))
        return (
            "bm-fallback",
            id(data),
            tuple(data.shape),
            _axis_signature(raw_data.get("kpar")),
            _axis_signature(raw_data.get("ev_arr")),
            round(float(params.ef_window), 8),
        )
    fs_arr = np.asarray(fs_data)
    return (
        "fs-volume",
        id(fs_arr),
        tuple(fs_arr.shape),
        _axis_signature(meta.get("fs_kx")),
        _axis_signature(meta.get("fs_ky")),
        _axis_signature(meta.get("fs_energy")),
        round(float(params.ef_window), 8),
        round(float(params.norm_ref_lo), 8),
        round(float(params.norm_ref_hi), 8),
        round(float(params.smooth_sigma), 8),
        bool(params.normalize_profile),
    )


class FermiSurfaceCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._fs_map_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = OrderedDict()
        self._fs_map_cache_max = 8
        self._mesh = None
        self._mesh_signature = None
        self._overlay_artists: list = []
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.toolbar = NavToolbar(self.canvas, self)
        act = self.toolbar.addAction("⤢ Vue init")
        act.setToolTip("Réinitialise les axes aux limites des données "
                       "(le graphe garde sa taille).")
        act.triggered.connect(self.reset_view)
        lay.addWidget(self.toolbar); lay.addWidget(self.canvas)
        self._dark()

    def reset_view(self):
        try:
            self.ax.set_aspect("auto")
            self.ax.relim()
            self.ax.autoscale(enable=True, axis="both", tight=False)
        except Exception:
            pass
        try:
            self.fig.set_layout_engine("tight")
        except Exception:
            pass
        self.canvas.draw_idle()

    def _dark(self):
        self.fig.set_facecolor("#2b2b2b"); self.ax.set_facecolor("#1a1a1a")

    def draw_fs(self, raw_data: dict[str, Any] | None, params: FSParams):
        if raw_data is None:
            self.ax.cla(); self._dark()
            self._mesh = None
            self._mesh_signature = None
            self._overlay_artists = []
            self.ax.text(0.5, 0.5, "Charge une FS", transform=self.ax.transAxes,
                         ha="center", va="center", color="w")
            self.canvas.draw_idle(); return "Aucune donnée"
        try:
            key = _fs_cache_key(raw_data, params)
            cached = self._fs_map_cache.pop(key, None)
            if cached is None:
                kx, ky, fs, title = extract_fs_map(raw_data, params)
                self._fs_map_cache[key] = (kx, ky, fs, title)
                while len(self._fs_map_cache) > self._fs_map_cache_max:
                    self._fs_map_cache.popitem(last=False)
            else:
                kx, ky, fs, title = cached
                self._fs_map_cache[key] = cached
            meta = raw_data.get("metadata", {}) or {}
            fs_kind = meta.get("fs_kind", "")
            x = kx - params.kx_center
            y = ky - params.ky_center
            signature = (
                tuple(np.asarray(fs).shape),
                _axis_signature(x),
                _axis_signature(y),
            )
            for artist in list(self._overlay_artists):
                try:
                    artist.remove()
                except Exception:
                    pass
            self._overlay_artists = []
            if self._mesh is not None and self._mesh_signature != signature:
                try:
                    self._mesh.remove()
                except Exception:
                    pass
                self._mesh = None
            if self._mesh is None:
                self.ax.cla(); self._dark()
                self._mesh = self.ax.pcolormesh(x, y, fs, cmap=params.cmap, shading="auto", vmin=0, vmax=1)
                self._mesh_signature = signature
            else:
                self._mesh.set_array(np.asarray(fs).ravel())
                self._mesh.set_cmap(params.cmap)
                self._mesh.set_clim(0, 1)
            has_kxky_axes = fs_kind == "kxky"
            self.ax.set_aspect("equal" if has_kxky_axes else "auto")
            self.ax.set_xlabel("kx (π/a)", color="w")
            ylabel = "ky (π/a)" if has_kxky_axes else "tilt (deg)"
            self.ax.set_ylabel(ylabel, color="w")
            self.ax.set_title(title, color="w", fontsize=10)
            if has_kxky_axes:
                self._overlay_bz(params)
            else:
                self.ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
                self.ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
            if has_kxky_axes and not params.overlay_bz:
                self.ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
                self.ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
            self.ax.tick_params(colors="w")
            for sp in self.ax.spines.values(): sp.set_edgecolor("#555")
            self.canvas.draw_idle()
            return f"{title} | shape={fs.shape}"
        except Exception as exc:
            self.ax.cla(); self._dark()
            self._mesh = None
            self._mesh_signature = None
            self._overlay_artists = []
            self.ax.text(0.5, 0.5, str(exc), transform=self.ax.transAxes,
                         ha="center", va="center", color="tomato", wrap=True)
            self.canvas.draw_idle(); return f"Erreur FS: {exc}"

    def detect_gamma(self, raw_data: dict[str, Any] | None, params: FSParams):
        kx, ky, fs, _ = extract_fs_map(raw_data, params)
        if len(ky) < 3:
            raise ValueError("Détection Γ FS impossible sans volume FS 2D.")
        meta = raw_data.get("metadata", {}) or {}
        if meta.get("fs_kind") != "kxky":
            raise ValueError("Détection Γ FS disponible seulement avec deux axes en π/a.")

        img = np.asarray(fs, dtype=float)
        kx_arr = np.asarray(kx, dtype=float)
        ky_arr = np.asarray(ky, dtype=float)
        kx_centers = []
        ky_centers = []

        def center_from_profile(axis, prof):
            y = np.asarray(prof, dtype=float)
            if not np.isfinite(y).any():
                return np.nan
            lo, hi = np.nanpercentile(y, [5, 99])
            if hi - lo <= 1e-12:
                return np.nan
            y = np.clip((y - lo) / (hi - lo), 0, None)
            left = axis < 0
            right = axis > 0
            if not left.any() or not right.any():
                return np.nan
            kl = axis[left][int(np.nanargmax(y[left]))]
            kr = axis[right][int(np.nanargmax(y[right]))]
            return float((kl + kr) / 2)

        y_samples = np.linspace(max(ky_arr.min(), -params.klim), min(ky_arr.max(), params.klim), 15)
        for y0 in y_samples:
            iy = int(np.argmin(np.abs(ky_arr - y0)))
            c = center_from_profile(kx_arr, img[iy, :])
            if np.isfinite(c): kx_centers.append(c)

        x_samples = np.linspace(max(kx_arr.min(), -params.klim), min(kx_arr.max(), params.klim), 15)
        for x0 in x_samples:
            ix = int(np.argmin(np.abs(kx_arr - x0)))
            c = center_from_profile(ky_arr, img[:, ix])
            if np.isfinite(c): ky_centers.append(c)

        if len(kx_centers) < 3 or len(ky_centers) < 3:
            raise ValueError("Pas assez de paires symétriques détectées sur la FS.")
        gx = float(np.nanmedian(kx_centers))
        gy = float(np.nanmedian(ky_centers))
        return {"kx": gx, "ky": gy,
                "gamma_kx_list": kx_centers, "gamma_ky_list": ky_centers}

    def _overlay_bz(self, p: FSParams):
        if not p.overlay_bz: return
        bx, by = p.bz_half_x, p.bz_half_y
        corners = bz_polygon(p.bz_shape, bx, by, p.bz_angle_deg)
        line, = self.ax.plot(corners[:,0], corners[:,1], color="white", lw=1.2, ls="--", alpha=0.85)
        self._overlay_artists.append(line)
        self._overlay_artists.append(self.ax.axhline(0, color="white", lw=0.5, ls=":", alpha=0.5))
        self._overlay_artists.append(self.ax.axvline(0, color="white", lw=0.5, ls=":", alpha=0.5))
        if p.show_hsym:
            def dot(x,y,name,color):
                scat = self.ax.scatter([x],[y], c=color, s=35, zorder=5, linewidths=0)
                ann = self.ax.annotate(name, (x,y), xytext=(4,4), textcoords="offset points", color=color, fontsize=9, fontweight="bold")
                self._overlay_artists.extend([scat, ann])
            for x, y, name, color in bz_high_symmetry_points(p.bz_shape, bx, by, p.bz_angle_deg):
                dot(x, y, name, color)
        self.ax.set_xlim(-p.klim, p.klim)
        if p.klim > 0: self.ax.set_ylim(-p.klim, p.klim)
