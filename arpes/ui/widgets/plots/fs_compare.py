"""Canvas comparaison FS LV vs LH côte-à-côte + diff/sum.

Widget Qt avec 3 canvas matplotlib + barre de contrôles (sélecteurs paires
polar, toggles diff/sum, sync axes, colormap). Pas de logique IO ici —
le controller fournit les raw_data dicts à ``draw_pair``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from arpes.physics.fs_ops import fs_diff, fs_sum, regrid_to_common


def _extract_fs_slice(raw_data: dict, ef_window: float = 0.030):
    """Récupère carte 2D FS @ EF depuis raw_data (volume kx-ky-ev)."""
    if raw_data is None:
        return None
    meta = raw_data.get("metadata", {}) or {}
    fs_data = meta.get("fs_data")
    if fs_data is None:
        return None
    fs = np.asarray(fs_data, dtype=float)
    kx = np.asarray(meta.get("fs_kx"), dtype=float)
    ky = np.asarray(meta.get("fs_ky"), dtype=float)
    ev = np.asarray(meta.get("fs_energy"), dtype=float)
    if fs.ndim != 3:
        return None
    mask = np.abs(ev) <= float(ef_window)
    if not mask.any():
        mask[np.argmin(np.abs(ev))] = True
    slice2d = np.nanmean(fs[:, :, mask], axis=2)  # (n_ky, n_kx)
    return kx, ky, slice2d


class FsCompareCanvas(QWidget):
    """Tab compare pol : 3 sous-canvas (A, B, A−B), sélecteurs + actions."""

    pair_load_requested = pyqtSignal()
    auto_suggest_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # --- barre contrôles ----------------------------------------------
        bar = QHBoxLayout()
        bar.addWidget(QLabel("A :"))
        self.cmb_a = QComboBox()
        self.cmb_a.setMinimumWidth(220)
        bar.addWidget(self.cmb_a)
        bar.addWidget(QLabel("B :"))
        self.cmb_b = QComboBox()
        self.cmb_b.setMinimumWidth(220)
        bar.addWidget(self.cmb_b)
        self.btn_load = QPushButton("Charger paire")
        self.btn_load.clicked.connect(self.pair_load_requested)
        bar.addWidget(self.btn_load)
        self.btn_suggest = QPushButton("Auto-suggérer LH/LV")
        self.btn_suggest.setToolTip(
            "Suggère B = partenaire de polarisation opposée du fichier A\n"
            "(via logbook : même (material, run_id), pol différente)."
        )
        self.btn_suggest.clicked.connect(self.auto_suggest_requested)
        bar.addWidget(self.btn_suggest)
        bar.addStretch(1)
        self.chk_diff = QCheckBox("Diff (A − B)")
        self.chk_diff.setChecked(True)
        bar.addWidget(self.chk_diff)
        self.chk_sum = QCheckBox("Somme (A + B)")
        bar.addWidget(self.chk_sum)
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["none", "max", "sum"])
        self.cmb_norm.setToolTip(
            "Normalisation diff :\n"
            "- none : A−B brut\n"
            "- max  : (A−B)/max(|A|,|B|)\n"
            "- sum  : (A−B)/(A+B) (dichroïsme)"
        )
        bar.addWidget(QLabel("Norm :"))
        bar.addWidget(self.cmb_norm)
        lay.addLayout(bar)

        # --- 3 canvas matplotlib horizontaux ------------------------------
        # IMPORTANT : forcer Expanding + min taille petite. Sinon
        # FigureCanvas demande figsize×dpi (400px×3=1200px) en sizeHint et
        # gonfle le QTabWidget → squashe le panneau droit de l'app.
        canv_row = QHBoxLayout()
        self.fig_a = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas_a = FigureCanvas(self.fig_a)
        self.canvas_a.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_a.setMinimumSize(120, 120)
        self.ax_a = self.fig_a.add_subplot(111)
        canv_row.addWidget(self.canvas_a, 1)

        self.fig_b = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas_b = FigureCanvas(self.fig_b)
        self.canvas_b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_b.setMinimumSize(120, 120)
        self.ax_b = self.fig_b.add_subplot(111)
        canv_row.addWidget(self.canvas_b, 1)

        self.fig_d = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas_d = FigureCanvas(self.fig_d)
        self.canvas_d.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_d.setMinimumSize(120, 120)
        self.ax_d = self.fig_d.add_subplot(111)
        canv_row.addWidget(self.canvas_d, 1)
        lay.addLayout(canv_row, 1)

        # --- status -------------------------------------------------------
        self.lbl_status = QLabel("Choisir deux fichiers et cliquer « Charger paire ».")
        self.lbl_status.setStyleSheet("color:#aaa; font-size:10px;")
        lay.addWidget(self.lbl_status)
        self._dark_all()

    def _dark_all(self):
        for fig, ax in (
            (self.fig_a, self.ax_a),
            (self.fig_b, self.ax_b),
            (self.fig_d, self.ax_d),
        ):
            fig.set_facecolor("#2b2b2b")
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="w")

    def populate_selectors(self, file_pairs: list[tuple[str, str]]):
        """Remplit les combos avec [(label_affiché, path), ...]."""
        for cmb in (self.cmb_a, self.cmb_b):
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem("— choisir —", None)
            for label, path in file_pairs:
                cmb.addItem(label, path)
            cmb.blockSignals(False)

    def selected_paths(self) -> tuple[str | None, str | None]:
        return self.cmb_a.currentData(), self.cmb_b.currentData()

    def set_suggested_partner_b(self, path: str | None):
        if not path:
            return
        idx = self.cmb_b.findData(path)
        if idx >= 0:
            self.cmb_b.setCurrentIndex(idx)

    def draw_pair(
        self,
        raw_a: dict | None, raw_b: dict | None,
        label_a: str = "A", label_b: str = "B",
        ef_window: float = 0.030,
        diff_normalize: str = "none",
        cmap: str = "inferno",
    ) -> str:
        for ax in (self.ax_a, self.ax_b, self.ax_d):
            ax.cla()
        self._dark_all()
        slice_a = _extract_fs_slice(raw_a, ef_window=ef_window) if raw_a else None
        slice_b = _extract_fs_slice(raw_b, ef_window=ef_window) if raw_b else None
        if slice_a is None or slice_b is None:
            for ax, msg in (
                (self.ax_a, "A : pas de volume FS" if slice_a is None else label_a),
                (self.ax_b, "B : pas de volume FS" if slice_b is None else label_b),
                (self.ax_d, "diff indisponible"),
            ):
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha="center", va="center", color="w")
            for c in (self.canvas_a, self.canvas_b, self.canvas_d):
                c.draw_idle()
            return "⚠ Compare pol : volume FS manquant"

        kx_a, ky_a, fs_a = slice_a
        kx_b, ky_b, fs_b = slice_b

        try:
            pair = regrid_to_common(
                kx_a, ky_a, fs_a, kx_b, ky_b, fs_b,
                label_a=label_a, label_b=label_b,
            )
        except ValueError as exc:
            for ax in (self.ax_a, self.ax_b, self.ax_d):
                ax.text(0.5, 0.5, str(exc), transform=ax.transAxes,
                        ha="center", va="center", color="tomato")
                self._dark_all()
            for c in (self.canvas_a, self.canvas_b, self.canvas_d):
                c.draw_idle()
            return f"⚠ Compare pol : {exc}"

        self.ax_a.pcolormesh(pair.kx, pair.ky, pair.fs_a, cmap=cmap, shading="auto")
        self.ax_a.set_title(label_a, color="w")
        self.ax_a.set_aspect("equal")
        self.ax_a.set_xlabel("kx", color="w"); self.ax_a.set_ylabel("ky", color="w")

        self.ax_b.pcolormesh(pair.kx, pair.ky, pair.fs_b, cmap=cmap, shading="auto")
        self.ax_b.set_title(label_b, color="w")
        self.ax_b.set_aspect("equal")
        self.ax_b.set_xlabel("kx", color="w")

        if self.chk_sum.isChecked():
            d = fs_sum(pair)
            title = f"{label_a} + {label_b}"
        else:
            d = fs_diff(pair, normalize=diff_normalize)
            title = f"{label_a} − {label_b}"
            if diff_normalize != "none":
                title += f" (norm={diff_normalize})"
        # cmap divergente pour diff/dichroïsme
        cmap_d = "RdBu_r" if not self.chk_sum.isChecked() else cmap
        vmax = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 1.0
        if self.chk_sum.isChecked():
            self.ax_d.pcolormesh(pair.kx, pair.ky, d, cmap=cmap_d, shading="auto")
        else:
            self.ax_d.pcolormesh(
                pair.kx, pair.ky, d, cmap=cmap_d, shading="auto",
                vmin=-vmax, vmax=vmax,
            )
        self.ax_d.set_title(title, color="w")
        self.ax_d.set_aspect("equal")
        self.ax_d.set_xlabel("kx", color="w")

        for c in (self.canvas_a, self.canvas_b, self.canvas_d):
            c.draw_idle()
        return (f"✓ Compare pol : overlap={pair.overlap_ratio*100:.0f} % "
                f"| grille {pair.kx.size}×{pair.ky.size}")

    def set_status(self, text: str):
        self.lbl_status.setText(text)
