"""Sélecteur de zone de Brillouin avec aperçu."""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from arpes.physics.bz import BZ_PRESETS, BZ_PRESET_ALIASES, bz_high_symmetry_points, bz_polygon
from arpes.ui.widgets.canvas import MplCanvas


class BZSelectorDialog(QDialog):
    def __init__(self, parent=None, current_key: str = "square"):
        super().__init__(parent)
        self.setWindowTitle("Choisir ZDB")
        self.resize(480, 520)
        self.selected_key = BZ_PRESET_ALIASES.get(current_key, current_key)
        if self.selected_key not in BZ_PRESETS:
            self.selected_key = "square"
        lay = QVBoxLayout(self)
        self._combo = QComboBox()
        for key, preset in BZ_PRESETS.items():
            self._combo.addItem(preset.label, key)
        self._combo.setCurrentIndex(max(0, list(BZ_PRESETS).index(self.selected_key)))
        self._combo.currentIndexChanged.connect(self._on_change)
        lay.addWidget(self._combo)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        lay.addWidget(self._note)
        self._canvas = MplCanvas(figsize=(4, 4))
        lay.addWidget(self._canvas, stretch=1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._draw()

    def _on_change(self) -> None:
        self.selected_key = str(self._combo.currentData())
        self._draw()

    def _draw(self) -> None:
        preset = BZ_PRESETS[self.selected_key]
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        poly = bz_polygon(preset.shape, preset.half_x, preset.half_y, preset.angle_deg)
        ax.plot(poly[:, 0], poly[:, 1], color="white", lw=1.4, ls="--")
        ax.axhline(0, color="#888", lw=0.6, ls=":")
        ax.axvline(0, color="#888", lw=0.6, ls=":")
        for x, y, label, color in bz_high_symmetry_points(
            preset.shape, preset.half_x, preset.half_y, preset.angle_deg
        ):
            ax.scatter([x], [y], c=color, s=35)
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points",
                        color=color, fontsize=9, fontweight="bold")
        lim = max(preset.half_x, preset.half_y) * 1.25
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.tick_params(colors="w", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        self._note.setText(f"{preset.label} : {preset.note}. Schéma indicatif en unités π/a.")
        self._canvas.redraw()
