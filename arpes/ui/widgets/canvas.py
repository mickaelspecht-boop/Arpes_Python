"""Matplotlib canvas wrapper Qt — fond sombre + toolbar optionnelle."""
from __future__ import annotations

from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar,
)
from matplotlib.figure import Figure


class MplCanvas(QWidget):
    def __init__(self, figsize=(5, 4), toolbar=False, nrows=1):
        super().__init__()
        # Optionnel : recalage de vue spécifique (ex BM pcolormesh dont
        # relim/autoscale matplotlib ne restaure pas l'étendue data).
        self.reset_callback = None
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = None
        if toolbar:
            self.toolbar = NavToolbar(self.canvas, self)
            act = self.toolbar.addAction("⤢ Vue init")
            act.setToolTip("Réinitialise les axes aux limites des données "
                           "(le graphe garde sa taille, seules les valeurs d'axes changent).")
            act.triggered.connect(self.reset_view)
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        if nrows == 1:
            self.ax  = self.fig.add_subplot(111)
            self.axes = [self.ax]
        else:
            self.axes = list(self.fig.subplots(nrows, 1))
            self.ax   = self.axes[0]
        self._dark()

    def reset_view(self):
        """Axes -> limites des données, aspect 'auto'. Pas de rétrécissement du cadre."""
        cb = self.reset_callback
        if callable(cb):
            try:
                cb()
                return
            except Exception:
                pass  # repli sur le reset générique ci-dessous
        for ax in self.axes:
            try:
                ax.set_aspect("auto")
                ax.relim()
                ax.autoscale(enable=True, axis="both", tight=False)
            except Exception:
                pass
        try:  # ré-applique une mise en page propre une fois
            self.fig.set_layout_engine("tight")
        except Exception:
            pass
        self.canvas.draw_idle()

    def _dark(self):
        self.fig.set_facecolor("#2b2b2b")
        for ax in self.axes:
            ax.set_facecolor("#1a1a1a")

    def redraw(self): self.canvas.draw_idle()
