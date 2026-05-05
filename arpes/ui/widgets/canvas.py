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
