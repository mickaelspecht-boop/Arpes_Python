"""Visual DFT band picker dialog."""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from arpes.theory.band_picker import (
    picker_band_curves,
    picker_k_axis,
    picker_segment_span,
    picker_ticks,
    validate_picker_data,
)
from arpes.theory.band_select import format_band_indices
from arpes.theory.models import TheoryBandData, TheoryOverlayConfig
from arpes.theory.plot import _band_color
from arpes.ui.widgets.canvas import MplCanvas


class TheoryBandPickerDialog(QDialog):
    selection_applied = pyqtSignal(object, str)

    def __init__(
        self,
        data: TheoryBandData | dict,
        config: TheoryOverlayConfig | dict,
        *,
        segments: list[str] | None = None,
        selected: list[int] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
        self.config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
        self._segments = list(segments or [])
        self._selected = {int(i) for i in (selected or []) if int(i) >= 0}
        self._lines = {}
        self._validation_error = validate_picker_data(self.data)
        self._default_xlim: tuple[float, float] | None = None
        self._default_ylim: tuple[float, float] | None = None

        title_id = self.data.material_id or "DFT"
        self.setWindowTitle(f"DFT Bands - {title_id}")
        self.resize(920, 640)
        self.setModal(True)

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Path:"))
        self.cmb_segment = QComboBox()
        self.cmb_segment.addItem("")
        self.cmb_segment.addItems(self._segments)
        if self.config.segment:
            self.cmb_segment.setCurrentText(self.config.segment)
        self.cmb_segment.currentIndexChanged.connect(self._redraw)
        top.addWidget(self.cmb_segment, 1)

        top.addWidget(QLabel("E_F +/-"))
        self.sp_efwin = QDoubleSpinBox()
        self.sp_efwin.setRange(0.0, 10.0)
        self.sp_efwin.setDecimals(3)
        self.sp_efwin.setSingleStep(0.05)
        self.sp_efwin.setValue(float(self.config.ef_window or 0.0))
        self.sp_efwin.valueChanged.connect(self._redraw)
        top.addWidget(self.sp_efwin)
        root.addLayout(top)

        self.canvas = MplCanvas(figsize=(8, 5), toolbar=True)
        self.canvas.reset_callback = self._reset_view
        root.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        self.lbl_selection = QLabel()
        bottom.addWidget(self.lbl_selection, 1)
        btn_clear = QPushButton("Deselect all")
        btn_clear.clicked.connect(self._clear_selection)
        bottom.addWidget(btn_clear)
        btn_invert = QPushButton("Invert")
        btn_invert.clicked.connect(self._invert_selection)
        bottom.addWidget(btn_invert)
        root.addLayout(bottom)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        root.addWidget(buttons)

        self.canvas.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self._redraw()

    def selected_band_indices(self) -> list[int]:
        return sorted(self._selected)

    def selected_segment(self) -> str:
        return self.cmb_segment.currentText().strip()

    def _curves(self):
        return picker_band_curves(
            self.data,
            self.config,
            segment=self.selected_segment(),
            ef_window=float(self.sp_efwin.value()),
        )

    def _reset_view(self):
        ax = self.canvas.ax
        if self._default_xlim is not None:
            ax.set_xlim(*self._default_xlim)
        if self._default_ylim is not None:
            ax.set_ylim(*self._default_ylim)
        self.canvas.canvas.draw_idle()

    def _redraw(self):
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        if self._validation_error:
            ax.text(
                0.5,
                0.5,
                self._validation_error,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#fca5a5",
            )
            self.lbl_selection.setText(self._validation_error)
            self.canvas.canvas.draw_idle()
            return
        ax.axhline(0.0, color="#67e8f9", lw=0.8, ls="--", alpha=0.75)
        ax.set_xlabel("DFT path")
        ax.set_ylabel("E - E_F (eV)")
        ax.tick_params(colors="#d1d5db")
        ax.xaxis.label.set_color("#d1d5db")
        ax.yaxis.label.set_color("#d1d5db")
        self._draw_path_guides(ax)
        self._lines = {}
        for curve in self._curves():
            finite = curve.energy[curve.energy == curve.energy]
            if finite.size == 0:
                continue
            selected = curve.band_index in self._selected
            color = _band_color(curve.band_index) if selected else "#1f77b4"
            alpha = 0.98 if selected else 0.78
            lw = 2.2 if selected else 1.05
            line, = ax.plot(
                curve.k,
                curve.energy,
                color=color,
                alpha=alpha,
                lw=lw,
                picker=5,
                zorder=8 if selected else 4,
            )
            line.set_gid(curve.band_index)
            self._lines[curve.band_index] = line
        self._set_initial_limits(ax)
        self._update_selection_label()
        self.canvas.fig.tight_layout()
        self.canvas.canvas.draw_idle()

    def _draw_path_guides(self, ax):
        ticks = picker_ticks(self.data, convention=self.config.path_convention)
        for tick in ticks:
            ax.axvline(tick.x, color="#4b5563", lw=0.7, alpha=0.65, zorder=1)
        if ticks:
            ax.set_xticks([tick.x for tick in ticks])
            ax.set_xticklabels([tick.label for tick in ticks])
        span = picker_segment_span(self.data, self.selected_segment())
        if span is not None:
            ax.axvspan(span[0], span[1], color="#38bdf8", alpha=0.08, zorder=0)

    def _set_initial_limits(self, ax):
        k = picker_k_axis(self.data)
        if k.size:
            self._default_xlim = (float(k.min()), float(k.max()))
            ax.set_xlim(*self._default_xlim)
        finite = []
        for curve in self._curves():
            vals = curve.energy[curve.energy == curve.energy]
            if vals.size:
                finite.append(vals)
        if not finite:
            return
        vals = np.concatenate(finite)
        finite_min = float(np.nanmin(vals))
        finite_max = float(np.nanmax(vals))
        if finite_min < -10.0 or finite_max > 15.0:
            # Materials Project band plots default to a compact EF-centered
            # window and leave deep semicore/high conduction bands clipped.
            self._default_ylim = (-5.0, 9.0)
        else:
            lo = float(np.nanpercentile(vals, 2))
            hi = float(np.nanpercentile(vals, 98))
            lo = min(lo, -1.0)
            hi = max(hi, 1.0)
            pad = 0.08 * max(hi - lo, 1e-9)
            self._default_ylim = (lo - pad, hi + pad)
        ax.set_ylim(*self._default_ylim)

    def _toolbar_navigating(self) -> bool:
        """True quand l'outil zoom/pan de la barre est actif (clic = navigation)."""
        toolbar = getattr(self.canvas, "toolbar", None)
        mode = getattr(toolbar, "mode", None)
        return bool(mode) and str(mode) != ""

    def _on_pick(self, event):
        if self._toolbar_navigating():
            return
        artist = getattr(event, "artist", None)
        idx = artist.get_gid() if artist is not None and hasattr(artist, "get_gid") else None
        if idx is None:
            return
        self._toggle_band(int(idx))

    def _toggle_band(self, idx: int):
        if idx in self._selected:
            self._selected.remove(idx)
        else:
            self._selected.add(idx)
        self._style_lines()

    def _on_button_press(self, event):
        if not getattr(event, "dblclick", False):
            return
        if self._toolbar_navigating():
            return
        if event.inaxes is not self.canvas.ax:
            return
        nearest = self._nearest_band(event.xdata, event.ydata)
        if nearest is None:
            return
        self._selected = {nearest}
        self._style_lines()

    def _on_scroll_zoom(self, event):
        if event.inaxes is not self.canvas.ax:
            return
        ax = self.canvas.ax
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        scale = 0.80 if event.button == "up" else 1.25
        self._zoom_axis(ax, xdata, ydata, scale)
        self.canvas.canvas.draw_idle()

    @staticmethod
    def _zoom_axis(ax, xcenter: float, ycenter: float, scale: float) -> None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        new_w = (x1 - x0) * scale
        new_h = (y1 - y0) * scale
        rx = (xcenter - x0) / (x1 - x0) if x1 != x0 else 0.5
        ry = (ycenter - y0) / (y1 - y0) if y1 != y0 else 0.5
        ax.set_xlim(xcenter - new_w * rx, xcenter + new_w * (1.0 - rx))
        ax.set_ylim(ycenter - new_h * ry, ycenter + new_h * (1.0 - ry))

    def _nearest_band(self, x, y):
        if x is None or y is None:
            return None
        best = None
        best_dist = float("inf")
        for curve in self._curves():
            finite = (curve.k == curve.k) & (curve.energy == curve.energy)
            if not finite.any():
                continue
            dist = ((curve.k[finite] - x) ** 2 + (curve.energy[finite] - y) ** 2) ** 0.5
            cur = float(dist.min())
            if cur < best_dist:
                best_dist = cur
                best = curve.band_index
        return best

    def _style_lines(self):
        for idx, line in self._lines.items():
            selected = idx in self._selected
            line.set_color(_band_color(idx) if selected else "#1f77b4")
            line.set_alpha(0.98 if selected else 0.78)
            line.set_linewidth(2.2 if selected else 1.05)
            line.set_zorder(8 if selected else 4)
        self._update_selection_label()
        self.canvas.canvas.draw_idle()

    def _update_selection_label(self):
        spec = format_band_indices(self.selected_band_indices())
        self.lbl_selection.setText(f"Selection: {spec or 'auto top-N'}")

    def _clear_selection(self):
        self._selected.clear()
        self._style_lines()

    def _invert_selection(self):
        all_indices = set(self._lines)
        self._selected = all_indices.difference(self._selected)
        self._style_lines()

    def _apply(self):
        self.selection_applied.emit(self.selected_band_indices(), self.selected_segment())
        self.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)
