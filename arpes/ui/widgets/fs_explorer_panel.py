"""FS Explorer tab widgets: FS map with a draggable cut line + cut BM view.

Three dumb components (state/compute live in FSExplorerController):
- FSExplorerMapView   : iso-E FS map + orientable line (drag center to move,
                        drag an end handle to rotate/resize)
- FSExplorerCutView   : the BM extracted along the line
- FSExplorerControlBar: E−EF slider, angle/length spins, mode, Play, speed
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from arpes.ui.widgets.canvas import MplCanvas

_HANDLE_PX = 12  # pick radius in screen pixels


class FSExplorerMapView(QWidget):
    """Iso-E FS map with a draggable/orientable cut line."""

    line_changed = pyqtSignal(float, float, float, float)  # cx, cy, angle, length
    drag_state = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(figsize=(5, 5), toolbar=True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.cx, self.cy, self.angle_deg, self.length = 0.0, 0.0, 0.0, 1.0
        self._mesh = None
        self._line_artists = []
        self._drag = None  # None | "move" | "end0" | "end1"
        mpl = self.canvas.canvas
        self._cids = [
            mpl.mpl_connect("button_press_event", self._on_press),
            mpl.mpl_connect("motion_notify_event", self._on_motion),
            mpl.mpl_connect("button_release_event", self._on_release),
        ]

    # ------------------------------------------------------------- drawing
    def draw_map(self, img, kx, ky, *, xlabel: str, ylabel: str, title: str,
                 equal_aspect: bool = False) -> None:
        ax = self.canvas.ax
        ax.cla()
        self._mesh = ax.pcolormesh(kx, ky, img, cmap="magma", shading="auto")
        # Both axes in k: equal aspect so the visual line angle IS the
        # physical angle in the zone (a 45° drag means Γ-M, not "whatever
        # the window stretch makes of it").
        ax.set_aspect("equal" if equal_aspect else "auto")
        ax.set_xlabel(xlabel, color="w", fontsize=9)
        ax.set_ylabel(ylabel, color="w", fontsize=9)
        ax.set_title(title, color="w", fontsize=9)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="w", labelsize=8)
        self._line_artists = []
        self._redraw_line()
        self.canvas.redraw()

    def show_placeholder(self, text: str) -> None:
        ax = self.canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                va="center", color="#9ca3af", fontsize=10, wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self._mesh = None
        self._line_artists = []
        self.canvas.redraw()

    def set_line(self, cx, cy, angle_deg, length) -> None:
        """Programmatic update (animation/spinboxes): no signal emitted."""
        self.cx, self.cy = float(cx), float(cy)
        self.angle_deg, self.length = float(angle_deg), float(length)
        self._redraw_line()
        self.canvas.redraw()

    def _endpoints(self):
        a = np.deg2rad(self.angle_deg)
        dx = 0.5 * self.length * np.cos(a)
        dy = 0.5 * self.length * np.sin(a)
        return (self.cx - dx, self.cy - dy), (self.cx + dx, self.cy + dy)

    def _redraw_line(self) -> None:
        ax = self.canvas.ax
        for art in self._line_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._line_artists = []
        if self._mesh is None:
            return
        (x0, y0), (x1, y1) = self._endpoints()
        ln, = ax.plot([x0, x1], [y0, y1], color="#22d3ee", lw=1.6, zorder=20)
        ends = ax.scatter([x0, x1], [y0, y1], s=45, marker="o",
                          facecolor="#22d3ee", edgecolor="w", zorder=21)
        ctr = ax.scatter([self.cx], [self.cy], s=60, marker="D",
                         facecolor="#fbbf24", edgecolor="k", zorder=22)
        self._line_artists = [ln, ends, ctr]

    # ---------------------------------------------------------------- drag
    def _hit(self, event) -> str | None:
        if self._mesh is None or event.xdata is None:
            return None
        trans = self.canvas.ax.transData.transform
        px, py = event.x, event.y
        (x0, y0), (x1, y1) = self._endpoints()
        for name, (wx, wy) in (("end0", (x0, y0)), ("end1", (x1, y1)),
                               ("move", (self.cx, self.cy))):
            sx, sy = trans((wx, wy))
            if (sx - px) ** 2 + (sy - py) ** 2 <= _HANDLE_PX ** 2:
                return name
        return None

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes is not self.canvas.ax:
            return
        # The pan/zoom toolbar modes own the mouse: don't fight them.
        if getattr(self.canvas.toolbar, "mode", "") not in ("", None):
            return
        self._drag = self._hit(event)
        if self._drag:
            self.drag_state.emit(True)

    def _on_motion(self, event) -> None:
        if self._drag is None or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self._drag == "move":
            self.cx, self.cy = x, y
        else:
            # Dragging an end: line pivots around the OTHER end.
            (x0, y0), (x1, y1) = self._endpoints()
            ox, oy = (x1, y1) if self._drag == "end0" else (x0, y0)
            self.cx, self.cy = 0.5 * (x + ox), 0.5 * (y + oy)
            self.length = float(np.hypot(x - ox, y - oy))
            self.angle_deg = float(np.rad2deg(np.arctan2(y - oy, x - ox)))
            if self._drag == "end0":
                self.angle_deg = (self.angle_deg + 180.0) % 360.0
        self._redraw_line()
        self.canvas.redraw()
        self.line_changed.emit(self.cx, self.cy, self.angle_deg, self.length)

    def _on_release(self, event) -> None:
        if self._drag is None:
            return
        self._drag = None
        self.drag_state.emit(False)
        self.line_changed.emit(self.cx, self.cy, self.angle_deg, self.length)


class FSExplorerCutView(QWidget):
    """BM extracted along the cut line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(figsize=(5, 5), toolbar=True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def draw_cut(self, image, k_along, e_ax, *, e_current: float | None,
                 xlabel: str, ylabel: str, title: str) -> None:
        ax = self.canvas.ax
        ax.cla()
        ax.pcolormesh(k_along, e_ax, np.asarray(image).T, cmap="magma",
                      shading="auto")
        if e_current is not None:
            ax.axhline(float(e_current), color="#22d3ee", lw=1.0, ls="--",
                       alpha=0.85)
        ax.set_xlabel(xlabel, color="w", fontsize=9)
        ax.set_ylabel(ylabel, color="w", fontsize=9)
        ax.set_title(title, color="w", fontsize=9)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="w", labelsize=8)
        self.canvas.redraw()

    def show_placeholder(self, text: str) -> None:
        ax = self.canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                va="center", color="#9ca3af", fontsize=10, wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.redraw()


class FSExplorerControlBar(QWidget):
    """E−EF slider + line spins + mode + Play/Pause + speed."""

    energy_changed = pyqtSignal(float)
    line_params_changed = pyqtSignal(float, float)   # angle_deg, length
    mode_changed = pyqtSignal(str)                   # "free" | "native"
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._e_ax = np.array([0.0])
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)

        lay.addWidget(QLabel("E−EF (eV):"))
        self.sl_e = QSlider(Qt.Orientation.Horizontal)
        self.sl_e.setRange(0, 0)
        self.sl_e.setToolTip(
            "Binding energy of the displayed iso-E map. Negative = occupied "
            "states. Independent from the EF integration of the main FS tab."
        )
        self.sl_e.valueChanged.connect(self._on_slider)
        lay.addWidget(self.sl_e, stretch=2)
        self.lbl_e = QLabel("—")
        self.lbl_e.setMinimumWidth(64)
        lay.addWidget(self.lbl_e)

        lay.addWidget(QLabel("Mode:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Free line", "free")
        self.cmb_mode.addItem("Native BMs", "native")
        self.cmb_mode.setToolTip(
            "Free line: arbitrary interpolated cut through the volume.\n"
            "Native BMs: snaps to the cuts actually measured (discrete "
            "tilt/scan steps), no interpolation."
        )
        self.cmb_mode.currentIndexChanged.connect(
            lambda _i: self.mode_changed.emit(self.cmb_mode.currentData()))
        lay.addWidget(self.cmb_mode)

        lay.addWidget(QLabel("Angle (°):"))
        self.sp_angle = QDoubleSpinBox()
        self.sp_angle.setRange(-360.0, 360.0)
        self.sp_angle.setDecimals(1)
        self.sp_angle.setSingleStep(5.0)
        self.sp_angle.setToolTip("Cut line angle (0° = +kx, 90° = +ky). "
                                 "Γ-X ≈ 0°, Γ-M ≈ 45° on a square zone.")
        lay.addWidget(self.sp_angle)
        lay.addWidget(QLabel("Length:"))
        self.sp_length = QDoubleSpinBox()
        self.sp_length.setRange(0.01, 100.0)
        self.sp_length.setDecimals(2)
        self.sp_length.setSingleStep(0.1)
        self.sp_length.setValue(1.0)
        lay.addWidget(self.sp_length)
        for sp in (self.sp_angle, self.sp_length):
            sp.valueChanged.connect(self._emit_line_params)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip(
            "Sweep the cut line through the Fermi surface, perpendicular to "
            "its direction.")
        self.btn_play.toggled.connect(self._on_play)
        lay.addWidget(self.btn_play)
        lay.addWidget(QLabel("Speed:"))
        self.sp_speed = QDoubleSpinBox()
        self.sp_speed.setRange(0.1, 10.0)
        self.sp_speed.setDecimals(1)
        self.sp_speed.setSingleStep(0.5)
        self.sp_speed.setValue(1.0)
        self.sp_speed.setToolTip("Sweep speed multiplier (steps per frame).")
        self.sp_speed.valueChanged.connect(
            lambda v: self.speed_changed.emit(float(v)))
        lay.addWidget(self.sp_speed)
        self.lbl_info = QLabel("")
        lay.addWidget(self.lbl_info, stretch=1)

    # ------------------------------------------------------------- energy
    def set_energy_axis(self, e_ax) -> None:
        self._e_ax = np.asarray(e_ax, dtype=float)
        self.sl_e.blockSignals(True)
        self.sl_e.setRange(0, max(0, self._e_ax.size - 1))
        idx = int(np.argmin(np.abs(self._e_ax)))  # default: EF
        self.sl_e.setValue(idx)
        self.sl_e.blockSignals(False)
        self.lbl_e.setText(f"{self._e_ax[idx]:+.3f}")

    def current_energy(self) -> float:
        return float(self._e_ax[int(self.sl_e.value())])

    def _on_slider(self, idx: int) -> None:
        e = float(self._e_ax[int(idx)])
        self.lbl_e.setText(f"{e:+.3f}")
        self.energy_changed.emit(e)

    # --------------------------------------------------------------- line
    def set_line_params(self, angle_deg: float, length: float) -> None:
        for sp, val in ((self.sp_angle, angle_deg), (self.sp_length, length)):
            sp.blockSignals(True)
            sp.setValue(float(val))
            sp.blockSignals(False)

    def _emit_line_params(self) -> None:
        self.line_params_changed.emit(
            float(self.sp_angle.value()), float(self.sp_length.value()))

    def _on_play(self, checked: bool) -> None:
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        self.play_toggled.emit(bool(checked))

    def stop_play(self) -> None:
        """Programmatic stop (file change, tab leave): no signal loop."""
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶ Play")
        self.btn_play.blockSignals(False)
