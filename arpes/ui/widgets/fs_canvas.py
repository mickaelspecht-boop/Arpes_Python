"""Fermi-surface matplotlib canvas widget."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from arpes.physics.bz import bz_high_symmetry_points, bz_polygon
from arpes.physics.fs import (
    FSParams,
    _axis_signature,
    _fs_cache_key,
    detect_gamma_from_fs_map,
    extract_fs_map,
)


class FermiSurfaceCanvas(QWidget):
    pocket_requested = pyqtSignal(float, float)
    pocket_mdc_requested = pyqtSignal(float, float)
    pairing_diagnose_requested = pyqtSignal()
    pocket_level_requested = pyqtSignal(float, float)
    pocket_preview_requested = pyqtSignal(float, float)
    pocket_preview_validate_requested = pyqtSignal()
    pocket_preview_cancel_requested = pyqtSignal()
    pockets_clear_requested = pyqtSignal()
    pockets_export_requested = pyqtSignal()
    pocket_open_requested = pyqtSignal(int)
    pocket_lasso_requested = pyqtSignal(float, float, float, float)
    pocket_preview_level_changed = pyqtSignal(float)
    pocket_manual_contour_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(80, 80)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(80, 80)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self._fs_map_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = OrderedDict()
        self._fs_map_cache_max = 8
        self._mesh = None
        self._mesh_signature = None
        self._overlay_artists: list = []
        self._bm_cut_artists: list = []
        self._pocket_artists: list = []
        self._bm_cut_center = (0.0, 0.0)
        self._fs_rotation_deg = 0.0
        self._pocket_preview_artists: list = []
        self._pocket_preview_active = False
        self.fs_display_name = ""
        self._last_fs: dict | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavToolbar(self.canvas, self)
        act = self.toolbar.addAction("⤢ Initial View")
        act.setToolTip("Reset axes to data limits (the plot keeps its size).")
        act.triggered.connect(self.reset_view)
        act_exp = self.toolbar.addAction("Export figure…")
        act_exp.setToolTip(
            "Save the current FS map as PNG (300 dpi, publication quality) or "
            "SVG (axes and labels stay vector; the map is embedded at 300 dpi). "
            "The figure is exported as displayed — pockets and BZ overlays included."
        )
        act_exp.triggered.connect(self.export_figure)
        from arpes.ui.widgets.fs_panel_pockets import (
            setup_manual_contour_tool,
            setup_pocket_action_bar,
            setup_pocket_lasso,
        )

        setup_pocket_lasso(self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        setup_pocket_action_bar(self)
        setup_manual_contour_tool(self)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        self.canvas.mpl_connect("pick_event", self._on_pick_event)
        self._hover_data = None
        self._pending_label = QLabel("Updating…", self.canvas)
        self._pending_label.setStyleSheet(
            "color: #ffb86c; background: rgba(43, 43, 43, 200);"
            "padding: 2px 8px; border-radius: 3px; font-weight: bold;"
        )
        self._pending_label.move(8, 8)
        self._pending_label.hide()
        self._dark()

    def _rotation_angle(self) -> float:
        return float(getattr(self, "_fs_rotation_deg", 0.0) or 0.0)

    def to_plot_points(self, points) -> np.ndarray:
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return arr.reshape((-1, 2)) if arr.ndim == 1 else arr
        orig_shape = arr.shape
        pts = arr.reshape((-1, 2))
        ang = np.deg2rad(self._rotation_angle())
        if abs(ang) <= 1e-12:
            return arr
        c, s = float(np.cos(ang)), float(np.sin(ang))
        out = np.column_stack((c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]))
        return out.reshape(orig_shape)

    def from_plot_points(self, points) -> np.ndarray:
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return arr.reshape((-1, 2)) if arr.ndim == 1 else arr
        orig_shape = arr.shape
        pts = arr.reshape((-1, 2))
        ang = -np.deg2rad(self._rotation_angle())
        if abs(ang) <= 1e-12:
            return arr
        c, s = float(np.cos(ang)), float(np.sin(ang))
        out = np.column_stack((c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]))
        return out.reshape(orig_shape)

    def set_pending(self, pending: bool) -> None:
        try:
            self._pending_label.setVisible(bool(pending))
        except RuntimeError:
            pass

    def export_figure(self) -> None:
        if not self._last_fs:
            QMessageBox.information(
                self, "Export figure", "Load and draw a Fermi surface first."
            )
            return
        src = self.fs_display_name or str(self._last_fs.get("title", "") or "")
        default_title = f"FS : {src}" if src else "FS"
        from arpes.ui.widgets.dialogs.fs_export_dialog import FsExportDialog

        dlg = FsExportDialog(self, default_title=default_title)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        opts = dlg.options()
        path, selected = QFileDialog.getSaveFileName(
            self, "Export figure", "fs_map.png",
            "PNG image, 300 dpi (*.png);;SVG vector (*.svg)",
        )
        if not path:
            return
        if "SVG" in selected and not path.lower().endswith(".svg"):
            path += ".svg"
        elif "PNG" in selected and not path.lower().endswith(".png"):
            path += ".png"
        try:
            from arpes.ui.widgets.fs_export import build_export_figure

            fig = build_export_figure(
                self._last_fs,
                add_hsym=opts["add_hsym"],
                title=opts["title"],
                to_plot=self.to_plot_points,
            )
            fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight")
        except Exception:
            QMessageBox.critical(
                self,
                "Export failed",
                f"The figure could not be saved to:\n{path}\n\n"
                "Check that the folder exists and is writable.",
            )

    def _format_coord(self, x, y) -> str:
        pt = self.from_plot_points(np.array([[x, y]], dtype=float))[0]
        base = f"kx = {pt[0]:.3f}   ky = {pt[1]:.3f}   (π/a)"
        if self._hover_data is None:
            return base
        ax_x, ax_y, fs = self._hover_data
        i = int(np.argmin(np.abs(ax_x - pt[0])))
        j = int(np.argmin(np.abs(ax_y - pt[1])))
        val = fs[j, i] if (j < fs.shape[0] and i < fs.shape[1]) else np.nan
        ival = f"{val:.3f}" if np.isfinite(val) else "—"
        return f"{base}   I = {ival}"

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
        self.fig.set_facecolor("#2b2b2b")
        self.ax.set_facecolor("#1a1a1a")

    def _rotated_mesh(self, x: np.ndarray, y: np.ndarray):
        angle = self._rotation_angle()
        if abs(angle) <= 1e-12:
            return x, y
        xx, yy = np.meshgrid(x, y)
        pts = self.to_plot_points(np.column_stack((xx.ravel(), yy.ravel())))
        return pts[:, 0].reshape(xx.shape), pts[:, 1].reshape(yy.shape)

    def draw_fs(self, raw_data: dict[str, Any] | None, params: FSParams):
        self._clear_bm_cut_artists()
        self._fs_rotation_deg = float(getattr(params, "fs_rotation_deg", 0.0) or 0.0)
        if raw_data is None:
            self.ax.cla()
            self._dark()
            self._mesh = None
            self._mesh_signature = None
            self._hover_data = None
            self._last_fs = None
            self._overlay_artists = []
            self._clear_pocket_artists()
            self.ax.text(0.5, 0.5, "Load an FS", transform=self.ax.transAxes, ha="center", va="center", color="w")
            self.canvas.draw_idle()
            return "No data"
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
            x = np.asarray(kx - params.kx_center, dtype=float)
            y = np.asarray(ky - params.ky_center, dtype=float)
            x_plot, y_plot = self._rotated_mesh(x, y)
            self._bm_cut_center = (float(params.kx_center), float(params.ky_center))
            signature = (
                tuple(np.asarray(fs).shape),
                _axis_signature(x),
                _axis_signature(y),
                round(float(params.kx_center), 8),
                round(float(params.ky_center), 8),
                round(self._rotation_angle(), 8),
            )
            for artist in list(self._overlay_artists):
                try:
                    artist.remove()
                except Exception:
                    pass
            self._overlay_artists = []
            self._clear_pocket_artists()
            if self._mesh is not None and self._mesh_signature != signature:
                try:
                    self._mesh.remove()
                except Exception:
                    pass
                self._mesh = None
            fresh_draw = self._mesh is None
            if fresh_draw:
                self.ax.cla()
                self._dark()
                self._mesh = self.ax.pcolormesh(x_plot, y_plot, fs, cmap=params.cmap, shading="auto", vmin=0, vmax=1)
                self._mesh_signature = signature
            else:
                self._mesh.set_array(np.asarray(fs).ravel())
                self._mesh.set_cmap(params.cmap)
                self._mesh.set_clim(0, 1)
            has_kxky_axes = fs_kind == "kxky"
            self.ax.set_aspect("equal" if has_kxky_axes else "auto")
            self.ax.set_xlabel(r"$k_x$ (π/a)", color="w")
            self.ax.set_ylabel(r"$k_y$ (π/a)" if has_kxky_axes else "tilt (deg)", color="w")
            suffix = f" | rot={self._rotation_angle():+.1f}°" if abs(self._rotation_angle()) > 1e-12 else ""
            self.ax.set_title(title + suffix, color="w", fontsize=10)
            if has_kxky_axes:
                self._overlay_bz(params)
                if params.overlay_bz_crystal or params.overlay_hs_crystal:
                    self._overlay_bz_crystal(params, raw_data)
            if fresh_draw:
                xp = np.asarray(x_plot, dtype=float)
                yp = np.asarray(y_plot, dtype=float)
                self.ax.set_xlim(float(np.nanmin(xp)), float(np.nanmax(xp)))
                self.ax.set_ylim(float(np.nanmin(yp)), float(np.nanmax(yp)))
            self.ax.tick_params(colors="w")
            for sp in self.ax.spines.values():
                sp.set_edgecolor("#555")
            self._hover_data = (x.ravel(), y.ravel(), np.asarray(fs))
            self._last_fs = {
                "x_plot": np.asarray(x_plot, dtype=float),
                "y_plot": np.asarray(y_plot, dtype=float),
                "fs": np.asarray(fs, dtype=float),
                "params": params,
                "fs_kind": fs_kind,
                "title": title,
            }
            self.ax.format_coord = self._format_coord
            self.canvas.draw_idle()
            return f"{title}{suffix} | shape={fs.shape}"
        except Exception as exc:
            self.ax.cla()
            self._dark()
            self._mesh = None
            self._mesh_signature = None
            self._hover_data = None
            self._last_fs = None
            self._overlay_artists = []
            self._clear_pocket_artists()
            self.ax.text(0.5, 0.5, str(exc), transform=self.ax.transAxes, ha="center", va="center", color="tomato", wrap=True)
            self.canvas.draw_idle()
            return f"FS error: {exc}"

    def _on_canvas_button_press(self, event) -> None:
        from arpes.ui.widgets.fs_panel_pockets import handle_manual_contour_click

        if handle_manual_contour_click(self, event):
            return
        if getattr(event, "button", None) != 3:
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        from arpes.ui.widgets.fs_panel_pockets import handle_canvas_right_click

        handle_canvas_right_click(self, event)

    def _on_pick_event(self, event) -> None:
        artist = getattr(event, "artist", None)
        idx = getattr(artist, "pocket_index", None)
        if idx is None:
            return
        self.pocket_open_requested.emit(int(idx))

    def _clear_pocket_artists(self) -> None:
        from arpes.ui.widgets.fs_panel_pockets import clear_pocket_artists

        clear_pocket_artists(self)

    def draw_pockets(self, pockets: list[dict] | None) -> None:
        from arpes.ui.widgets.fs_panel_pockets import draw_pockets

        draw_pockets(self, pockets)

    def draw_pocket_preview(self, contour) -> None:
        from arpes.ui.widgets.fs_panel_pockets import draw_pocket_preview

        draw_pocket_preview(self, contour)
        self._pocket_preview_active = True

    def clear_pocket_preview(self) -> None:
        from arpes.ui.widgets.fs_panel_pockets import clear_pocket_preview

        clear_pocket_preview(self)
        self._pocket_preview_active = False

    def _clear_bm_cut_artists(self) -> None:
        from arpes.ui.widgets.fs_panel_bm_cuts import clear_bm_cut_artists

        clear_bm_cut_artists(self)

    def draw_bm_cuts(self, cuts: list) -> None:
        from arpes.ui.widgets.fs_panel_bm_cuts import draw_bm_cuts

        draw_bm_cuts(self, cuts)

    def detect_gamma(self, raw_data: dict[str, Any] | None, params: FSParams):
        kx, ky, fs, _ = extract_fs_map(raw_data, params)
        if len(ky) < 3:
            raise ValueError("FS Γ detection is impossible without a 2D FS volume.")
        meta = raw_data.get("metadata", {}) or {}
        if meta.get("fs_kind") != "kxky":
            raise ValueError("FS Γ detection is available only with two axes in π/a.")
        return detect_gamma_from_fs_map(kx, ky, fs, params).as_dict()

    def _overlay_bz_crystal(self, p: FSParams, raw_data):
        from arpes.ui.widgets.fs_panel_bz_crystal import overlay_bz_crystal

        overlay_bz_crystal(self, p, raw_data)

    def _overlay_bz(self, p: FSParams):
        if not p.overlay_bz:
            return
        bx, by = p.bz_half_x, p.bz_half_y
        corners = self.to_plot_points(bz_polygon(p.bz_shape, bx, by, p.bz_angle_deg))
        line, = self.ax.plot(corners[:, 0], corners[:, 1], color="white", lw=1.2, ls="--", alpha=0.85)
        self._overlay_artists.append(line)
        h = self.to_plot_points(np.array([[-10.0, 0.0], [10.0, 0.0]], dtype=float))
        v = self.to_plot_points(np.array([[0.0, -10.0], [0.0, 10.0]], dtype=float))
        self._overlay_artists.append(self.ax.plot(h[:, 0], h[:, 1], color="white", lw=0.5, ls=":", alpha=0.5)[0])
        self._overlay_artists.append(self.ax.plot(v[:, 0], v[:, 1], color="white", lw=0.5, ls=":", alpha=0.5)[0])
        if p.show_hsym:
            def dot(x, y, name, color):
                pt = self.to_plot_points(np.array([[x, y]], dtype=float))[0]
                scat = self.ax.scatter([pt[0]], [pt[1]], c=color, s=35, zorder=5, linewidths=0)
                ann = self.ax.annotate(name, (pt[0], pt[1]), xytext=(4, 4), textcoords="offset points", color=color, fontsize=9, fontweight="bold")
                self._overlay_artists.extend([scat, ann])

            for x, y, name, color in bz_high_symmetry_points(
                p.bz_shape,
                bx,
                by,
                p.bz_angle_deg,
                label_overrides=(getattr(p, "bz_label_overrides", None) or None),
            ):
                dot(x, y, name, color)
