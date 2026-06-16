"""Pocket overlay drawer for FermiSurfaceCanvas (free functions)."""
from __future__ import annotations

import numpy as np


_POCKET_COLORS = (
    "#22d3ee",  # cyan
    "#f97316",  # orange
    "#a3e635",  # lime
    "#f472b6",  # pink
    "#c084fc",  # violet
    "#facc15",  # yellow
    "#60a5fa",  # blue
    "#fb7185",  # rose
)


def _pocket_color(idx: int) -> str:
    return _POCKET_COLORS[int(idx) % len(_POCKET_COLORS)]


def setup_pocket_lasso(canvas) -> None:
    """Toolbar toggle: drag a box around one pocket → pocket_lasso_requested.

    The RectangleSelector is created lazily on each activation so it always
    binds to the *current* axes content (draw_fs may have cla()'d the axes
    since the last use)."""
    canvas._pocket_lasso_selector = None
    act = canvas.toolbar.addAction("▭ Pocket")
    act.setCheckable(True)
    act.setToolTip(
        "Drag a box around ONE pocket: the seed point and iso-level are "
        "derived automatically from the selection. Then fine-tune the Level "
        "if needed and validate the preview (right-click)."
    )

    def _on_select(eclick, erelease):
        act.setChecked(False)  # one-shot; also deactivates via _toggle
        if eclick.xdata is None or erelease.xdata is None:
            return
        corners = np.array(
            [
                [float(eclick.xdata), float(eclick.ydata)],
                [float(eclick.xdata), float(erelease.ydata)],
                [float(erelease.xdata), float(eclick.ydata)],
                [float(erelease.xdata), float(erelease.ydata)],
            ],
            dtype=float,
        )
        if hasattr(canvas, "from_plot_points"):
            corners = canvas.from_plot_points(corners)
        canvas.pocket_lasso_requested.emit(
            float(np.nanmin(corners[:, 0])), float(np.nanmax(corners[:, 0])),
            float(np.nanmin(corners[:, 1])), float(np.nanmax(corners[:, 1])),
        )

    def _toggle(on: bool):
        sel = canvas._pocket_lasso_selector
        if not on:
            if sel is not None:
                sel.set_active(False)
            canvas._pocket_lasso_selector = None
            return
        from matplotlib.widgets import RectangleSelector
        canvas._pocket_lasso_selector = RectangleSelector(
            canvas.ax, _on_select, useblit=True, button=[1],
            interactive=False,
            props=dict(facecolor="none", edgecolor="#00d4ff", linestyle="--"),
        )

    act.toggled.connect(_toggle)
    canvas._act_pocket_lasso = act


def setup_pocket_action_bar(canvas) -> None:
    """Inline action bar shown under the FS canvas while a pocket preview is
    active: a Level slider (live contour redraw, 80 ms debounce) plus explicit
    Validate / Cancel buttons. Replaces the hidden right-click-only flow; the
    context menu remains as an alternative path."""
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget,
    )

    bar = QWidget()
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(4, 2, 4, 2)
    lay.addWidget(QLabel("Level:"))
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 1000)
    spin = QDoubleSpinBox()
    spin.setDecimals(3); spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.005); spin.setFixedWidth(70)
    spin.setKeyboardTracking(False)
    btn_ok = QPushButton("✓ Validate (MDC fit)")
    btn_ok.setToolTip("Run the radial MDC fit from this contour. "
                      "Produces kF ± σ. May take a few seconds.")
    btn_no = QPushButton("✗ Cancel")
    btn_no.setToolTip("Discard this preview without adding a pocket.")
    lay.addWidget(slider, stretch=3)
    lay.addWidget(spin)
    lay.addWidget(btn_ok)
    lay.addWidget(btn_no)
    bar.setVisible(False)

    # Debounced live redraw: a fast drag emits once per 80 ms, not per tick.
    timer = QTimer(bar); timer.setSingleShot(True); timer.setInterval(80)
    timer.timeout.connect(lambda: canvas.pocket_preview_level_changed.emit(spin.value()))
    canvas._pocket_bar_range = (0.0, 1.0)

    def _slider_to_level(v: int) -> float:
        lo, hi = canvas._pocket_bar_range
        return lo + (hi - lo) * (v / 1000.0)

    def _level_to_slider(lvl: float) -> int:
        lo, hi = canvas._pocket_bar_range
        span = (hi - lo) or 1.0
        return int(round(1000.0 * (float(lvl) - lo) / span))

    def _on_slider(v: int):
        spin.blockSignals(True); spin.setValue(_slider_to_level(v)); spin.blockSignals(False)
        timer.start()

    def _on_spin(val: float):
        slider.blockSignals(True); slider.setValue(_level_to_slider(val)); slider.blockSignals(False)
        timer.start()

    slider.valueChanged.connect(_on_slider)
    spin.valueChanged.connect(_on_spin)
    btn_ok.clicked.connect(canvas.pocket_preview_validate_requested)
    btn_no.clicked.connect(canvas.pocket_preview_cancel_requested)

    def set_state(visible: bool, level: float | None = None,
                  lo: float | None = None, hi: float | None = None) -> None:
        """Controller hook: show/hide the bar and calibrate slider mapping to
        the actual intensity range of the previewed map."""
        if lo is not None and hi is not None and hi > lo:
            canvas._pocket_bar_range = (float(lo), float(hi))
            spin.blockSignals(True)
            spin.setRange(float(lo), float(hi)); spin.blockSignals(False)
        if level is not None:
            spin.blockSignals(True); spin.setValue(float(level)); spin.blockSignals(False)
            slider.blockSignals(True)
            slider.setValue(_level_to_slider(float(level))); slider.blockSignals(False)
        bar.setVisible(bool(visible))

    canvas.pocket_action_bar = bar
    canvas.set_pocket_bar_state = set_state
    canvas.layout().addWidget(bar)


def setup_manual_contour_tool(canvas) -> None:
    """Manual contour mode: left-click crosses, then validate a closed contour."""
    from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget

    canvas._manual_contour_points = []
    canvas._manual_contour_artists = []
    canvas._manual_contour_active = False
    act = canvas.toolbar.addAction("✚ Contour")
    act.setCheckable(True)
    act.setToolTip(
        "Manual contour: left-click points around one pocket, then Validate. "
        "Use this when nearby small pockets confuse automatic contours."
    )

    bar = QWidget()
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(4, 2, 4, 2)
    label = QLabel("Manual contour: 0 points")
    chk_snap = QCheckBox("Snap")
    chk_snap.setChecked(True)
    chk_snap.setToolTip("Snap each cross to the strongest local edge before computing metrics.")
    btn_undo = QPushButton("Undo")
    btn_ok = QPushButton("Validate")
    btn_no = QPushButton("Cancel")
    lay.addWidget(label)
    lay.addWidget(chk_snap)
    lay.addWidget(btn_undo)
    lay.addWidget(btn_ok)
    lay.addWidget(btn_no)
    lay.addStretch(1)
    bar.setVisible(False)

    def _clear_artists() -> None:
        for art in list(canvas._manual_contour_artists):
            try:
                art.remove()
            except Exception:
                pass
        canvas._manual_contour_artists = []

    def _redraw() -> None:
        _clear_artists()
        pts = np.asarray(canvas._manual_contour_points, dtype=float)
        label.setText(f"Manual contour: {len(pts)} points")
        if pts.size:
            plot = canvas.to_plot_points(pts) if hasattr(canvas, "to_plot_points") else pts
            scat = canvas.ax.scatter(
                plot[:, 0], plot[:, 1], marker="x", s=52, color="#ffffff",
                linewidths=1.4, zorder=14,
            )
            canvas._manual_contour_artists.append(scat)
            if plot.shape[0] >= 2:
                line, = canvas.ax.plot(
                    plot[:, 0], plot[:, 1], color="#ffffff", lw=0.9,
                    ls=":", alpha=0.85, zorder=13,
                )
                canvas._manual_contour_artists.append(line)
        canvas.canvas.draw_idle()

    def _set_active(on: bool) -> None:
        canvas._manual_contour_active = bool(on)
        bar.setVisible(bool(on))
        if not on:
            canvas._manual_contour_points = []
            _clear_artists()
            canvas.canvas.draw_idle()
        label.setText(f"Manual contour: {len(canvas._manual_contour_points)} points")

    def _undo() -> None:
        if canvas._manual_contour_points:
            canvas._manual_contour_points.pop()
            _redraw()

    def _validate() -> None:
        pts = np.asarray(canvas._manual_contour_points, dtype=float)
        if pts.shape[0] >= 5:
            canvas.pocket_manual_contour_requested.emit({
                "points": pts.tolist(),
                "snap": bool(chk_snap.isChecked()),
            })
        act.setChecked(False)

    act.toggled.connect(_set_active)
    btn_undo.clicked.connect(_undo)
    btn_ok.clicked.connect(_validate)
    btn_no.clicked.connect(lambda: act.setChecked(False))

    canvas._act_manual_contour = act
    canvas._redraw_manual_contour = _redraw
    canvas.manual_contour_bar = bar
    canvas.layout().addWidget(bar)


def handle_manual_contour_click(canvas, event) -> bool:
    if not bool(getattr(canvas, "_manual_contour_active", False)):
        return False
    if getattr(event, "button", None) != 1:
        return True
    if event.inaxes is not canvas.ax or event.xdata is None or event.ydata is None:
        return True
    x, y = float(event.xdata), float(event.ydata)
    if hasattr(canvas, "from_plot_points"):
        x, y = canvas.from_plot_points([[x, y]])[0]
    canvas._manual_contour_points.append([float(x), float(y)])
    redraw = getattr(canvas, "_redraw_manual_contour", None)
    if redraw is not None:
        redraw()
    return True


def handle_canvas_right_click(canvas, event) -> None:
    """Open right-click pocket menu and emit the selected canvas signal."""
    from PyQt6.QtWidgets import QMenu
    from PyQt6.QtGui import QCursor

    menu = QMenu(canvas)
    act_preview = menu.addAction("Preview pocket here (ISO contour)")
    act = menu.addAction("Quick ISO (no fit)")
    act.setToolTip("Heuristic iso-contour only — fast quicklook, no kF ± σ.")
    act_validate = None
    act_cancel = None
    if canvas._pocket_preview_active:
        menu.addSeparator()
        act_validate = menu.addAction("Validate → MDC fit")
        act_cancel = menu.addAction("Cancel preview")
    menu.addSeparator()
    act_diag = menu.addAction("Diagnostic pairing FS ↔ BMs")
    menu.addSeparator()
    act_export = menu.addAction("Export pockets CSV")
    act_clear = menu.addAction("Clear pockets")
    chosen = menu.exec(QCursor.pos())
    x, y = float(event.xdata), float(event.ydata)
    if hasattr(canvas, "from_plot_points"):
        x, y = canvas.from_plot_points([[x, y]])[0]
    if chosen == act:
        canvas.pocket_requested.emit(x, y)
    elif chosen == act_preview:
        canvas.pocket_preview_requested.emit(x, y)
    elif chosen is not None and chosen is act_validate:
        canvas.pocket_preview_validate_requested.emit()
    elif chosen is not None and chosen is act_cancel:
        canvas.pocket_preview_cancel_requested.emit()
    elif chosen == act_diag:
        canvas.pairing_diagnose_requested.emit()
    elif chosen == act_export:
        canvas.pockets_export_requested.emit()
    elif chosen == act_clear:
        canvas.pockets_clear_requested.emit()


def clear_pocket_preview(canvas) -> None:
    for art in list(canvas._pocket_preview_artists):
        try:
            art.remove()
        except Exception:
            pass
    canvas._pocket_preview_artists = []
    canvas.canvas.draw_idle()


def draw_pocket_preview(canvas, contour) -> None:
    clear_pocket_preview(canvas)
    arr = np.asarray([] if contour is None else contour, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        canvas.canvas.draw_idle()
        return
    arr_plot = canvas.to_plot_points(arr) if hasattr(canvas, "to_plot_points") else arr
    pts = canvas.ax.scatter(
        arr_plot[:, 0], arr_plot[:, 1],
        s=16, marker="o", color="#00ffff", alpha=0.95,
        linewidths=0.0, zorder=12,
    )
    canvas._pocket_preview_artists.append(pts)
    canvas.canvas.draw_idle()


def clear_pocket_artists(canvas) -> None:
    for art in list(canvas._pocket_artists):
        try:
            art.remove()
        except Exception:
            pass
    canvas._pocket_artists = []


def draw_pockets(canvas, pockets: list[dict] | None) -> None:
    clear_pocket_artists(canvas)
    for idx, pocket in enumerate(pockets or [], start=1):
        contour = np.asarray(pocket.get("contour") or [], dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2 or contour.shape[0] < 3:
            continue
        color = _pocket_color(idx - 1)
        contour_plot = canvas.to_plot_points(contour) if hasattr(canvas, "to_plot_points") else contour
        pts = canvas.ax.scatter(
            contour_plot[:, 0], contour_plot[:, 1],
            s=18, marker="o", color=color, alpha=0.9,
            linewidths=0.0, zorder=10,
            picker=True,
        )
        pts.set_pickradius(5)
        setattr(pts, "pocket_index", idx - 1)
        label = str(pocket.get("hs_label_nearest") or f"P{idx}")
        cx = float(pocket.get("centroid_kx", np.nan)) - canvas._bm_cut_center[0]
        cy = float(pocket.get("centroid_ky", np.nan)) - canvas._bm_cut_center[1]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            cx = float(np.nanmean(contour[:, 0]))
            cy = float(np.nanmean(contour[:, 1]))
        if hasattr(canvas, "to_plot_points"):
            cx, cy = canvas.to_plot_points([[cx, cy]])[0]
        ann = canvas.ax.annotate(
            label,
            (cx, cy),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            zorder=11,
            picker=True,
        )
        setattr(ann, "pocket_index", idx - 1)
        canvas._pocket_artists.extend([pts, ann])
    canvas.canvas.draw_idle()
