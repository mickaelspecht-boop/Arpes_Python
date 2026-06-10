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
        canvas.pocket_lasso_requested.emit(
            float(eclick.xdata), float(erelease.xdata),
            float(eclick.ydata), float(erelease.ydata),
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
    arr = np.asarray(contour or [], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        canvas.canvas.draw_idle()
        return
    cx, cy = getattr(canvas, "_bm_cut_center", (0.0, 0.0))
    pts = canvas.ax.scatter(
        arr[:, 0] - cx, arr[:, 1] - cy,
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
        pts = canvas.ax.scatter(
            contour[:, 0], contour[:, 1],
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
