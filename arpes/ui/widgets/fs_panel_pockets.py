"""Pocket overlay drawer for FermiSurfaceCanvas (free functions)."""
from __future__ import annotations

import numpy as np


def handle_canvas_right_click(canvas, event) -> None:
    """Open right-click pocket menu and emit the selected canvas signal."""
    from PyQt6.QtWidgets import QMenu
    from PyQt6.QtGui import QCursor

    menu = QMenu(canvas)
    act_wiz = menu.addAction("Caractérisation guidée (wizard)")
    act = menu.addAction("Caractériser poche ici (iso-contour)")
    act_mdc = menu.addAction("Caractériser par MDC radial (publication)")
    act_preview = menu.addAction("Aperçu poche ici (slider)")
    act_validate = None
    act_cancel = None
    if canvas._pocket_preview_active:
        menu.addSeparator()
        act_validate = menu.addAction("Valider l'aperçu")
        act_cancel = menu.addAction("Annuler l'aperçu")
    menu.addSeparator()
    act_diag = menu.addAction("Diagnostic pairing FS ↔ BMs")
    menu.addSeparator()
    act_export = menu.addAction("Exporter poches CSV")
    act_clear = menu.addAction("Effacer poches")
    chosen = menu.exec(QCursor.pos())
    x, y = float(event.xdata), float(event.ydata)
    if chosen == act_wiz:
        canvas.pocket_wizard_requested.emit(x, y)
    elif chosen == act:
        canvas.pocket_requested.emit(x, y)
    elif chosen == act_mdc:
        canvas.pocket_mdc_requested.emit(x, y)
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
    line, = canvas.ax.plot(
        arr[:, 0] - cx, arr[:, 1] - cy,
        color="#00ffff", lw=1.8, linestyle=(0, (4, 3)),
        alpha=0.95, zorder=12,
    )
    canvas._pocket_preview_artists.append(line)
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
        line, = canvas.ax.plot(
            contour[:, 0], contour[:, 1],
            color="#39ff88", lw=1.5, alpha=0.9, zorder=10,
            picker=True, pickradius=5,
        )
        setattr(line, "pocket_index", idx - 1)
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
            color="#39ff88",
            fontsize=9,
            fontweight="bold",
            zorder=11,
            picker=True,
        )
        setattr(ann, "pocket_index", idx - 1)
        canvas._pocket_artists.extend([line, ann])
    canvas.canvas.draw_idle()
