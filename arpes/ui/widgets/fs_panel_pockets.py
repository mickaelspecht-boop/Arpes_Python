"""Pocket overlay drawer for FermiSurfaceCanvas (free functions)."""
from __future__ import annotations

import numpy as np


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
