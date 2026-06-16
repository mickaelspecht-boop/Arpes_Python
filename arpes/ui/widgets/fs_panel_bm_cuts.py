"""BM cut overlay drawer for FermiSurfaceCanvas (free functions)."""
from __future__ import annotations

import math


_COLOR = {
    "exact": "#00d4ff",
    "rotated": "#ffae42",
    "scaled": "#ff5544",
    "incompatible": "#888888",
}


def _line_key(cut) -> tuple:
    if cut.kx_points.size == 0:
        return ()
    return (
        round(float(cut.kx_points[0]), 5),
        round(float(cut.ky_points[0]), 5),
        round(float(cut.kx_points[-1]), 5),
        round(float(cut.ky_points[-1]), 5),
    )


def _duplicate_offsets(cuts: list) -> dict[int, float]:
    groups: dict[tuple, list[int]] = {}
    for i, cut in enumerate(cuts):
        key = _line_key(cut)
        if key:
            groups.setdefault(key, []).append(i)
    offsets: dict[int, float] = {}
    for idxs in groups.values():
        n = len(idxs)
        if n <= 1:
            continue
        for rank, idx in enumerate(idxs):
            offsets[idx] = (rank - (n - 1) / 2.0) * 0.018
    return offsets


def clear_bm_cut_artists(canvas) -> None:
    for art in list(canvas._bm_cut_artists):
        try:
            art.remove()
        except Exception:
            pass
    canvas._bm_cut_artists = []


def draw_bm_cuts(canvas, cuts: list) -> None:
    """B.3 — overlay des lignes BM cuts sur la FS courante.

    cuts : list[BMCutLine] (cf arpes/physics/bm_cut_overlay.py).
    Couleurs : cyan (exact), orange (rotated azi), rouge pointillé (scaled hv).
    Lignes pickables (5 px) — attach `bm_cut_path` pour interaction click.
    """
    clear_bm_cut_artists(canvas)
    if not cuts:
        canvas.canvas.draw_idle()
        return
    cx, cy = getattr(canvas, "_bm_cut_center", (0.0, 0.0))
    offsets = _duplicate_offsets(cuts)
    for i, cut in enumerate(cuts):
        if cut.kx_points.size == 0:
            continue
        kx_plot = cut.kx_points - cx
        ky_plot = cut.ky_points - cy
        offset = offsets.get(i, 0.0)
        if offset:
            dx = float(kx_plot[-1] - kx_plot[0])
            dy = float(ky_plot[-1] - ky_plot[0])
            norm = math.hypot(dx, dy)
            if norm > 1e-12:
                kx_plot = kx_plot + (-dy / norm) * offset
                ky_plot = ky_plot + (dx / norm) * offset
        if hasattr(canvas, "to_plot_points"):
            pts_plot = canvas.to_plot_points(
                [[float(x), float(y)] for x, y in zip(kx_plot, ky_plot)]
            )
            kx_plot = pts_plot[:, 0]
            ky_plot = pts_plot[:, 1]
        color = _COLOR.get(cut.quality, "white")
        linestyle = "--" if cut.quality == "scaled" else "-"
        line, = canvas.ax.plot(
            kx_plot, ky_plot,
            color=color, linestyle=linestyle,
            linewidth=1.2, alpha=0.78,
            picker=True, pickradius=5,
            zorder=8,
        )
        setattr(line, "bm_cut_path", cut.bm_path)
        setattr(line, "bm_cut_label", cut.label)
        canvas._bm_cut_artists.append(line)
        ann = canvas.ax.annotate(
            cut.label,
            (kx_plot[0], ky_plot[0]),
            color=color, fontsize=7, alpha=0.85,
            xytext=(4, 4), textcoords="offset points",
            zorder=9,
        )
        canvas._bm_cut_artists.append(ann)
    canvas.canvas.draw_idle()
