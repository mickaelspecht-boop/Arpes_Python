"""BM cut overlay drawer for FermiSurfaceCanvas (free functions)."""
from __future__ import annotations


_COLOR = {
    "exact": "#00d4ff",
    "rotated": "#ffae42",
    "scaled": "#ff5544",
    "incompatible": "#888888",
}


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
    for cut in cuts:
        if cut.kx_points.size == 0:
            continue
        kx_plot = cut.kx_points - cx
        ky_plot = cut.ky_points - cy
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
