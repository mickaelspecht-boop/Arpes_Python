"""Build a print-style Fermi-surface figure for export.

The interactive canvas is dark (white labels on a black axes). For a figure
meant to go on a white page or into a paper, the export uses a transparent
background and black axes, labels, ticks and title. The map itself, the
optional Brillouin-zone outline and the high-symmetry markers are rebuilt on
a fresh ``Figure`` so the live canvas is never mutated.

No PyQt here: only matplotlib and the pure Brillouin-zone helpers.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from arpes.physics.bz import bz_high_symmetry_points, bz_polygon


def build_export_figure(ctx: dict, *, add_hsym: bool, title: str, to_plot) -> Figure:
    """Return a transparent, black-axes ``Figure`` of the current FS map.

    ``ctx`` is the canvas' last-draw context:
    ``{x_plot, y_plot, fs, params, fs_kind, title}``. ``to_plot`` is the
    canvas' rotation transform, so the overlay matches the displayed map.
    """
    x_plot = np.asarray(ctx["x_plot"], dtype=float)
    y_plot = np.asarray(ctx["y_plot"], dtype=float)
    fs = np.asarray(ctx["fs"], dtype=float)
    params = ctx["params"]
    is_kxky = ctx.get("fs_kind") == "kxky"

    fig = Figure(figsize=(7, 6), tight_layout=True)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("none")
    ax.pcolormesh(x_plot, y_plot, fs, cmap=params.cmap, shading="auto", vmin=0, vmax=1)
    ax.set_aspect("equal" if is_kxky else "auto")
    ax.set_xlabel(r"$k_x$ (π/a)", color="k")
    ax.set_ylabel(r"$k_y$ (π/a)" if is_kxky else "tilt (deg)", color="k")
    if title:
        ax.set_title(title, color="k", fontsize=11)
    ax.tick_params(colors="k")
    for sp in ax.spines.values():
        sp.set_edgecolor("k")

    if add_hsym and is_kxky:
        _draw_hsym(ax, params, to_plot)
    return fig


def _draw_hsym(ax, p, to_plot) -> None:
    """Brillouin-zone outline plus black high-symmetry markers and labels."""
    corners = to_plot(bz_polygon(p.bz_shape, p.bz_half_x, p.bz_half_y, p.bz_angle_deg))
    corners = np.asarray(corners, dtype=float)
    ax.plot(corners[:, 0], corners[:, 1], color="k", lw=1.0, ls="--", alpha=0.7)
    for x, y, name, _color in bz_high_symmetry_points(
        p.bz_shape,
        p.bz_half_x,
        p.bz_half_y,
        p.bz_angle_deg,
        label_overrides=(getattr(p, "bz_label_overrides", None) or None),
    ):
        pt = np.asarray(to_plot(np.array([[x, y]], dtype=float)), dtype=float)[0]
        ax.scatter([pt[0]], [pt[1]], c="k", s=30, zorder=5, linewidths=0)
        ax.annotate(
            name, (pt[0], pt[1]), xytext=(4, 4), textcoords="offset points",
            color="k", fontsize=10, fontweight="bold",
        )
