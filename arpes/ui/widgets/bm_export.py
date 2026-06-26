"""Print-style band-map figure for export.

Like the Fermi-surface export, the band map is rebuilt on a fresh figure with a
transparent background and black axes so it can go straight onto a white page.
The map is drawn in the display mode currently selected (raw, second
derivative, or curvature) and the only overlay is the Fermi-level line.

No PyQt here: matplotlib only.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def build_bm_export_figure(disp, kpar, ev, *, cmap, color_kwargs,
                           gamma: float = 1.0, title: str = "",
                           ef_eV: float = 0.0) -> Figure:
    """Return a transparent, black-axes band map with the EF line only.

    ``disp`` is the displayed array (k, E); it is transposed for pcolormesh to
    match the live canvas. ``cmap`` and ``color_kwargs`` are the same the canvas
    uses for the current mode, so the exported image matches what is shown.
    """
    disp = np.asarray(disp, dtype=float)
    kpar = np.asarray(kpar, dtype=float)
    ev = np.asarray(ev, dtype=float)

    fig = Figure(figsize=(6, 5), tight_layout=True)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("none")

    kw = dict(color_kwargs)
    if float(gamma) != 1.0:
        from matplotlib.colors import PowerNorm
        kw = {"norm": PowerNorm(gamma=float(gamma),
                                vmin=color_kwargs.get("vmin"),
                                vmax=color_kwargs.get("vmax"))}
    ax.pcolormesh(kpar, ev, disp.T, cmap=cmap, shading="auto", **kw)
    ax.axhline(float(ef_eV), color="k", lw=0.8, ls="--", alpha=0.8)

    ax.set_xlabel(r"$k$ ($\pi/a$)", color="k")
    ax.set_ylabel(r"$E - E_F$ (eV)", color="k")
    if title:
        ax.set_title(title, color="k", fontsize=10)
    ax.tick_params(colors="k")
    for sp in ax.spines.values():
        sp.set_edgecolor("k")
    return fig
