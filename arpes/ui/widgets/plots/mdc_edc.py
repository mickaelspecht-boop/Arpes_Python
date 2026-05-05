"""MDC/EDC plotting and fitting façade."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from .mdc_fit import *
from .mdc_diagnostics import *
from .mdc_regions import *
from .edc_fit import *

def mdc_waterfall(
    data_cut, kpar, ev_arr,
    ev_start=-0.5, ev_end=0.0, delta_ev=0.05,
    smooth_sigma=1.5, offset_scale=1.0, normalize="each",
    ax=None, cmap="coolwarm_r", lw=1.2,
    fill=True, fill_alpha=0.25, title=None,
):
    """
    Waterfall de MDCs : une courbe I(k) par tranche en energie.

    Parametres
    ----------
    data_cut : np.ndarray (nk, ne)
    kpar : np.ndarray (nk,)
    ev_arr : np.ndarray (ne,)
    ev_start, ev_end : float
        Fenetre en energie.
    delta_ev : float
        Pas entre deux MDCs.
    smooth_sigma : float
        Lissage gaussien 1D le long de k (pixels).
    offset_scale : float
        Facteur sur le decalage vertical.
    normalize : str
        "each", "global" ou "none".
    ax : Axes ou None
    cmap : str
    lw, fill, fill_alpha, title : display params

    Retourne
    --------
    fig, ax, energies
    """
    energies = []
    ev = ev_start
    while ev <= ev_end + 1e-9:
        energies.append(ev)
        ev += delta_ev
    if not energies:
        raise ValueError(f"Aucune energie dans [{ev_start}, {ev_end}]")

    global_max = 1.0
    if normalize == "global":
        global_max = float(np.nanmax(np.abs(data_cut))) or 1.0

    cmap_fn = plt.get_cmap(cmap)
    n = len(energies)
    colors = [cmap_fn(i / max(n - 1, 1)) for i in range(n)]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    for i, ev_i in enumerate(energies):
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        mdc = data_cut[:, ie].astype(float)

        if smooth_sigma > 0:
            mdc = gaussian_filter1d(mdc, sigma=smooth_sigma)

        if normalize == "each":
            mdc_range = mdc.max() - mdc.min()
            if mdc_range > 0:
                mdc = (mdc - mdc.min()) / mdc_range
        elif normalize == "global":
            mdc = mdc / global_max

        offset = i * delta_ev * offset_scale
        y = mdc + offset
        c = colors[i]

        ax.plot(kpar, y, color=c, lw=lw, label=f"{ev_arr[ie]:.3f} eV", zorder=n - i)
        if fill:
            ax.fill_between(kpar, offset, y, color=c,
                            alpha=fill_alpha, zorder=n - i)
        ax.axhline(offset, color=c, lw=0.4, ls='--', alpha=0.4, zorder=0)

    tick_step = max(1, n // 10)
    tick_idx = list(range(0, n, tick_step))
    if tick_idx[-1] != n - 1:
        tick_idx.append(n - 1)
    ax.set_yticks([i * delta_ev * offset_scale for i in tick_idx])
    ax.set_yticklabels([f"{energies[i]:.3f}" for i in tick_idx], fontsize=8)

    ax.set_xlabel("k// (pi/a)", fontsize=11)
    ax.set_ylabel("E - EF (eV)", fontsize=11)
    ax.set_xlim(kpar[0], kpar[-1])
    ax.set_title(title or f"Waterfall MDC  [{ev_start:.2f} -> {ev_end:.2f} eV]",
                 fontsize=11)

    if created_fig:
        plt.tight_layout()

    return fig, ax, energies


# =============================================================================
#  4. Seconde derivee et courbure 2D (Zhang et al., 2011)
# =============================================================================
