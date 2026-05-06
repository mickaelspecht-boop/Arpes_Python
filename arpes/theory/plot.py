"""Drawing helpers for optional theoretical band overlays."""
from __future__ import annotations

from typing import Any

from .models import TheoryBandData, TheoryOverlayConfig, filter_bands_for_view


def draw_theory_overlay(ax, overlay: dict[str, Any] | None) -> int:
    """Draw DFT bands on an existing ARPES band-map axis.

    Returns the number of drawn bands. Invalid or disabled overlays are no-op.
    """
    overlay = overlay or {}
    if not overlay.get("enabled", False):
        return 0
    data_raw = overlay.get("data") or {}
    config_raw = overlay.get("config") or {}
    data = TheoryBandData.from_dict(data_raw)
    config = TheoryOverlayConfig.from_dict({**config_raw, "enabled": overlay.get("enabled", False)})
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    curves = filter_bands_for_view(data, config, xlim=xlim, ylim=ylim)
    best = (overlay.get("comparison") or [{}])[0] if overlay.get("comparison") else {}
    import numpy as _np
    y0, y1 = sorted((float(ylim[0]), float(ylim[1])))
    margin = 0.1 * max(y1 - y0, 1e-9)
    for k, band in curves:
        b = _np.asarray(band, dtype=float).copy()
        out = (b < y0 - margin) | (b > y1 + margin)
        b[out] = _np.nan
        ax.plot(
            k,
            b,
            color="#f8fafc",
            lw=0.85,
            alpha=float(config.alpha),
            zorder=6,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if curves:
        label = _overlay_label(data, config, best)
        ax.text(
            0.99,
            0.98,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#f8fafc",
            fontsize=7,
            bbox={"facecolor": "#111827", "edgecolor": "#64748b", "alpha": 0.70, "pad": 2},
            zorder=9,
        )
    return len(curves)


def _overlay_label(data: TheoryBandData, config: TheoryOverlayConfig, best: dict[str, Any] | None = None) -> str:
    segment = f" | {config.segment}" if config.segment else ""
    label = (
        f"DFT MP {data.material_id}{segment} | "
        f"dE={config.energy_shift:+.2f} eV dk={config.k_shift:+.2f}"
    )
    if best:
        label += f"\nmeilleur score: bande {best.get('band_index')} rms={float(best.get('rms_e', 0.0))*1000:.0f} meV"
    return label
