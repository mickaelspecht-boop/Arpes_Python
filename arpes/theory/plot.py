"""Drawing helpers for optional theoretical band overlays."""
from __future__ import annotations

from typing import Any

from .alignment import effective_mu_shift, effective_z_scale
from .models import TheoryBandData, TheoryOverlayConfig, select_bands_for_view


def _band_color(idx: int):
    """Stable color by band index (cyclic tab20)."""
    import matplotlib as _mpl

    return _mpl.colormaps["tab20"](int(idx) % 20)


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
    curves = select_bands_for_view(data, config, xlim=xlim, ylim=ylim)
    best = (overlay.get("comparison") or [{}])[0] if overlay.get("comparison") else {}
    import numpy as _np
    y0, y1 = sorted((float(ylim[0]), float(ylim[1])))
    margin = 0.1 * max(y1 - y0, 1e-9)
    chars = list(data.band_character or [])
    drawn_idx: list[int] = []
    for idx, k, band in curves:
        b = _np.asarray(band, dtype=float).copy()
        out = (b < y0 - margin) | (b > y1 + margin)
        b[out] = _np.nan
        color = _band_color(idx) if config.color_by_band else "#f8fafc"
        ax.plot(
            k,
            b,
            color=color,
            lw=0.85,
            alpha=float(config.alpha),
            zorder=6,
        )
        if idx not in drawn_idx:
            drawn_idx.append(idx)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if curves and config.color_by_band:
        _draw_band_legend(ax, drawn_idx, chars)
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


def _draw_band_legend(ax, drawn_idx: list[int], chars: list[str]) -> None:
    """Compact legend: color swatch + b{idx} + character."""
    from matplotlib.lines import Line2D

    if not drawn_idx:
        return
    handles = []
    labels = []
    for idx in drawn_idx[:14]:  # overflow guard
        ch = chars[idx] if 0 <= idx < len(chars) else ""
        labels.append(f"b{idx}" + (f"·{ch}" if ch else ""))
        handles.append(Line2D([0], [0], color=_band_color(idx), lw=2))
    leg = ax.legend(
        handles, labels, loc="upper left", fontsize=6, ncol=1,
        framealpha=0.70, facecolor="#111827", edgecolor="#64748b",
        labelcolor="#f8fafc", handlelength=1.2, borderpad=0.3,
        labelspacing=0.25, title="DFT bands",
    )
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_color("#f8fafc")
        leg.get_title().set_fontsize(6)
    leg.set_zorder(9)


def _overlay_label(data: TheoryBandData, config: TheoryOverlayConfig, best: dict[str, Any] | None = None) -> str:
    segment = f" | {config.segment}" if config.segment else ""
    source_label = "DFT MP" if data.source == "materials_project" else "local DFT"
    label = (
        f"{source_label} {data.material_id}{segment} | "
        f"mu={effective_mu_shift(config):+.2f} eV Z={effective_z_scale(config):.2f} "
        f"dk={config.k_shift:+.2f}"
    )
    if best:
        label += f"\nbest score: band {best.get('band_index')} rms={float(best.get('rms_e', 0.0))*1000:.0f} meV"
    return label
