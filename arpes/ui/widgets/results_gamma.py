"""Γ(E) lifetime panel drawer (free functions, panel-first).

Split out of ``results.py`` (LOC cap). Draws the MDC linewidth Γ_k(E) of every
visible fitted file with its σ band, plus a selectable trend overlay:

- ``quadratic``: Fermi-liquid form Γ = Γ₀ + a·E²  (default)
- ``linear``:    marginal-FL-like form Γ = a + b·E

The trend model is read from ``panel._cmb_gamma_model`` (index 0 = quadratic,
1 = linear). The y/x limits are set from robust percentiles so a single bad
slice (huge Γ) no longer compresses the whole curve into a flat band — the
trend stays visible.
"""
from __future__ import annotations

import numpy as np

from arpes.analysis.results import weighted_linear_fit

_E_WINDOW = 0.30


def _trend_model(panel) -> str:
    cmb = getattr(panel, "_cmb_gamma_model", None)
    return "linear" if (cmb is not None and cmb.currentIndex() == 1) else "quadratic"


def _gamma_trend(e, g, sg, *, model: str, e_range=None):
    """Weighted Γ(E) trend over the chosen window.

    Returns ``(e_grid, g_grid, intercept, slope)`` or None if too few points.
    ``e_range=(lo, hi)`` overrides the default symmetric ``|E| ≤ _E_WINDOW``.
    """
    if e_range is not None:
        lo, hi = (e_range[0], e_range[1]) if e_range[0] <= e_range[1] else (e_range[1], e_range[0])
        in_window = (e >= lo) & (e <= hi)
    else:
        in_window = np.abs(e) <= _E_WINDOW
    valid = np.isfinite(e) & np.isfinite(g) & in_window
    if int(valid.sum()) < 3:
        return None
    ev, gv = e[valid], g[valid]
    w = None
    if sg is not None and np.size(sg):
        sv = np.asarray(sg, dtype=float)[: len(e)][valid]
        if sv.size == gv.size and np.all(np.isfinite(sv) & (sv > 0)):
            w = sv
    x = ev if model == "linear" else ev ** 2
    fit = weighted_linear_fit(x, gv, sigma=w)
    e_grid = np.linspace(float(ev.min()), float(ev.max()), 80)
    xg = e_grid if model == "linear" else e_grid ** 2
    return e_grid, fit.intercept + fit.slope * xg, float(fit.intercept), float(fit.slope)


def _apply_robust_limits(ax, xs, ys) -> None:
    """Frame the axes on the bulk of the data (2–98 pct) with margin so the
    trend is readable instead of zoomed onto a cluster or a few outliers."""
    if ys:
        allg = np.concatenate([np.ravel(np.asarray(v, dtype=float)) for v in ys])
        allg = allg[np.isfinite(allg)]
        if allg.size:
            # Asymmetric robust clip: keep the low end (Γ → Γ₀) but trim the few
            # bad slices with a runaway Γ that otherwise compress the trend.
            lo, hi = np.percentile(allg, 2), np.percentile(allg, 95)
            if hi <= lo:
                lo, hi = float(allg.min()), float(allg.max())
            pad = 0.12 * (hi - lo) if hi > lo else max(abs(hi), 1e-3) * 0.1
            ax.set_ylim(lo - pad, hi + pad)
    if xs:
        allx = np.concatenate([np.ravel(np.asarray(v, dtype=float)) for v in xs])
        allx = allx[np.isfinite(allx)]
        if allx.size:
            x0, x1 = float(allx.min()), float(allx.max())
            xpad = 0.05 * (x1 - x0) if x1 > x0 else 0.02
            ax.set_xlim(x0 - xpad, x1 + xpad)


def draw_gamma_panel(panel, colors) -> None:
    from arpes.analysis.results import gamma_reliability_mask
    from arpes.ui.widgets import results_bands
    visible = panel._visible_files()
    model = _trend_model(panel)
    ax = panel._canvas_gamma.ax
    ax.cla(); ax.set_facecolor("#1a1a1a")
    panel._canvas_gamma.fig.set_facecolor("#2b2b2b")
    plotted = 0
    panel._gamma_sigma_missing = False
    panel._gamma_unreliable_shown = False
    xs: list = []
    ys: list = []
    unreliable_labeled = False
    erange = panel._gamma_e_range() if hasattr(panel, "_gamma_e_range") else None
    eq_lines: list = []
    for ci, (name, entry) in enumerate(panel._session.files.items()):
        if entry.fit_result is None or name not in visible:
            continue
        fr = entry.fit_result
        ev = np.asarray(fr.get("e_fitted", []), dtype=float)
        g_arrays = fr.get("gamma_corrige") or fr.get("gamma") or []
        sg_arrays = fr.get("sigma_gamma") or []
        if not sg_arrays:
            sg_arrays = (fr.get("ensemble") or {}).get("gamma_std") or []
        base_color = panel._color_for_file(name, ci, colors)
        for i, g_raw in enumerate(g_arrays):
            if not results_bands.band_visible(panel, name, i):
                continue
            style = results_bands.band_style(base_color, i)
            color = style["color"]
            pairs = list(getattr(entry.fit_params, "pairs", None) or [])
            gmax = pairs[i].get("gamma_max") if i < len(pairs) else entry.fit_params.gamma_max
            g = np.asarray(g_raw, dtype=float)
            n = min(len(ev), len(g))
            if n == 0:
                continue
            e_n, g_n = ev[:n], g[:n]
            finite = np.isfinite(e_n) & np.isfinite(g_n)
            if int(finite.sum()) < 3:
                continue
            # Reliable = peaks resolved AND not saturated (gamma_reliability_mask).
            reliable = gamma_reliability_mask(fr, pair_index=i, gamma_max=gmax)[:n] & finite
            unreliable = finite & ~reliable
            # Reliable points: solid line + markers, framed by the y-limits.
            if reliable.any():
                ax.plot(
                    e_n[reliable], g_n[reliable],
                    marker=style["marker_plus"], linestyle=style["linestyle"],
                    ms=3, lw=0.8, color=color, alpha=0.85,
                    label=results_bands.band_label(name, entry, i)
                    if plotted < 8 else "_",
                )
                xs.append(e_n[reliable]); ys.append(g_n[reliable])
            # Unreliable points: grey crosses, no line — visible but flagged.
            if unreliable.any():
                ax.plot(e_n[unreliable], g_n[unreliable], "x", ms=4, color="#777",
                        alpha=0.55, label=("unreliable (peaks merged / saturated)"
                                           if not unreliable_labeled else "_"))
                unreliable_labeled = True
                panel._gamma_unreliable_shown = True
            sg_i = None
            if i < len(sg_arrays):
                sg = np.asarray(sg_arrays[i], dtype=float)[:n]
                sg_i = sg
                band_valid = reliable & np.isfinite(sg) & (sg > 0)
                if band_valid.any():
                    ax.fill_between(e_n[band_valid],
                                    g_n[band_valid] - sg[band_valid],
                                    g_n[band_valid] + sg[band_valid],
                                    color=color, alpha=0.18, lw=0)
                    idxs = np.flatnonzero(band_valid)[::max(1, int(band_valid.sum()) // 12)]
                    ax.errorbar(e_n[idxs], g_n[idxs], yerr=sg[idxs],
                                fmt="none", ecolor=color, elinewidth=0.9,
                                capsize=2.5, alpha=0.9, zorder=3)
            else:
                panel._gamma_sigma_missing = True
            # Trend fit only on the reliable region within the chosen window.
            sg_rel = sg_i[reliable] if sg_i is not None else None
            trend = _gamma_trend(e_n[reliable], g_n[reliable], sg_rel,
                                 model=model, e_range=erange)
            if trend is not None:
                ax.plot(trend[0], trend[1], "--", color=color, lw=1.3, alpha=0.85)
                intercept, slope = trend[2], trend[3]
                if model == "linear":
                    eq = (
                        f"{results_bands.band_label(name, entry, i)}:  "
                        f"Γ = {intercept:.3f} + {slope:.2f}·E"
                    )
                else:
                    eq = (
                        f"{results_bands.band_label(name, entry, i)}:  "
                        f"Γ₀ = {intercept:.3f}, a = {slope:.1f}"
                    )
                eq_lines.append((eq, color))
            plotted += 1
    # Chosen fit equations, one coloured line each (also written to the export).
    if eq_lines:
        ax.text(0.015, 0.975, "Fit (" + ("a + b·E" if model == "linear"
                else "Γ₀ + a·E²") + f")  E∈[{erange[0]:.3f}, {erange[1]:.3f}] eV"
                if erange else "Fit", transform=ax.transAxes, ha="left", va="top",
                color="#cfd8e3", fontsize=7.5, fontweight="bold")
        for j, (eq, col) in enumerate(eq_lines[:8]):
            ax.text(0.015, 0.93 - 0.045 * j, eq, transform=ax.transAxes,
                    ha="left", va="top", color=col, fontsize=7.5)
    panel._gamma_equations = [(eq, model) for eq, _ in eq_lines]
    _apply_robust_limits(ax, xs, ys)
    form = "a + b·E" if model == "linear" else r"$\Gamma_0 + a E^2$"
    ax.set_xlabel(r"$E - E_F$ (eV)", fontsize=10, color="w")
    ax.set_ylabel(r"$\Gamma_k$ (HWHM, π/a)", fontsize=10, color="w")
    ax.set_title(rf"$\Gamma_k(E)$ — bands ±σ and {('linear' if model=='linear' else 'Fermi-liquid')} trend ({form})",
                 fontsize=10, color="w")
    ax.tick_params(colors="w")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    if panel._gamma_sigma_missing:
        ax.text(0.02, 0.97, "σ not stored in this fit — re-run the MDC fit "
                "to get uncertainty bars", transform=ax.transAxes,
                ha="left", va="top", color="#e6b35a", fontsize=8)
    if plotted > 0:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 8:
            leg = ax.legend(fontsize=7, facecolor="#333", labelcolor="w",
                            loc="best", frameon=True, framealpha=0.75)
            leg.set_draggable(True)
    else:
        panel._canvas_gamma.fig.subplots_adjust(right=0.97)
    panel._canvas_gamma.redraw()
