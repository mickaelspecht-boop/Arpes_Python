"""Waterfall (BM stacked-curves) data prep + matplotlib draw routine.

Extracted from plot_compute.py to respect the architect's 700-LOC ceiling.
Pure-numpy / pure-matplotlib (no PyQt).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class WaterfallData:
    k_cut: np.ndarray
    ev_sel: np.ndarray
    data_cut: np.ndarray
    indices: list[int]
    bounds: tuple[float, float, float, float]
    spacing: float
    amp_scale: float

    @property
    def n_curves(self) -> int:
        return len(self.indices)


def prepare_waterfall_data(
    data,
    kpar,
    ev,
    *,
    bounds: tuple[float, float, float, float],
    n_target: int = 32,
    amp_scale: float = 1.8,
    spacing: float = 0.72,
) -> WaterfallData | None:
    kpar = np.asarray(kpar, dtype=float)
    ev = np.asarray(ev, dtype=float)
    data = np.asarray(data, dtype=float)
    k0, k1, e0, e1 = bounds
    k0, k1 = sorted((float(k0), float(k1)))
    e0, e1 = sorted((float(e0), float(e1)))
    k_mask = (kpar >= k0) & (kpar <= k1)
    e_mask = (ev >= e0) & (ev <= e1)
    if not k_mask.any() or not e_mask.any():
        return None

    k_cut = np.asarray(kpar[k_mask], dtype=float)
    ev_sel = np.asarray(ev[e_mask], dtype=float)
    data_cut = np.asarray(data[np.ix_(k_mask, e_mask)], dtype=float)

    n_target = max(1, int(n_target))
    step = max(1, int(np.ceil(ev_sel.size / n_target)))
    indices = list(range(0, ev_sel.size, step))
    if indices[-1] != ev_sel.size - 1:
        indices.append(ev_sel.size - 1)
    return WaterfallData(
        k_cut=k_cut,
        ev_sel=ev_sel,
        data_cut=data_cut,
        indices=indices,
        bounds=(k0, k1, e0, e1),
        spacing=float(spacing),
        amp_scale=float(amp_scale),
    )


def draw_waterfall_axes(
    ax,
    data,
    kpar,
    ev,
    *,
    bounds: tuple[float, float, float, float],
    n_target: int = 32,
    amp_scale: float = 1.8,
    smooth_sigma: float = 0.0,
    fit_result: dict | None = None,
    n_pairs: int = 0,
    pair_colors: list[str] | tuple[str, ...] = (),
    gamma_center: float = 0.0,
    residual_ax=None,
) -> bool:
    """Dessine le waterfall MDC sur un axe Matplotlib.

    Retourne True si des courbes ont été dessinées, False si la plage est vide.
    """
    ax.cla()
    ax.set_facecolor("#1a1a1a")
    if residual_ax is not None:
        residual_ax.cla()
        residual_ax.set_facecolor("#1a1a1a")
    wf = prepare_waterfall_data(
        data, kpar, ev,
        bounds=bounds,
        n_target=n_target,
        amp_scale=amp_scale,
    )
    if wf is None:
        ax.text(0.5, 0.5, "Plage waterfall vide", transform=ax.transAxes,
                ha="center", va="center", color="tomato")
        if residual_ax is not None:
            residual_ax.text(0.5, 0.5, "Plage waterfall vide", transform=residual_ax.transAxes,
                             ha="center", va="center", color="tomato")
        return False

    from scipy.ndimage import gaussian_filter1d
    import matplotlib

    k0, k1, e0, e1 = wf.bounds
    cmap = matplotlib.colormaps.get_cmap("plasma")
    smooth_sigma = max(0.0, float(smooth_sigma))

    for rank, j in enumerate(wf.indices):
        mdc = wf.data_cut[:, j].astype(float)
        if smooth_sigma > 0 and mdc.size > 3:
            mdc = gaussian_filter1d(np.nan_to_num(mdc), sigma=smooth_sigma)
        finite = np.isfinite(mdc)
        if not finite.any():
            continue
        lo, hi = np.nanpercentile(mdc[finite], [1, 99])
        if hi - lo > 1e-12:
            mdc_n = np.clip((mdc - lo) / (hi - lo), 0, 1)
        else:
            mdc_n = np.zeros_like(mdc)
        offset = rank * wf.spacing
        color = cmap(rank / max(1, wf.n_curves - 1))
        ax.plot(wf.k_cut, wf.amp_scale * mdc_n + offset, color=color, lw=1.05)
        ax.axhline(offset, color=color, lw=0.25, alpha=0.25)

    if fit_result is not None:
        e_fit = np.asarray(fit_result.get("e_fitted", []), dtype=float)
        residuals = fit_result.get("residuals") or []
        fit_curves = fit_result.get("fit_curves") or []
        fit_kpar = np.asarray(fit_result.get("fit_kpar", kpar), dtype=float)
        residual_rms = np.nan
        if residual_ax is not None and residuals:
            values = []
            for raw in residuals:
                arr = np.asarray(raw, dtype=float)
                values.append(arr[np.isfinite(arr)])
            if values:
                flat = np.concatenate(values)
                if flat.size:
                    residual_rms = float(np.sqrt(np.nanmean(flat ** 2)))
        if e_fit.size:
            residual_ticks: list[tuple[int, float]] = []
            for i in range(int(n_pairs)):
                color = pair_colors[i % len(pair_colors)] if pair_colors else "white"
                for key, marker in (("kF_minus", "o"), ("kF_plus", "^")):
                    series = fit_result.get(key, [])
                    if i >= len(series):
                        continue
                    k_series = np.asarray(series[i], dtype=float)
                    xs, ys = [], []
                    for ee, kk in zip(e_fit, k_series):
                        if not np.isfinite(ee) or not np.isfinite(kk):
                            continue
                        if ee < e0 or ee > e1 or kk < k0 or kk > k1:
                            continue
                        rel = int(np.argmin(np.abs(wf.ev_sel - ee)))
                        rank = min(range(wf.n_curves), key=lambda r: abs(wf.indices[r] - rel))
                        mdc_rank = wf.data_cut[:, wf.indices[rank]].astype(float)
                        finite = np.isfinite(mdc_rank)
                        if finite.any():
                            lo, hi = np.nanpercentile(mdc_rank[finite], [1, 99])
                            val = wf.amp_scale * float(
                                np.interp(
                                    kk,
                                    wf.k_cut,
                                    np.clip((mdc_rank - lo) / (hi - lo + 1e-12), 0, 1),
                                )
                            )
                        else:
                            val = 0.0
                        xs.append(kk)
                        ys.append(rank * wf.spacing + val)
                    if xs:
                        ax.scatter(xs, ys, s=14, color=color, marker=marker,
                                   edgecolors="none", alpha=0.9, zorder=5)
            if residual_ax is not None and residuals:
                for ii, ee in enumerate(e_fit):
                    if not np.isfinite(ee) or ee < e0 or ee > e1:
                        continue
                    rel = int(np.argmin(np.abs(wf.ev_sel - ee)))
                    rank = min(range(wf.n_curves), key=lambda r: abs(wf.indices[r] - rel))
                    offset = rank * wf.spacing
                    if ii < len(fit_curves):
                        fit_y = np.asarray(fit_curves[ii], dtype=float)
                        if fit_y.size == fit_kpar.size:
                            ax.plot(fit_kpar, wf.amp_scale * fit_y + offset,
                                    color="white", lw=0.55, alpha=0.65)
                    residual = np.asarray(residuals[ii], dtype=float)
                    if residual.size != fit_kpar.size:
                        continue
                    residual_ax.plot(fit_kpar, residual + offset,
                                     color="#fde047", lw=0.75, alpha=0.9)
                    residual_ax.axhline(offset, color="#777", lw=0.45, alpha=0.55)
                    if np.isfinite(residual_rms):
                        residual_ax.axhline(offset + residual_rms, color="#999",
                                            lw=0.35, ls="--", alpha=0.45)
                        residual_ax.axhline(offset - residual_rms, color="#999",
                                            lw=0.35, ls="--", alpha=0.45)
                    residual_ticks.append((rank, float(ee)))
                if residual_ticks:
                    step = max(1, len(residual_ticks) // 6)
                    shown = residual_ticks[::step]
                    if shown[-1] != residual_ticks[-1]:
                        shown.append(residual_ticks[-1])
                    residual_ax.set_yticks([r * wf.spacing for r, _ in shown])
                    residual_ax.set_yticklabels([f"{ee:.3f}" for _, ee in shown],
                                                fontsize=7, color="w")
                    residual_ax.set_title(
                        "Résidus MDC" if not np.isfinite(residual_rms)
                        else f"Résidus MDC  rms={residual_rms:.3f}",
                        fontsize=8, color="w",
                    )
                else:
                    residual_ax.text(0.5, 0.5, "Résidus hors plage",
                                     transform=residual_ax.transAxes,
                                     ha="center", va="center", color="#aaa", fontsize=8)
            elif residual_ax is not None:
                residual_ax.text(0.5, 0.5, "Résidus indisponibles",
                                 transform=residual_ax.transAxes,
                                 ha="center", va="center", color="#aaa", fontsize=8)

    tick_step = max(1, wf.n_curves // 9)
    tick_idx = list(range(0, wf.n_curves, tick_step))
    if tick_idx[-1] != wf.n_curves - 1:
        tick_idx.append(wf.n_curves - 1)
    ax.set_yticks([i * wf.spacing for i in tick_idx])
    ax.set_yticklabels([f"{wf.ev_sel[wf.indices[i]]:.3f}" for i in tick_idx],
                       fontsize=8, color="w")
    ax.axvline(0.0, color="w", lw=0.5, ls="--", alpha=0.35)
    ax.axvline(float(gamma_center), color="cyan", lw=0.6, ls=":", alpha=0.65)
    ax.set_xlim(k0, k1)
    ax.set_ylim(-0.25, max(1.0, (wf.n_curves - 1) * wf.spacing + wf.amp_scale + 0.2))
    ax.set_xlabel("k// (π/a)", fontsize=9, color="w")
    ax.set_ylabel("E − EF (eV)", fontsize=9, color="w")
    ax.set_title(
        f"Waterfall MDC  {e0:.3f}→{e1:.3f} eV  |  {wf.n_curves} courbes",
        fontsize=9, color="w")
    ax.tick_params(colors="w", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    if residual_ax is not None:
        residual_ax.set_xlim(k0, k1)
        residual_ax.set_xlabel("k// (π/a)", fontsize=8, color="w")
        residual_ax.set_ylabel("MDC-fit", fontsize=8, color="w")
        residual_ax.tick_params(colors="w", labelsize=7)
        for sp in residual_ax.spines.values():
            sp.set_edgecolor("#555")
    return True


