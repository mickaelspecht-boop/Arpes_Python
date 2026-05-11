"""Préparation des données de plot ARPES — sans PyQt.

Première tranche de l'Update L : extraire les calculs de données d'affichage
BM sans déplacer les canvases Matplotlib ni les callbacks souris.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np


def apply_edcnorm(data: np.ndarray) -> np.ndarray:
    edc = np.nanmean(data, axis=0, keepdims=True)
    safe = np.where((np.abs(edc) > 1e-12) & np.isfinite(edc), edc, 1.0)
    return data / safe


def compute_secdev(data: np.ndarray, kpar, ev_arr, sigma_k=2.0, sigma_e=2.0) -> np.ndarray:
    """-d²I/dE² lissée."""
    from scipy.ndimage import gaussian_filter

    d = gaussian_filter(data.astype(float), sigma=[sigma_k, sigma_e])
    de = np.gradient(np.gradient(d, ev_arr, axis=1), ev_arr, axis=1)
    return -de


def compute_curvature(
    data: np.ndarray,
    kpar,
    ev_arr,
    sigma_k=2.0,
    sigma_e=2.0,
    c0_fraction=0.05,
    border_clip=3,
) -> np.ndarray:
    """Courbure 2D ARPES type Zhang et al. (RSI 2011)."""
    from scipy.ndimage import gaussian_filter

    arr = data.astype(float)
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        arr = arr.copy()
        finite = arr[np.isfinite(arr)]
        arr[nan_mask] = float(np.nanmedian(finite)) if finite.size else 0.0

    d = gaussian_filter(arr, sigma=[sigma_k, sigma_e])
    dI_dE = np.gradient(d, ev_arr, axis=1)
    dI_dk = np.gradient(d, kpar, axis=0)
    d2I_dk2 = np.gradient(dI_dk, kpar, axis=0)
    d2I_dkdE = np.gradient(dI_dk, ev_arr, axis=1)

    bc = max(0, int(border_clip))
    interior = (slice(bc, -bc or None), slice(bc, -bc or None))
    gk_ref = np.abs(dI_dk[interior])
    ge_ref = np.abs(dI_dE[interior])
    if gk_ref.size == 0 or ge_ref.size == 0:
        gk_ref = np.abs(dI_dk)
        ge_ref = np.abs(dI_dE)
    C0 = float(c0_fraction) * (np.nanmax(gk_ref) ** 2 + np.nanmax(ge_ref) ** 2)

    numer = (C0 + dI_dE**2) * d2I_dk2 - dI_dk * dI_dE * d2I_dkdE
    denom = (C0 + dI_dk**2 + dI_dE**2) ** 1.5
    curv = -numer / (denom + 1e-30)

    if bc > 0 and min(curv.shape) > 2 * bc:
        curv[:bc, :] = np.nan
        curv[-bc:, :] = np.nan
        curv[:, :bc] = np.nan
        curv[:, -bc:] = np.nan
    return curv


def _compute_below_ef_only(compute_fn, data: np.ndarray, kpar, ev_arr) -> np.ndarray:
    """Calcule un mode derive seulement pour E <= EF, masque E > EF."""
    ev_arr = np.asarray(ev_arr, dtype=float)
    data = np.asarray(data)
    out = np.full_like(data, np.nan, dtype=float)
    below = ev_arr <= 0.0
    if below.sum() < 3:
        return out
    out[:, below] = compute_fn(data[:, below], kpar, ev_arr[below])
    return out


def display_grid_config(cfg: dict | None) -> dict:
    cfg = cfg or {}
    try:
        strength = float(cfg.get("strength", 0.85))
    except Exception:
        strength = 0.85
    return {
        "method": "fft2mask",
        "grid_freq": None,
        "grid_period_px": None,
        "notch_width": 2,
        "notch_sigma": 0.8,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "fft2_center_radius": 18.0,
        "fft2_peak_sensitivity": 2.5,
        "fft2_plane": "display",
    }


@dataclass
class BandmapDisplayResult:
    data: np.ndarray
    grid_info: dict = field(default_factory=dict)


@dataclass
class BandmapAxesState:
    """Matplotlib handles for a reusable band-map axis."""
    mesh: object | None = None
    signature: tuple = field(default_factory=tuple)
    base_artists: list = field(default_factory=list)


def _axis_signature(axis) -> tuple:
    arr = np.asarray(axis, dtype=float)
    if arr.size == 0:
        return (0,)
    payload = np.ascontiguousarray(arr, dtype=np.float64)
    digest = hashlib.sha256(payload.tobytes()).hexdigest()
    return (tuple(payload.shape), digest)


def compute_bandmap_display(
    raw_data: dict,
    *,
    mode: str,
    edc_norm_enabled: bool,
    grid_correction: dict | None = None,
    grid_artifact_fn=None,
) -> BandmapDisplayResult:
    """Prépare la carte BM affichée pour le mode demandé."""
    raw = np.asarray(raw_data["data"])
    if mode == "Raw":
        disp = raw
    elif mode == "EDCnorm":
        disp = apply_edcnorm(raw)
    elif mode == "SecDev":
        norm = apply_edcnorm(raw) if edc_norm_enabled else raw
        disp = _compute_below_ef_only(compute_secdev, norm, raw_data["kpar"], raw_data["ev_arr"])
    elif mode == "Curvature":
        norm = apply_edcnorm(raw) if edc_norm_enabled else raw
        disp = _compute_below_ef_only(compute_curvature, norm, raw_data["kpar"], raw_data["ev_arr"])
    else:
        disp = raw

    grid_info: dict = {}
    if grid_correction:
        grid_cfg = display_grid_config(grid_correction)
        if grid_artifact_fn is not None:
            try:
                disp, info = grid_artifact_fn(np.asarray(disp, dtype=float), axis=0, **grid_cfg)
                info.update({
                    "method": "display_fft2mask",
                    "view_mode": mode,
                    "target": "display",
                    "shape": tuple(np.asarray(disp).shape),
                    "strength": grid_cfg["strength"],
                })
                grid_info = info
            except Exception as exc:
                grid_info = {
                    "method": "display_fft2mask",
                    "error": str(exc),
                    "view_mode": mode,
                    "strength": grid_cfg["strength"],
                }

    return BandmapDisplayResult(data=disp, grid_info=grid_info)


def fit_roi_bounds(
    kpar,
    ev_arr,
    *,
    k_min: float,
    k_max: float,
    ev_start: float,
    ev_end: float,
) -> tuple[float, float, float, float] | None:
    kpar = np.asarray(kpar, dtype=float)
    ev_arr = np.asarray(ev_arr, dtype=float)
    k0, k1 = sorted((float(k_min), float(k_max)))
    e0, e1 = sorted((float(ev_start), float(ev_end)))
    k0 = float(np.clip(k0, np.nanmin(kpar), np.nanmax(kpar)))
    k1 = float(np.clip(k1, np.nanmin(kpar), np.nanmax(kpar)))
    e0 = float(np.clip(e0, np.nanmin(ev_arr), np.nanmax(ev_arr)))
    e1 = float(np.clip(e1, np.nanmin(ev_arr), np.nanmax(ev_arr)))
    if k1 <= k0 or e1 <= e0:
        return None
    return k0, k1, e0, e1


def fit_roi_data(
    disp: np.ndarray,
    kpar,
    ev_arr,
    bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if bounds is None:
        return np.asarray(disp)
    k0, k1, e0, e1 = bounds
    kpar = np.asarray(kpar, dtype=float)
    ev_arr = np.asarray(ev_arr, dtype=float)
    mk = (kpar >= k0) & (kpar <= k1)
    me = (ev_arr >= e0) & (ev_arr <= e1)
    if not mk.any() or not me.any():
        return np.asarray(disp)
    return np.asarray(disp)[np.ix_(mk, me)]


def map_color_kwargs(
    disp: np.ndarray,
    *,
    mode: str,
    roi_ref: np.ndarray | None = None,
) -> tuple[str, dict]:
    ref = np.asarray(roi_ref) if roi_ref is not None else np.asarray(disp)
    if mode in ("Raw", "EDCnorm"):
        finite = ref[np.isfinite(ref)]
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        return "inferno", {"vmin": 0, "vmax": max(vmax, 1e-12)}
    pos = ref[np.isfinite(ref) & (ref > 0)]
    vmax = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
    return "hot_r", {"vmin": 0, "vmax": max(vmax, 1e-12)}


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


def draw_bandmap_axes(
    ax,
    *,
    kpar,
    ev,
    disp,
    cmap: str,
    color_kwargs: dict,
    gamma: float = 1.0,
    sel_ev: float,
    sel_k: float,
    int_win: float,
    title: str,
    title_size: int = 9,
    label_size: int = 10,
    tick_label_size: int | None = None,
    show_k_zero: bool = True,
    state: BandmapAxesState | None = None,
    reset_limits: bool = False,
):
    """Dessine le fond commun d'une carte BM sur un axe Matplotlib."""
    c_arr = np.asarray(disp).T
    signature = (tuple(c_arr.shape), _axis_signature(kpar), _axis_signature(ev))
    if state is None:
        ax.cla()
        state_out = None
    else:
        state_out = state
        needs_limit_reset = bool(reset_limits) or state_out.mesh is None
        for artist in list(state_out.base_artists):
            try:
                artist.remove()
            except Exception:
                pass
        state_out.base_artists = []
        if state_out.mesh is not None and state_out.signature != signature:
            try:
                state_out.mesh.remove()
            except Exception:
                pass
            state_out.mesh = None
            needs_limit_reset = True

    ax.set_facecolor("#1a1a1a")
    if getattr(ax, "figure", None) is not None:
        ax.figure.set_facecolor("#2b2b2b")

    kw = dict(color_kwargs)
    if float(gamma) != 1.0:
        from matplotlib.colors import PowerNorm

        kw = {"norm": PowerNorm(gamma=float(gamma), vmin=kw["vmin"], vmax=kw["vmax"])}
    if state_out is None or state_out.mesh is None:
        mesh = ax.pcolormesh(kpar, ev, c_arr, cmap=cmap, shading="auto", **kw)
        if state_out is not None:
            state_out.mesh = mesh
            state_out.signature = signature
    else:
        mesh = state_out.mesh
        mesh.set_array(c_arr.ravel())
        mesh.set_cmap(cmap)
        if "norm" in kw:
            mesh.set_norm(kw["norm"])
        else:
            mesh.set_norm(None)
            mesh.set_clim(kw.get("vmin"), kw.get("vmax"))
    if state_out is not None and needs_limit_reset:
        ax.autoscale(enable=True, axis="both", tight=True)

    base_artists = []
    base_artists.append(ax.axhline(0, color="cyan", lw=0.8, ls="--", alpha=0.6))
    if show_k_zero:
        base_artists.append(ax.axvline(0, color="w", lw=0.5, ls="--", alpha=0.4))
    base_artists.append(ax.axhspan(float(sel_ev) - float(int_win), float(sel_ev) + float(int_win),
                                   alpha=0.14, color="lime", zorder=2, lw=0))
    base_artists.append(ax.axhline(float(sel_ev), color="lime", lw=0.8, ls="--", zorder=3))
    base_artists.append(ax.axvline(float(sel_k), color="lime", lw=1.0, ls=":", zorder=3))
    if state_out is not None:
        state_out.base_artists = base_artists

    ax.set_xlabel("k// (π/a)", fontsize=label_size, color="w")
    ax.set_ylabel("E − EF (eV)", fontsize=label_size, color="w")
    ax.set_title(title, fontsize=title_size, color="w")
    if tick_label_size is None:
        ax.tick_params(colors="w")
    else:
        ax.tick_params(colors="w", labelsize=tick_label_size)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    return state_out


def mdc_curve(
    raw_data: dict,
    *,
    selected_ev: float,
    int_window: float,
    edc_norm_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    data = apply_edcnorm(raw_data["data"]) if edc_norm_enabled else np.asarray(raw_data["data"])
    ev_arr = np.asarray(raw_data["ev_arr"], dtype=float)
    mask_e = np.abs(ev_arr - float(selected_ev)) <= float(int_window)
    if not mask_e.any():
        mask_e[int(np.argmin(np.abs(ev_arr - float(selected_ev))))] = True
    mdc = np.nanmean(data[:, mask_e], axis=1).astype(float)
    return np.asarray(raw_data["kpar"], dtype=float), mdc


def edc_curve(
    raw_data: dict,
    *,
    selected_k: float,
    edc_norm_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    data = apply_edcnorm(raw_data["data"]) if edc_norm_enabled else np.asarray(raw_data["data"])
    kpar = np.asarray(raw_data["kpar"], dtype=float)
    idx = int(np.argmin(np.abs(kpar - float(selected_k))))
    return np.asarray(raw_data["ev_arr"], dtype=float), data[idx, :].astype(float)


def scroll_zoom_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    xdata: float,
    ydata: float,
    step: float,
    button: str | None = None,
    zoom_base: float = 0.94,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calcule les nouvelles limites d'axes pour un zoom molette Matplotlib."""
    step = float(step or 0.0)
    if step == 0.0:
        step = 1.0 if button == "up" else -1.0
    step_mag = min(abs(step), 3.0)
    scale = float(zoom_base) ** step_mag if step > 0 else (1.0 / (float(zoom_base) ** step_mag))

    def zoom_limits(lims, center):
        lo, hi = float(lims[0]), float(lims[1])
        direction = 1.0 if hi >= lo else -1.0
        lo_s, hi_s = (lo, hi) if direction > 0 else (hi, lo)
        span = hi_s - lo_s
        if not np.isfinite(span) or abs(span) <= 1e-15:
            return lims
        new_span = span * scale
        min_span = max(abs(span) * 1e-3, 1e-6)
        if new_span < min_span:
            return lims
        rel = (float(center) - lo_s) / span
        rel = float(np.clip(rel, 0.0, 1.0))
        new_lo = float(center) - rel * new_span
        new_hi = float(center) + (1.0 - rel) * new_span
        return (new_lo, new_hi) if direction > 0 else (new_hi, new_lo)

    return zoom_limits(xlim, xdata), zoom_limits(ylim, ydata)


def draw_fit_roi_overlay(ax, bounds: tuple[float, float, float, float] | None):
    if bounds is None:
        return None
    from matplotlib.patches import Rectangle

    k0, k1, e0, e1 = bounds
    rect = Rectangle(
        (k0, e0), k1 - k0, e1 - e0,
        fill=False, edgecolor="#7dd3fc", linewidth=1.1,
        linestyle="--", alpha=0.95, zorder=8,
    )
    ax.add_patch(rect)
    return rect


def draw_ef_label(ax, text: str, *, horizontal: bool = True):
    if horizontal:
        x0, x1 = ax.get_xlim()
        x = x0 + 0.012 * (x1 - x0)
        return ax.text(
            x, 0.0, text,
            color="cyan", fontsize=8, va="bottom", ha="left",
            bbox=dict(facecolor="#1a1a1a", edgecolor="none", alpha=0.65, pad=1.5),
            zorder=9,
        )
    y0, y1 = ax.get_ylim()
    y = y0 + 0.88 * (y1 - y0)
    return ax.text(
        0.0, y, text,
        color="cyan", fontsize=7, va="top", ha="left", rotation=90,
        bbox=dict(facecolor="#1a1a1a", edgecolor="none", alpha=0.65, pad=1.2),
        zorder=9,
    )


def apply_ef_correction_to_dict(d: dict, cfg: dict) -> tuple[dict, dict]:
    """Applique une correction EF par colonne (poly) au dict legacy bandmap.

    Renvoie (dict_corrigé, info) où info contient ef_smooth, ef_at_center, etc.
    Modifie une copie : ne touche pas l'objet d'origine.
    """
    if not cfg or cfg.get("mode") != "poly":
        return d, {}
    coefs = np.asarray(cfg.get("poly_coefs", []), dtype=float)
    if coefs.size == 0:
        return d, {}
    kpar = np.asarray(d["kpar"], dtype=float)
    ev = np.asarray(d["ev_arr"], dtype=float)
    data = np.asarray(d["data"], dtype=float)
    ef_smooth = np.polyval(coefs, kpar)
    try:
        from arpes.ui.widgets.plots import apply_ef_correction_per_column as _apply
    except Exception:
        return d, {}
    data_corr = _apply(data, kpar, ev, ef_smooth)
    out = dict(d)
    out["data"] = data_corr
    info = {"ef_smooth": ef_smooth, "ef_center": float(np.interp(0.0, kpar, ef_smooth))}
    return out, info
