"""ARPES plot-data preparation without PyQt.

First Update L slice: extract BM display-data calculations without moving the
Matplotlib canvases or mouse callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np


def apply_edcnorm(data: np.ndarray) -> np.ndarray:
    edc = np.nanmean(data, axis=0, keepdims=True)
    safe = np.where((np.abs(edc) > 1e-12) & np.isfinite(edc), edc, 1.0)
    return data / safe


@dataclass
class DerivParams:
    """User-tunable parameters for the SecDev / Curvature display modes.

    Smoothing widths are in *physical* units (eV and π/a), converted to pixels
    per map from the local axis spacing, so the same value behaves consistently
    across datasets with different sampling. ``c0_alpha`` is the curvature
    regularization fraction; ``ef_margin_eV`` is how far above EF the derivative
    is still shown (a hard cut exactly at EF creates a spurious derivative edge).
    """
    sigma_e_eV: float = 0.025
    sigma_k_inv_a: float = 0.04
    c0_alpha: float = 0.05
    ef_margin_eV: float = 0.05


def _sigma_px(coord, sigma_phys: float, fallback: float = 2.0) -> float:
    """Convert a physical smoothing width to pixels using the median axis step."""
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2 or sigma_phys <= 0:
        return 0.0 if sigma_phys <= 0 else fallback
    step = abs(float(np.median(np.diff(coord))))
    if step <= 0 or not np.isfinite(step):
        return fallback
    return float(sigma_phys / step)


def _smooth_masked(arr: np.ndarray, sigma) -> tuple[np.ndarray, np.ndarray]:
    """NaN-safe Gaussian smoothing via normalized convolution.

    The trapezoid-corrected maps carry a NaN border whose sample↔background
    cliff is the largest gradient in the frame; filling it with a median (the
    old behaviour) leaks the background inward and poisons every downstream
    derivative and the C0 estimate. Here NaNs are excluded from the convolution
    (``Gauss(I·M)/Gauss(M)``) so the border never contaminates the interior.
    Returns the smoothed field and the smoothed validity mask (0 outside data).
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(arr, dtype=float)
    mask = np.isfinite(a).astype(float)
    filled = np.where(mask > 0, a, 0.0)
    den = gaussian_filter(mask, sigma)
    smooth = gaussian_filter(filled, sigma) / (den + 1e-10)
    return smooth, den


def compute_secdev(data, kpar, ev_arr, *, sigma_k_px=2.0, sigma_e_px=2.0,
                   border_clip=3, **_ignored) -> np.ndarray:
    """Smoothed -d²I/dE² (peaks → positive). NaN-safe smoothing first."""
    ev_arr = np.asarray(ev_arr, dtype=float)
    smooth, mask_sm = _smooth_masked(data, [sigma_k_px, sigma_e_px])
    d2 = np.gradient(np.gradient(smooth, ev_arr, axis=1), ev_arr, axis=1)
    out = -d2
    out[mask_sm < 0.5] = np.nan  # re-propagate the data mask (drop the border)
    _clip_border(out, border_clip)
    return out


def _clip_border(arr: np.ndarray, border_clip: int) -> None:
    bc = max(0, int(border_clip))
    if bc > 0 and min(arr.shape) > 2 * bc:
        arr[:bc, :] = np.nan
        arr[-bc:, :] = np.nan
        arr[:, :bc] = np.nan
        arr[:, -bc:] = np.nan


def compute_curvature(
    data, kpar, ev_arr, *,
    sigma_k_px=2.0, sigma_e_px=2.0, c0_alpha=0.05, border_clip=3, **_ignored,
) -> np.ndarray:
    """Full 2D curvature, Zhang et al. RSI 82, 043712 (2011).

        C = -[ (C0 + I_E²)·I_kk − 2·I_k·I_E·I_kE + (C0 + I_k²)·I_EE ]
             / (C0 + I_k² + I_E²)^(3/2)

    Both cross term (factor 2) and the I_EE term are included — the previous
    implementation dropped them, so it only enhanced k-direction structure and
    looked flat on steep bands. C0 is the regularization scale: set from the
    95th percentile of the gradient magnitude on the *eroded interior* (never
    the max, which sits on the trapezoid border cliff and would wash the band).
    """
    from scipy.ndimage import binary_erosion

    kpar = np.asarray(kpar, dtype=float)
    ev_arr = np.asarray(ev_arr, dtype=float)
    smooth, mask_sm = _smooth_masked(data, [sigma_k_px, sigma_e_px])

    I_E = np.gradient(smooth, ev_arr, axis=1)
    I_k = np.gradient(smooth, kpar, axis=0)
    I_kk = np.gradient(I_k, kpar, axis=0)
    I_EE = np.gradient(I_E, ev_arr, axis=1)
    I_kE = np.gradient(I_k, ev_arr, axis=1)

    valid = mask_sm >= 0.99  # fully inside data (no border smoothing bleed)
    interior = binary_erosion(valid, iterations=5)
    if not interior.any():
        interior = valid
    if interior.any():
        gk = np.abs(I_k[interior])
        ge = np.abs(I_E[interior])
        C0 = float(c0_alpha) * (
            float(np.percentile(gk, 95)) ** 2 + float(np.percentile(ge, 95)) ** 2
        )
    else:
        C0 = float(c0_alpha) * (np.nanmax(np.abs(I_k)) ** 2 + np.nanmax(np.abs(I_E)) ** 2)
    C0 = max(C0, 1e-30)

    numer = (C0 + I_E**2) * I_kk - 2.0 * I_k * I_E * I_kE + (C0 + I_k**2) * I_EE
    denom = (C0 + I_k**2 + I_E**2) ** 1.5
    curv = -numer / (denom + 1e-30)

    curv[mask_sm < 0.5] = np.nan
    _clip_border(curv, border_clip)
    return curv


def _compute_deriv_below_ef(kind: str, data: np.ndarray, kpar, ev_arr,
                            dp: "DerivParams") -> np.ndarray:
    """Run a derivative display mode for E ≤ EF + margin, NaN above.

    A hard cut exactly at EF injects a spurious derivative edge; the configurable
    ``ef_margin_eV`` keeps EF and the thermally populated states just above it
    visible while still dropping the noisy high-energy tail.
    """
    ev_arr = np.asarray(ev_arr, dtype=float)
    data = np.asarray(data)
    out = np.full(data.shape, np.nan, dtype=float)
    below = ev_arr <= float(dp.ef_margin_eV)
    if below.sum() < 3:
        return out
    ev_b = ev_arr[below]
    sk_px = _sigma_px(kpar, dp.sigma_k_inv_a)
    se_px = _sigma_px(ev_b, dp.sigma_e_eV)
    if kind == "SecDev":
        out[:, below] = compute_secdev(
            data[:, below], kpar, ev_b, sigma_k_px=sk_px, sigma_e_px=se_px)
    elif kind == "Curvature":
        out[:, below] = compute_curvature(
            data[:, below], kpar, ev_b,
            sigma_k_px=sk_px, sigma_e_px=se_px, c0_alpha=dp.c0_alpha)
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
    distortion_info: dict = field(default_factory=dict)
    kpar: np.ndarray | None = None
    ev: np.ndarray | None = None


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
    distortion_correction: dict | None = None,
    deriv_params: "DerivParams | None" = None,
) -> BandmapDisplayResult:
    """Prépare la carte BM affichée pour le mode demandé.

    Pipeline : raw → distortion (trapèze + parabole) → EDC norm/secdev/curv
    → grid (FFT). La distorsion est appliquée *en premier* pour que toutes
    les corrections aval travaillent sur la BM redressée.
    """
    raw = np.asarray(raw_data["data"])
    kpar_disp = np.asarray(raw_data["kpar"])
    ev_disp = np.asarray(raw_data["ev_arr"])
    distortion_info: dict = {}
    if distortion_correction:
        from arpes.physics.distortion import apply_distortion

        try:
            raw, distortion_info = apply_distortion(
                raw, kpar_disp, ev_disp, distortion_correction,
            )
            if distortion_info.get("applied"):
                kpar_disp = np.asarray(distortion_info.get("kpar_axis", kpar_disp))
                ev_disp = np.asarray(distortion_info.get("ev_axis", ev_disp))
        except Exception as exc:
            distortion_info = {"applied": False, "error": str(exc)}
    dp = deriv_params or DerivParams()
    if mode == "Raw":
        disp = raw
    elif mode == "EDCnorm":
        disp = apply_edcnorm(raw)
    elif mode == "SecDev":
        norm = apply_edcnorm(raw) if edc_norm_enabled else raw
        disp = _compute_deriv_below_ef("SecDev", norm, kpar_disp, ev_disp, dp)
    elif mode == "Curvature":
        norm = apply_edcnorm(raw) if edc_norm_enabled else raw
        disp = _compute_deriv_below_ef("Curvature", norm, kpar_disp, ev_disp, dp)
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

    return BandmapDisplayResult(
        data=disp, grid_info=grid_info, distortion_info=distortion_info,
        kpar=kpar_disp, ev=ev_disp,
    )


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


def make_bandmap_format_coord(kpar, ev, c_arr):
    """Toolbar hover readout: exact (k, E) plus the intensity of the nearest
    data pixel. Reads the true array (not the rendered image) so the value
    shown matches the science; NaN shows as an em dash, never 0."""
    kp = np.asarray(kpar, dtype=float).ravel()
    e = np.asarray(ev, dtype=float).ravel()
    arr = np.asarray(c_arr)

    def _fmt(x, y):
        i = int(np.argmin(np.abs(kp - x)))
        j = int(np.argmin(np.abs(e - y)))
        val = arr[j, i] if (j < arr.shape[0] and i < arr.shape[1]) else np.nan
        ival = f"{val:.4g}" if np.isfinite(val) else "—"
        return f"k = {x:.3f}   E = {y:.4f} eV   I = {ival}"

    return _fmt


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
    axis_labels: tuple[str, str] | None = None,
    axis_note: str | None = None,
):
    """Dessine le fond commun d'une carte BM sur un axe Matplotlib."""
    c_arr = np.asarray(disp).T
    signature = (tuple(c_arr.shape), _axis_signature(kpar), _axis_signature(ev))
    if state is None:
        ax.cla()
        state_out = None
        needs_limit_reset = True
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
    if needs_limit_reset:
        # Reset bounds to the current file's data extent, then disable autoscale:
        # otherwise (a) a reused mesh keeps old bounds, and (b) overlays added
        # later (EF axhline, DFT theory, Gamma halo, kf...) can expand axes.
        # This ensures loading a new file restores the correct frame without
        # touching an active zoom when only parameters changed.
        kp = np.asarray(kpar, dtype=float)
        ee = np.asarray(ev, dtype=float)
        try:
            if np.isfinite(kp).any():
                ax.set_xlim(float(np.nanmin(kp)), float(np.nanmax(kp)))
            if np.isfinite(ee).any():
                ax.set_ylim(float(np.nanmin(ee)), float(np.nanmax(ee)))
        except (ValueError, TypeError):
            pass
        ax.set_autoscale_on(False)

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

    ax.format_coord = make_bandmap_format_coord(kpar, ev, c_arr)
    xlabel, ylabel = axis_labels or ("k// (π/a)", "E − EF (eV)")
    ax.set_xlabel(xlabel, fontsize=label_size, color="w")
    ax.set_ylabel(ylabel, fontsize=label_size, color="w")
    if axis_note:
        note = ax.text(
            0.99, 0.01, axis_note,
            transform=ax.transAxes, ha="right", va="bottom",
            color="#fbbf24", fontsize=max(6, label_size - 3), zorder=30,
        )
        base_artists.append(note)
        if state_out is not None:
            state_out.base_artists = base_artists
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
    except Exception as exc:
        # Import KO → données retournées NON corrigées EF : signaler
        # explicitement (utilisateur croyait avoir appliqué une correction).
        import warnings
        warnings.warn(
            f"apply_ef_correction_to_dict: import unavailable ({exc}); "
            f"EF correction SKIPPED - returning raw data.",
            RuntimeWarning, stacklevel=2,
        )
        return d, {}
    data_corr = _apply(data, kpar, ev, ef_smooth)
    out = dict(d)
    out["data"] = data_corr
    info = {"ef_smooth": ef_smooth, "ef_center": float(np.interp(0.0, kpar, ef_smooth))}
    return out, info


# Re-exports kept for back-compat (waterfall logic moved to waterfall_compute.py
# in the LOC-ceiling split, but existing importers still reference these names
# via arpes.physics.plot_compute).
from arpes.physics.waterfall_compute import (  # noqa: E402
    WaterfallData,
    prepare_waterfall_data,
    draw_waterfall_axes,
)
