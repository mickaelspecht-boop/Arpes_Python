#!/usr/bin/env python3
"""
Batch review des Band Maps (BM) ARPES pour BaNi2As2.

Objectif
--------
- lire un logbook expérimental (.csv, format Solaris/DA30)
- sélectionner les scans de type BM (par défaut: fixed cut / Normal)
- charger chaque fichier .pxt ou .ibw
- auto-calibrer EF de façon sample-based (sans gold)
- produire pour chaque BM un panneau de diagnostic:
    1) BM lissée
    2) Waterfall MDC proche EF (~40 MDC)
    3) Waterfall MDC profond (-2 → -1 eV)
    4) -d2I/dE2
    5) Courbure 2D
- écrire un CSV récapitulatif avec métadonnées réelles (hν, T, polarisation, direction)

Dépendances
-----------
numpy, pandas, matplotlib, scipy, xarray, erlab
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.optimize import curve_fit

try:
    import erlab.io
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Impossible d'importer erlab.io. Active d'abord ton environnement avec erlab.\n"
        f"Erreur: {exc}"
    )


# -----------------------------------------------------------------------------
# Paramètres par défaut
# -----------------------------------------------------------------------------
DEFAULTS = {
    "work_func": 4.0310,         # eV — BaNi2As2 URANOS/Solaris (session oct. 2025)
    "a_lattice": 3.960,          # Å, BaNi2As2
    "temperature_K": 28.0,
    "ef_search": (-0.50, 0.20),  # large pour couvrir les sessions avec φ imprécis
    "ef_half_width": 0.10,
    "ef_n_windows": 9,
    "ef_sigma_init": 0.025,
    "ef_fix_kBT": True,
    "bm_ev_range": None,          # None = toute la fenêtre énergie disponible
    # Waterfall proche EF (bandes basses énergie, delta fin)
    "waterfall_nef_start": None,   # None = auto (-0.35 eV)
    "waterfall_nef_end":   None,   # None = auto (+0.05 eV)
    "waterfall_nef_delta": None,   # None = auto (0.010 eV → ~40 MDC)
    # Waterfall profond (bandes de valence, -2 → -1 eV environ)
    "waterfall_deep_start":        None,  # None = auto (-1.0 eV)
    "waterfall_deep_end":          None,  # None = auto (-0.35 eV)
    "waterfall_deep_delta":        None,  # None = auto (0.015 eV → ~65 MDC)
    "waterfall_deep_smooth_sigma": None,  # None = auto (3.5 — plus lissé, MDC peu dispersives)
    "waterfall_deep_offset_scale": None,  # None = auto (8.0 — serré pour loger toutes les courbes)
    "waterfall_smooth_sigma": 1.8,
    "waterfall_offset_scale": 14.0,
    "waterfall_fill_alpha": 0.10,
    "secdev_sigma_k": 2.0,
    "secdev_sigma_e": 2.0,
    "secdev_c0_fraction": 0.20,
    "secdev_border_clip": 5,
    "k_crop": None,              # ex: [-1.0, 1.0]
    "only_spectrum_name": "fixed cut",
    "only_mode": "Normal",
    "dpi": 120,
}

_KB = 8.617333e-5


# -----------------------------------------------------------------------------
# Utilitaires logbook
# -----------------------------------------------------------------------------
def clean_filename(value: object) -> str:
    """Nettoie le nom de fichier venant du logbook Solaris."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = s.replace(" ", "")
    s = s.replace(",", "")
    # Ex: "BaNi2As2_0001.pxt.ibw" -> on garde les deux possibilités plus bas.
    return s


def candidate_paths(data_dir: Path, file_cell: str) -> List[Path]:
    """
    Génère des chemins candidats robustes à partir de la colonne File du logbook.

    Cas Solaris observé:
      "BaNi2As2_0001.pxt .ibw"
    On veut alors tester:
      BaNi2As2_0001.pxt
      BaNi2As2_0001.ibw
      BaNi2As2_0001.zip
    et aussi faire un fallback par stem si besoin.
    """
    raw0 = "" if pd.isna(file_cell) else str(file_cell).strip()
    if not raw0:
        return []

    # Normalisation douce
    raw = raw0.replace(",", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    candidates: List[Path] = []
    seen_names = set()

    def add_name(name: str) -> None:
        name = name.strip()
        if not name:
            return
        p = data_dir / name
        key = str(p)
        if key not in seen_names:
            seen_names.add(key)
            candidates.append(p)

    # 1) extraire explicitement les noms du type xxx_0001.ext
    #    Permet de traiter "BaNi2As2_0001.pxt .ibw"
    explicit = re.findall(r"[^\s]+?\.(?:pxt|ibw|zip)", raw, flags=re.IGNORECASE)
    for name in explicit:
        add_name(name)

    # 2) si on n'a récupéré qu'un nom composite du style .pxt.ibw, reconstruire
    compact = raw.replace(" ", "")
    stem = re.sub(r"(\.(?:pxt|ibw|zip))+$", "", compact, flags=re.IGNORECASE)
    if stem:
        for ext in (".pxt", ".ibw", ".zip"):
            add_name(stem + ext)

    # 3) fallback supplémentaire: chercher par stem dans le dossier
    #    utile si le logbook et les fichiers ne matchent pas exactement
    if stem:
        for ext in ("*.pxt", "*.ibw", "*.zip"):
            for p in sorted(data_dir.glob(f"{stem}{ext[1:]}")):
                add_name(p.name)
        # recherche plus large insensible aux doublons d'extension
        for p in sorted(data_dir.glob(f"{stem}*")):
            if p.suffix.lower() in {".pxt", ".ibw", ".zip"}:
                add_name(p.name)

    return candidates


# -----------------------------------------------------------------------------
# Chargement BM DA30 -> k-space
# -----------------------------------------------------------------------------
def load_da30_bm(filepath: Path, work_func: float, a_lattice: float,
                  hv_logbook: float = np.nan) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Charge un fixed cut DA30 et convertit en coordonnées kx / E-EF approximatives.
    Retourne: da_k, kpar, ev_arr, data (nk, ne), hv_file
    hv_logbook : valeur du logbook pour vérification de cohérence.
    """
    import warnings

    da = erlab.io.load(str(filepath))
    hv = float(da.attrs.get("hv", np.nan))
    if not np.isfinite(hv):
        raise ValueError(f"hv absent dans {filepath.name}")

    # Vérification cohérence logbook vs fichier
    if np.isfinite(hv_logbook) and abs(hv - hv_logbook) > 2.0:
        print(f"  ⚠ hv logbook ({hv_logbook:.1f} eV) ≠ hv fichier ({hv:.1f} eV) pour {filepath.name}")

    ef_kin = hv - work_func
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da_be = da.assign_coords(eV=da.eV - ef_kin, hv=hv, xi=0.0)
    da_be.attrs["configuration"] = 1
    da_be.kspace.work_function = work_func

    da_k = da_be.kspace.convert()
    # erlab retourne kx en 1/Å -> conversion en pi/a
    da_k = da_k.assign_coords(kx=da_k.kx * a_lattice / np.pi)
    da_k.attrs["kx_unit"] = "pi/a"

    kpar = np.asarray(da_k.kx.values, dtype=float)
    ev_arr = np.asarray(da_k.eV.values, dtype=float)
    data = np.asarray(da_k.values, dtype=float)
    if data.shape != (len(kpar), len(ev_arr)):
        data = np.squeeze(data)
        if data.shape != (len(kpar), len(ev_arr)):
            raise ValueError(
                f"Shape inattendue pour {filepath.name}: {data.shape}, attendu ({len(kpar)}, {len(ev_arr)})"
            )
    return da_k, kpar, ev_arr, data, hv


# -----------------------------------------------------------------------------
# Auto-calibration EF sample-based
# -----------------------------------------------------------------------------
def average_edc(data2d: np.ndarray) -> np.ndarray:
    I = np.nanmean(data2d, axis=0).astype(float)
    if np.isfinite(I).any():
        med = float(np.nanmedian(I[np.isfinite(I)]))
    else:
        med = 0.0
    return np.nan_to_num(I, nan=med, posinf=med, neginf=med)


def fit_fermi_edge(
    ev_arr: np.ndarray,
    I_arr: np.ndarray,
    temperature_K: float = 28.0,
    fit_range: Tuple[float, float] = (-0.08, 0.08),
    sigma_resolution_init: float = 0.020,
    fix_kBT: bool = True,
) -> Dict[str, float]:
    """Fit FD convoluée simple. Retourne EF et score de résidu."""
    mask = (ev_arr >= fit_range[0]) & (ev_arr <= fit_range[1])
    if mask.sum() < 12:
        raise ValueError("Fenêtre de fit EF trop petite")

    e = ev_arr[mask].astype(float)
    I = I_arr[mask].astype(float)
    I = np.nan_to_num(I, nan=np.nanmedian(I[np.isfinite(I)]) if np.isfinite(I).any() else 0.0)
    I = I / (np.max(I) or 1.0)
    dE = abs(float(e[1] - e[0]))
    kBT_fixed = _KB * temperature_K

    def _fd_conv(e_arr, EF, A, slope, bg, sigma):
        raw = np.exp(np.clip((e_arr - EF) / kBT_fixed, -500, 500))
        fd = 1.0 / (raw + 1.0)
        pix = sigma / max(dE, 1e-9)
        if pix > 0.3:
            fd = ndi.gaussian_filter1d(fd, sigma=pix)
        return A * fd + slope * (e_arr - EF) + bg

    grad = np.gradient(I, e)
    EF_guess = float(e[int(np.argmax(np.abs(grad)))])
    p0 = [EF_guess, 0.8, -0.3, 0.05, sigma_resolution_init]
    lo = [EF_guess - 0.15, 0.0, -5.0, -1.0, 0.003]
    hi = [EF_guess + 0.15, 3.0, 5.0, 2.0, 0.10]

    popt, pcov = curve_fit(_fd_conv, e, I, p0=p0, bounds=(lo, hi), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    I_fit = _fd_conv(e, *popt)
    residual = float(np.sqrt(np.mean((I - I_fit) ** 2)))
    return {
        "EF": float(popt[0]),
        "EF_err": float(perr[0]) if np.isfinite(perr[0]) else np.nan,
        "sigma_res": float(popt[4]),
        "residual": residual,
    }


def auto_fit_ef_sample(
    ev_axis: np.ndarray,
    I_edc: np.ndarray,
    temperature_K: float,
    search: Tuple[float, float],
    half_width: float,
    n_windows: int,
    sigma_init: float,
    fix_kBT: bool,
) -> Dict[str, float]:
    """Teste plusieurs fenêtres autour du gradient max et garde le meilleur fit FD."""
    mask_search = (ev_axis >= search[0]) & (ev_axis <= search[1])
    if mask_search.sum() < 20:
        raise ValueError(f"Fenêtre de recherche EF trop petite: {search}")

    e_s = ev_axis[mask_search]
    I_s = I_edc[mask_search]
    grad = -np.gradient(I_s, e_s)
    i0 = int(np.nanargmax(np.abs(grad)))
    e0 = float(e_s[i0])

    centers = np.linspace(
        max(search[0] + half_width, e0 - 0.06),
        min(search[1] - half_width, e0 + 0.06),
        max(n_windows, 3),
    )

    fits = []
    for center in centers:
        fit_range = (center - half_width, center + half_width)
        try:
            r = fit_fermi_edge(
                ev_axis,
                I_edc,
                temperature_K=temperature_K,
                fit_range=fit_range,
                sigma_resolution_init=sigma_init,
                fix_kBT=fix_kBT,
            )
            r["fit_range"] = fit_range
            fits.append(r)
        except Exception:
            continue

    if not fits:
        raise RuntimeError("Auto-fit EF impossible")

    # Score pénalisé comme dans V6 : résidu + pénalité si FWHM aberrante ou EF très loin du centre
    for r in fits:
        penalty = 0.0
        sigma = r.get("sigma_res", np.nan)
        if not np.isfinite(sigma) or sigma <= 0:
            penalty += 1.0
        if np.isfinite(sigma) and sigma > 0.18:
            penalty += 0.20
        # Si EF s'est éloigné de plus de 80 meV du centre de la fenêtre testée,
        # c'est probablement un faux minimum.
        center_dist = abs(r["EF"] - (r["fit_range"][0] + r["fit_range"][1]) / 2)
        if center_dist > 0.08:
            penalty += 0.10
        r["_score"] = r["residual"] + penalty

    fits.sort(key=lambda x: x["_score"])
    return fits[0]



def adaptive_ef_fit(ev_arr: np.ndarray, data: np.ndarray, row: pd.Series, cfg: Dict[str, object]) -> Dict[str, float]:
    """EF fit robuste : adapte la fenêtre si possible, sinon saute proprement."""
    ev_min = float(np.nanmin(ev_arr))
    ev_max = float(np.nanmax(ev_arr))
    edc = average_edc(data)

    # Fenêtre demandée par config, tronquée à l'axe réel
    s0, s1 = tuple(cfg['ef_search'])
    search = (max(ev_min, float(s0)), min(ev_max, float(s1)))

    def _enough_points(rng):
        mask = (ev_arr >= rng[0]) & (ev_arr <= rng[1])
        return int(mask.sum()) >= 20

    # Si la fenêtre standard ne marche pas, tenter le bord supérieur du scan
    if not _enough_points(search):
        width = min(0.25, max(0.08, (ev_max - ev_min) * 0.15))
        search = (max(ev_min, ev_max - width), ev_max)

    if not _enough_points(search):
        return {
            'EF': 0.0,
            'EF_err': np.nan,
            'sigma_res': np.nan,
            'residual': np.nan,
            'status': 'skipped_no_ef_window',
        }

    half_width = min(float(cfg['ef_half_width']), max((search[1]-search[0]) * 0.45, 0.02))
    try:
        r = auto_fit_ef_sample(
            ev_arr,
            edc,
            temperature_K=float(row.get('Sample temperature [K]', cfg['temperature_K'])),
            search=search,
            half_width=half_width,
            n_windows=int(cfg['ef_n_windows']),
            sigma_init=float(cfg['ef_sigma_init']),
            fix_kBT=bool(cfg['ef_fix_kBT']),
        )
        r['status'] = 'fit_ok'
        return r
    except Exception:
        return {
            'EF': 0.0,
            'EF_err': np.nan,
            'sigma_res': np.nan,
            'residual': np.nan,
            'status': 'fit_failed_fallback_zero',
        }

# -----------------------------------------------------------------------------
# Prétraitement et diagnostics BM
# -----------------------------------------------------------------------------
def crop_ke(
    data: np.ndarray,
    kpar: np.ndarray,
    ev_arr: np.ndarray,
    k_crop: Optional[Tuple[float, float]],
    ev_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_mask = np.ones_like(kpar, dtype=bool)
    if k_crop is not None:
        k_mask = (kpar >= k_crop[0]) & (kpar <= k_crop[1])
    e_mask = (ev_arr >= ev_range[0]) & (ev_arr <= ev_range[1])
    return data[k_mask][:, e_mask], kpar[k_mask], ev_arr[e_mask]


def robust_normalize(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    p1, p99 = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        return np.nan_to_num(x)
    y = (x - p1) / (p99 - p1)
    return np.clip(np.nan_to_num(y), 0.0, 1.0)


def secdev_curvature(
    data_cut: np.ndarray,
    kpar: np.ndarray,
    ev_arr: np.ndarray,
    sigma_k: float = 2.0,
    sigma_e: float = 2.0,
    c0_fraction: float = 0.20,
    border_clip: int = 5,
) -> Dict[str, np.ndarray]:
    d = np.asarray(data_cut, dtype=float)
    if np.isnan(d).any():
        d = d.copy()
        d[np.isnan(d)] = np.nanmedian(d)
    I = ndi.gaussian_filter(d, sigma=[sigma_k, sigma_e])

    dI_dE = np.gradient(I, ev_arr, axis=1)
    d2I_dE2 = np.gradient(dI_dE, ev_arr, axis=1)
    dI_dk = np.gradient(I, kpar, axis=0)
    d2I_dk2 = np.gradient(dI_dk, kpar, axis=0)
    d2I_dkdE = np.gradient(dI_dk, ev_arr, axis=1)

    secdev = -d2I_dE2
    bc = max(0, int(border_clip))
    interior = (slice(bc, -bc or None), slice(bc, -bc or None))
    C0 = c0_fraction * (
        np.abs(dI_dk[interior]).max() ** 2 + np.abs(dI_dE[interior]).max() ** 2
    )
    numer = (C0 + dI_dE**2) * d2I_dk2 - dI_dk * dI_dE * d2I_dkdE
    denom = (C0 + dI_dk**2 + dI_dE**2) ** 1.5
    curv2d = -numer / (denom + 1e-30)

    if bc > 0:
        for arr in (I, secdev, curv2d):
            arr[:bc, :] = np.nan
            arr[-bc:, :] = np.nan
            arr[:, :bc] = np.nan
            arr[:, -bc:] = np.nan

    return {"smoothed": I, "secdev": secdev, "curvature": curv2d}


def waterfall_plot(
    data_cut: np.ndarray,
    kpar: np.ndarray,
    ev_arr: np.ndarray,
    ev_start: float,
    ev_end: float,
    delta_ev: float,
    smooth_sigma: float,
    offset_scale: float,
    normalize: str = "each",
    fill_alpha: float = 0.12,
    title: str = "BM — Waterfall MDC",
    ax: Optional[plt.Axes] = None,
    fill: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    energies = []
    ev = ev_start
    while ev <= ev_end + 1e-12:
        energies.append(ev)
        ev += delta_ev
    if not energies:
        raise ValueError("Aucune énergie sélectionnée pour le waterfall")

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
    else:
        fig = ax.figure

    cmap = plt.get_cmap("plasma")
    global_max = float(np.nanmax(np.abs(data_cut))) or 1.0
    vertical_step = delta_ev * offset_scale

    for i, ev_i in enumerate(energies):
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        mdc = np.asarray(data_cut[:, ie], dtype=float)
        if smooth_sigma > 0:
            mdc = ndi.gaussian_filter1d(mdc, sigma=smooth_sigma)

        if normalize == "each":
            mmin = float(np.nanmin(mdc))
            mmax = float(np.nanmax(mdc))
            r = mmax - mmin
            if np.isfinite(r) and r > 0:
                mdc = (mdc - mmin) / r
            else:
                mdc = np.zeros_like(mdc)
        elif normalize == "global":
            mdc = mdc / global_max

        offset = i * vertical_step
        y = mdc + offset
        c = cmap(i / max(1, len(energies) - 1))
        ax.plot(kpar, y, color=c, lw=1.25, zorder=10 + i)
        if fill:
            ax.fill_between(kpar, offset, y, color=c, alpha=fill_alpha, zorder=1 + i)
        ax.axhline(offset, color=c, lw=0.35, ls="--", alpha=0.35, zorder=0)

    tick_step = max(1, len(energies) // 10)
    tick_idx = list(range(0, len(energies), tick_step))
    if tick_idx[-1] != len(energies) - 1:
        tick_idx.append(len(energies) - 1)
    ax.set_yticks([i * vertical_step for i in tick_idx])
    ax.set_yticklabels([f"{energies[i]:.3f}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("k// (pi/a)")
    ax.set_ylabel("E - EF (eV)")
    ax.set_xlim(float(kpar[0]), float(kpar[-1]))
    ax.set_title(title)
    ax.grid(False)
    if created_fig:
        fig.tight_layout()
    return fig, ax




def choose_energy_windows(ev_arr: np.ndarray, cfg: Dict[str, object]) -> Dict[str, float]:
    """
    Choisit deux fenêtres waterfall indépendantes :
      - nef  : proche de EF (delta fin, ~40 MDC) pour voir les bandes basses EB
      - deep : bandes profondes (-2 → -1 eV typiquement)
    Chaque fenêtre est clippée aux bornes réelles du scan.
    """
    ev_min = float(np.nanmin(ev_arr))
    ev_max = float(np.nanmax(ev_arr))

    bm_ev_range = cfg.get("bm_ev_range", None)
    if bm_ev_range is None:
        display_lo = ev_min
        display_hi = min(ev_max, 0.10)
    else:
        display_lo = max(ev_min, float(bm_ev_range[0]))
        display_hi = min(ev_max, float(bm_ev_range[1]))

    # ── Fenêtre d'analyse (BM / SecDev / Curvature) ───────────────────────────
    # Clippée à -1 eV : on n'a pas besoin du bulk pour l'analyse des bandes proches EF
    analysis_lo = max(ev_min, -1.0)
    analysis_hi = min(ev_max, 0.10)

    # ── Fenêtre proche EF ─────────────────────────────────────────────────────
    nef_start_cfg = cfg.get("waterfall_nef_start", None)
    nef_end_cfg   = cfg.get("waterfall_nef_end",   None)
    nef_delta_cfg = cfg.get("waterfall_nef_delta", None)

    nef_start = max(ev_min, float(nef_start_cfg) if nef_start_cfg is not None else -0.35)
    nef_end   = min(ev_max, float(nef_end_cfg)   if nef_end_cfg   is not None else  0.05)
    nef_delta = float(nef_delta_cfg) if nef_delta_cfg is not None else 0.010

    # ── Fenêtre profonde (-2 → -1 eV) ─────────────────────────────────────────
    deep_start_cfg = cfg.get("waterfall_deep_start", None)
    deep_end_cfg   = cfg.get("waterfall_deep_end",   None)
    deep_delta_cfg = cfg.get("waterfall_deep_delta", None)

    deep_start = max(ev_min, float(deep_start_cfg) if deep_start_cfg is not None else -1.0)
    deep_end   = min(ev_max, float(deep_end_cfg)   if deep_end_cfg   is not None else -0.35)
    deep_delta = float(deep_delta_cfg) if deep_delta_cfg is not None else 0.015

    # Smooth/offset dédiés au waterfall profond
    deep_smooth_cfg = cfg.get("waterfall_deep_smooth_sigma", None)
    deep_offset_cfg = cfg.get("waterfall_deep_offset_scale", None)
    deep_smooth = float(deep_smooth_cfg) if deep_smooth_cfg is not None else 3.5
    deep_offset = float(deep_offset_cfg) if deep_offset_cfg is not None else 8.0

    return {
        "display_lo":        display_lo,
        "display_hi":        display_hi,
        "analysis_lo":       analysis_lo,
        "analysis_hi":       analysis_hi,
        "nef_start":         nef_start,
        "nef_end":           nef_end,
        "nef_delta":         nef_delta,
        "deep_start":        deep_start,
        "deep_end":          deep_end,
        "deep_delta":        deep_delta,
        "deep_smooth_sigma": deep_smooth,
        "deep_offset_scale": deep_offset,
    }



def _REMOVED_quality_metrics(data_cut: np.ndarray, kpar: np.ndarray, ev_arr: np.ndarray, derived: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    [SUPPRIMÉ — fonction conservée comme stub pour éviter les erreurs de merge]

    Composantes (pondération totale = 100):
      1. SNR near EF    (25 pts) : rapport signal/bruit dans la fenêtre juste sous EF.
      2. SecDev strength (20 pts) : intensité de -d²I/dE² → visibilité des bandes.
      3. Dynamic range   (15 pts) : étendue du signal lissé (scan peu bruité).
      4. FD sharpness    (15 pts) : pic de dérivée du profil EDC étroit → bonne résolution.
      5. BM symmetry     (15 pts) : symétrie I(k,E) ≈ I(-k,E) → scan centré sur Γ.
      6. NaN penalty     (10 pts) : pénalité si beaucoup de pixels manquants.

    Toutes les métriques sont auto-normalisées sur les données du scan lui-même
    pour éviter que les valeurs absolues d'intensité (qui varient d'une session
    à l'autre) biaisant le classement.
    """
    ev_min = float(np.nanmin(ev_arr))
    ev_max = float(np.nanmax(ev_arr))
    smooth = derived["smoothed"]
    secdev = derived["secdev"]

    # ── 1. SNR near EF ─────────────────────────────────────────────────────────
    # Fenêtre "top" = juste sous EF ; fenêtre "noise" = toute la valence
    top_width = min(0.25, (ev_max - ev_min) * 0.15)
    e_mask_top = (ev_arr >= max(ev_min, ev_max - top_width)) & (ev_arr <= ev_max)
    if e_mask_top.sum() < 5:
        e_mask_top = ev_arr >= np.nanpercentile(ev_arr, 85)

    top_signal = float(np.nanpercentile(smooth[:, e_mask_top], 90))
    # Bruit : écart-type spatial sur chaque tranche d'énergie, médianné
    noise_per_e = np.nanstd(smooth, axis=0)           # std spatiale pour chaque E
    noise_level = float(np.nanmedian(noise_per_e))
    snr = top_signal / max(noise_level, 1e-9)
    # Un SNR de 10 est acceptable, 30+ est excellent → normaliser sur [5, 40]
    snr_score = float(np.clip((snr - 5.0) / 35.0, 0.0, 1.0))

    # ── 2. SecDev strength ──────────────────────────────────────────────────────
    # Ratio 98th percentile / 50th percentile de |secdev| :
    # grand ratio = pics très marqués par rapport au fond = bandes visibles.
    secdev_abs = np.abs(secdev)
    finite_sec = secdev_abs[np.isfinite(secdev_abs)]
    if finite_sec.size > 10:
        p98 = float(np.nanpercentile(finite_sec, 98))
        p50 = float(np.nanpercentile(finite_sec, 50))
        sec_ratio = p98 / max(p50, 1e-9)
        # Ratio ~2 = mediocre, ~8+ = excellent
        sec_score = float(np.clip((sec_ratio - 2.0) / 10.0, 0.0, 1.0))
    else:
        sec_ratio = 0.0
        sec_score = 0.0

    # ── 3. Dynamic range ────────────────────────────────────────────────────────
    finite_sm = smooth[np.isfinite(smooth)]
    if finite_sm.size > 10:
        p99 = float(np.nanpercentile(finite_sm, 99))
        p01 = float(np.nanpercentile(finite_sm, 1))
        p95 = float(np.nanpercentile(finite_sm, 95))
        dynamic_range = p99 - p01
        dyn_score = float(np.clip((dynamic_range / max(p95, 1e-9) - 0.5) / 1.5, 0.0, 1.0))
    else:
        dynamic_range = 0.0
        dyn_score = 0.0

    # ── 4. FD sharpness (netteté du bord de Fermi) ─────────────────────────────
    # On intègre le signal sur k, puis on prend le gradient : le pic doit être fin.
    edc = np.nanmean(data_cut, axis=0).astype(float)
    edc_finite = edc[np.isfinite(edc)]
    if edc_finite.size > 10:
        edc_norm = edc / max(np.nanmax(edc_finite), 1e-9)
        grad_edc = np.abs(np.gradient(edc_norm, ev_arr))
        grad_max = float(np.nanmax(grad_edc))
        # FWHM approx : fraction des points dont le gradient > 50% du max
        # Fenêtre autour du max uniquement (évite les faux pics dans la valence)
        i_max = int(np.nanargmax(grad_edc))
        half_win = max(5, len(ev_arr) // 8)
        i_lo = max(0, i_max - half_win)
        i_hi = min(len(ev_arr), i_max + half_win)
        grad_win = grad_edc[i_lo:i_hi]
        ev_win = ev_arr[i_lo:i_hi]
        de = abs(float(ev_win[1] - ev_win[0])) if len(ev_win) > 1 else 0.01
        above_half = float(np.sum(grad_win >= 0.5 * np.nanmax(grad_win)))
        fwhm_ev = above_half * de
        # FWHM < 0.04 eV = très net, > 0.15 eV = flou
        fd_score = float(np.clip(1.0 - (fwhm_ev - 0.04) / 0.12, 0.0, 1.0))
        ef_sharpness = grad_max
    else:
        fd_score = 0.0
        ef_sharpness = 0.0
        fwhm_ev = np.nan

    # ── 5. BM symmetry (centrage sur Γ, kpar centré sur 0) ────────────────────
    # Compare I(k>0, E) et I(k<0, E) par interpolation.
    k_pos_mask = kpar >= 0.0
    k_neg_mask = kpar < 0.0
    if k_pos_mask.sum() >= 3 and k_neg_mask.sum() >= 3:
        k_pos = kpar[k_pos_mask]
        k_neg = kpar[k_neg_mask]
        k_shared = np.linspace(0, min(float(k_pos.max()), float(-k_neg.min())), 40)
        if k_shared.size > 3:
            I_pos = np.nanmean(
                RegularGridInterpolator(
                    (k_pos, ev_arr), data_cut[k_pos_mask, :],
                    method='linear', bounds_error=False, fill_value=np.nan
                )(np.c_[k_shared, np.zeros(len(k_shared))]),
                axis=0 if False else 0,   # dummy keepdim trick
            )
            # Interpolation simplifiée : moyenne sur E après interpolation sur k
            def _interp_strip(k_arr, data_strip, k_query):
                """Interpolation 1D sur l'axe k, moyennée sur E."""
                out = np.full(len(k_query), np.nan)
                for j, kq in enumerate(k_query):
                    i = np.searchsorted(k_arr, kq)
                    if 0 < i < len(k_arr):
                        w = (kq - k_arr[i-1]) / (k_arr[i] - k_arr[i-1] + 1e-30)
                        out[j] = float(np.nanmean((1-w)*data_strip[i-1] + w*data_strip[i]))
                return out

            i_pos = _interp_strip(k_pos, data_cut[k_pos_mask, :], k_shared)
            i_neg = _interp_strip(-k_neg[::-1], data_cut[k_neg_mask[::-1], :], k_shared)

            valid = np.isfinite(i_pos) & np.isfinite(i_neg)
            if valid.sum() > 3:
                sym_err = float(np.nanmean(np.abs(i_pos[valid] - i_neg[valid])) /
                                max(np.nanmean(np.abs(i_pos[valid]) + np.abs(i_neg[valid])) / 2, 1e-9))
                # sym_err = 0 → parfaitement symétrique, ~0.5+ → très asymétrique
                sym_score = float(np.clip(1.0 - sym_err * 2.5, 0.0, 1.0))
                symmetry_error = sym_err
            else:
                sym_score = 0.5
                symmetry_error = np.nan
        else:
            sym_score = 0.5
            symmetry_error = np.nan
    else:
        sym_score = 0.5
        symmetry_error = np.nan

    # ── 6. NaN penalty ──────────────────────────────────────────────────────────
    nan_frac = float(np.isnan(data_cut).mean())
    # nan_frac < 0.02 → plein score, nan_frac > 0.20 → score 0
    nan_score = float(np.clip(1.0 - (nan_frac - 0.02) / 0.18, 0.0, 1.0))

    # ── Calcul du score total ───────────────────────────────────────────────────
    score = 0.0
    score += 25.0 * snr_score
    score += 20.0 * sec_score
    score += 15.0 * dyn_score
    score += 15.0 * fd_score
    score += 15.0 * sym_score
    score += 10.0 * nan_score

    return {
        "score": float(np.clip(score, 0, 100)),
        # Détail des sous-scores
        "score_snr_nef": round(25.0 * snr_score, 2),
        "score_secdev":  round(20.0 * sec_score, 2),
        "score_dynrange": round(15.0 * dyn_score, 2),
        "score_fd_sharp": round(15.0 * fd_score, 2),
        "score_symmetry": round(15.0 * sym_score, 2),
        "score_nan_pen":  round(10.0 * nan_score, 2),
        # Métriques brutes pour débogage
        "snr_near_ef": round(snr, 3),
        "secdev_ratio_98_50": round(sec_ratio, 3),
        "dynamic_range": round(dynamic_range, 5),
        "fd_fwhm_ev": round(fwhm_ev, 4) if np.isfinite(fwhm_ev) else np.nan,
        "symmetry_error": round(symmetry_error, 4) if np.isfinite(symmetry_error) else np.nan,
        "nan_fraction": round(nan_frac, 4),
    }


# -----------------------------------------------------------------------------
# Extraction métadonnées logbook
# -----------------------------------------------------------------------------
def extract_metadata(row: pd.Series, cfg: Dict[str, object]) -> Dict[str, object]:
    """
    Extrait les métadonnées réelles du logbook Solaris/DA30.
    Aucune valeur générique silencieuse : un avertissement est émis si une colonne manque.

    Polarisation via phase ondulateur (convention URANOS Solaris) :
      0°  → LH (linéaire horizontale)
      60° → LV (linéaire verticale)
      30° → L30 (linéaire 30°)

    Direction de scan via R2 :
      ~2°  → ΓX
      ~47° → ΓM
      ~92° → inhabituel
    """
    def _get(col, fallback=np.nan, warn=True):
        val = row.get(col, None)
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            if warn:
                print(f"  ⚠ colonne '{col}' absente ou NaN → fallback={fallback}")
            return fallback
        return val

    # ── Température ──────────────────────────────────────────────────────────
    temp_raw = _get("Sample temperature [K]", fallback=None, warn=False)
    if temp_raw is None or (not isinstance(temp_raw, str) and pd.isna(temp_raw)):
        temp_K = float(cfg.get("temperature_K", 28.0))
        temp_source = "config_fallback"
        print(f"  ⚠ température absente du logbook → utilisation config ({temp_K} K)")
    else:
        try:
            temp_K = float(temp_raw)
            temp_source = "logbook"
        except (ValueError, TypeError):
            temp_K = float(cfg.get("temperature_K", 28.0))
            temp_source = "config_fallback"

    # ── Énergie du photon ─────────────────────────────────────────────────────
    hv_raw = _get("Monochromator energy [eV]", fallback=np.nan)
    try:
        hv = float(hv_raw)
    except (ValueError, TypeError):
        hv = np.nan

    # ── Direction de scan (R2) ────────────────────────────────────────────────
    r2_raw = _get("R2", fallback=np.nan, warn=False)
    try:
        r2_deg = float(r2_raw)
        if abs(r2_deg - 47) < 10:
            scan_dir = "ΓM"
        elif abs(r2_deg - 2) < 8 or abs(r2_deg - 5) < 8:
            scan_dir = "ΓX"
        elif abs(r2_deg - 92) < 10:
            scan_dir = "ΓX?"
        else:
            scan_dir = f"R2={r2_deg:.0f}°"
    except (ValueError, TypeError):
        r2_deg = np.nan
        scan_dir = "?"

    # ── Polarisation (phase ondulateur) ──────────────────────────────────────
    phase_raw = _get("Undulator phase", fallback=np.nan, warn=False)
    try:
        phase_deg = float(phase_raw)
        if abs(phase_deg - 0) < 5:
            pol = "LH"
        elif abs(phase_deg - 60) < 5:
            pol = "LV"
        elif abs(phase_deg - 30) < 5:
            pol = "L30"
        else:
            pol = f"phase={phase_deg:.0f}°"
    except (ValueError, TypeError):
        phase_deg = np.nan
        pol = "?"

    # ── Autres colonnes utiles ────────────────────────────────────────────────
    meas_no       = _get("Measurement NO",   fallback="?",  warn=False)
    date          = _get("Date",             fallback="?",  warn=False)
    time_         = _get("Time",             fallback="?",  warn=False)
    beam_current  = _get("Beam current [mA]",fallback=np.nan, warn=False)
    pressure      = _get("Pressure [mbar]",  fallback=np.nan, warn=False)
    comments      = _get("Comments",         fallback="",   warn=False)

    return {
        "meas_no":           meas_no,
        "date":              date,
        "time":              time_,
        "hv":                hv,
        "temperature_K":     temp_K,
        "temperature_source": temp_source,
        "scan_direction":    scan_dir,
        "R2_deg":            r2_deg,
        "polarization":      pol,
        "undulator_phase":   phase_deg,
        "beam_current":      beam_current,
        "pressure":          pressure,
        "comments":          comments,
    }


# -----------------------------------------------------------------------------
# Figure 5 panneaux
# -----------------------------------------------------------------------------
def add_panel_image(ax: plt.Axes, kpar: np.ndarray, ev_arr: np.ndarray, img: np.ndarray, title: str, mode: str) -> None:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        vmin = vmax = None
        cmap = "inferno"
    elif mode == "bm":
        vmin, vmax = np.nanpercentile(finite, [1, 99])
        cmap = "inferno"
    elif mode in ("secdev", "curv"):
        # -d²I/dE² et -∇²I sont positifs aux pics (bandes) → on garde seulement les valeurs
        # positives (vmin=0) : le fond tombe à zéro, les bandes ressortent clairement.
        pos = finite[finite > 0]
        vmin = 0.0
        vmax = float(np.nanpercentile(pos, 99)) if pos.size > 0 else 1.0
        cmap = "hot_r"
    else:
        vm = np.nanpercentile(np.abs(finite), 99)
        vmin, vmax = -vm, vm
        cmap = "RdBu_r"
    ax.pcolormesh(kpar, ev_arr, img.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.axhline(0, color="cyan", lw=0.9, ls="--")
    ax.set_xlabel("k// (pi/a)")
    ax.set_ylabel("E - EF (eV)")
    ax.set_title(title)


def _draw_waterfall(ax: plt.Axes, data_cut: np.ndarray, kpar: np.ndarray,
                    ev_arr: np.ndarray, ev_start: float, ev_end: float,
                    delta_ev: float, cfg: Dict[str, object], title: str) -> None:
    """Trace un waterfall MDC sur ax. Affiche un message si la fenêtre est trop petite."""
    mask = (ev_arr >= ev_start) & (ev_arr <= ev_end)
    if mask.sum() < 5:
        ax.text(0.5, 0.5, f"Fenêtre insuffisante\n({ev_start:.2f} → {ev_end:.2f} eV)",
                ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return
    waterfall_plot(
        data_cut[:, mask],
        kpar,
        ev_arr[mask],
        ev_start=ev_start,
        ev_end=ev_end,
        delta_ev=delta_ev,
        smooth_sigma=float(cfg["waterfall_smooth_sigma"]),
        offset_scale=float(cfg["waterfall_offset_scale"]),
        normalize="each",
        fill_alpha=float(cfg["waterfall_fill_alpha"]),
        title=title,
        ax=ax,
        fill=False,
    )


def make_review_figure(
    file_label: str,
    meta: Dict[str, object],
    kpar: np.ndarray,
    ev_arr: np.ndarray,
    data_cut: np.ndarray,       # données proche EF (≥ -1 eV) : BM / SecDev / Curv / waterfall-nef
    kpar_full: np.ndarray,
    ev_full: np.ndarray,
    data_full: np.ndarray,      # données complètes (jusqu'à ev_min) : waterfall profond uniquement
    derived: Dict[str, np.ndarray],
    cfg: Dict[str, object],
) -> plt.Figure:
    # Layout 2×3 :
    #   [0,0] BM lissée (≤1 eV)  |  [0,1] Waterfall proche EF (~40 MDC)  |  [0,2] Waterfall profond (-2→-1 eV)
    #   [1,0] -d²I/dE²  (≤1 eV) |  [1,1] Courbure 2D (≤1 eV)            |  [1,2] libre
    #
    # Lecture des panneaux :
    #   BM lissée      : couleur = intensité. Bandes = traînées qui montent/descendent avec k.
    #   Waterfall MDC  : chaque ligne = une MDC à énergie fixe. Pic sur la ligne = bande à ce k.
    #                    Le décalage des pics d'une ligne à l'autre montre la dispersion.
    #   -d²I/dE²       : zones claires = bandes accentuées (fond supprimé par vmin=0).
    #   Courbure 2D    : C = −∇²I / (1+|∇I|²)^(3/2). Max positif = centre d'une bande.
    #                    Plus robuste que SecDev pour bandes dispersives. Ignorer bords.
    fig = plt.figure(figsize=(22, 9), dpi=int(cfg["dpi"]))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    ax_bm   = fig.add_subplot(gs[0, 0])
    ax_nef  = fig.add_subplot(gs[0, 1])
    ax_deep = fig.add_subplot(gs[0, 2])
    ax_sec  = fig.add_subplot(gs[1, 0])
    ax_curv = fig.add_subplot(gs[1, 1])
    # gs[1,2] libre

    add_panel_image(ax_bm,   kpar, ev_arr, derived["smoothed"], "BM lissée  (proche EF)",              mode="bm")
    add_panel_image(ax_sec,  kpar, ev_arr, derived["secdev"],   "-d²I/dE²  (zones claires = bandes)",  mode="secdev")
    add_panel_image(ax_curv, kpar, ev_arr, derived["curvature"],
                    "Courbure 2D  (max local = centre de bande)\nignorer artéfacts de bord", mode="curv")

    nef_start  = float(cfg["waterfall_nef_start"])
    nef_end    = float(cfg["waterfall_nef_end"])
    nef_delta  = float(cfg["waterfall_nef_delta"])
    deep_start = float(cfg["waterfall_deep_start"])
    deep_end   = float(cfg["waterfall_deep_end"])
    deep_delta = float(cfg["waterfall_deep_delta"])
    n_nef  = int(round((nef_end  - nef_start)  / nef_delta))
    n_deep = int(round((deep_end - deep_start) / deep_delta))

    # Waterfall proche EF : données proche EF (data_cut)
    _draw_waterfall(ax_nef, data_cut, kpar, ev_arr,
                    nef_start, nef_end, nef_delta, cfg,
                    f"Waterfall MDC — proche EF\n"
                    f"({nef_start:.2f}→{nef_end:.2f} eV, Δ={nef_delta*1000:.0f} meV, ~{n_nef} MDC)")

    # Waterfall profond : données complètes (data_full) + smooth/offset dédiés
    deep_cfg = dict(cfg)
    deep_cfg["waterfall_smooth_sigma"] = float(cfg["waterfall_deep_smooth_sigma"])
    deep_cfg["waterfall_offset_scale"] = float(cfg["waterfall_deep_offset_scale"])
    _draw_waterfall(ax_deep, data_full, kpar_full, ev_full,
                    deep_start, deep_end, deep_delta, deep_cfg,
                    f"Waterfall MDC — profond\n"
                    f"({deep_start:.1f}→{deep_end:.1f} eV, Δ={deep_delta*1000:.0f} meV, ~{n_deep} MDC)")

    hv    = meta.get("hv", np.nan)
    temp  = meta.get("temperature_K", np.nan)
    pol   = meta.get("polarization", "?")
    direc = meta.get("scan_direction", "?")
    meas  = meta.get("meas_no", "?")

    hv_str   = f"{hv:.1f}" if np.isfinite(float(hv))   else "?"
    temp_str = f"{temp:.1f}" if np.isfinite(float(temp)) else "?"

    fig.suptitle(
        f"{file_label}  |  #{meas}  hν={hv_str} eV  T={temp_str} K  pol={pol}  dir={direc}",
        fontsize=13,
        fontweight="bold",
    )
    return fig


# -----------------------------------------------------------------------------
# Pipeline principal fichier par fichier
# -----------------------------------------------------------------------------
def process_file(filepath: Path, row: pd.Series, out_dir: Path, cfg: Dict[str, object]) -> Dict[str, object]:
    # Extraire les métadonnées réelles du logbook (pas de valeurs génériques silencieuses)
    meta = extract_metadata(row, cfg)

    da_k, kpar, ev_arr, data, hv_file = load_da30_bm(
        filepath,
        float(cfg["work_func"]),
        float(cfg["a_lattice"]),
        hv_logbook=meta["hv"],
    )

    ef_fit = adaptive_ef_fit(ev_arr, data, row, cfg)
    ev_corr = ev_arr - float(ef_fit["EF"])

    windows = choose_energy_windows(ev_corr, cfg)
    local_cfg = dict(cfg)
    local_cfg["waterfall_nef_start"]         = windows["nef_start"]
    local_cfg["waterfall_nef_end"]           = windows["nef_end"]
    local_cfg["waterfall_nef_delta"]         = windows["nef_delta"]
    local_cfg["waterfall_deep_start"]        = windows["deep_start"]
    local_cfg["waterfall_deep_end"]          = windows["deep_end"]
    local_cfg["waterfall_deep_delta"]        = windows["deep_delta"]
    local_cfg["waterfall_deep_smooth_sigma"] = windows["deep_smooth_sigma"]
    local_cfg["waterfall_deep_offset_scale"] = windows["deep_offset_scale"]

    k_crop_tup = tuple(local_cfg["k_crop"]) if local_cfg.get("k_crop") is not None else None

    # ── Crop proche EF (BM affichée / SecDev / Curvature / waterfall-nef) ────
    # Borné à -1 eV : on observe uniquement les bandes proches de EF, pas le bulk
    data_cut, k_cut, e_cut = crop_ke(
        data, kpar, ev_corr, k_crop_tup,
        (windows["analysis_lo"], windows["analysis_hi"]),
    )
    data_cut = robust_normalize(data_cut)

    # ── Crop complet (waterfall profond -2 → -1 eV) ────────────────────────────
    data_full, k_full, e_full = crop_ke(
        data, kpar, ev_corr, k_crop_tup,
        (windows["display_lo"], windows["display_hi"]),
    )
    data_full = robust_normalize(data_full)

    derived = secdev_curvature(
        data_cut, k_cut, e_cut,
        sigma_k=float(cfg["secdev_sigma_k"]),
        sigma_e=float(cfg["secdev_sigma_e"]),
        c0_fraction=float(cfg["secdev_c0_fraction"]),
        border_clip=int(cfg["secdev_border_clip"]),
    )

    fig = make_review_figure(
        filepath.name, meta,
        k_cut, e_cut, data_cut,      # proche EF : BM / SecDev / Curv / waterfall-nef
        k_full, e_full, data_full,   # complet   : waterfall profond
        derived, local_cfg,
    )
    png_path = out_dir / f"{filepath.stem}_review.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "file":             filepath.name,
        "path":             str(filepath),
        "meas_no":          meta["meas_no"],
        "date":             meta["date"],
        "time":             meta["time"],
        "hv_eV":            hv_file,
        "hv_logbook_eV":    meta["hv"],
        "temperature_K":    meta["temperature_K"],
        "temp_source":      meta["temperature_source"],
        "scan_direction":   meta["scan_direction"],
        "R2_deg":           meta["R2_deg"],
        "polarization":     meta["polarization"],
        "undulator_phase":  meta["undulator_phase"],
        "beam_current_mA":  meta["beam_current"],
        "pressure_mbar":    meta["pressure"],
        "comments":         meta["comments"],
        "ef_shift_eV":      float(ef_fit["EF"]),
        "ef_err_eV":        float(ef_fit.get("EF_err", np.nan)),
        "ef_residual":      float(ef_fit.get("residual", np.nan)),
        "ef_status":        str(ef_fit.get("status", "unknown")),
        "review_png":       str(png_path),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch review des band maps ARPES.")
    p.add_argument("--data-dir", type=Path, required=True, help="Dossier contenant les fichiers .pxt/.ibw")
    p.add_argument("--logbook", type=Path, required=True, help="CSV logbook Solaris")
    p.add_argument("--out-dir", type=Path, default=Path("bm_review_out"), help="Dossier de sortie")
    p.add_argument("--config", type=Path, default=None, help="JSON optionnel pour surcharger les paramètres")
    p.add_argument("--all-normal", action="store_true", help="Traiter tous les scans 'Normal' même si le Spectrum Name diffère")
    p.add_argument("--files", nargs="*", default=None, help="Liste explicite de fichiers à traiter")
    p.add_argument("--max-files", type=int, default=None, help="Limiter le nombre de fichiers traités")
    return p.parse_args()


def load_config(config_path: Optional[Path]) -> Dict[str, object]:
    cfg = dict(DEFAULTS)
    if config_path is None:
        return cfg
    payload = json.loads(config_path.read_text())
    cfg.update(payload)
    return cfg


def select_rows(df: pd.DataFrame, cfg: Dict[str, object], explicit_files: Optional[List[str]], all_normal: bool) -> pd.DataFrame:
    out = df.copy()
    if explicit_files:
        explicit = {Path(x).name for x in explicit_files}
        mask = out["File"].fillna("").apply(lambda x: any(Path(str(c)).name in explicit for c in candidate_paths(Path("."), str(x))))
        return out.loc[mask].copy()

    if all_normal:
        return out.loc[out["Mode"].astype(str).str.strip().eq(str(cfg["only_mode"]))].copy()

    return out.loc[
        out["Mode"].astype(str).str.strip().str.lower().eq(str(cfg["only_mode"]).lower())
        & out["Spectrum Name"].astype(str).str.strip().str.lower().eq(str(cfg["only_spectrum_name"]).lower())
    ].copy()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    erlab.io.set_loader("da30")

    # Logbook : gestion BOM UTF-8 (\xef\xbb\xbf) + auto-détection séparateur
    # NB : ne pas ouvrir avec Excel en double-cliquant — Excel utilise la virgule par défaut
    #      et affiche tout dans la colonne A. Utiliser Data > Depuis Texte/CSV > séparateur=Tab.
    import io
    raw_bytes = args.logbook.read_bytes()
    raw_text = raw_bytes.decode("utf-8-sig", errors="replace")
    n_tabs = raw_text.count("\t")
    n_semi = raw_text.count(";")
    sep = "\t" if n_tabs > n_semi else ";"
    print(f"Logbook : {args.logbook.name}  |  séparateur détecté = {'TAB' if sep == chr(9) else ';'}  "
          f"(tabs={n_tabs}, semicolons={n_semi})")
    df = pd.read_csv(io.StringIO(raw_text), sep=sep)
    rows = select_rows(df, cfg, args.files, args.all_normal)
    if rows.empty:
        print("Aucune ligne du logbook ne correspond aux critères.")
        return 1

    results: List[Dict[str, object]] = []
    missing: List[Tuple[object, str]] = []

    if args.max_files is not None:
        rows = rows.head(args.max_files)

    print(f"{len(rows)} fichiers BM candidats trouvés dans le logbook.")

    for _, row in rows.iterrows():
        file_cell = row.get("File", "")
        cands = candidate_paths(args.data_dir, str(file_cell))
        chosen = next((p for p in cands if p.exists()), None)
        if chosen is None:
            missing.append((row.get("Measurement NO", "?"), str(file_cell)))
            continue

        print(f"→ Traitement {chosen.name}")
        try:
            r = process_file(chosen, row, args.out_dir, cfg)
            # recopier quelques métadonnées utiles
            for col in [
                "Measurement NO",
                "Date",
                "Time",
                "Mode",
                "Spectrum Name",
                "Monochromator energy [eV]",
                "Sample temperature [K]",
                "Beam current [mA]",
            ]:
                r[col] = row.get(col, np.nan)
            results.append(r)
        except Exception as exc:
            results.append(
                {
                    "file": chosen.name,
                    "path": str(chosen),
                    "error": str(exc),
                    "Measurement NO": row.get("Measurement NO", np.nan),
                    "Mode": row.get("Mode", np.nan),
                    "Spectrum Name": row.get("Spectrum Name", np.nan),
                }
            )
            print(f"   ✗ échec: {exc}")

    if missing:
        miss_path = args.out_dir / "missing_files.csv"
        pd.DataFrame(missing, columns=["Measurement NO", "File_from_logbook"]).to_csv(miss_path, index=False)
        print(f"{len(missing)} fichiers du logbook introuvables -> {miss_path}")

    if not results:
        print("Aucun fichier traité avec succès.")
        return 1

    res_df = pd.DataFrame(results)
    out_csv = args.out_dir / "bm_review_summary.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Résumé écrit: {out_csv}")

    show_cols = [c for c in [
        "file", "meas_no", "hv_eV", "temperature_K", "polarization",
        "scan_direction", "ef_shift_eV", "ef_status",
    ] if c in res_df.columns]
    print("\nRésumé :")
    print(res_df[show_cols].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
