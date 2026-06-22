"""Controleur de fit MDC sans dependance PyQt."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
import hashlib
import json
import warnings

import numpy as np

from arpes.physics.dispersion_fit import (
    CURVATURE_MAX,
    MIN_DISP_POINTS,
    curvature_ratio,
    linear_dispersion_fit,
)


def compute_fit_params_hash(
    fp: Any,
    *,
    ef_offset: float = 0.0,
    view_mode: str = "",
    hv: float | None = None,
    bm_distortion: dict | None = None,
    grid_correction: dict | None = None,
    ef_correction: dict | None = None,
    ensemble_settings: dict | None = None,
) -> str:
    """Empreinte stable des paramètres ayant un impact sur le fit MDC.

    Stockée dans fit_result et recalculée à l'affichage : si l'état
    courant diffère du hash mémorisé, le fit affiché est marqué STALE
    (paramètres modifiés depuis le fit → résultats potentiellement
    incohérents). Cohérence cache (arpes-redteam).
    """
    fp_dict: dict
    if is_dataclass(fp):
        fp_dict = asdict(fp)
    elif isinstance(fp, dict):
        fp_dict = dict(fp)
    else:
        fp_dict = {k: getattr(fp, k) for k in dir(fp)
                    if not k.startswith("_") and not callable(getattr(fp, k))}
    payload = {
        "fp": _sanitize(fp_dict),
        "ef_offset": float(ef_offset or 0.0),
        "view_mode": str(view_mode or ""),
        "hv": float(hv) if hv is not None else None,
        "bm_distortion": _sanitize(bm_distortion or {}),
        "grid_correction": _sanitize(grid_correction or {}),
        "ef_correction": _sanitize(ef_correction or {}),
        "ensemble_settings": _sanitize(ensemble_settings or {}),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# P2.2 — rigueur quantitative vF/kF/m* (fit partagé physics.dispersion_fit).
_HBAR2_OVER_ME = 7.6199682          # eV·Å² = ℏ²/m_e


def compute_fermi_velocity_mstar(
    fit_result: dict,
    crystal_a: float,
    *,
    branch: str = "kF_minus",
    pair_index: int = 0,
    window_eV: float = 0.05,
) -> dict[str, float]:
    """vF (eV·Å), kF (Å⁻¹), m*/m_e + incertitudes à partir de kF(E).

    Ajuste E ≈ slope·k + intercept (k en π/a) sur les points dans
    ±window_eV de E_F, puis kF=−intercept/slope, vF=|slope|/(π/a),
    m*/m_e=7.6199·kF/vF (ℏ²/m_e=7.6199 eV·Å²).

    Rigueur P2.2 :
    - Refuse (NaN dans vF/kF/m*) si < 5 points : ni la gate quadratique
      ni l'ODR n'ont de sens sous ce seuil.
    - Gate linéarité : fit quadratique E=a·k²+b·k+c ; si la courbure
      domine (|a|·Δk/|b| > 0.10) la bande n'est pas linéaire près de E_F
      (kink / coupure Fermi) → kF=−b/a non fiable, refus.
    - Régression orthogonale (TLS) pondérée par Γ si disponible, sinon OLS
      pondéré vertical (``polyfit(cov=True)``).
    - σ_vF, σ_kF, σ_m* propagées via la covariance 2×2 (corrélation
      slope↔intercept conservée).

    ``sigma_type`` distingue ``orthogonal_tls`` (vraie pondération) de
    ``ols_regression`` (incertitude de régression seule, PAS une propagation
    d'erreurs de mesure). Clés héritées vF_eV_A / kF_inv_A / mstar_over_me
    conservées (NaN si refus) ; nouvelles clés additives.
    """
    nan = float("nan")
    out = {
        "vF_eV_A": nan, "kF_inv_A": nan, "mstar_over_me": nan,
        "vF_sigma_eV_A": nan, "kF_inv_A_sigma": nan, "mstar_sigma": nan,
        "linear_ok": False, "refused_reason": "", "sigma_type": "none",
        "n_points": 0, "curvature_ratio": nan,
    }

    def _refuse(reason: str) -> dict[str, float]:
        out["refused_reason"] = reason
        return out

    if not fit_result or crystal_a <= 0:
        return _refuse("fit_result vide ou a invalide")
    e_raw = fit_result.get("e_fitted")
    e = np.asarray([] if e_raw is None else e_raw, dtype=float)
    branches = fit_result.get(branch)
    if branches is None or not (0 <= pair_index < len(branches)):
        return _refuse("branche/paire absente")
    k = np.asarray(branches[pair_index], dtype=float)  # en π/a
    mask = np.isfinite(e) & np.isfinite(k) & (np.abs(e) <= float(window_eV))
    n = int(mask.sum())
    out["n_points"] = n
    if n < MIN_DISP_POINTS:
        return _refuse(f"too few points ({n} < {MIN_DISP_POINTS})")
    ew = e[mask]
    kw = k[mask]

    # Linearity gate: quadratic curvature relative to the linear slope.
    if float(np.ptp(kw)) <= 0:
        return _refuse("constant k (vertical slope)")
    curv = curvature_ratio(kw, ew)
    out["curvature_ratio"] = float(curv)
    if not np.isfinite(curv) or curv > CURVATURE_MAX:
        return _refuse(f"nonlinear band (curvature {curv:.2f} > {CURVATURE_MAX})")

    # σ_k per point: proxy = Γ (HWHM in π/a) if available in the fit.
    sk = None
    for gkey in ("gamma_corrige", "gamma_brut", "gamma"):
        g_all = fit_result.get(gkey)
        if g_all is not None and 0 <= pair_index < len(g_all):
            g = np.asarray(g_all[pair_index], dtype=float)
            if g.size == e.size:
                gm = g[mask]
                if np.all(np.isfinite(gm)) and np.all(gm > 0):
                    sk = gm
                    break

    fit = linear_dispersion_fit(kw, ew, sk)
    if not fit["ok"]:
        return _refuse("regression did not converge / degenerate")

    slope = fit["slope"]
    intercept = fit["intercept"]
    cov = fit["cov"]
    var_s = float(cov[0, 0])
    var_i = float(cov[1, 1])
    cov_si = float(cov[0, 1])

    pi_over_a = np.pi / float(crystal_a)
    k0_pi_a = -intercept / slope
    kF_inv_A = float(abs(k0_pi_a) * pi_over_a)
    vF_eV_A = float(abs(slope) / pi_over_a)  # (π/a)→Å⁻¹ : ÷(π/a)
    mstar = _HBAR2_OVER_ME * kF_inv_A / vF_eV_A if vF_eV_A > 0 else nan

    # Propagation σ via dérivées partielles sur (slope, intercept).
    # vF = |slope|/(π/a) → σ_vF = √var_s / (π/a)
    sigma_vF = float(np.sqrt(max(var_s, 0.0)) / pi_over_a)
    # kF = |intercept/slope|·(π/a). ∂k0/∂s = intercept/slope², ∂k0/∂i = −1/slope
    dk0_ds = intercept / (slope * slope)
    dk0_di = -1.0 / slope
    var_k0 = (dk0_ds * dk0_ds * var_s + dk0_di * dk0_di * var_i
              + 2.0 * dk0_ds * dk0_di * cov_si)
    sigma_kF = float(np.sqrt(max(var_k0, 0.0)) * pi_over_a)
    # m* = C·(π/a)²·|intercept|/slope². ∂/∂i = m*/intercept, ∂/∂s = −2·m*/slope
    if np.isfinite(mstar) and intercept != 0.0:
        dm_di = mstar / intercept
        dm_ds = -2.0 * mstar / slope
        var_m = (dm_di * dm_di * var_i + dm_ds * dm_ds * var_s
                 + 2.0 * dm_di * dm_ds * cov_si)
        sigma_mstar = float(np.sqrt(max(var_m, 0.0)))
    else:
        sigma_mstar = nan

    out.update({
        "vF_eV_A": vF_eV_A, "kF_inv_A": kF_inv_A, "mstar_over_me": float(mstar),
        "vF_sigma_eV_A": sigma_vF, "kF_inv_A_sigma": sigma_kF,
        "mstar_sigma": sigma_mstar, "linear_ok": True,
        "sigma_type": fit["method"], "refused_reason": "",
    })
    return out


def gamma_to_hwhm_factor(fit_result) -> float:
    """Multiplier converting a fit's stored gamma to HWHM.

    Modern fits tag ``width_convention="HWHM"`` (gamma already HWHM → 1.0).
    Legacy/untagged fits stored the FWHM under the same keys → 0.5.
    """
    fr = fit_result or {}
    conv = str(fr.get("width_convention", "")).strip().upper()
    units = str(fr.get("gamma_units", "")).strip().upper()
    if conv == "HWHM" or (conv == "" and "HWHM" in units):
        return 1.0
    return 0.5


def _scale_gamma_nested(value, factor: float):
    """Multiply every numeric leaf of a (possibly nested) gamma array."""
    if isinstance(value, (list, tuple)):
        return [_scale_gamma_nested(v, factor) for v in value]
    if isinstance(value, np.ndarray):
        return value * factor
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value * factor
    return value


def migrate_fit_result_to_hwhm(fit_result) -> bool:
    """In-place: rescale a legacy (FWHM) fit_result's gamma arrays to HWHM.

    Idempotent — already-HWHM-tagged fits are left untouched. Scales every
    ``gamma*`` array (plus the ensemble's ``gamma*`` entries) by 0.5: widths,
    sigmas, medians and the resolution floor all share the FWHM→HWHM factor.
    Then stamps the HWHM tags so the whole app reads one convention.
    ``gamma_units`` (a string label) is skipped. Returns True if converted.
    """
    fr = fit_result
    if not isinstance(fr, dict) or not fr:
        return False
    if gamma_to_hwhm_factor(fr) == 1.0:
        return False  # already HWHM
    # "gamma" in k catches sigma_gamma / gamma_left / gamma_min too; the only
    # gamma-named non-width entry is the gamma_units label, excluded.
    for k in list(fr.keys()):
        if "gamma" in k and k != "gamma_units":
            fr[k] = _scale_gamma_nested(fr[k], 0.5)
    ens = fr.get("ensemble")
    if isinstance(ens, dict):
        for k in list(ens.keys()):
            if "gamma" in k and k != "gamma_units":
                ens[k] = _scale_gamma_nested(ens[k], 0.5)
    fr["width_convention"] = "HWHM"
    fr.setdefault("gamma_units", "pi/a HWHM")
    return True


def imaginary_self_energy(
    fit_result: dict,
    crystal_a: float,
    *,
    pair_index: int = 0,
    use_corrected: bool = True,
    side: str = "mean",
) -> dict:
    """Im Sigma(E) = vF * Gamma_k_HWHM(E), in eV.

    Physical relation: an MDC at fixed E is a Lorentzian in k of HWHM = |Σ''|/vF,
    so |Σ''| = vF · HWHM_k (Σ'' = (vF/2)·FWHM_k, the textbook form).

    ``side``: 'mean' (averaged gamma, default), 'left' (gamma on kF- side),
    'right' (gamma on kF+ side). Left/right sources are populated only when
    the fit ran in ``width_mode='independent'``; for other modes, left and
    right equal mean.

    Width convention: modern fits tag ``fit_result["width_convention"]="HWHM"``
    (gamma stored as HWHM). Legacy/untagged fits stored the FWHM under the same
    key, so for those Σ'' = (vF/2)·gamma. This is resolved from the tag here so
    both vintages stay correct. Conversion: Gamma_k[A^-1] = Gamma[pi/a]*pi/a.
    vF tiré de compute_fermi_velocity_mstar (kF_minus, paire 0 par
    défaut). Renvoie ``{"energy": e, "im_sigma": Σ, "vF_eV_A": vF,
    "pair_index": pair_index, "side": side}``. Tableaux vides si
    pré-requis manquent.
    """
    empty = {"energy": np.array([]), "im_sigma": np.array([]),
              "vF_eV_A": float("nan"), "pair_index": int(pair_index),
              "side": str(side), "error": ""}
    if not fit_result or crystal_a <= 0:
        empty["error"] = "missing lattice parameter a"
        return empty
    e_raw = fit_result.get("e_fitted")
    e = np.asarray([] if e_raw is None else e_raw, dtype=float)
    # Choix de la source γ selon side
    if str(side).lower() == "left":
        keys = (("gamma_left_corrige" if use_corrected else "gamma_left_brut"),
                "gamma_corrige", "gamma")
    elif str(side).lower() == "right":
        keys = (("gamma_right_corrige" if use_corrected else "gamma_right_brut"),
                "gamma_corrige", "gamma")
    else:
        keys = (("gamma_corrige" if use_corrected else "gamma_brut"),
                "gamma_corrige", "gamma")
    g_all = None
    for k in keys:
        cand = fit_result.get(k)
        if cand is not None:
            g_all = cand
            break
    if g_all is None or not (0 <= pair_index < len(g_all)):
        empty["error"] = "missing Gamma arrays"
        return empty
    g_pi_a = np.asarray(g_all[pair_index], dtype=float)
    if g_pi_a.size != e.size or g_pi_a.size == 0:
        empty["error"] = "Gamma/E arrays misaligned"
        return empty
    vfd = compute_fermi_velocity_mstar(
        fit_result, crystal_a, pair_index=pair_index)
    vF = vfd.get("vF_eV_A", float("nan"))
    if not np.isfinite(vF) or vF <= 0:
        empty["error"] = vfd.get("refused_reason") or "missing vF"
        return empty
    pi_over_a = np.pi / float(crystal_a)
    g_inv_A = g_pi_a * pi_over_a
    # Σ'' = vF·HWHM. HWHM-tagged fits use vF·γ; legacy/untagged fits stored the
    # FWHM (HWHM=γ/2) so they use (vF/2)·γ. gamma_to_hwhm_factor encodes this.
    width_coef = vF * gamma_to_hwhm_factor(fit_result)
    im_sigma = width_coef * g_inv_A  # eV
    # σ propagée depuis l'ensemble fit : Im Σ linéaire en Γ → σ(ImΣ)=coef·σ_Γ
    im_sigma_std = np.full_like(im_sigma, np.nan)
    ens = fit_result.get("ensemble") or {}
    gstd_all = ens.get("gamma_std") or []
    if 0 <= pair_index < len(gstd_all):
        g_std_pi_a = np.asarray(gstd_all[pair_index], dtype=float)
        if g_std_pi_a.size == e.size:
            im_sigma_std = width_coef * (g_std_pi_a * pi_over_a)
    finite = np.isfinite(e) & np.isfinite(im_sigma)
    return {
        "energy": e[finite],
        "im_sigma": im_sigma[finite],
        "im_sigma_std": im_sigma_std[finite],
        "vF_eV_A": float(vF),
        "pair_index": int(pair_index),
        "side": str(side),
        "error": "",
    }


def ensemble_fit(
    mdc_fitter,
    data,
    kpar,
    ev,
    fp,
    *,
    n_runs: int = 30,
    jitter_pct: float = 0.10,
    resolution_source: str = "",
    seed: int | None = None,
) -> dict:
    """I1: refit N fois avec perturbation des initiaux, agrège statistiquement.

    Hypothèse : 1 paire = 1 bande (modèle Lorentzien symétrique).
    L'ensemble réduit la sensibilité aux initiaux et donne σ statistique
    plus fiable que la covariance de l'optimiseur seule.

    - kF_init et γ_init perturbés relatif (×(1 + N(0, jitter_pct))).
    - Filtre outliers via MAD (>3σ) avant moyenne.
    - Renvoie dict ``ensemble`` à injecter dans fit_result :
      ``{n_runs, n_ok, jitter_pct, e_fitted,
         kF_minus_med/std, kF_plus_med/std,
         gamma_brut_med, gamma_corrige_med/std}`` + side widths and resolution
         metadata when provided by the per-run full fit.
    """
    from copy import deepcopy

    rng = np.random.default_rng(seed)
    runs: list[dict] = []
    base_pairs = list(getattr(fp, "pairs", None) or [])
    for _ in range(int(n_runs)):
        fp_j = deepcopy(fp)
        new_pairs = []
        for pp in base_pairs:
            kfj = float(pp.get("kF_init", 0.30)) * (
                1.0 + jitter_pct * float(rng.standard_normal()))
            gij = float(pp.get("gamma_init", 0.08)) * (
                1.0 + jitter_pct * float(rng.standard_normal()))
            new_pairs.append({**pp, "kF_init": max(0.0, kfj),
                              "gamma_init": max(1e-4, gij)})
        try:
            object.__setattr__(fp_j, "pairs", new_pairs)
        except Exception:
            fp_j.pairs = new_pairs
        try:
            fr = mdc_fitter.run_full_fit(
                data, kpar, ev, fp_j,
                resolution_source=resolution_source,
            )
        except Exception as exc:
            # Run individuel échoué → on continue (intention) mais on
            # signale pour debug : N runs perdues peuvent biaiser σ.
            warnings.warn(
                f"ensemble_fit: jitter run failed ({exc}); ignored.",
                RuntimeWarning, stacklevel=2,
            )
            continue
        if not fr or fr.get("e_fitted") is None:
            continue
        runs.append(fr)
    if not runs:
        return {"n_runs": int(n_runs), "n_ok": 0,
                "jitter_pct": float(jitter_pct), "ensemble": True}
    # Référence taille e_fitted (1ère run convergée)
    _e0 = runs[0].get("e_fitted")
    e_ref = np.asarray([] if _e0 is None else _e0, dtype=float)
    n_e = e_ref.size
    if n_e == 0:
        return {"n_runs": int(n_runs), "n_ok": 0,
                "jitter_pct": float(jitter_pct), "ensemble": True}
    # Empile les branches dim (n_runs_ok, n_pairs, n_e)
    n_pairs = max(
        max(len(fr.get("kF_minus") or []) for fr in runs),
        max(len(fr.get("kF_plus") or []) for fr in runs),
        1,
    )

    def _stack(key: str) -> np.ndarray:
        arr = np.full((len(runs), n_pairs, n_e), np.nan, dtype=float)
        for ri, fr in enumerate(runs):
            branches = fr.get(key) or []
            for pi in range(min(len(branches), n_pairs)):
                v = np.asarray(branches[pi], dtype=float)
                if v.size == n_e:
                    arr[ri, pi, :] = v
        return arr

    km = _stack("kF_minus")
    kp = _stack("kF_plus")
    gb = _stack("gamma_brut")
    gc = _stack("gamma_corrige")
    gm = _stack("gamma_min")
    glb = _stack("gamma_left_brut")
    grb = _stack("gamma_right_brut")
    glc = _stack("gamma_left_corrige")
    grc = _stack("gamma_right_corrige")
    if np.all(np.isnan(gb)):
        gb = _stack("gamma")
    if np.all(np.isnan(gc)):
        gc = gb
    skm = _stack("sigma_kF_minus")
    skp = _stack("sigma_kF_plus")
    sgc = _stack("sigma_gamma")
    sgl = _stack("sigma_gamma_left")
    sgr = _stack("sigma_gamma_right")

    def _agg(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Median + std after MAD filtering (>3*MAD) per (pair, slice)."""
        if np.all(np.isnan(stack)):
            empty = np.full(stack.shape[1:], np.nan, dtype=float)
            return empty, empty.copy()
        med = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - med[np.newaxis, ...]), axis=0)
        # MAD * 1.4826 approximates sigma for a Gaussian distribution.
        sigma = 1.4826 * mad
        mask_ok = np.abs(stack - med[np.newaxis, ...]) <= 3.0 * sigma[np.newaxis, ...] + 1e-12
        filt = np.where(mask_ok, stack, np.nan)
        return np.nanmedian(filt, axis=0), np.nanstd(filt, axis=0)

    def _combine_sigma(between: np.ndarray, within: np.ndarray) -> np.ndarray:
        """Combine run-to-run spread with per-run optimizer covariance sigma."""
        within = np.asarray(within, dtype=float)
        finite = np.isfinite(within) & (within > 0)
        if not finite.any():
            return between
        within_med = np.nanmedian(np.where(finite, within, np.nan), axis=0)
        return np.sqrt(np.nan_to_num(between, nan=0.0) ** 2
                       + np.nan_to_num(within_med, nan=0.0) ** 2)

    kF_minus_med, kF_minus_std = _agg(km)
    kF_plus_med, kF_plus_std = _agg(kp)
    gamma_brut_med, gamma_brut_std = _agg(gb)
    gamma_med, gamma_std = _agg(gc)
    gamma_min_med, _ = _agg(gm)
    gamma_left_brut_med, gamma_left_brut_std = _agg(glb)
    gamma_right_brut_med, gamma_right_brut_std = _agg(grb)
    gamma_left_corrige_med, gamma_left_corrige_std = _agg(glc)
    gamma_right_corrige_med, gamma_right_corrige_std = _agg(grc)
    kF_minus_std = _combine_sigma(kF_minus_std, skm)
    kF_plus_std = _combine_sigma(kF_plus_std, skp)
    gamma_std = _combine_sigma(gamma_std, sgc)
    gamma_left_corrige_std = _combine_sigma(gamma_left_corrige_std, sgl)
    gamma_right_corrige_std = _combine_sigma(gamma_right_corrige_std, sgr)
    out = {
        "ensemble": True,
        "n_runs": int(n_runs),
        "n_ok": int(len(runs)),
        "jitter_pct": float(jitter_pct),
        "e_fitted": e_ref.tolist(),
        "kF_minus_med": [row.tolist() for row in kF_minus_med],
        "kF_minus_std": [row.tolist() for row in kF_minus_std],
        "kF_plus_med": [row.tolist() for row in kF_plus_med],
        "kF_plus_std": [row.tolist() for row in kF_plus_std],
        "gamma_brut_med": [row.tolist() for row in gamma_brut_med],
        "gamma_brut_std": [row.tolist() for row in gamma_brut_std],
        "gamma_med": [row.tolist() for row in gamma_med],
        "gamma_std": [row.tolist() for row in gamma_std],
    }
    if not np.all(np.isnan(gm)):
        out["gamma_min_med"] = [row.tolist() for row in gamma_min_med]
    if not np.all(np.isnan(glb)):
        out["gamma_left_brut_med"] = [row.tolist() for row in gamma_left_brut_med]
        out["gamma_left_brut_std"] = [row.tolist() for row in gamma_left_brut_std]
    if not np.all(np.isnan(grb)):
        out["gamma_right_brut_med"] = [row.tolist() for row in gamma_right_brut_med]
        out["gamma_right_brut_std"] = [row.tolist() for row in gamma_right_brut_std]
    if not np.all(np.isnan(glc)):
        out["gamma_left_corrige_med"] = [row.tolist() for row in gamma_left_corrige_med]
        out["gamma_left_corrige_std"] = [row.tolist() for row in gamma_left_corrige_std]
    if not np.all(np.isnan(grc)):
        out["gamma_right_corrige_med"] = [row.tolist() for row in gamma_right_corrige_med]
        out["gamma_right_corrige_std"] = [row.tolist() for row in gamma_right_corrige_std]
    first = runs[0]
    for key in ("resolution", "width_mode", "shape", "eta", "fit_kpar", "kpar", "ev_arr", "n_pairs"):
        if key in first:
            out[key] = _sanitize(first[key])
    return out


def detect_n_pairs(
    k_arr,
    mdc,
    *,
    k_min: float,
    k_max: float,
    center_init: float = 0.0,
    smooth_sigma: float = 3.0,
    min_height: float = 0.10,
    max_pairs: int = 8,
) -> int:
    """Compte les paires de pics symétriques autour de ``center_init``.

    Lisse + ``find_peaks`` dans la fenêtre k_min..k_max ; sépare gauche/
    droite du centre ; renvoie min(n_gauche, n_droite). Si un seul côté
    a des pics → renvoie ce nombre (pic symétrisé). 0 → 1 par défaut
    (toujours au moins une paire). Borné par ``max_pairs``.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    k = np.asarray(k_arr, dtype=float)
    m = np.asarray(mdc, dtype=float)
    mask = (k >= float(k_min)) & (k <= float(k_max))
    if int(mask.sum()) < 10:
        return 1
    kw, mw = k[mask], m[mask]
    s = max(1, int(smooth_sigma))
    mw_sm = gaussian_filter1d(np.nan_to_num(mw), sigma=s)
    lo, hi = float(np.nanmin(mw_sm)), float(np.nanmax(mw_sm))
    if hi - lo < 1e-10:
        return 1
    mn = (mw_sm - lo) / (hi - lo)
    pks, _ = find_peaks(mn, height=float(min_height), distance=max(3, s))
    if pks.size == 0:
        return 1
    kpks = kw[pks]
    left = int(np.sum(kpks < float(center_init)))
    right = int(np.sum(kpks >= float(center_init)))
    if left == 0 and right == 0:
        return 1
    if left == 0 or right == 0:
        return max(1, min(int(max_pairs), max(left, right)))
    return max(1, min(int(max_pairs), min(left, right)))


def _sanitize(obj):
    """JSON-safe conversion: numpy scalars, lists, recursive dicts."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass(frozen=True)
class FitSummary:
    n_points: int
    n_ok: int
    xg_mean: float
    label_text: str
    status_text: str
    resolution_dominates: bool = False


class MdcFitter:
    """Prepare les arguments, appelle arpes_plots, resume le resultat."""

    def __init__(self, arpes_plots_module: Any):
        self.ap = arpes_plots_module

    @staticmethod
    def fit_kwargs(fp: Any, resolution_source: str = "") -> dict[str, Any]:
        kF_init_list = [p.get("kF_init", 0.30) for p in (getattr(fp, "pairs", None) or [])]
        return {
            "n_pairs": fp.n_pairs,
            "ev_start": fp.ev_start,
            "ev_end": fp.ev_end,
            "smooth_fit": fp.smooth_fit,
            "smooth_detect": fp.smooth_detect,
            "gamma_init": fp.gamma_init,
            "gamma_max": fp.gamma_max,
            "kF_init": kF_init_list or None,
            "center_init": fp.center_init,
            "xg_range": fp.xg_range,
            "min_amplitude": fp.min_amplitude,
            "max_jump": fp.max_jump,
            "mdc_energy_window": getattr(fp, "mdc_energy_window", 0.0),
            "scan_direction": fp.scan_direction,
            "width_mode": fp.width_mode,
            "k_min": fp.k_min,
            "k_max": fp.k_max,
            "k0_max": fp.k0_max,
            "dE_eV": fp.dE_meV / 1000.0,
            "dk_inv_a": fp.dk_inv_a,
            "resolution_source": resolution_source,
            "shape": getattr(fp, "shape", "lorentzian"),
            "hold_center": bool(getattr(fp, "hold_center", False)),
            "hold_gamma": bool(getattr(fp, "hold_gamma", False)),
            "verbose": False,
        }

    def run_full_fit(self, data, kpar, ev, fp: Any, resolution_source: str = "") -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.ap.fit_mdc_peak_pairs(
                data,
                kpar,
                ev,
                **self.fit_kwargs(fp, resolution_source=resolution_source),
            )

    @staticmethod
    def summarize(fr: dict, *, crystal_a: float = 0.0) -> FitSummary:
        e_fitted = fr.get("e_fitted", [])
        n_e = len(e_fitted)
        kf0 = np.asarray((fr.get("kF_minus") or [[np.nan]])[0], dtype=float)
        n_ok = int(np.isfinite(kf0).sum())
        xg_arr = np.asarray(fr.get("xg") or [], dtype=float)
        xg_mean = float(np.nanmean(xg_arr)) if np.isfinite(xg_arr).any() else float("nan")
        gamma_note = ""
        resolution_dominates = False
        if fr.get("gamma_brut") and fr.get("gamma_corrige"):
            gb = np.asarray(fr["gamma_brut"][0], dtype=float)
            gc = np.asarray(fr["gamma_corrige"][0], dtype=float)
            if np.isfinite(gb).any() and np.isfinite(gc).any():
                resolution_dominates = float(np.nanmedian(gc)) < 0.3 * float(np.nanmedian(gb))
                warn = " Warning" if resolution_dominates else ""
                gamma_note = (
                    f"\nΓ med = {float(np.nanmedian(gb)):.4f} raw / "
                    f"{float(np.nanmedian(gc)):.4f} corrected{warn}"
                )
        sigma_note = ""
        sigma_k = fr.get("sigma_kF_plus") or fr.get("sigma_kF_minus") or []
        sigma_g = fr.get("sigma_gamma") or []
        if sigma_k:
            sk = np.asarray(sigma_k[0], dtype=float)
            sg = np.asarray(sigma_g[0], dtype=float) if sigma_g else np.asarray([], dtype=float)
            parts = []
            if np.isfinite(sk).any():
                parts.append(f"σkF med = {float(np.nanmedian(sk)):.4g} π/a")
            if sg.size and np.isfinite(sg).any():
                parts.append(f"σΓ med = {float(np.nanmedian(sg)):.4g} π/a")
            if parts:
                sigma_note = "\n" + " | ".join(parts)
        vf_line = ""
        if float(crystal_a or 0.0) > 0:
            vfd = compute_fermi_velocity_mstar(fr, float(crystal_a))
            vF = vfd["vF_eV_A"]
            ms = vfd["mstar_over_me"]
            kF = vfd["kF_inv_A"]
            if np.isfinite(vF) and np.isfinite(ms):
                vf_line = (f"\nvF = {vF:.2f} eV·Å | m*/m_e = {ms:.2f} | "
                            f"kF = {kF:.3f} Å⁻¹")
        label_text = (
            f"OK  Fit complet  {n_ok}/{n_e} points\n"
            f"xg = {xg_mean:.4f} π/a"
            f"{gamma_note}{sigma_note}{vf_line}"
        )
        status_text = (
            f"Fit OK — {n_ok}/{n_e}  xg={xg_mean:.4f}"
            f"{gamma_note.replace(chr(10), ' | ')}"
            f"{sigma_note.replace(chr(10), ' | ')}"
        )
        return FitSummary(
            n_points=n_e,
            n_ok=n_ok,
            xg_mean=xg_mean,
            label_text=label_text,
            status_text=status_text,
            resolution_dominates=resolution_dominates,
        )

    @staticmethod
    def update_entry_after_fit(
        entry: Any,
        fp: Any,
        *,
        ef_offset: float,
        edcnorm: bool,
        view_mode: str,
        hv: float | None,
    ) -> None:
        entry.fit_params = fp
        entry.ef_offset = ef_offset
        entry.edcnorm = bool(edcnorm)
        entry.view_mode = view_mode
        entry.meta.hv = hv
