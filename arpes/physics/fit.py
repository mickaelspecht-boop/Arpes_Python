"""Controleur de fit MDC sans dependance PyQt."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
import hashlib
import json
import warnings

import numpy as np


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


def compute_fermi_velocity_mstar(
    fit_result: dict,
    crystal_a: float,
    *,
    branch: str = "kF_minus",
    pair_index: int = 0,
    window_eV: float = 0.05,
) -> dict[str, float]:
    """vF (eV·Å) + m*/m_e à partir de kF(E) près de E_F.

    Régression linéaire E ≈ vF·(k - k_F0) sur les points e_fitted dans
    ±window_eV. kF en π/a converti en Å⁻¹ via a. m*/m_e=7.6199·kF/vF
    (ℏ²/m_e = 7.6199 eV·Å²). Renvoie NaN si données insuffisantes.
    """
    nan = float("nan")
    out = {"vF_eV_A": nan, "kF_inv_A": nan, "mstar_over_me": nan}
    if not fit_result or crystal_a <= 0:
        return out
    e_raw = fit_result.get("e_fitted")
    e = np.asarray([] if e_raw is None else e_raw, dtype=float)
    branches = fit_result.get(branch)
    if branches is None or not (0 <= pair_index < len(branches)):
        return out
    k = np.asarray(branches[pair_index], dtype=float)  # en π/a
    mask = np.isfinite(e) & np.isfinite(k) & (np.abs(e) <= float(window_eV))
    if int(mask.sum()) < 3:
        return out
    ew = e[mask]
    kw = k[mask]
    # Régression : E = vF_natural·(k - k0) avec k en π/a → vF_pi_a = pente
    slope_pi_a, intercept = np.polyfit(kw, ew, 1)
    if not np.isfinite(slope_pi_a) or abs(slope_pi_a) < 1e-9:
        return out
    # k0 (kF à E=0) en π/a, puis Å⁻¹
    k0_pi_a = -intercept / slope_pi_a
    pi_over_a = np.pi / float(crystal_a)
    kF_inv_A = float(abs(k0_pi_a) * pi_over_a)
    vF_eV_A = float(abs(slope_pi_a) / pi_over_a)  # (π/a)→Å⁻¹ : ÷(π/a)
    HBAR2_OVER_ME = 7.6199682  # eV·Å²
    mstar = HBAR2_OVER_ME * kF_inv_A / vF_eV_A if vF_eV_A > 0 else nan
    out["vF_eV_A"] = vF_eV_A
    out["kF_inv_A"] = kF_inv_A
    out["mstar_over_me"] = float(mstar)
    return out


def imaginary_self_energy(
    fit_result: dict,
    crystal_a: float,
    *,
    pair_index: int = 0,
    use_corrected: bool = True,
) -> dict:
    """Im Σ(E) = (vF/2)·Γ_k(E) en eV.

    Γ stocké en π/a (HWHM). Conversion : Γ_k[Å⁻¹] = Γ[π/a]·π/a.
    vF tiré de compute_fermi_velocity_mstar (kF_minus, paire 0 par
    défaut). Renvoie ``{"energy": e, "im_sigma": Σ, "vF_eV_A": vF,
    "pair_index": pair_index}``. Tableaux vides si pré-requis manquent.
    """
    empty = {"energy": np.array([]), "im_sigma": np.array([]),
              "vF_eV_A": float("nan"), "pair_index": int(pair_index)}
    if not fit_result or crystal_a <= 0:
        return empty
    e_raw = fit_result.get("e_fitted")
    e = np.asarray([] if e_raw is None else e_raw, dtype=float)
    src = "gamma_corrige" if use_corrected else "gamma_brut"
    g_all = fit_result.get(src)
    if g_all is None:
        g_all = fit_result.get("gamma_corrige") or fit_result.get("gamma")
    if g_all is None or not (0 <= pair_index < len(g_all)):
        return empty
    g_pi_a = np.asarray(g_all[pair_index], dtype=float)
    if g_pi_a.size != e.size or g_pi_a.size == 0:
        return empty
    vfd = compute_fermi_velocity_mstar(
        fit_result, crystal_a, pair_index=pair_index)
    vF = vfd.get("vF_eV_A", float("nan"))
    if not np.isfinite(vF) or vF <= 0:
        return empty
    pi_over_a = np.pi / float(crystal_a)
    g_inv_A = g_pi_a * pi_over_a
    im_sigma = (vF / 2.0) * g_inv_A  # eV
    # σ propagée depuis l'ensemble fit : Im Σ linéaire en Γ → σ(ImΣ)=(vF/2)·σ_Γ
    im_sigma_std = np.full_like(im_sigma, np.nan)
    ens = fit_result.get("ensemble") or {}
    gstd_all = ens.get("gamma_std") or []
    if 0 <= pair_index < len(gstd_all):
        g_std_pi_a = np.asarray(gstd_all[pair_index], dtype=float)
        if g_std_pi_a.size == e.size:
            im_sigma_std = (vF / 2.0) * (g_std_pi_a * pi_over_a)
    finite = np.isfinite(e) & np.isfinite(im_sigma)
    return {
        "energy": e[finite],
        "im_sigma": im_sigma[finite],
        "im_sigma_std": im_sigma_std[finite],
        "vF_eV_A": float(vF),
        "pair_index": int(pair_index),
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
         gamma_corrige_med/std}`` + ``kF_minus``/``kF_plus``/``gamma_*``
         écrasés par les médianes (consommé par pipeline aval).
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
        except Exception:
            continue
        if not fr or fr.get("e_fitted") is None:
            continue
        runs.append(fr)
    if not runs:
        return {"n_runs": int(n_runs), "n_ok": 0,
                "jitter_pct": float(jitter_pct), "ensemble": True}
    # Référence taille e_fitted (1ère run convergée)
    e_ref = np.asarray(runs[0].get("e_fitted") or [], dtype=float)
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
    gc = _stack("gamma_corrige")
    if np.all(np.isnan(gc)):
        gc = _stack("gamma_brut")

    def _agg(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Médiane + std après filtrage MAD (>3·MAD) par (paire, slice)."""
        med = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - med[np.newaxis, ...]), axis=0)
        # MAD * 1.4826 ≈ σ pour distribution gaussienne
        sigma = 1.4826 * mad
        mask_ok = np.abs(stack - med[np.newaxis, ...]) <= 3.0 * sigma[np.newaxis, ...] + 1e-12
        filt = np.where(mask_ok, stack, np.nan)
        return np.nanmedian(filt, axis=0), np.nanstd(filt, axis=0)

    kF_minus_med, kF_minus_std = _agg(km)
    kF_plus_med, kF_plus_std = _agg(kp)
    gamma_med, gamma_std = _agg(gc)
    return {
        "ensemble": True,
        "n_runs": int(n_runs),
        "n_ok": int(len(runs)),
        "jitter_pct": float(jitter_pct),
        "e_fitted": e_ref.tolist(),
        "kF_minus_med": [row.tolist() for row in kF_minus_med],
        "kF_minus_std": [row.tolist() for row in kF_minus_std],
        "kF_plus_med": [row.tolist() for row in kF_plus_med],
        "kF_plus_std": [row.tolist() for row in kF_plus_std],
        "gamma_med": [row.tolist() for row in gamma_med],
        "gamma_std": [row.tolist() for row in gamma_std],
    }


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
    """JSON-safe : numpy scalaires, listes, dicts récursifs."""
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
            "scan_direction": fp.scan_direction,
            "width_mode": fp.width_mode,
            "k_min": fp.k_min,
            "k_max": fp.k_max,
            "k0_max": fp.k0_max,
            "dE_eV": fp.dE_meV / 1000.0,
            "dk_inv_a": fp.dk_inv_a,
            "resolution_source": resolution_source,
            "shape": getattr(fp, "shape", "lorentzian"),
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
        xg_mean = float(np.nanmean(fr.get("xg", [np.nan])))
        gamma_note = ""
        resolution_dominates = False
        if fr.get("gamma_brut") and fr.get("gamma_corrige"):
            gb = np.asarray(fr["gamma_brut"][0], dtype=float)
            gc = np.asarray(fr["gamma_corrige"][0], dtype=float)
            if np.isfinite(gb).any() and np.isfinite(gc).any():
                resolution_dominates = float(np.nanmedian(gc)) < 0.3 * float(np.nanmedian(gb))
                warn = " Attention" if resolution_dominates else ""
                gamma_note = (
                    f"\nΓ med = {float(np.nanmedian(gb)):.4f} brut / "
                    f"{float(np.nanmedian(gc)):.4f} corrigé{warn}"
                )
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
            f"{gamma_note}{vf_line}"
        )
        status_text = f"Fit OK — {n_ok}/{n_e}  xg={xg_mean:.4f}{gamma_note.replace(chr(10), ' | ')}"
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
