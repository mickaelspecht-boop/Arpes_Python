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
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


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
    def summarize(fr: dict) -> FitSummary:
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
        label_text = (
            f"OK  Fit complet  {n_ok}/{n_e} points\n"
            f"xg = {xg_mean:.4f} π/a"
            f"{gamma_note}"
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
