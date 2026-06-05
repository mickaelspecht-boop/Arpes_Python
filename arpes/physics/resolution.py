"""Estimation simple de la resolution instrumentale ARPES.

Les fichiers disponibles ne stockent pas directement la resolution energie/k.
On estime donc une FWHM energie depuis les parametres analyseur Scienta et une
FWHM k depuis le pas angulaire quand il existe, sinon un defaut conservateur.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

DEFAULT_DE_MEV = 15.0
DEFAULT_DK_INV_A = 0.005


def _positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _lens_defaults(lens_mode: str) -> tuple[float, float, str] | None:
    lm = (lens_mode or "").strip()
    lml = lm.lower()
    if lml.startswith("da30l") or "da30" in lml:
        return 200.0, 0.2, "DA30"
    if "angular30" in lml or "r8000" in lml:
        return 250.0, 0.3, "R8000"
    return None


def estimate_resolutions(meta: dict) -> dict:
    """Retourne dE_meV/dk_inv_a/source estimes depuis les metadata loader."""
    meta = meta or {}
    pe = _positive_float(meta.get("pass_energy_eV", meta.get("pass_energy")))
    lens_mode = str(meta.get("lens_mode") or meta.get("Lens Mode") or "")
    defaults = _lens_defaults(lens_mode)

    if pe is not None and defaults is not None:
        radius_mm, slit_mm, instrument = defaults
        dE_meV = pe * slit_mm / (2.0 * radius_mm) * 1000.0
        source = f"estime PE={pe:g} {instrument} slit~{slit_mm:g}mm"
    elif pe is not None:
        dE_meV = DEFAULT_DE_MEV
        source = f"defaut energie (lens inconnu, PE={pe:g})"
    else:
        dE_meV = DEFAULT_DE_MEV
        source = "defaut energie (PE absent)"

    angle_step = _positive_float(meta.get("angle_step_deg"))
    if angle_step is not None:
        hv = _positive_float(meta.get("hv"))
        work_func = _positive_float(meta.get("work_function_eV"))
        ef_kin = _positive_float(meta.get("ef_kinetic_from_hv"))
        if ef_kin is None and hv is not None:
            ef_kin = max(hv - (work_func or 4.5), 1e-9)
        if ef_kin is not None:
            a_lattice = _positive_float(meta.get("a_lattice"))
            dk_inv_a = 0.51233 * math.sqrt(ef_kin) * math.cos(0.0) * math.radians(angle_step) * a_lattice / math.pi
            source += f"; dk depuis angle_step={angle_step:g}deg"
        else:
            dk_inv_a = DEFAULT_DK_INV_A
            source += "; dk defaut (hv absent)"
    else:
        dk_inv_a = DEFAULT_DK_INV_A
        source += "; dk defaut"

    return {
        "dE_meV": float(dE_meV),
        "dk_inv_a": float(dk_inv_a),
        "source": source,
    }
