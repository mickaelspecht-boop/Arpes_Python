"""Pure energy-alignment helpers for DFT/ARPES overlays."""
from __future__ import annotations

from typing import Any

import numpy as np


def effective_mu_shift(config: Any) -> float:
    """Return physical chemical-potential shift in eV.

    New configs store ``mu_shift`` and use ``E_overlay = Z * (E_DFT - mu)``.
    Legacy configs stored an additive ``energy_shift`` where
    ``E_overlay = E_DFT + energy_shift``. For ``Z=1`` this corresponds to
    ``mu_shift = -energy_shift``.
    """
    mu = getattr(config, "mu_shift", None)
    if mu is not None:
        return float(mu)
    return -float(getattr(config, "energy_shift", 0.0) or 0.0)


def effective_z_scale(config: Any) -> float:
    z = float(getattr(config, "z_scale", 1.0) or 1.0)
    return z if np.isfinite(z) and z > 0 else 1.0


def apply_energy_transform(energies: Any, config: Any | None = None, *, mu_shift: float | None = None, z_scale: float | None = None):
    """Apply option-A ARPES/DFT energy transform.

    ``E_overlay = Z * (E_DFT_rel - mu_shift)``.
    """
    arr = np.asarray(energies, dtype=float)
    if config is not None:
        mu = effective_mu_shift(config)
        z = effective_z_scale(config)
    else:
        mu = float(mu_shift or 0.0)
        z = float(z_scale or 1.0)
    return z * (arr - mu)


def alignment_warnings(mu_shift: float, z_scale: float) -> list[str]:
    warnings: list[str] = []
    if abs(float(mu_shift)) > 0.3:
        warnings.append("|mu| > 0.3 eV: shift grand pour un semi-metal.")
    if float(z_scale) < 0.2 or float(z_scale) > 1.5:
        warnings.append("Z hors plage 0.2-1.5: renormalisation suspecte.")
    return warnings
