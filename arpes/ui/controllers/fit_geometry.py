"""Pure MDC fit-geometry preparation shared by the fit controller."""
from __future__ import annotations

from arpes.physics.mdc_geometry import symmetric_k0_ceiling


def geometry_warning(fp) -> str:
    """Explain invalid or clipped symmetric peak-pair geometry."""
    try:
        center = float(fp.center_init)
        lo, hi = sorted((float(fp.k_min), float(fp.k_max)))
    except Exception:
        return ""
    if not (lo <= center <= hi):
        return (
            f"⚠ centre Γ ({center:+.3f}) hors fenêtre k [{lo:+.2f}, {hi:+.2f}] : "
            "la paire symétrique ne peut pas tomber sur la bande — "
            "élargis la fenêtre k ou recale le centre."
        )
    ceiling = symmetric_k0_ceiling(lo, hi, center)
    too_wide = [
        abs(float(pp.get("kF_init", 0.30)))
        for pp in (getattr(fp, "pairs", None) or [])
        if abs(float(pp.get("kF_init", 0.30))) > ceiling
    ]
    if too_wide:
        return (
            f"⚠ |kF−centre|={max(too_wide):.3f} dépasse la plage symétrique "
            f"disponible ({ceiling:.3f}) dans k=[{lo:+.2f}, {hi:+.2f}] : "
            "le fit bornera kF au bord le plus proche."
        )
    return ""


def debug_mdc_kwargs(fp) -> dict:
    """Parameters making the one-slice diagnostic match the full fit."""
    pairs = list(getattr(fp, "pairs", None) or [])
    return {
        "n_pairs": fp.n_pairs,
        "smooth_fit": fp.smooth_fit,
        "smooth_detect": fp.smooth_detect,
        "gamma_init": [p.get("gamma_init", fp.gamma_init) for p in pairs]
        or fp.gamma_init,
        "gamma_max": [p.get("gamma_max", fp.gamma_max) for p in pairs]
        or fp.gamma_max,
        "kF_init": [p.get("kF_init", 0.30) for p in pairs] or None,
        "center_init": fp.center_init,
        "xg_range": fp.xg_range,
        "k_min": fp.k_min,
        "k_max": fp.k_max,
        "k0_max": fp.k0_max,
        "width_mode": fp.width_mode,
        "mdc_energy_window": fp.mdc_energy_window,
    }
