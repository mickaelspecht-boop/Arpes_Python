"""Kink analysis — electron-boson coupling λ from ARPES dispersion.

Pipeline:
1. Extract experimental dispersion E_exp(k) from MDC fits (positions of pic
   per energy slice).
2. Build a "bare band" E_bare(k) — parabolic or tight-binding.
3. Compute real part of self-energy Re Σ(E) = E_exp(k) - E_bare(k), evaluated
   on the experimental dispersion.
4. Compute imaginary part Im Σ(E) ≈ (v_bare / 2) · Γ_MDC(E)  (FWHM in k).
5. Extract coupling λ = -∂ReΣ/∂ω|_{ω=0}, where ω = E - E_F.

Conventions:
- Binding energy positive below E_F (ω = E - E_F, so ω < 0 means occupied).
- λ > 0 for electron-phonon (Migdal limit).
- v_bare in eV·Å, Γ in Å⁻¹, Im Σ in eV.

Physicist warnings (surface in UI):
- λ value depends on fit window — recommend testing 2 windows (e.g., 50 meV
  and 200 meV) and reporting sensitivity.
- Parabolic bare-band is a crude approximation; TB bare-band (from tb_fit
  result) is more rigorous when available.
- MDC fit invalid near band-bottom (vertical dispersion) or if 2 bands are
  unresolved — chi² check should flag.
- Kramers-Kronig consistency between Re Σ and Im Σ is *not* enforced here;
  optional post-validation in caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Bare-band models
# ---------------------------------------------------------------------------

def bare_parabolic(k: np.ndarray, k0: float, v_F: float, alpha: float) -> np.ndarray:
    """Linear + quadratic bare band: E_bare(k) = v_F (k - k0) + alpha (k - k0)²."""
    dk = k - k0
    return v_F * dk + alpha * dk * dk


def fit_bare_parabolic(
    k: np.ndarray,
    E: np.ndarray,
    *,
    window_eV: tuple[float, float] = (-0.3, -0.05),
    p0: tuple[float, float, float] | None = None,
) -> dict:
    """Fit linear+quadratic bare band on (k, E) restricted to a deep window
    where many-body renormalization is weak (deep below E_F).

    window_eV: (E_min, E_max) in binding convention (negative below E_F).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy.optimize required")
    k = np.asarray(k, float)
    E = np.asarray(E, float)
    mask = np.isfinite(k) & np.isfinite(E) & (E >= window_eV[0]) & (E <= window_eV[1])
    if mask.sum() < 4:
        raise ValueError(
            f"Bare-band fit window {window_eV} has <4 points; "
            "widen the deep-energy window."
        )
    kf, Ef = k[mask], E[mask]
    if p0 is None:
        k0_guess = float(np.mean(kf))
        vF_guess = float((Ef[-1] - Ef[0]) / (kf[-1] - kf[0] + 1e-9))
        p0 = (k0_guess, vF_guess, 0.0)
    popt, pcov = curve_fit(bare_parabolic, kf, Ef, p0=p0)
    return {
        "k0": float(popt[0]),
        "v_F": float(popt[1]),    # eV·Å (signed)
        "alpha": float(popt[2]),  # eV·Å²
        "perr": [float(v) for v in np.sqrt(np.diag(pcov))],
        "window_eV": tuple(window_eV),
        "n_points": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Self-energy
# ---------------------------------------------------------------------------

@dataclass
class KinkResult:
    E_exp: np.ndarray            # binding energies of dispersion samples (eV)
    k_exp: np.ndarray            # extracted momentum at each E (Å⁻¹)
    E_bare: np.ndarray           # bare-band energy at k_exp (eV)
    re_sigma: np.ndarray         # Re Σ(E) = E_exp - E_bare (eV)
    im_sigma: np.ndarray | None  # Im Σ(E) from Γ_MDC × v_bare/2 (eV)
    v_bare: float                # bare Fermi velocity (eV·Å) used for Im Σ
    lambda_coupling: float | None  # λ = -dReΣ/dω at ω=0
    lambda_err: float | None
    bare_model: str
    bare_params: dict
    notes: list[str] = field(default_factory=list)


def compute_self_energy(
    E_exp: np.ndarray,
    k_exp: np.ndarray,
    *,
    bare_fn: Callable[[np.ndarray], np.ndarray],
    bare_v_F: float,
    gamma_mdc: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute Re Σ and (optionally) Im Σ from experimental dispersion.

    Inputs:
        E_exp: binding energies (eV).
        k_exp: experimental k at each E (Å⁻¹), from MDC fit.
        bare_fn: callable k → E_bare(k).
        bare_v_F: bare Fermi velocity (eV·Å) at the relevant k_F.
        gamma_mdc: optional FWHM_k(E) (Å⁻¹) from MDC fit.

    Returns:
        (Re Σ, Im Σ or None) — both arrays aligned with E_exp.
    """
    E_exp = np.asarray(E_exp, float)
    k_exp = np.asarray(k_exp, float)
    E_bare_at_kexp = bare_fn(k_exp)
    re_sigma = E_exp - E_bare_at_kexp
    im_sigma = None
    if gamma_mdc is not None:
        g = np.asarray(gamma_mdc, float)
        im_sigma = 0.5 * abs(bare_v_F) * g
    return re_sigma, im_sigma


def extract_lambda(
    omega: np.ndarray,
    re_sigma: np.ndarray,
    *,
    window_eV: float = 0.05,
) -> tuple[float, float] | tuple[None, None]:
    """Extract λ = -∂ReΣ/∂ω at ω=0 via linear fit on |ω| < window_eV.

    ω = E - E_F (so ω < 0 occupied).
    """
    if not _HAS_SCIPY:
        return None, None
    omega = np.asarray(omega, float)
    re = np.asarray(re_sigma, float)
    mask = np.isfinite(omega) & np.isfinite(re) & (np.abs(omega) <= window_eV)
    if mask.sum() < 3:
        return None, None

    def _lin(w, a, b):
        return a + b * w

    popt, pcov = curve_fit(_lin, omega[mask], re[mask])
    slope = float(popt[1])
    slope_err = float(np.sqrt(np.diag(pcov))[1])
    return -slope, slope_err


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def run_kink_analysis(
    E_exp: np.ndarray,
    k_exp: np.ndarray,
    *,
    E_F: float = 0.0,
    bare: str = "parabolic",
    bare_window_eV: tuple[float, float] = (-0.3, -0.08),
    bare_model_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    bare_v_F: float | None = None,
    gamma_mdc: np.ndarray | None = None,
    lambda_window_eV: float = 0.05,
) -> KinkResult:
    """Full kink pipeline.

    Inputs:
        E_exp, k_exp: extracted dispersion (E in binding convention, k in Å⁻¹).
        E_F: Fermi level offset on the E axis (default 0 if already referenced).
        bare: "parabolic" (fit on bare_window_eV) or "custom" (use bare_model_fn).
        bare_model_fn: optional callable k → E_bare(k), required if bare="custom".
        bare_v_F: optional explicit v_F for Im Σ; if None and bare="parabolic",
                  taken from the parabolic fit.
        gamma_mdc: optional Γ_k(E) FWHM for Im Σ.
        lambda_window_eV: half-width around E_F for λ slope fit.
    """
    E_exp = np.asarray(E_exp, float)
    k_exp = np.asarray(k_exp, float)
    notes: list[str] = []

    if bare == "parabolic":
        fitp = fit_bare_parabolic(k_exp, E_exp, window_eV=bare_window_eV)

        def _bare_fn(k):
            return bare_parabolic(k, fitp["k0"], fitp["v_F"], fitp["alpha"])

        bare_fn = _bare_fn
        bare_params = fitp
        v_F_used = bare_v_F if bare_v_F is not None else fitp["v_F"]
    elif bare == "custom":
        if bare_model_fn is None:
            raise ValueError("bare='custom' requires bare_model_fn")
        if bare_v_F is None:
            raise ValueError("bare='custom' requires bare_v_F")
        bare_fn = bare_model_fn
        bare_params = {}
        v_F_used = bare_v_F
    else:
        raise ValueError(f"Unknown bare={bare!r}")

    re, im = compute_self_energy(
        E_exp, k_exp, bare_fn=bare_fn, bare_v_F=v_F_used, gamma_mdc=gamma_mdc
    )
    omega = E_exp - E_F
    lam, lam_err = extract_lambda(omega, re, window_eV=lambda_window_eV)

    if lam is None:
        notes.append("λ extraction failed — too few points within window.")
    elif lam < 0:
        notes.append("λ < 0 — unphysical for electron-boson coupling; "
                     "check bare-band window and dispersion sign.")
    if E_exp.ptp() < 0.05:
        notes.append("Dispersion span <50 meV — λ estimate fragile, widen range.")

    return KinkResult(
        E_exp=E_exp,
        k_exp=k_exp,
        E_bare=bare_fn(k_exp),
        re_sigma=re,
        im_sigma=im,
        v_bare=float(v_F_used),
        lambda_coupling=lam,
        lambda_err=lam_err,
        bare_model=bare,
        bare_params=bare_params,
        notes=notes,
    )


def dispersion_from_mdc_peaks(
    mdc_fit_payload: list[dict],
    *,
    branch_key: str = "kF_minus",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Helper: extract (E, k, Γ_FWHM) arrays from a list of MDC fit results.

    Each entry must contain:
        "E": float (binding energy)
        branch_key: float (k position in Å⁻¹)
        optionally "gamma_<branch>": float (FWHM in Å⁻¹)
    """
    Es, ks, gs = [], [], []
    gkey = "gamma_" + branch_key.replace("kF_", "")
    for entry in mdc_fit_payload:
        E = entry.get("E")
        k = entry.get(branch_key)
        if E is None or k is None:
            continue
        Es.append(float(E))
        ks.append(float(k))
        gs.append(float(entry.get(gkey, np.nan)))
    Es = np.array(Es)
    ks = np.array(ks)
    gs = np.array(gs)
    order = np.argsort(Es)
    Es, ks, gs = Es[order], ks[order], gs[order]
    if not np.any(np.isfinite(gs)):
        gs = None
    return Es, ks, gs
