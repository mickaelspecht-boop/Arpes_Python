"""Gap extraction — Δ from EDC at k_F via symmetrization + Dynes fit.

Works for: superconducting gap (BCS s, s±, d-wave), CDW gap, pseudogap,
topological surface gap, etc.

Method:
1. Take EDC I(E) at a chosen k_F.
2. Symmetrize:  I_sym(ω) = I(ω) + I(-ω),  ω = E - E_F.
   Eliminates Fermi-Dirac cutoff if particle-hole symmetric locally.
3. Fit Dynes function:
      I_D(ω) = Re[(ω - i Γ) / sqrt((ω - i Γ)² - Δ²)]
   where Γ is the pair-breaking (broadening) parameter.
4. Optionally convolve with instrumental energy resolution (Gaussian).
5. For multi-gap (s± pnictides): sum of two Dynes spectra weighted.

Conventions:
- ω in meV (positive above E_F).
- Δ in meV, Γ in meV (Γ ≥ 0).
- Δ must be extracted STRICTLY at k_F — back-bending of Bogoliubov quasiparticle
  dispersion away from k_F leads to systematic overestimation of Δ.

Physicist warnings (surface in UI):
- Symmetrization assumes particle-hole symmetry locally → valid at k_F, breaks
  far from k_F.
- Matrix-element effects may suppress EDC intensity at k_F without affecting Δ.
- If Γ > Δ, gap is filled (pseudogap-like) — extraction fragile.
- Instrumental resolution must be convolved; otherwise Γ absorbs it and
  overestimates pair-breaking.
- For pnictides (s± multi-gap), allow Δ per pocket (not a single global Δ).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.signal import fftconvolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Dynes model
# ---------------------------------------------------------------------------

def dynes(omega: np.ndarray, Delta: float, Gamma: float) -> np.ndarray:
    """Dynes quasiparticle DOS ``N(ω)=Re[(ω-iΓ)/√((ω-iΓ)²-Δ²)]``.

    Observable: *tunneling* DOS (STS/jonction), pas la fonction spectrale
    ARPES (pour un EDC symétrisé à k_F utiliser ``norman_spectral``).

    P2.5 — l'ancien ``|Re[…]|`` (magnitude) déformait la forme de raie. On
    retire l'``abs`` et on fixe la branche de la racine : Im(denom) ≤ 0 (même
    signe que Im(ω-iΓ)=−Γ) → N(ω) ≥ 0 partout sans hack de magnitude
    (le naïf ``Re`` seul donne N<0 dans le gap à cause de la branche
    principale de ``np.sqrt``).
    """
    w = np.asarray(omega, complex)
    denom = np.sqrt((w - 1j * Gamma) ** 2 - Delta ** 2)
    denom = np.where(np.imag(denom) > 0.0, -denom, denom)  # force Im(denom) ≤ 0
    return np.real((w - 1j * Gamma) / denom)


def norman_spectral(
    omega: np.ndarray, Delta: float, Gamma1: float, Gamma0: float = 1.0
) -> np.ndarray:
    """Fonction spectrale ARPES à k_F — Norman et al. PRB 57, R11093 (1998).

    Σ(ω) = −iΓ₁ + Δ²/(ω + iΓ₀) ; A(k_F,ω) = −(1/π)·Im G, G=1/(ω−Σ).
    Définie-positive partout (pas d'``abs``). ``Γ₀`` régularise la
    singularité Σ'=Δ²/ω en ω=0 (fixer ≳ résolution, jamais 0).

    Limites : Δ→0 → lorentzienne unique en ω=0 ; Γ→0 → deux pics en ±Δ.
    """
    w = np.asarray(omega, dtype=float)
    w2 = w * w + float(Gamma0) ** 2
    # ReΣ = Δ²ω/(ω²+Γ₀²) → ω−ReΣ = ω(1−Δ²/(ω²+Γ₀²)) : zéros en ±√(Δ²−Γ₀²)≈±Δ.
    re_denom = w - (Delta ** 2) * w / w2          # ω − ReΣ
    im_denom = float(Gamma1) + (Delta ** 2) * float(Gamma0) / w2  # −ImΣ > 0
    return im_denom / (np.pi * (re_denom ** 2 + im_denom ** 2))


def norman_multi(
    omega: np.ndarray,
    deltas: list[float],
    gammas: list[float],
    weights: list[float],
    gamma0: float = 1.0,
) -> np.ndarray:
    """Weighted sum of Norman spectral functions (multi-gap s+-)."""
    omega = np.asarray(omega, float)
    out = np.zeros_like(omega)
    norm = sum(weights) or 1.0
    for D, G, w in zip(deltas, gammas, weights):
        out = out + (w / norm) * norman_spectral(omega, D, G, gamma0)
    return out


def dynes_multi(
    omega: np.ndarray,
    deltas: list[float],
    gammas: list[float],
    weights: list[float],
) -> np.ndarray:
    """Sum of Dynes spectra (multi-gap s±-like)."""
    omega = np.asarray(omega, float)
    out = np.zeros_like(omega)
    norm = sum(weights) or 1.0
    for D, G, w in zip(deltas, gammas, weights):
        out = out + (w / norm) * dynes(omega, D, G)
    return out


def gaussian_kernel(
    omega: np.ndarray,
    fwhm_meV: float,
) -> np.ndarray:
    """Normalized Gaussian kernel on the same grid as omega."""
    sigma = fwhm_meV / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    g = np.exp(-(omega ** 2) / (2.0 * sigma * sigma))
    return g / g.sum()


def convolve_resolution(
    spectrum: np.ndarray,
    omega: np.ndarray,
    resolution_meV: float,
) -> np.ndarray:
    """Convolve spectrum with Gaussian of FWHM resolution_meV (in meV)."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy.signal.fftconvolve required")
    if resolution_meV <= 0:
        return spectrum
    # Build kernel centered on 0 on a symmetric grid matching omega spacing
    dw = float(np.mean(np.diff(omega)))
    half = max(1, int(round(3.0 * resolution_meV / abs(dw))))
    grid = np.arange(-half, half + 1) * dw
    k = gaussian_kernel(grid, resolution_meV)
    return fftconvolve(spectrum, k, mode="same")


# ---------------------------------------------------------------------------
# Symmetrization
# ---------------------------------------------------------------------------

def symmetrize_edc(
    E: np.ndarray,
    I: np.ndarray,
    *,
    E_F: float = 0.0,
    omega_max_meV: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetrized EDC I_sym(ω) = I(ω) + I(-ω).

    Inputs:
        E: energy axis (eV).
        I: intensity (arbitrary units).
        E_F: Fermi level on E axis (eV).
        omega_max_meV: half-window around E_F (meV).

    Returns: (omega_meV, I_sym) on a symmetric grid.
    """
    E = np.asarray(E, float)
    I = np.asarray(I, float)
    if E.shape != I.shape:
        raise ValueError(f"E and I shapes differ: {E.shape} vs {I.shape}")
    omega_eV = E - E_F
    mask = np.abs(omega_eV) <= (omega_max_meV * 1e-3) * 1.05
    if mask.sum() < 6:
        raise ValueError("Too few points in symmetrization window")
    w = omega_eV[mask]
    y = I[mask]
    order = np.argsort(w)
    w, y = w[order], y[order]
    # Interpolate -ω onto the same grid as ω
    y_neg = np.interp(-w, w, y, left=np.nan, right=np.nan)
    sym = y + y_neg
    good = np.isfinite(sym)
    return w[good] * 1e3, sym[good]   # ω in meV


# ---------------------------------------------------------------------------
# Fit drivers
# ---------------------------------------------------------------------------

@dataclass
class GapFitResult:
    omega_meV: np.ndarray
    I_sym: np.ndarray
    I_fit: np.ndarray
    deltas_meV: list[float]
    gammas_meV: list[float]
    weights: list[float]
    delta_err_meV: list[float]
    n_gaps: int
    resolution_meV: float
    chi2_red: float
    notes: list[str] = field(default_factory=list)


def fit_dynes_single(
    omega_meV: np.ndarray,
    I_sym: np.ndarray,
    *,
    resolution_meV: float = 0.0,
    Delta_guess_meV: float = 5.0,
    Gamma_guess_meV: float = 1.0,
    amplitude_guess: float | None = None,
) -> GapFitResult:
    """Fit single-gap Dynes (with optional amplitude + offset) to I_sym."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required")
    omega_meV = np.asarray(omega_meV, float)
    I_sym = np.asarray(I_sym, float)
    A0 = amplitude_guess if amplitude_guess is not None else float(np.max(I_sym))
    bg0 = float(np.median(I_sym[: max(1, len(I_sym) // 8)]))

    def _model(w, D, G, A, B):
        spec = A * dynes(w, abs(D), abs(G)) + B
        if resolution_meV > 0:
            spec = convolve_resolution(spec, w, resolution_meV)
        return spec

    p0 = (Delta_guess_meV, Gamma_guess_meV, A0, bg0)
    # P2.5 — borne Δ 50→100 meV (régime pseudogap, Bi2212 sous-dopé).
    bounds = ([0.05, 0.0, 0.0, -np.inf], [100.0, 20.0, 10.0 * A0 + 1e-6, np.inf])
    popt, pcov = curve_fit(_model, omega_meV, I_sym, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    D, G, A, B = popt
    I_fit = _model(omega_meV, *popt)
    residuals = I_sym - I_fit
    dof = max(1, len(I_sym) - len(popt))
    chi2 = float(np.sum(residuals ** 2) / dof)

    notes: list[str] = []
    if abs(G) > abs(D):
        notes.append("Γ > Δ — gap filled, extraction fragile (pseudogap regime).")
    if resolution_meV == 0:
        notes.append("No instrumental resolution convolved — Γ overestimated.")

    return GapFitResult(
        omega_meV=omega_meV,
        I_sym=I_sym,
        I_fit=I_fit,
        deltas_meV=[abs(float(D))],
        gammas_meV=[abs(float(G))],
        weights=[1.0],
        delta_err_meV=[float(perr[0])],
        n_gaps=1,
        resolution_meV=float(resolution_meV),
        chi2_red=chi2,
        notes=notes,
    )


def fit_dynes_two_gap(
    omega_meV: np.ndarray,
    I_sym: np.ndarray,
    *,
    resolution_meV: float = 0.0,
    D1_guess_meV: float = 3.0,
    D2_guess_meV: float = 8.0,
    Gamma_guess_meV: float = 1.0,
) -> GapFitResult:
    """Fit 2-gap (s±-like) Dynes."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required")
    A0 = float(np.max(I_sym))
    bg0 = float(np.median(I_sym[: max(1, len(I_sym) // 8)]))

    def _model(w, D1, D2, G1, G2, w1, A, B):
        spec = A * dynes_multi(w, [abs(D1), abs(D2)],
                               [abs(G1), abs(G2)],
                               [abs(w1), max(0.0, 1.0 - abs(w1))]) + B
        if resolution_meV > 0:
            spec = convolve_resolution(spec, w, resolution_meV)
        return spec

    p0 = (D1_guess_meV, D2_guess_meV, Gamma_guess_meV, Gamma_guess_meV, 0.5, A0, bg0)
    # P2.5 — borne Δ 50→100 meV (pseudogap).
    bounds = ([0.05, 0.05, 0.0, 0.0, 0.0, 0.0, -np.inf],
              [100.0, 100.0, 20.0, 20.0, 1.0, 10.0 * A0 + 1e-6, np.inf])
    popt, pcov = curve_fit(_model, omega_meV, I_sym, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    D1, D2, G1, G2, w1, A, B = popt
    w2 = max(0.0, 1.0 - abs(w1))
    I_fit = _model(omega_meV, *popt)
    chi2 = float(np.sum((I_sym - I_fit) ** 2) / max(1, len(I_sym) - len(popt)))

    notes = []
    if abs(abs(D1) - abs(D2)) < 0.1:
        notes.append("Δ₁ ≈ Δ₂ — 2-gap fit indistinguishable from 1-gap.")

    return GapFitResult(
        omega_meV=omega_meV,
        I_sym=I_sym,
        I_fit=I_fit,
        deltas_meV=[abs(float(D1)), abs(float(D2))],
        gammas_meV=[abs(float(G1)), abs(float(G2))],
        weights=[abs(float(w1)), w2],
        delta_err_meV=[float(perr[0]), float(perr[1])],
        n_gaps=2,
        resolution_meV=float(resolution_meV),
        chi2_red=chi2,
        notes=notes,
    )


def _gamma0_for(resolution_meV: float) -> float:
    """Norman Gamma0 regularization: near resolution scale, never 0."""
    return float(max(1.0, 0.3 * float(resolution_meV)))


def fit_norman_single(
    omega_meV: np.ndarray,
    I_sym: np.ndarray,
    *,
    resolution_meV: float = 0.0,
    Delta_guess_meV: float = 5.0,
    Gamma_guess_meV: float = 1.0,
    amplitude_guess: float | None = None,
) -> GapFitResult:
    """Fit gap ARPES sur EDC symétrisé via fonction spectrale Norman (1998).

    Modèle physiquement correct pour un EDC ARPES symétrisé à k_F (vs Dynes
    qui est une DOS tunnel). Γ₀ fixé (régularisation), params libres
    (Δ, Γ₁, A, B).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required")
    omega_meV = np.asarray(omega_meV, float)
    I_sym = np.asarray(I_sym, float)
    A0 = amplitude_guess if amplitude_guess is not None else float(np.max(I_sym))
    bg0 = float(np.median(I_sym[: max(1, len(I_sym) // 8)]))
    g0 = _gamma0_for(resolution_meV)

    def _model(w, D, G, A, B):
        spec = A * norman_spectral(w, abs(D), abs(G), g0) + B
        if resolution_meV > 0:
            spec = convolve_resolution(spec, w, resolution_meV)
        return spec

    p0 = (Delta_guess_meV, Gamma_guess_meV, A0, bg0)
    bounds = ([0.05, 0.0, 0.0, -np.inf], [100.0, 50.0, 10.0 * A0 + 1e-6, np.inf])
    popt, pcov = curve_fit(_model, omega_meV, I_sym, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    D, G, A, B = popt
    I_fit = _model(omega_meV, *popt)
    dof = max(1, len(I_sym) - len(popt))
    chi2 = float(np.sum((I_sym - I_fit) ** 2) / dof)

    notes = [f"Norman PRB 57 R11093 (1998), Γ₀={g0:.1f} meV fixed."]
    if abs(D) > 1e-9 and abs(G) / abs(D) > 0.5:
        notes.append("Γ₁/Δ > 0.5 - merged peaks, Δ unreliable (filled gap).")
    if abs(D) >= 99.0:
        notes.append("Δ at the 100 meV bound - check possible gap/phonon-kink confusion (~60-70 meV).")
    if resolution_meV == 0:
        notes.append("No convoluted resolution - Γ overestimated.")

    return GapFitResult(
        omega_meV=omega_meV, I_sym=I_sym, I_fit=I_fit,
        deltas_meV=[abs(float(D))], gammas_meV=[abs(float(G))], weights=[1.0],
        delta_err_meV=[float(perr[0])], n_gaps=1,
        resolution_meV=float(resolution_meV), chi2_red=chi2, notes=notes,
    )


def fit_norman_two_gap(
    omega_meV: np.ndarray,
    I_sym: np.ndarray,
    *,
    resolution_meV: float = 0.0,
    D1_guess_meV: float = 3.0,
    D2_guess_meV: float = 8.0,
    Gamma_guess_meV: float = 1.0,
) -> GapFitResult:
    """Fit 2-gap (s±) ARPES via somme de fonctions spectrales Norman."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required")
    omega_meV = np.asarray(omega_meV, float)
    I_sym = np.asarray(I_sym, float)
    A0 = float(np.max(I_sym))
    bg0 = float(np.median(I_sym[: max(1, len(I_sym) // 8)]))
    g0 = _gamma0_for(resolution_meV)

    def _model(w, D1, D2, G1, G2, w1, A, B):
        spec = A * norman_multi(w, [abs(D1), abs(D2)], [abs(G1), abs(G2)],
                                [abs(w1), max(0.0, 1.0 - abs(w1))], g0) + B
        if resolution_meV > 0:
            spec = convolve_resolution(spec, w, resolution_meV)
        return spec

    p0 = (D1_guess_meV, D2_guess_meV, Gamma_guess_meV, Gamma_guess_meV, 0.5, A0, bg0)
    bounds = ([0.05, 0.05, 0.0, 0.0, 0.0, 0.0, -np.inf],
              [100.0, 100.0, 50.0, 50.0, 1.0, 10.0 * A0 + 1e-6, np.inf])
    popt, pcov = curve_fit(_model, omega_meV, I_sym, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    D1, D2, G1, G2, w1, A, B = popt
    w2 = max(0.0, 1.0 - abs(w1))
    I_fit = _model(omega_meV, *popt)
    chi2 = float(np.sum((I_sym - I_fit) ** 2) / max(1, len(I_sym) - len(popt)))

    notes = [f"Norman PRB 57 R11093 (1998), Γ₀={g0:.1f} meV fixed."]
    if abs(abs(D1) - abs(D2)) < 0.1:
        notes.append("Δ₁ ≈ Δ₂ - 2-gap model indistinguishable from 1-gap.")

    return GapFitResult(
        omega_meV=omega_meV, I_sym=I_sym, I_fit=I_fit,
        deltas_meV=[abs(float(D1)), abs(float(D2))],
        gammas_meV=[abs(float(G1)), abs(float(G2))],
        weights=[abs(float(w1)), w2],
        delta_err_meV=[float(perr[0]), float(perr[1])], n_gaps=2,
        resolution_meV=float(resolution_meV), chi2_red=chi2, notes=notes,
    )


# ---------------------------------------------------------------------------
# k_F scan (Δ(angle) for symmetry of pairing)
# ---------------------------------------------------------------------------

def scan_gap_over_kf(
    edcs: list[dict],
    *,
    resolution_meV: float = 0.0,
    omega_max_meV: float = 30.0,
    n_gaps: int = 1,
) -> dict:
    """Iterate gap extraction over a list of EDCs at different k_F.

    Each item: {"angle_deg": float, "E": np.ndarray, "I": np.ndarray, "E_F": float}.
    Returns dict with arrays of angle, Δ (and Δ₂ if n_gaps=2), errors.
    """
    angles, Ds, Errs = [], [], []
    Ds2, Errs2 = [], []
    for item in edcs:
        try:
            w, sym = symmetrize_edc(
                item["E"], item["I"],
                E_F=item.get("E_F", 0.0),
                omega_max_meV=omega_max_meV,
            )
            if n_gaps == 1:
                r = fit_dynes_single(w, sym, resolution_meV=resolution_meV)
                Ds.append(r.deltas_meV[0])
                Errs.append(r.delta_err_meV[0])
            else:
                r = fit_dynes_two_gap(w, sym, resolution_meV=resolution_meV)
                Ds.append(r.deltas_meV[0])
                Errs.append(r.delta_err_meV[0])
                Ds2.append(r.deltas_meV[1])
                Errs2.append(r.delta_err_meV[1])
            angles.append(float(item.get("angle_deg", np.nan)))
        except (ValueError, RuntimeError):
            continue
    out = {
        "angle_deg": np.array(angles),
        "delta_meV": np.array(Ds),
        "delta_err_meV": np.array(Errs),
    }
    if n_gaps == 2:
        out["delta2_meV"] = np.array(Ds2)
        out["delta2_err_meV"] = np.array(Errs2)
    return out
