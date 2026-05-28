"""Tight-binding fit of ARPES dispersion E(k) — pure numpy/scipy, no Qt.

Generic for any periodic crystal (cuprate, pnictide, TMD, graphene, kagome).
Lattice geometry parameterized via ``lattice_type``.

Conventions:
- k in inverse Angstrom (Å⁻¹).
- E in eV, binding energy convention (positive below E_F).
- Hopping t > 0 by convention (bandwidth W = 8t for square 2D nearest-neighbor).
- t' < 0 typical for cuprate-like d-orbital (t'/t ≈ -0.3).
- Effective mass m*/m computed at band bottom (parabolic limit only).

Physicist warnings (must surface in UI):
- Multi-band materials (pnictides, TMDs) require *per-pocket* fits — a single
  1-band global fit is physically meaningless when multiple bands cross E_F.
- Extracted t are *effective* (phenomenological), NOT directly comparable to
  ab-initio hoppings without Wannier downfolding.
- Folding (e.g., 2 Fe per unit cell in pnictides) requires explicit choice
  of crystalline vs unfolded BZ — caller responsibility.
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


HBAR_EV_S = 6.582119569e-16  # eV·s
ME_KG = 9.1093837015e-31     # kg
HBAR2_OVER_2M_EVA2 = 3.80998   # ℏ²/(2 m_e) in eV·Å²


# ---------------------------------------------------------------------------
# TB models (k in Å⁻¹, lattice in Å)
# ---------------------------------------------------------------------------

def tb_1d_chain(k: np.ndarray, eps0: float, t: float, a: float) -> np.ndarray:
    """1D nearest-neighbor chain: E = eps0 - 2 t cos(k a)."""
    return eps0 - 2.0 * t * np.cos(k * a)


def tb_2d_square(
    kx: np.ndarray,
    ky: np.ndarray,
    eps0: float,
    t: float,
    tp: float,
    tpp: float,
    a: float,
) -> np.ndarray:
    """2D square lattice up to 3rd nearest neighbor.

    E(k) = eps0
           - 2 t  [cos(kx a) + cos(ky a)]
           - 4 t' cos(kx a) cos(ky a)
           - 2 t''[cos(2 kx a) + cos(2 ky a)]
    """
    ka, la = kx * a, ky * a
    return (
        eps0
        - 2.0 * t * (np.cos(ka) + np.cos(la))
        - 4.0 * tp * np.cos(ka) * np.cos(la)
        - 2.0 * tpp * (np.cos(2.0 * ka) + np.cos(2.0 * la))
    )


def tb_2d_hex(
    kx: np.ndarray,
    ky: np.ndarray,
    eps0: float,
    t: float,
    tp: float,
    a: float,
) -> np.ndarray:
    """2D hexagonal lattice (triangular Bravais), 1st + 2nd neighbor.

    E(k) = eps0
           - 2 t  [cos(k·a1) + cos(k·a2) + cos(k·a3)]
           - 2 t' [cos(k·b1) + cos(k·b2) + cos(k·b3)]
    """
    s3 = np.sqrt(3.0) / 2.0
    a1k = kx * a
    a2k = 0.5 * kx * a + s3 * ky * a
    a3k = -0.5 * kx * a + s3 * ky * a
    nn = np.cos(a1k) + np.cos(a2k) + np.cos(a3k)
    b1k = s3 * kx * a + 1.5 * ky * a
    b2k = -s3 * kx * a + 1.5 * ky * a
    b3k = np.sqrt(3.0) * kx * a
    nnn = np.cos(b1k) + np.cos(b2k) + np.cos(b3k)
    return eps0 - 2.0 * t * nn - 2.0 * tp * nnn


def tb_2d_rect(
    kx: np.ndarray,
    ky: np.ndarray,
    eps0: float,
    tx: float,
    ty: float,
    tp: float,
    a: float,
    b: float,
) -> np.ndarray:
    """2D rectangular lattice with anisotropic hoppings."""
    return (
        eps0
        - 2.0 * tx * np.cos(kx * a)
        - 2.0 * ty * np.cos(ky * b)
        - 4.0 * tp * np.cos(kx * a) * np.cos(ky * b)
    )


# ---------------------------------------------------------------------------
# Fit results
# ---------------------------------------------------------------------------

@dataclass
class TBFitResult:
    model: str                       # "1d_chain" | "2d_square" | "2d_hex" | "2d_rect"
    lattice_type: str                # "square" | "hex" | "rect" | "chain"
    params: dict[str, float]         # fitted parameters (eps0, t, t', ...)
    perr: dict[str, float]           # 1-sigma errors
    a: float                         # lattice parameter (Å)
    b: float | None = None           # second lattice parameter for "rect"
    m_eff_over_me: float | None = None   # effective mass at band bottom
    bandwidth_eV: float | None = None    # W = E_max - E_min over fit window
    chi2_red: float | None = None
    n_points: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fit drivers
# ---------------------------------------------------------------------------

def fit_dispersion_1d(
    k: np.ndarray,
    E: np.ndarray,
    a: float,
    *,
    p0: tuple[float, float] | None = None,
    bounds: tuple[tuple, tuple] | None = None,
    sigma: np.ndarray | None = None,
) -> TBFitResult:
    """Fit 1D chain model on extracted dispersion E(k).

    Inputs:
        k: array of momenta (Å⁻¹).
        E: array of binding energies at each k (eV, positive below E_F).
        a: lattice parameter (Å).
        p0: optional (eps0, t) initial guess.
        bounds: optional ((eps0_lo, t_lo), (eps0_hi, t_hi)).
        sigma: optional 1-sigma uncertainty on E.

    Returns: TBFitResult with params, errors, m_eff.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy.optimize required for TB fit")
    k = np.asarray(k, float)
    E = np.asarray(E, float)
    mask = np.isfinite(k) & np.isfinite(E)
    if mask.sum() < 3:
        raise ValueError("Need at least 3 finite (k, E) points")
    kf, Ef = k[mask], E[mask]
    sf = None if sigma is None else np.asarray(sigma, float)[mask]

    if p0 is None:
        eps0_guess = float(np.median(Ef))
        t_guess = max(1e-3, 0.25 * float(np.ptp(Ef)))
        p0 = (eps0_guess, t_guess)
    if bounds is None:
        bounds = ((-10.0, 1e-4), (10.0, 5.0))

    def _model(kk, eps0, t):
        return tb_1d_chain(kk, eps0, t, a)

    popt, pcov = curve_fit(_model, kf, Ef, p0=p0, bounds=bounds, sigma=sf,
                           absolute_sigma=sf is not None)
    perr = np.sqrt(np.diag(pcov))
    eps0, t = popt
    # m* at band bottom: E ≈ eps0 - 2t + t a² k² → m* = ℏ²/(2 t a²)
    m_eff = HBAR2_OVER_2M_EVA2 / (t * a * a) if t > 0 else None
    residuals = Ef - _model(kf, *popt)
    dof = max(1, len(Ef) - len(popt))
    chi2 = float(np.sum(residuals ** 2) / dof)

    return TBFitResult(
        model="1d_chain",
        lattice_type="chain",
        params={"eps0": float(eps0), "t": float(t)},
        perr={"eps0": float(perr[0]), "t": float(perr[1])},
        a=float(a),
        m_eff_over_me=m_eff,
        bandwidth_eV=4.0 * float(t),
        chi2_red=chi2,
        n_points=int(len(Ef)),
    )


def fit_dispersion_2d(
    kx: np.ndarray,
    ky: np.ndarray,
    E: np.ndarray,
    a: float,
    *,
    lattice_type: str = "square",
    b: float | None = None,
    p0: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    sigma: np.ndarray | None = None,
) -> TBFitResult:
    """Fit 2D TB model (square / hex / rect) on dispersion E(kx, ky).

    Inputs are flat arrays of equal length (one point per measurement).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy.optimize required for TB fit")
    kx = np.asarray(kx, float)
    ky = np.asarray(ky, float)
    E = np.asarray(E, float)
    mask = np.isfinite(kx) & np.isfinite(ky) & np.isfinite(E)
    if mask.sum() < 4:
        raise ValueError("Need at least 4 finite (kx, ky, E) points")
    kxf, kyf, Ef = kx[mask], ky[mask], E[mask]
    sf = None if sigma is None else np.asarray(sigma, float)[mask]

    lt = (lattice_type or "square").lower()
    notes: list[str] = []

    if lt == "square":
        names = ["eps0", "t", "tp", "tpp"]
        p0_def = {"eps0": float(np.median(Ef)), "t": 0.3, "tp": -0.1, "tpp": 0.0}
        bounds_def = {"eps0": (-10.0, 10.0), "t": (1e-4, 5.0),
                      "tp": (-2.0, 2.0), "tpp": (-1.0, 1.0)}
        def _model(K, eps0, t, tp, tpp):
            return tb_2d_square(K[0], K[1], eps0, t, tp, tpp, a)
    elif lt == "hex":
        names = ["eps0", "t", "tp"]
        p0_def = {"eps0": float(np.median(Ef)), "t": 0.3, "tp": -0.05}
        bounds_def = {"eps0": (-10.0, 10.0), "t": (1e-4, 5.0), "tp": (-2.0, 2.0)}
        def _model(K, eps0, t, tp):
            return tb_2d_hex(K[0], K[1], eps0, t, tp, a)
    elif lt == "rect":
        if b is None:
            raise ValueError("lattice_type='rect' requires parameter b")
        names = ["eps0", "tx", "ty", "tp"]
        p0_def = {"eps0": float(np.median(Ef)), "tx": 0.3, "ty": 0.3, "tp": -0.05}
        bounds_def = {"eps0": (-10.0, 10.0), "tx": (1e-4, 5.0),
                      "ty": (1e-4, 5.0), "tp": (-2.0, 2.0)}
        def _model(K, eps0, tx, ty, tp):
            return tb_2d_rect(K[0], K[1], eps0, tx, ty, tp, a, b)
    else:
        raise ValueError(f"Unknown lattice_type={lattice_type!r}")

    p0_used = {**p0_def, **(p0 or {})}
    bnd = {**bounds_def, **(bounds or {})}
    p0_vec = [p0_used[n] for n in names]
    lo = [bnd[n][0] for n in names]
    hi = [bnd[n][1] for n in names]

    K = np.vstack([kxf, kyf])
    popt, pcov = curve_fit(_model, K, Ef, p0=p0_vec, bounds=(lo, hi),
                           sigma=sf, absolute_sigma=sf is not None)
    perr = np.sqrt(np.diag(pcov))
    params = dict(zip(names, [float(v) for v in popt]))
    perrs = dict(zip(names, [float(v) for v in perr]))

    # m* at band bottom (Γ): expand cosines → m* = ℏ²/(2 t_eff a²) where t_eff
    # is the curvature along principal axis at k=0.
    t_eff = None
    if lt == "square":
        # ∂²E/∂kx² at Γ = 2 t a² + 8 t' a² + 8 t'' a² → curvature coefficient
        t_eff = params["t"] + 2.0 * params["tp"] + 4.0 * params["tpp"]
    elif lt == "hex":
        t_eff = 1.5 * params["t"] + 4.5 * params["tp"]
    elif lt == "rect":
        t_eff = 0.5 * (params["tx"] + params["ty"]) + params["tp"]
    m_eff = (HBAR2_OVER_2M_EVA2 / (t_eff * a * a)) if (t_eff and t_eff > 0) else None
    if t_eff is not None and t_eff <= 0:
        notes.append("Curvature at Γ non-positive — band is hole-like or saddle.")

    residuals = Ef - _model(K, *popt)
    dof = max(1, len(Ef) - len(popt))
    chi2 = float(np.sum(residuals ** 2) / dof)

    # Multi-band sanity flag: dispersion span vs single-band bandwidth
    if lt == "square":
        W = 8.0 * params["t"]
        if np.ptp(Ef) > 1.5 * W:
            notes.append(
                "Dispersion span exceeds single-band bandwidth 1.5×W — "
                "likely multi-band data; fit per pocket recommended."
            )

    return TBFitResult(
        model=f"2d_{lt}",
        lattice_type=lt,
        params=params,
        perr=perrs,
        a=float(a),
        b=float(b) if b is not None else None,
        m_eff_over_me=m_eff,
        bandwidth_eV=(8.0 * params["t"]) if lt == "square" else None,
        chi2_red=chi2,
        n_points=int(len(Ef)),
        notes=notes,
    )


def evaluate_tb_model(
    result: TBFitResult,
    kx: np.ndarray,
    ky: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a fitted TB model on arbitrary k-grid (for overlay)."""
    p = result.params
    a = result.a
    lt = result.lattice_type
    if lt == "chain":
        return tb_1d_chain(kx, p["eps0"], p["t"], a)
    if ky is None:
        raise ValueError("2D model needs ky")
    if lt == "square":
        return tb_2d_square(kx, ky, p["eps0"], p["t"], p["tp"], p["tpp"], a)
    if lt == "hex":
        return tb_2d_hex(kx, ky, p["eps0"], p["t"], p["tp"], a)
    if lt == "rect":
        return tb_2d_rect(kx, ky, p["eps0"], p["tx"], p["ty"], p["tp"], a, result.b)
    raise ValueError(f"Unknown lattice_type={lt!r}")


def renormalization_vs_dft(
    t_exp: float,
    t_dft: float,
) -> float:
    """Mass renormalization Z = t_DFT / t_exp (>1 means correlated narrowing)."""
    if t_exp <= 0 or t_dft <= 0:
        return float("nan")
    return float(t_dft) / float(t_exp)
