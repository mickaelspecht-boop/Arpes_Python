"""Extraction de résultats physiques + incertitudes depuis un fit MDC.

Tous les calculs partent du dict ``fit_result`` produit par mdc_fit. Les
incertitudes statistiques par slice (sigma_kF_*, sigma_gamma) sont propagées
via régression linéaire pondérée pour les grandeurs globales (kF à E_F, vF,
m*, Γ_FL).

Conventions
-----------
* ``kF`` est exprimé en π/a (axe k de la BM). La conversion vers Å⁻¹ se fait
  hors de ce module via le paramètre du cristal ``a`` si voulu.
* ``vF`` est exprimé en eV·(π/a) — dérivée dE/dk dans le repère natif. Pour
  obtenir ℏvF en eV·Å, multiplier par ``a/π``.
* ``m_star`` rapporté en unités de masse de l'électron via la formule
  ``m*/m_e = ℏ² · kF / vF`` après conversion d'unités. La conversion utilise
  la constante ``HBAR2_OVER_ME = 7.6199682e-2 eV·Å²``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math

import numpy as np

HBAR2_OVER_ME_eV_A2 = 7.6199682e-2  # ℏ² / m_e en eV·Å²


@dataclass(frozen=True)
class LinearFit:
    slope: float = float("nan")
    slope_sigma: float = float("nan")
    intercept: float = float("nan")
    intercept_sigma: float = float("nan")
    n_points: int = 0


@dataclass(frozen=True)
class BranchResult:
    """Résultats par branche kF (kF_minus ou kF_plus) d'une paire donnée."""
    branch: str = ""
    pair_index: int = 0
    kF_at_EF: float = float("nan")
    kF_at_EF_sigma: float = float("nan")
    vF_eV_pi_a: float = float("nan")
    vF_sigma: float = float("nan")
    m_star_over_me: float = float("nan")
    m_star_sigma: float = float("nan")
    luttinger_density_pi_a2: float = float("nan")
    luttinger_density_sigma: float = float("nan")
    luttinger_units: str = ""
    n_points_used: int = 0


@dataclass(frozen=True)
class GammaFermiLiquid:
    """Fit Γ(E) = Γ₀ + a·E² (Fermi liquid) sur une paire."""
    pair_index: int = 0
    gamma_zero: float = float("nan")
    gamma_zero_sigma: float = float("nan")
    coef_E2: float = float("nan")
    coef_E2_sigma: float = float("nan")
    n_points_used: int = 0


@dataclass(frozen=True)
class AsymmetryCheck:
    pair_index: int = 0
    delta_kF: float = float("nan")
    delta_kF_sigma: float = float("nan")
    is_symmetric: bool = False


@dataclass(frozen=True)
class ResultsBundle:
    branches: tuple[BranchResult, ...] = ()
    gamma_fl: tuple[GammaFermiLiquid, ...] = ()
    asymmetry: tuple[AsymmetryCheck, ...] = ()
    crystal_a_angstrom: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "branches": [asdict(b) for b in self.branches],
            "gamma_fl": [asdict(g) for g in self.gamma_fl],
            "asymmetry": [asdict(a) for a in self.asymmetry],
            "crystal_a_angstrom": float(self.crystal_a_angstrom),
        }


def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray | None = None,
) -> LinearFit:
    """Régression linéaire ``y = slope·x + intercept`` avec σ statistiques.

    Renvoie pente, ordonnée et leurs écart-types via la matrice de covariance
    standard. Skip NaN. Sigma=None → pondération uniforme.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if sigma is None:
        sigma = np.ones_like(x)
    sigma = np.asarray(sigma, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if int(valid.sum()) < 2:
        return LinearFit(n_points=int(valid.sum()))
    x = x[valid]
    y = y[valid]
    sigma = sigma[valid]
    w = 1.0 / (sigma ** 2)
    sw = float(np.sum(w))
    swx = float(np.sum(w * x))
    swy = float(np.sum(w * y))
    swxx = float(np.sum(w * x * x))
    swxy = float(np.sum(w * x * y))
    delta = sw * swxx - swx ** 2
    if abs(delta) < 1e-30:
        return LinearFit(n_points=int(valid.sum()))
    slope = (sw * swxy - swx * swy) / delta
    intercept = (swxx * swy - swx * swxy) / delta
    slope_var = sw / delta
    intercept_var = swxx / delta
    return LinearFit(
        slope=float(slope),
        slope_sigma=float(math.sqrt(max(slope_var, 0.0))),
        intercept=float(intercept),
        intercept_sigma=float(math.sqrt(max(intercept_var, 0.0))),
        n_points=int(valid.sum()),
    )


def _branch_arrays(fit_result: dict, branch: str, pair_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = np.asarray(fit_result.get("e_fitted", []), dtype=float)
    arrays = fit_result.get(branch) or []
    if not (0 <= pair_index < len(arrays)):
        return np.array([]), np.array([]), np.array([])
    k = np.asarray(arrays[pair_index], dtype=float)
    sigma_arrays = fit_result.get(f"sigma_{branch}") or []
    if 0 <= pair_index < len(sigma_arrays):
        sk = np.asarray(sigma_arrays[pair_index], dtype=float)
    else:
        sk = np.full_like(k, fill_value=np.nan, dtype=float)
    n = min(len(e), len(k), len(sk))
    return e[:n], k[:n], sk[:n]


def extract_branch_result(
    fit_result: dict,
    *,
    branch: str,
    pair_index: int,
    e_window: float = 0.10,
    crystal_a_angstrom: float = 0.0,
) -> BranchResult:
    """kF₀, vF, m*, n_Luttinger pour ``(branch, pair_index)``.

    Sélectionne les slices |E| ≤ ``e_window`` autour de E_F=0, ajuste
    ``E = α + β·k`` (régression pondérée par σ_k). On a vF = β et
    kF = -α/β. Propagation linéaire pour σ_kF, σ_vF, σ_m*.
    """
    e, k, sk = _branch_arrays(fit_result, branch, pair_index)
    valid = np.isfinite(k) & np.isfinite(e) & (np.abs(e) <= float(e_window))
    e_w, k_w = e[valid], k[valid]
    sk_w = sk[valid] if sk.size else None
    if int(valid.sum()) < 3:
        return BranchResult(branch=branch, pair_index=pair_index, n_points_used=int(valid.sum()))

    # Régresse E = alpha + beta * k. β = dE/dk = vF (eV·π/a).
    # σ_k joue le rôle d'incertitude horizontale ; on l'utilise comme poids
    # 1/σ² approximatif (régression orthogonale serait plus rigoureuse).
    fit = weighted_linear_fit(k_w, e_w, sigma=sk_w)
    if fit.n_points < 3 or not math.isfinite(fit.slope) or abs(fit.slope) < 1e-9:
        return BranchResult(branch=branch, pair_index=pair_index, n_points_used=fit.n_points)

    alpha = fit.intercept
    beta = fit.slope
    sigma_alpha = fit.intercept_sigma
    sigma_beta = fit.slope_sigma
    kF = -alpha / beta
    sigma_kF = math.sqrt(
        (sigma_alpha / beta) ** 2 + (alpha * sigma_beta / (beta ** 2)) ** 2
    )

    # vF en eV·(π/a). m*/m_e calculé seulement si crystal_a_angstrom > 0.
    vF = beta
    sigma_vF = sigma_beta
    m_star_ratio = float("nan")
    sigma_m_star = float("nan")
    luttinger = float("nan")
    sigma_luttinger = float("nan")

    if crystal_a_angstrom > 0:
        # kF en Å⁻¹ : kF_A = kF * π / a
        kF_A = kF * math.pi / crystal_a_angstrom
        sigma_kF_A = sigma_kF * math.pi / crystal_a_angstrom
        # vF en eV·Å : vF_A = vF * a / π
        vF_A = vF * crystal_a_angstrom / math.pi
        sigma_vF_A = sigma_vF * crystal_a_angstrom / math.pi
        # m*/m_e = ℏ² kF / (m_e · ℏvF) = (ℏ²/m_e) · kF / (ℏvF)
        # Avec vF en eV·Å, ℏvF est implicite (unités atomiques). On utilise :
        #   m*/m_e ≈ HBAR2_OVER_ME_eV_A2 · kF / vF_eV_A
        # (kF en Å⁻¹, vF en eV·Å → ratio sans dimension)
        m_star_ratio = HBAR2_OVER_ME_eV_A2 * abs(kF_A) / abs(vF_A) if abs(vF_A) > 0 else float("nan")
        if math.isfinite(m_star_ratio):
            rel = math.sqrt((sigma_kF_A / kF_A) ** 2 + (sigma_vF_A / vF_A) ** 2)
            sigma_m_star = m_star_ratio * rel
        # Densité Luttinger 2D avec dégénérescence spin.
        luttinger = 2.0 * kF_A ** 2 / (2.0 * math.pi)
        sigma_luttinger = 2.0 * abs(2.0 * kF_A * sigma_kF_A) / (2.0 * math.pi)
        luttinger_units = "A^-2"
    else:
        # Densité Luttinger en unités réduites avec dégénérescence spin.
        luttinger = 2.0 * (kF ** 2) / (2.0 * math.pi)
        sigma_luttinger = 2.0 * abs(2.0 * kF * sigma_kF) / (2.0 * math.pi)
        luttinger_units = "(pi/a)^2"

    return BranchResult(
        branch=branch,
        pair_index=pair_index,
        kF_at_EF=float(kF),
        kF_at_EF_sigma=float(sigma_kF),
        vF_eV_pi_a=float(vF),
        vF_sigma=float(sigma_vF),
        m_star_over_me=float(m_star_ratio),
        m_star_sigma=float(sigma_m_star),
        luttinger_density_pi_a2=float(luttinger),
        luttinger_density_sigma=float(sigma_luttinger),
        luttinger_units=luttinger_units,
        n_points_used=int(fit.n_points),
    )


def fit_gamma_fermi_liquid(
    fit_result: dict,
    *,
    pair_index: int,
    e_window: float = 0.30,
) -> GammaFermiLiquid:
    """Fit Γ(E) = Γ₀ + a·E² par régression pondérée sur σ_gamma.

    Utilise ``gamma_corrige`` si dispo (résolution déconvoluée), sinon
    ``gamma`` brut.
    """
    e = np.asarray(fit_result.get("e_fitted", []), dtype=float)
    g_arrays = fit_result.get("gamma_corrige") or fit_result.get("gamma") or []
    sg_arrays = fit_result.get("sigma_gamma") or []
    if not (0 <= pair_index < len(g_arrays)):
        return GammaFermiLiquid(pair_index=pair_index)
    g = np.asarray(g_arrays[pair_index], dtype=float)
    sg = np.asarray(sg_arrays[pair_index], dtype=float) if 0 <= pair_index < len(sg_arrays) else np.full_like(g, np.nan)
    n = min(len(e), len(g), len(sg))
    e, g, sg = e[:n], g[:n], sg[:n]
    valid = np.isfinite(e) & np.isfinite(g) & (np.abs(e) <= float(e_window))
    if int(valid.sum()) < 3:
        return GammaFermiLiquid(pair_index=pair_index, n_points_used=int(valid.sum()))
    x = (e[valid]) ** 2
    y = g[valid]
    w = sg[valid] if sg.size else None
    if w is not None and not np.all(np.isfinite(w) & (w > 0)):
        w = None
    fit = weighted_linear_fit(x, y, sigma=w)
    return GammaFermiLiquid(
        pair_index=pair_index,
        gamma_zero=float(fit.intercept),
        gamma_zero_sigma=float(fit.intercept_sigma),
        coef_E2=float(fit.slope),
        coef_E2_sigma=float(fit.slope_sigma),
        n_points_used=fit.n_points,
    )


def compute_asymmetry(
    fit_result: dict,
    *,
    pair_index: int,
    e_window: float = 0.10,
    significance_sigma: float = 2.0,
) -> AsymmetryCheck:
    """Compare ``kF_plus`` vs ``|kF_minus|`` près de E_F.

    Renvoie ``delta_kF = kF_plus - |kF_minus|`` ± σ et un booléen
    ``is_symmetric`` vrai si ``|delta| ≤ significance_sigma · σ``.
    """
    bm = extract_branch_result(fit_result, branch="kF_minus", pair_index=pair_index, e_window=e_window)
    bp = extract_branch_result(fit_result, branch="kF_plus", pair_index=pair_index, e_window=e_window)
    if not (math.isfinite(bm.kF_at_EF) and math.isfinite(bp.kF_at_EF)):
        return AsymmetryCheck(pair_index=pair_index)
    delta = bp.kF_at_EF - abs(bm.kF_at_EF)
    sigma = math.sqrt(bm.kF_at_EF_sigma ** 2 + bp.kF_at_EF_sigma ** 2)
    is_sym = bool(abs(delta) <= significance_sigma * sigma) if sigma > 0 else False
    return AsymmetryCheck(
        pair_index=pair_index,
        delta_kF=float(delta),
        delta_kF_sigma=float(sigma),
        is_symmetric=is_sym,
    )


def compute_results(
    fit_result: dict | None,
    *,
    e_window_kF: float = 0.10,
    e_window_gamma: float = 0.30,
    crystal_a_angstrom: float = 0.0,
) -> ResultsBundle:
    """Calcule tous les résultats physiques + incertitudes depuis un fit."""
    if not fit_result:
        return ResultsBundle()
    n_pairs = int(fit_result.get("n_pairs", 0) or 0)
    if n_pairs <= 0:
        n_pairs = max(len(fit_result.get("kF_minus") or []), len(fit_result.get("kF_plus") or []))
    branches: list[BranchResult] = []
    for i in range(n_pairs):
        for br in ("kF_minus", "kF_plus"):
            branches.append(extract_branch_result(
                fit_result, branch=br, pair_index=i,
                e_window=e_window_kF, crystal_a_angstrom=crystal_a_angstrom,
            ))
    gamma_fl = tuple(
        fit_gamma_fermi_liquid(fit_result, pair_index=i, e_window=e_window_gamma)
        for i in range(n_pairs)
    )
    asym = tuple(
        compute_asymmetry(fit_result, pair_index=i, e_window=e_window_kF)
        for i in range(n_pairs)
    )
    return ResultsBundle(
        branches=tuple(branches),
        gamma_fl=gamma_fl,
        asymmetry=asym,
        crystal_a_angstrom=float(crystal_a_angstrom),
    )
