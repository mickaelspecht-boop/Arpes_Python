"""Bootstrap des résultats physiques par rééchantillonnage des points fittés.

Plutôt que refaire ``curve_fit`` Lorentzien par tranche (très coûteux), on
bootstrap la régression linéaire au niveau analyse :

    1. Tirer N rééchantillonnages avec remplacement de la liste de paires
       (k_i, e_i, σ_k_i) près de E_F.
    2. Pour chaque rééchantillon, fit ``E = α + β k`` pondéré.
    3. Extraire les distributions de kF₀ = -α/β et vF = β.
    4. Renvoyer médiane + (p84 - p16)/2 comme σ robuste.

Robuste face aux outliers résiduels (points aberrants restants après
suppression manuelle). Plus large que les σ statistiques de
``curve_fit`` quand la dispersion résiduelle dépasse l'incertitude
gaussienne attendue.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .results import (
    BranchResult,
    HBAR2_OVER_ME_eV_A2,
    _branch_arrays,
    weighted_linear_fit,
)


@dataclass(frozen=True)
class BootstrapBranchResult(BranchResult):
    sigma_method: str = "bootstrap"
    n_iter: int = 0


def _block_length(n: int) -> int:
    """Longueur de bloc 3–5 slices selon n (moving-block bootstrap)."""
    return int(min(5, max(3, math.ceil(n / 3.0))))


def _moving_block_indices(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    """Indices d'un rééchantillon moving-block de taille ~n.

    Tire des blocs CONTIGUS de ``block_len`` slices : préserve la
    corrélation entre slices voisines en E (le rééchantillonnage iid
    point-à-point la casse → σ sous-estimée). cf P2.2.
    """
    block_len = int(max(1, min(block_len, n)))
    n_blocks = int(math.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def bootstrap_branch_result(
    fit_result: dict,
    *,
    branch: str,
    pair_index: int,
    e_window: float = 0.10,
    crystal_a_angstrom: float = 0.0,
    n_iter: int = 500,
    seed: int | None = 0,
    center: float = 0.0,
) -> BootstrapBranchResult:
    e, k, sk = _branch_arrays(fit_result, branch, pair_index)
    valid = np.isfinite(k) & np.isfinite(e) & (np.abs(e) <= float(e_window))
    if int(valid.sum()) < 4:
        return BootstrapBranchResult(
            branch=branch, pair_index=pair_index,
            n_points_used=int(valid.sum()), n_iter=0,
        )
    e_w = e[valid]
    k_w = k[valid]
    sk_w = sk[valid] if sk.size else np.ones_like(k_w)
    sk_w = np.where(np.isfinite(sk_w) & (sk_w > 0), sk_w, np.median(sk_w[np.isfinite(sk_w) & (sk_w > 0)]) if np.any(np.isfinite(sk_w) & (sk_w > 0)) else 1.0)

    rng = np.random.default_rng(seed)
    n = len(k_w)
    block_len = _block_length(n)
    kF_samples = []
    vF_samples = []
    for _ in range(int(n_iter)):
        idx = _moving_block_indices(rng, n, block_len)
        fit = weighted_linear_fit(k_w[idx], e_w[idx], sigma=sk_w[idx])
        if (fit.n_points < 3 or not math.isfinite(fit.slope)
                or abs(fit.slope) < 1e-9):
            continue
        kF_samples.append(-fit.intercept / fit.slope - float(center))
        vF_samples.append(fit.slope)
    if len(kF_samples) < 5:
        return BootstrapBranchResult(
            branch=branch, pair_index=pair_index,
            n_points_used=n, n_iter=len(kF_samples),
        )
    kF_arr = np.asarray(kF_samples)
    vF_arr = np.asarray(vF_samples)
    kF_med = float(np.nanmedian(kF_arr))
    vF_med = float(np.nanmedian(vF_arr))
    kF_sigma = float(0.5 * (np.nanpercentile(kF_arr, 84) - np.nanpercentile(kF_arr, 16)))
    vF_sigma = float(0.5 * (np.nanpercentile(vF_arr, 84) - np.nanpercentile(vF_arr, 16)))

    m_star = float("nan")
    m_star_sigma = float("nan")
    luttinger = float("nan")
    luttinger_sigma = float("nan")
    luttinger_units = ""
    if crystal_a_angstrom > 0:
        a = float(crystal_a_angstrom)
        kF_A_arr = kF_arr * math.pi / a
        vF_A_arr = vF_arr * a / math.pi
        ratio = HBAR2_OVER_ME_eV_A2 * np.abs(kF_A_arr) / np.where(np.abs(vF_A_arr) > 0, np.abs(vF_A_arr), np.nan)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size >= 5:
            m_star = float(np.nanmedian(ratio))
            m_star_sigma = float(0.5 * (np.nanpercentile(ratio, 84) - np.nanpercentile(ratio, 16)))
        n_arr = 2.0 * (kF_A_arr ** 2) / (2.0 * math.pi)
        luttinger_units = "A^-2"
        n_arr = n_arr[np.isfinite(n_arr)]
        if n_arr.size >= 5:
            luttinger = float(np.nanmedian(n_arr))
            luttinger_sigma = float(0.5 * (np.nanpercentile(n_arr, 84) - np.nanpercentile(n_arr, 16)))
    else:
        n_arr = 2.0 * (kF_arr ** 2) / (2.0 * math.pi)
        luttinger_units = "(pi/a)^2"
        n_arr = n_arr[np.isfinite(n_arr)]
        if n_arr.size >= 5:
            luttinger = float(np.nanmedian(n_arr))
            luttinger_sigma = float(0.5 * (np.nanpercentile(n_arr, 84) - np.nanpercentile(n_arr, 16)))

    return BootstrapBranchResult(
        branch=branch,
        pair_index=pair_index,
        kF_at_EF=kF_med,
        kF_at_EF_sigma=kF_sigma,
        vF_eV_pi_a=vF_med,
        vF_sigma=vF_sigma,
        m_star_over_me=m_star,
        m_star_sigma=m_star_sigma,
        luttinger_density_pi_a2=luttinger,
        luttinger_density_sigma=luttinger_sigma,
        luttinger_units=luttinger_units,
        n_points_used=n,
        n_iter=len(kF_samples),
    )
