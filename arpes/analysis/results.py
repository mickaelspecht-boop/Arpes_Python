"""Extraction of physical results and uncertainties from an MDC fit.

All calculations start from the ``fit_result`` dict produced by mdc_fit. The
per-slice statistical uncertainties (sigma_kF_*, sigma_gamma) are propagated
through weighted linear regression for the global quantities (kF at E_F, vF,
m*, Γ_FL).

Conventions
-----------
* ``kF`` is expressed in π/a (BM k axis). Conversion to Å⁻¹ is done outside
  this module through the crystal parameter ``a`` if needed.
* ``vF`` is expressed in eV·(π/a), the dE/dk derivative in the native frame.
  To obtain ℏvF in eV·Å, multiply by ``a/π``.
* ``m_star`` is reported in electron-mass units through
  ``m*/m_e = ℏ² · kF / vF`` after unit conversion. The conversion uses the
  constant ``HBAR2_OVER_ME = 7.6199682 eV·Å²`` (= ℏ²/m_e; ref ℏ²/2m_e
  = 3.80998 eV·Å²).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math

import numpy as np

from arpes.physics.dispersion_fit import (
    CURVATURE_MAX,
    MIN_DISP_POINTS,
    curvature_ratio,
    linear_dispersion_fit,
)

# ℏ²/m_e in eV·Å² (ℏ²/2m_e = 3.80998 eV·Å²). Fixed bug: previously 7.62e-2
# (100× too small) -> exported m* was underestimated by 100×. Consistent with physics/fit.py.
HBAR2_OVER_ME_eV_A2 = 7.6199682


@dataclass(frozen=True)
class LinearFit:
    slope: float = float("nan")
    slope_sigma: float = float("nan")
    intercept: float = float("nan")
    intercept_sigma: float = float("nan")
    n_points: int = 0


@dataclass(frozen=True)
class BranchResult:
    """Results per kF branch (kF_minus or kF_plus) for a given pair."""
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
    linear_ok: bool = True
    refused_reason: str = ""


@dataclass(frozen=True)
class GammaFermiLiquid:
    """Fit Γ(E) = Γ₀ + a·E² (Fermi liquid) for one pair."""
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
    """Linear regression ``y = slope·x + intercept`` with statistical σ.

    Returns slope, intercept, and their standard deviations from the standard
    covariance matrix. Skips NaN. Sigma=None -> uniform weighting.
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
    if not sigma_arrays:
        ensemble = fit_result.get("ensemble") or {}
        sigma_arrays = ensemble.get(f"{branch}_std") or []
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
    center: float = 0.0,
) -> BranchResult:
    """kF₀, vF, m*, n_Luttinger for ``(branch, pair_index)``.

    Selects slices with |E| ≤ ``e_window`` around E_F=0 and fits
    ``E = α + β·k`` (weighted regression by σ_k). Then vF = β and the Fermi
    crossing is -α/β; the **Fermi momentum** is measured from the band centre
    Γ: ``kF = -α/β − center`` (``center`` = (kF₊+kF₋)/2, the pair midpoint).
    Without this, an off-centre band (Γ not at k=0 in the detector window) gives
    a kF measured from k=0 — wrong m*/density. Uses linear propagation for
    σ_kF, σ_vF, σ_m* (a constant centre shift leaves the σ unchanged).
    """
    e, k, sk = _branch_arrays(fit_result, branch, pair_index)
    valid = np.isfinite(k) & np.isfinite(e) & (np.abs(e) <= float(e_window))
    e_w, k_w = e[valid], k[valid]
    sk_w = sk[valid] if sk.size else None
    n_valid = int(valid.sum())
    # P2.2: ≥5 points required (quadratic gate + ODR is meaningless below this threshold).
    if n_valid < MIN_DISP_POINTS:
        return BranchResult(
            branch=branch, pair_index=pair_index, n_points_used=n_valid,
            linear_ok=False,
            refused_reason=f"too few points ({n_valid} < {MIN_DISP_POINTS})",
        )

    # P2.2: linearity gate: quadratic curvature relative to the slope.
    # kF=−α/β only makes sense if E(k) is linear near E_F (otherwise a kink or
    # Fermi cutoff contaminates the extrapolation).
    curv = curvature_ratio(k_w, e_w)
    if not math.isfinite(curv) or curv > CURVATURE_MAX:
        return BranchResult(
            branch=branch, pair_index=pair_index, n_points_used=n_valid,
            linear_ok=False,
            refused_reason=f"nonlinear band (curvature {curv:.2f} > {CURVATURE_MAX})",
        )

    # P2.2: orthogonal regression (scipy.odr), weighted by real σ_k when
    # available: both E and k carry noise, so vertical OLS underestimates β.
    # Weighted OLS fallback (polyfit cov) without σ_k. Returns β=dE/dk=vF
    # (eV·π/a) and the 2×2 covariance (α↔β correlation kept for σ_kF/σ_m*).
    sk_use = (
        sk_w if (sk_w is not None and np.all(np.isfinite(sk_w)) and np.all(sk_w > 0))
        else None
    )
    fit = linear_dispersion_fit(k_w, e_w, sk_use)
    if not fit["ok"]:
        return BranchResult(
            branch=branch, pair_index=pair_index, n_points_used=n_valid,
            linear_ok=False, refused_reason="regression did not converge / degenerate",
        )

    beta = float(fit["slope"])        # vF in eV·(π/a)
    alpha = float(fit["intercept"])
    cov = fit["cov"]
    var_b = float(cov[0, 0])
    var_a = float(cov[1, 1])
    cov_ab = float(cov[0, 1])
    sigma_beta = math.sqrt(max(var_b, 0.0))
    kF = (-alpha / beta) - float(center)  # Fermi momentum from the band centre Γ
    # σ_kF with full covariance: ∂kF/∂α=−1/β, ∂kF/∂β=α/β².
    dkF_da = -1.0 / beta
    dkF_db = alpha / (beta * beta)
    var_kF = (dkF_da ** 2 * var_a + dkF_db ** 2 * var_b
              + 2.0 * dkF_da * dkF_db * cov_ab)
    sigma_kF = math.sqrt(max(var_kF, 0.0))

    # vF in eV·(π/a). m*/m_e is computed only if crystal_a_angstrom > 0.
    vF = beta
    sigma_vF = sigma_beta
    m_star_ratio = float("nan")
    sigma_m_star = float("nan")
    luttinger = float("nan")
    sigma_luttinger = float("nan")

    if crystal_a_angstrom > 0:
        # kF in Å⁻¹: kF_A = kF * π / a
        kF_A = kF * math.pi / crystal_a_angstrom
        sigma_kF_A = sigma_kF * math.pi / crystal_a_angstrom
        # vF in eV·Å: vF_A = vF * a / π
        vF_A = vF * crystal_a_angstrom / math.pi
        sigma_vF_A = sigma_vF * crystal_a_angstrom / math.pi
        # m*/m_e = ℏ² kF / (m_e · ℏvF) = (ℏ²/m_e) · kF / (ℏvF)
        # With vF in eV·Å, ℏvF is implicit (atomic units). Use:
        #   m*/m_e ≈ HBAR2_OVER_ME_eV_A2 · kF / vF_eV_A
        # (kF in Å⁻¹, vF in eV·Å -> dimensionless ratio)
        m_star_ratio = HBAR2_OVER_ME_eV_A2 * abs(kF_A) / abs(vF_A) if abs(vF_A) > 0 else float("nan")
        if math.isfinite(m_star_ratio) and kF != 0.0 and beta != 0.0:
            # m* ∝ |kF|/|vF|, with kF = −α/β − center and vF = β. Propagate σ_m*
            # through (kF, vF) — both already computed with the correct
            # derivatives — using their covariance. The earlier |α|/β² form
            # dropped `center`: for an off-Γ pocket (center≠0) it divided by the
            # k=0 intercept α≈0 and blew σ_m* up spuriously even though kF/vF are
            # well determined (e.g. C05). cov(kF,vF) = ∂kF/∂α·cov_ab + ∂kF/∂β·var_β.
            cov_kf_vf = dkF_da * cov_ab + dkF_db * var_b
            rel_var = ((sigma_kF / kF) ** 2 + (sigma_vF / beta) ** 2
                       - 2.0 * cov_kf_vf / (kF * beta))
            sigma_m_star = m_star_ratio * math.sqrt(max(rel_var, 0.0))
        # 2D Luttinger density with spin degeneracy.
        luttinger = 2.0 * kF_A ** 2 / (2.0 * math.pi)
        sigma_luttinger = 2.0 * abs(2.0 * kF_A * sigma_kF_A) / (2.0 * math.pi)
        luttinger_units = "A^-2"
    else:
        # Luttinger density in reduced units with spin degeneracy.
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
        n_points_used=n_valid,
        linear_ok=True,
        refused_reason="",
    )


def gamma_reliability_mask(
    fit_result: dict,
    *,
    pair_index: int = 0,
    gamma_max: float | None = None,
    merge_factor: float = 1.0,
) -> np.ndarray:
    """Per-slice boolean mask: True where Γ(E) is physically reliable.

    A slice is *unreliable* when:

    * **peaks merged** — the ±kF Lorentzians overlap, i.e. their full width
      exceeds their separation: ``2·Γ_HWHM ≥ merge_factor·|kF₊ − kF₋|``. This is
      what happens near the band top / E_F: the single fitted width then
      describes one blob, not a lifetime.
    * **saturated** — Γ is pinned at the optimizer bound ``gamma_max`` (when
      provided): the value is the constraint, not a measurement.

    Returns a bool array aligned to ``e_fitted`` (all-False if inputs missing).
    """
    fr = fit_result or {}
    e = np.asarray(fr.get("e_fitted", []), dtype=float)
    n = e.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    def _arr(key: str) -> np.ndarray:
        a = fr.get(key) or []
        if 0 <= pair_index < len(a):
            v = np.asarray(a[pair_index], dtype=float)
            if v.size >= n:
                return v[:n]
            return np.concatenate([v, np.full(n - v.size, np.nan)])
        return np.full(n, np.nan)

    g = _arr("gamma_corrige")
    if not np.isfinite(g).any():
        g = _arr("gamma")
    sep = np.abs(_arr("kF_plus") - _arr("kF_minus"))
    with np.errstate(invalid="ignore"):
        resolved = np.isfinite(sep) & np.isfinite(g) & (sep > (2.0 * g / max(merge_factor, 1e-9)))
    mask = resolved
    if gamma_max is not None and float(gamma_max) > 0:
        g_raw = _arr("gamma_brut")
        if not np.isfinite(g_raw).any():
            g_raw = g
        with np.errstate(invalid="ignore"):
            mask = mask & ~(g_raw >= 0.98 * float(gamma_max))
    return mask


def fit_gamma_fermi_liquid(
    fit_result: dict,
    *,
    pair_index: int,
    e_window: float = 0.30,
    gamma_max: float | None = None,
    e_lo: float | None = None,
    e_hi: float | None = None,
) -> GammaFermiLiquid:
    """Fit Γ(E) = Γ₀ + a·E² by weighted regression on σ_gamma.

    Uses ``gamma_corrige`` if available (deconvolved resolution), otherwise raw
    ``gamma``. The trend is fit only on the **reliable** slices
    (``gamma_reliability_mask``): merged-peak / saturated slices are excluded so
    Γ₀ is extrapolated from the resolved region, not the near-E_F blow-up.

    Energy window: ``[e_lo, e_hi]`` when both are given (user-chosen fit range),
    otherwise the symmetric ``|E| ≤ e_window``.
    """
    e = np.asarray(fit_result.get("e_fitted", []), dtype=float)
    g_arrays = fit_result.get("gamma_corrige") or fit_result.get("gamma") or []
    sg_arrays = fit_result.get("sigma_gamma") or []
    if not sg_arrays:
        sg_arrays = (fit_result.get("ensemble") or {}).get("gamma_std") or []
    if not (0 <= pair_index < len(g_arrays)):
        return GammaFermiLiquid(pair_index=pair_index)
    g = np.asarray(g_arrays[pair_index], dtype=float)
    sg = np.asarray(sg_arrays[pair_index], dtype=float) if 0 <= pair_index < len(sg_arrays) else np.full_like(g, np.nan)
    n = min(len(e), len(g), len(sg))
    e, g, sg = e[:n], g[:n], sg[:n]
    reliable = gamma_reliability_mask(
        fit_result, pair_index=pair_index, gamma_max=gamma_max)[:n]
    if e_lo is not None and e_hi is not None:
        lo, hi = (float(e_lo), float(e_hi)) if e_lo <= e_hi else (float(e_hi), float(e_lo))
        in_window = (e >= lo) & (e <= hi)
    else:
        in_window = np.abs(e) <= float(e_window)
    valid = np.isfinite(e) & np.isfinite(g) & in_window & reliable
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


def _pair_center(fit_result: dict, pair_index: int) -> float:
    """Band centre Γ in π/a = median pair midpoint (kF₊+kF₋)/2 over slices.

    Used so kF is measured from the band's own centre, not from k=0 — required
    when the band sits off-centre in the detector window. 0.0 if undecidable.
    """
    km_all = fit_result.get("kF_minus") or []
    kp_all = fit_result.get("kF_plus") or []
    if not (0 <= pair_index < len(km_all) and 0 <= pair_index < len(kp_all)):
        return 0.0
    km = np.asarray(km_all[pair_index], dtype=float)
    kp = np.asarray(kp_all[pair_index], dtype=float)
    n = min(km.size, kp.size)
    if n == 0:
        return 0.0
    mid = (km[:n] + kp[:n]) / 2.0
    c = float(np.nanmedian(mid)) if np.isfinite(mid).any() else 0.0
    return c if math.isfinite(c) else 0.0


def compute_asymmetry(
    fit_result: dict,
    *,
    pair_index: int,
    e_window: float = 0.10,
    significance_sigma: float = 2.0,
    center: float = 0.0,
) -> AsymmetryCheck:
    """Compare ``kF_plus`` vs ``|kF_minus|`` near E_F (both from the band Γ).

    Returns ``delta_kF = kF_plus - |kF_minus|`` ± σ and a boolean
    ``is_symmetric`` true if ``|delta| ≤ significance_sigma · σ``.
    """
    bm = extract_branch_result(fit_result, branch="kF_minus", pair_index=pair_index, e_window=e_window, center=center)
    bp = extract_branch_result(fit_result, branch="kF_plus", pair_index=pair_index, e_window=e_window, center=center)
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
    gamma_max: float | None = None,
    gamma_e_lo: float | None = None,
    gamma_e_hi: float | None = None,
) -> ResultsBundle:
    """Compute all physical results and uncertainties from a fit.

    ``gamma_max`` (the fit's width bound) lets the Γ₀ trend flag saturated
    slices; pass it from the entry's fit_params when available.
    ``gamma_e_lo``/``gamma_e_hi`` set the user-chosen Γ(E) fit window.
    """
    if not fit_result:
        return ResultsBundle()
    n_pairs = int(fit_result.get("n_pairs", 0) or 0)
    if n_pairs <= 0:
        n_pairs = max(len(fit_result.get("kF_minus") or []), len(fit_result.get("kF_plus") or []))
    centers = [_pair_center(fit_result, i) for i in range(n_pairs)]
    branches: list[BranchResult] = []
    for i in range(n_pairs):
        for br in ("kF_minus", "kF_plus"):
            branches.append(extract_branch_result(
                fit_result, branch=br, pair_index=i,
                e_window=e_window_kF, crystal_a_angstrom=crystal_a_angstrom,
                center=centers[i],
            ))
    gamma_fl = tuple(
        fit_gamma_fermi_liquid(
            fit_result, pair_index=i, e_window=e_window_gamma, gamma_max=gamma_max,
            e_lo=gamma_e_lo, e_hi=gamma_e_hi)
        for i in range(n_pairs)
    )
    asym = tuple(
        compute_asymmetry(fit_result, pair_index=i, e_window=e_window_kF,
                          center=centers[i])
        for i in range(n_pairs)
    )
    return ResultsBundle(
        branches=tuple(branches),
        gamma_fl=gamma_fl,
        asymmetry=asym,
        crystal_a_angstrom=float(crystal_a_angstrom),
    )


def select_representative_branch(branches) -> "BranchResult | None":
    """Pick the single branch best summarising a file (multi-file aggregation).

    The mirror branches (kF_minus / kF_plus of a pair) measure the SAME |kF|,
    |vF| and m*; but m* ∝ |α|/β² is ill-conditioned when the dispersion's k=0
    intercept α ≈ 0, so one branch can carry a huge σ_m* while its mirror is
    tight (e.g. C05: kF_plus 48 % vs kF_minus 2 %). Among the branches with a
    finite ``kF_at_EF``, return the one with the smallest *relative* m*
    uncertainty; fall back to the first finite-kF branch when no m* is finite.
    Returns ``None`` if no branch has a finite kF.
    """
    finite = [br for br in (branches or []) if np.isfinite(br.kF_at_EF)]
    if not finite:
        return None

    def _rel_sigma(br: "BranchResult") -> float:
        m, s = br.m_star_over_me, br.m_star_sigma
        if np.isfinite(m) and m != 0.0 and np.isfinite(s):
            return abs(s / m)
        return float("inf")

    return min(finite, key=_rel_sigma)
