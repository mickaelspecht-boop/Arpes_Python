"""Real self-energy Re Sigma from an ARPES fit and a DFT band."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arpes.physics.kink_analysis import extract_lambda
from arpes.theory.alignment import apply_energy_transform
from arpes.theory.models import (
    TheoryBandData,
    TheoryOverlayConfig,
    compare_fit_to_theory,
    displayed_k_axis,
    selected_segment_mask,
)


@dataclass(frozen=True)
class RealSelfEnergyResult:
    energy: np.ndarray
    re_sigma: np.ndarray
    k_exp: np.ndarray
    e_dft: np.ndarray
    branch: str
    pair_index: int
    band_index: int
    rms_e: float
    kink_energy: float = float("nan")
    lambda_eff: float = float("nan")
    lambda_err: float = float("nan")
    notes: tuple[str, ...] = ()


def real_self_energy(
    fit_result: dict[str, Any] | None,
    theory_overlay: dict[str, Any] | None,
    *,
    branch: str = "",
    pair_index: int | None = None,
    band_index: int | None = None,
    min_points: int = 4,
) -> RealSelfEnergyResult:
    """Compute ``Re Sigma(E) = E_exp - E_DFT(k_exp)``.

    If ``branch/pair/band`` are not provided, the best DFT band is selected by
    ``compare_fit_to_theory``.
    """
    fr = fit_result or {}
    overlay = theory_overlay or {}
    if not overlay.get("data"):
        raise ValueError("Import a DFT band before computing Re Sigma.")
    if not fr.get("e_fitted"):
        raise ValueError("Run an MDC fit before computing Re Sigma.")

    data = TheoryBandData.from_dict(overlay.get("data") or {})
    config = TheoryOverlayConfig.from_dict(overlay.get("config") or {})
    assignment = _select_assignment(
        data, config, fr,
        branch=branch,
        pair_index=pair_index,
        band_index=band_index,
        min_points=min_points,
    )
    branch = assignment["branch"]
    pair_index = int(assignment["pair_index"])
    band_index = int(assignment["band_index"])

    e_exp = np.asarray(fr.get("e_fitted", []), dtype=float)
    branches = fr.get(branch) or []
    if not (0 <= pair_index < len(branches)):
        raise ValueError(f"Branch {branch} pair {pair_index + 1} not found.")
    k_exp = np.asarray(branches[pair_index], dtype=float)

    k_dft = displayed_k_axis(data, config)
    bands = apply_energy_transform(data.bands, config)
    if bands.ndim != 2 or not (0 <= band_index < bands.shape[0]):
        raise ValueError(f"DFT band {band_index} not found.")
    order = np.argsort(k_dft)
    k_ref = k_dft[order]
    e_ref = bands[band_index][order]
    segment = selected_segment_mask(data, config, k_dft.size)[order]
    valid_ref = segment & np.isfinite(k_ref) & np.isfinite(e_ref)
    if int(valid_ref.sum()) < 2:
        raise ValueError("Invalid DFT band for interpolation.")
    k_ref = k_ref[valid_ref]
    e_ref = e_ref[valid_ref]
    lo, hi = float(np.nanmin(k_ref)), float(np.nanmax(k_ref))

    n = min(e_exp.size, k_exp.size)
    valid = (
        np.isfinite(e_exp[:n]) & np.isfinite(k_exp[:n])
        & (k_exp[:n] >= lo) & (k_exp[:n] <= hi)
    )
    if int(valid.sum()) < int(min_points):
        raise ValueError("Insufficient DFT/fit overlap for Re Sigma.")
    e_axis = e_exp[:n][valid]
    k_axis = k_exp[:n][valid]
    e_dft = np.interp(k_axis, k_ref, e_ref)
    re_sigma = e_axis - e_dft
    kink_energy, lambda_eff, lambda_err = _estimate_kink(e_axis, re_sigma)
    return RealSelfEnergyResult(
        energy=e_axis,
        re_sigma=re_sigma,
        k_exp=k_axis,
        e_dft=e_dft,
        branch=branch,
        pair_index=pair_index,
        band_index=band_index,
        rms_e=float(np.sqrt(np.nanmean(re_sigma ** 2))),
        kink_energy=kink_energy,
        lambda_eff=lambda_eff,
        lambda_err=lambda_err,
        notes=_double_counting_notes(data.source),
    )


# P2.3: DFT functionals that already renormalize correlations: comparing
# Re Σ = E_exp − E_DFT to one of them double-counts many-body effects.
_RENORM_FUNCTIONALS = (
    "gga+u", "dft+u", "+u", "hybrid", "hse", "pbe0", "b3lyp",
    "scan", "meta-gga", "mbj", "gw",
)


def _double_counting_notes(source: str) -> tuple[str, ...]:
    src = str(source or "").lower()
    if any(tok in src for tok in _RENORM_FUNCTIONALS):
        return (
            "DFT source appears renormalized (GGA+U / hybrid / GW): Re Σ "
            "= E_exp − E_DFT double-counts correlations. Compare against a "
            "bare GGA/PBE band instead.",
        )
    return ()


def _select_assignment(
    data: TheoryBandData,
    config: TheoryOverlayConfig,
    fr: dict[str, Any],
    *,
    branch: str,
    pair_index: int | None,
    band_index: int | None,
    min_points: int,
) -> dict[str, Any]:
    if branch and pair_index is not None and band_index is not None:
        return {"branch": branch, "pair_index": int(pair_index), "band_index": int(band_index)}
    matches = compare_fit_to_theory(data, config, fr, max_results=1, min_points=min_points)
    if not matches:
        raise ValueError("No DFT band overlaps the fit enough.")
    return matches[0]


def _estimate_kink(energy: np.ndarray, re_sigma: np.ndarray) -> tuple[float, float, float]:
    """Kink energy (maximum curvature) + canonical λ.

    P2.3: λ = −∂ReΣ/∂ω|_{ω=0} via ``kink_analysis.extract_lambda`` (linear fit
    over |ω| < 50 meV), instead of the old ad-hoc difference between near/deep
    slope medians. There is a single λ definition in the code.
    Returns ``(kink_energy, lambda_eff, lambda_err)``.
    """
    order = np.argsort(energy)
    e = np.asarray(energy, dtype=float)[order]
    s = np.asarray(re_sigma, dtype=float)[order]
    valid = np.isfinite(e) & np.isfinite(s)
    e, s = e[valid], s[valid]
    if e.size < 5:
        return float("nan"), float("nan"), float("nan")
    d1 = np.gradient(s, e)
    d2 = np.gradient(d1, e)
    kink_energy = float(e[int(np.nanargmax(np.abs(d2)))])
    lam, lam_err = extract_lambda(e, s, window_eV=0.05)
    if lam is None:
        return kink_energy, float("nan"), float("nan")
    return kink_energy, float(lam), float(lam_err)
