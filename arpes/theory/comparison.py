"""Fit-vs-DFT comparison helpers."""
from __future__ import annotations

from typing import Any

import numpy as np

from arpes.theory.alignment import (
    apply_energy_transform,
    effective_mu_shift,
    effective_z_scale,
)
from arpes.theory.data import TheoryBandData, TheoryOverlayConfig
from arpes.theory.selection import (
    displayed_k_axis,
    parse_band_indices,
    selected_segment_mask,
)


def compare_fit_to_theory(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    fit_result: dict[str, Any] | None,
    *,
    max_results: int = 6,
    min_points: int = 3,
) -> list[dict[str, Any]]:
    """Score fitted experimental kF branches against DFT bands."""
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
    fr = fit_result or {}
    e_exp = np.asarray(fr.get("e_fitted", []), dtype=float)
    if e_exp.size == 0 or not data.k_distance or not data.bands:
        return []
    k_dft = displayed_k_axis(data, config)
    bands = apply_energy_transform(data.bands, config)
    if bands.ndim != 2 or bands.shape[1] != k_dft.size:
        return []
    segment_mask = selected_segment_mask(data, config, k_dft.size)
    order = np.argsort(k_dft)
    k_sorted = k_dft[order]
    valid_segment_sorted = segment_mask[order]
    out: list[dict[str, Any]] = []
    for branch_name in ("kF_minus", "kF_plus"):
        branches = fr.get(branch_name) or []
        for pair_index, k_branch_raw in enumerate(branches):
            k_exp = np.asarray(k_branch_raw, dtype=float)
            n = min(k_exp.size, e_exp.size)
            if n == 0:
                continue
            k_exp_n = k_exp[:n]
            e_exp_n = e_exp[:n]
            valid_exp = np.isfinite(k_exp_n) & np.isfinite(e_exp_n)
            if int(valid_exp.sum()) < int(min_points):
                continue
            for band_index, band in enumerate(bands):
                band_sorted = np.asarray(band, dtype=float)[order]
                valid_band = valid_segment_sorted & np.isfinite(k_sorted) & np.isfinite(band_sorted)
                if int(valid_band.sum()) < 2:
                    continue
                k_ref = k_sorted[valid_band]
                e_ref = band_sorted[valid_band]
                lo, hi = float(np.nanmin(k_ref)), float(np.nanmax(k_ref))
                valid = valid_exp & (k_exp_n >= lo) & (k_exp_n <= hi)
                if int(valid.sum()) < int(min_points):
                    continue
                e_interp = np.interp(k_exp_n[valid], k_ref, e_ref)
                residual = e_exp_n[valid] - e_interp
                rms_e = float(np.sqrt(np.nanmean(residual**2)))
                med_e = float(np.nanmedian(residual))
                out.append({
                    "branch": branch_name,
                    "pair_index": int(pair_index),
                    "band_index": int(band_index),
                    "n_points": int(valid.sum()),
                    "rms_e": rms_e,
                    "median_e": med_e,
                })
    out.sort(key=lambda item: (item["rms_e"], -item["n_points"]))
    return out[: int(max_results)]


def fit_mu_shift(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    fit_result: dict[str, Any] | None,
    *,
    band_index: int | None = None,
    robust: bool = True,
    min_points: int = 3,
) -> dict[str, Any] | None:
    """Compute the μ that best aligns a DFT band onto the ARPES fit.

    Closed form, no iterative optimiser. With
    ``E_overlay = Z·(E_DFT − μ)`` the residual ``e_exp − E_overlay`` shifts
    by ``Z·Δμ`` uniformly, so the L2-optimal additive correction is
    ``μ_new = μ_cur − ⟨residual⟩ / Z``. ``robust`` uses the median of the
    residual instead of the mean (resistant to kF outliers).

    The candidate band is the best-scoring one among the currently
    *selected* bands (``config.band_indices``); ``band_index`` forces a
    specific band. Returns ``None`` when there is no usable overlap.
    """
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config

    ranked = compare_fit_to_theory(
        data, config, fit_result, max_results=10**6, min_points=min_points
    )
    if not ranked:
        return None

    n_bands = len(data.bands or [])
    selected = set(parse_band_indices(str(config.band_indices or ""), n_bands))
    if band_index is not None:
        ranked = [r for r in ranked if r["band_index"] == int(band_index)]
    elif selected:
        ranked = [r for r in ranked if r["band_index"] in selected] or ranked
    if not ranked:
        return None
    best = ranked[0]

    z = effective_z_scale(config)
    if not np.isfinite(z) or abs(z) < 1e-9:
        return None

    # Recompute the residual vector for the chosen band/branch/pair.
    k_dft = displayed_k_axis(data, config)
    bands = apply_energy_transform(data.bands, config)
    if bands.ndim != 2 or bands.shape[1] != k_dft.size:
        return None
    segment_mask = selected_segment_mask(data, config, k_dft.size)
    order = np.argsort(k_dft)
    k_sorted = k_dft[order]
    band_sorted = np.asarray(bands[best["band_index"]], dtype=float)[order]
    valid_band = segment_mask[order] & np.isfinite(k_sorted) & np.isfinite(band_sorted)
    if int(valid_band.sum()) < 2:
        return None
    k_ref = k_sorted[valid_band]
    e_ref = band_sorted[valid_band]

    fr = fit_result or {}
    e_exp_all = np.asarray(fr.get("e_fitted", []), dtype=float)
    k_branch = np.asarray((fr.get(best["branch"]) or [])[best["pair_index"]], dtype=float)
    n = min(k_branch.size, e_exp_all.size)
    k_exp = k_branch[:n]
    e_exp = e_exp_all[:n]
    lo, hi = float(np.nanmin(k_ref)), float(np.nanmax(k_ref))
    valid = np.isfinite(k_exp) & np.isfinite(e_exp) & (k_exp >= lo) & (k_exp <= hi)
    if int(valid.sum()) < int(min_points):
        return None
    residual = e_exp[valid] - np.interp(k_exp[valid], k_ref, e_ref)

    shift = float(np.median(residual)) if robust else float(np.mean(residual))
    mu_cur = effective_mu_shift(config)
    mu_new = mu_cur - shift / z
    rms_before = float(np.sqrt(np.nanmean(residual**2)))
    rms_after = float(np.sqrt(np.nanmean((residual - shift) ** 2)))
    return {
        "mu": float(mu_new),
        "mu_before": float(mu_cur),
        "band_index": int(best["band_index"]),
        "branch": best["branch"],
        "pair_index": int(best["pair_index"]),
        "n_points": int(valid.sum()),
        "rms_before": rms_before,
        "rms_after": rms_after,
        "robust": bool(robust),
    }
