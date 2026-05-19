"""Fit-vs-DFT comparison helpers."""
from __future__ import annotations

from typing import Any

import numpy as np

from arpes.theory.alignment import apply_energy_transform
from arpes.theory.data import TheoryBandData, TheoryOverlayConfig
from arpes.theory.selection import displayed_k_axis, selected_segment_mask


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
