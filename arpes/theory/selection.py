"""Band selection and displayed-coordinate helpers for DFT overlays."""
from __future__ import annotations

from typing import Any

import numpy as np

from arpes.theory.alignment import apply_energy_transform
from arpes.theory.data import TheoryBandData, TheoryOverlayConfig, _finite_float
from arpes.theory.labels import _branch_index_for_segment, _clean_segment_name


def parse_band_indices(spec: str, n_bands: int) -> list[int]:
    """Parse `'1,3,5-8'` -> [1,3,5,6,7,8]. Skip out-of-range."""
    out: list[int] = []
    seen: set[int] = set()
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            try:
                lo_s, hi_s = chunk.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            for idx in range(lo, hi + 1):
                if 0 <= idx < n_bands and idx not in seen:
                    seen.add(idx)
                    out.append(idx)
        else:
            try:
                idx = int(chunk)
            except ValueError:
                continue
            if 0 <= idx < n_bands and idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def displayed_k_axis(data: TheoryBandData | dict[str, Any], config: TheoryOverlayConfig | dict[str, Any]) -> np.ndarray:
    """Return the k axis used by visible DFT overlays."""
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
    k_raw = np.asarray(data.k_distance, dtype=float)
    k = _branch_local_k(data, config, k_raw)
    return k * float(config.k_scale) + float(config.k_shift)


def displayed_gamma_k(
    data: TheoryBandData | dict[str, Any], config: TheoryOverlayConfig | dict[str, Any]
) -> float | None:
    """Displayed k (π/a) of the DFT Γ point, or None.

    ``Γ_display = local_k(Γ) * k_scale + k_shift`` — the same transform the bands
    use. This is the physically correct pivot for the Γ mirror and, after
    alignment, equals the manual Γ center placed on the BM. None when there is no
    Γ label or it falls off the current branch.
    """
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
    raw = None
    for item in (data.labels or []):
        name = str(item.get("label") or "").strip().upper().replace("GAMMA", "Γ")
        if name == "Γ" and item.get("k") is not None:
            try:
                raw = float(item["k"])
            except (TypeError, ValueError):
                raw = None
            break
    if raw is None:
        return None
    k_full = np.asarray(data.k_distance, dtype=float)
    if k_full.size == 0:
        return None
    loc = _branch_local_k(data, config, k_full)
    idx = int(np.argmin(np.abs(k_full - raw)))
    if idx >= loc.size:
        return None
    u = float(loc[idx])
    if not np.isfinite(u):
        return None
    return u * float(config.k_scale) + float(config.k_shift)


def selected_segment_mask(data: TheoryBandData | dict[str, Any], config: TheoryOverlayConfig | dict[str, Any], n_k: int) -> np.ndarray:
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
    return _segment_mask(data, config, n_k)


def select_bands_for_view(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return ``[(band_index, k_display, E_display), ...]`` for the view."""
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    config = TheoryOverlayConfig.from_dict(config) if isinstance(config, dict) else config
    if not config.enabled or not data.k_distance or not data.bands:
        return []
    k = displayed_k_axis(data, config)
    bands = apply_energy_transform(data.bands, config)
    if bands.ndim != 2 or bands.shape[1] != k.size:
        return []
    segment_mask = _segment_mask(data, config, k.size)
    x0, x1 = sorted((float(xlim[0]), float(xlim[1])))
    y0, y1 = sorted((float(ylim[0]), float(ylim[1])))
    mask_x = (k >= x0) & (k <= x1) & segment_mask
    if not mask_x.any():
        mask_x = segment_mask
    scored: list[tuple[float, float, int]] = []
    y_center = 0.5 * (y0 + y1)
    for idx, band in enumerate(bands):
        visible = band[mask_x]
        finite = visible[np.isfinite(visible)]
        if finite.size == 0:
            continue
        overlap = np.mean((finite >= y0) & (finite <= y1))
        distance = float(np.nanmin(np.abs(finite - y_center)))
        scored.append((-float(overlap), distance, idx))
    scored.sort()
    explicit = parse_band_indices(config.band_indices, len(bands))
    selected = explicit if explicit else [idx for *_rest, idx in scored[: int(config.max_bands)]]
    win = float(config.ef_window)
    if win > 0.0:
        kept: list[int] = []
        for idx in selected:
            finite = bands[idx][np.isfinite(bands[idx])]
            if finite.size and float(np.nanmin(finite)) <= win and float(np.nanmax(finite)) >= -win:
                kept.append(idx)
        selected = kept
    # Mirror about the DFT Γ position (= the manual Γ center once aligned), not
    # about k=0 — otherwise a non-zero Γ reflects to the wrong side.
    pivot = 0.0
    if config.mirror_gamma:
        g = displayed_gamma_k(data, config)
        pivot = g if g is not None else 0.0
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for idx in selected:
        band = bands[idx].copy()
        band[~segment_mask] = np.nan
        curves.append((idx, k, band))
        if config.mirror_gamma:
            curves.append((idx, 2.0 * pivot - k, band.copy()))
    return curves


def filter_bands_for_view(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compat: ``[(k, band), ...]`` without source band index."""
    return [(k, band) for _idx, k, band in select_bands_for_view(
        data, config, xlim=xlim, ylim=ylim)]


def _branch_local_k(
    data: TheoryBandData, config: TheoryOverlayConfig, k_raw: np.ndarray
) -> np.ndarray:
    """Local branch coordinate used by overlays."""
    if not data.branches or not config.segment:
        return k_raw
    br = _branch_index_for_segment(data.branches, config.segment)
    if br is None:
        return k_raw
    try:
        s = int(br.get("start", 0))
        e = int(br.get("end", k_raw.size - 1))
    except (TypeError, ValueError):
        return k_raw
    s, e = max(0, min(s, e)), min(k_raw.size - 1, max(s, e))
    loc = np.full(k_raw.size, np.nan, dtype=float)
    name = _clean_segment_name(str(br.get("name", "")))
    left, _, right = name.partition("-")
    gamma_at_end = right.strip() == "Γ" and left.strip() != "Γ"

    abs_dist = np.asarray(data.k_distance_abs, dtype=float)
    a_cryst = float(getattr(config, "crystal_a", 0.0) or 0.0)
    if abs_dist.size == k_raw.size and a_cryst > 0.0:
        seg = abs_dist[s:e + 1] - abs_dist[s]
        kpa = seg * (a_cryst / np.pi)
        if gamma_at_end:
            kpa = kpa[-1] - kpa
        loc[s:e + 1] = kpa
        return loc

    span = max(e - s, 1)
    frac = (np.arange(s, e + 1, dtype=float) - s) / span
    if gamma_at_end:
        frac = 1.0 - frac
    loc[s:e + 1] = frac
    return loc


def _segment_mask(data: TheoryBandData, config: TheoryOverlayConfig, n_k: int) -> np.ndarray:
    if not config.segment:
        return np.ones(n_k, dtype=bool)
    if data.branches:
        br = _branch_index_for_segment(data.branches, config.segment)
        if br is not None:
            try:
                s = int(br.get("start", 0))
                e = int(br.get("end", n_k - 1))
            except (TypeError, ValueError):
                s, e = 0, n_k - 1
            s, e = max(0, min(s, e)), min(n_k - 1, max(s, e))
            mask = np.zeros(n_k, dtype=bool)
            mask[s:e + 1] = True
            return mask
        return np.ones(n_k, dtype=bool)
    if "-" not in config.segment:
        return np.ones(n_k, dtype=bool)
    left, right = [
        x.strip().upper().replace("GAMMA", "Γ")
        for x in config.segment.split("-", 1)
    ]
    label_positions = {
        str(item.get("label") or "").upper().replace("GAMMA", "Γ"): item.get("k")
        for item in data.labels
    }
    if left not in label_positions or right not in label_positions:
        return np.ones(n_k, dtype=bool)
    lo = _finite_float(label_positions[left], float("nan"))
    hi = _finite_float(label_positions[right], float("nan"))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.ones(n_k, dtype=bool)
    a, b = sorted((lo, hi))
    raw_k = np.asarray(data.k_distance, dtype=float)
    return (raw_k >= a) & (raw_k <= b)
