"""Pure geometry helpers for symmetric MDC peak-pair fits."""
from __future__ import annotations

import numpy as np


def symmetric_k0_ceiling(
    k_min: float,
    k_max: float,
    center: float,
    *,
    edge_fraction: float = 0.95,
) -> float:
    """Largest symmetric half-separation fitting inside both k-window edges."""
    lo, hi = sorted((float(k_min), float(k_max)))
    distance = min(float(center) - lo, hi - float(center))
    return max(0.0, float(edge_fraction) * distance)


def relative_k0_guesses(
    peak_positions,
    center: float,
    n_pairs: int,
) -> list[float]:
    """Return nearest right-side peak distances from ``center``."""
    peaks = np.asarray(peak_positions, dtype=float)
    distances = peaks[np.isfinite(peaks) & (peaks > float(center))] - float(center)
    if distances.size == 0:
        return []
    return np.sort(distances)[:max(0, int(n_pairs))].astype(float).tolist()


def symmetric_peak_positions(center: float, k0: float) -> tuple[float, float]:
    """Absolute left/right peak positions for a positive center-relative k0."""
    distance = abs(float(k0))
    center = float(center)
    return center - distance, center + distance


def per_pair_values(value, n_pairs: int, default: float) -> list[float]:
    """Normalize a scalar or short sequence to one finite value per pair."""
    n_pairs = max(1, int(n_pairs))
    if np.isscalar(value):
        values = [float(value)]
    else:
        values = [float(v) for v in np.ravel(value)]
    values = [v if np.isfinite(v) else float(default) for v in values]
    if not values:
        values = [float(default)]
    values.extend([values[-1]] * (n_pairs - len(values)))
    return values[:n_pairs]


def energy_window_plan(
    energy_axis,
    targets,
    full_width: float,
) -> list[tuple[int, slice | np.ndarray]]:
    """Precompute nearest rows and integration selectors for many MDC slices.

    Monotonic axes use vectorized ``searchsorted``; irregular axes retain the
    exact mask-based behavior. This removes repeated O(N_E) scans in fit loops.
    """
    axis = np.asarray(energy_axis, dtype=float)
    target = np.asarray(targets, dtype=float)
    if axis.ndim != 1 or axis.size == 0:
        return []
    increasing = bool(axis.size < 2 or np.all(np.diff(axis) >= 0))
    decreasing = bool(axis.size < 2 or np.all(np.diff(axis) <= 0))
    if not (increasing or decreasing):
        out = []
        half = 0.5 * max(0.0, float(full_width))
        for value in target:
            index = int(np.argmin(np.abs(axis - value)))
            mask = np.abs(axis - axis[index]) <= half
            if not mask.any():
                mask[index] = True
            out.append((index, mask))
        return out

    ordered = axis if increasing else axis[::-1]
    insertion = np.searchsorted(ordered, target, side="left")
    right = np.clip(insertion, 0, ordered.size - 1)
    left = np.clip(insertion - 1, 0, ordered.size - 1)
    choose_left = np.abs(target - ordered[left]) <= np.abs(ordered[right] - target)
    ordered_idx = np.where(choose_left, left, right)
    half = 0.5 * max(0.0, float(full_width))
    lo = np.searchsorted(ordered, ordered[ordered_idx] - half, side="left")
    hi = np.searchsorted(ordered, ordered[ordered_idx] + half, side="right")

    out = []
    for idx, start, stop in zip(ordered_idx, lo, hi):
        start, stop = int(start), int(stop)
        center = ordered[int(idx)]
        while start < stop and abs(ordered[start] - center) > half:
            start += 1
        while stop > start and abs(ordered[stop - 1] - center) > half:
            stop -= 1
        if increasing:
            out.append((int(idx), slice(start, stop)))
        else:
            out.append((
                int(axis.size - 1 - idx),
                slice(int(axis.size - stop), int(axis.size - start)),
            ))
    return out


def fit_pair_parameter_lists(fp) -> tuple[list[float], object, object]:
    """Extract center-relative positions and per-pair widths from FitParams."""
    pairs = list(getattr(fp, "pairs", None) or [])
    k0 = [p.get("kF_init", 0.30) for p in pairs]
    gamma_init = [p.get("gamma_init", fp.gamma_init) for p in pairs]
    gamma_max = [p.get("gamma_max", fp.gamma_max) for p in pairs]
    if str(fp.width_mode) == "free":
        gamma_init, gamma_max = fp.gamma_init, fp.gamma_max
    return k0, gamma_init or fp.gamma_init, gamma_max or fp.gamma_max
