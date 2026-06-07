"""Pure DFT band selection / characterization logic.

No Qt, no network, no I/O. Used by the checkable UI list and the MP loader.
Testable in isolation.

- ``compute_band_meta``: per band {idx, e_min, e_max, crosses_ef}.
- ``bands_crossing_ef``: indices crossing +/-window around E=0.
- ``format_band_indices``: inverse of ``parse_band_indices``
  ([1,3,5,6,7,8] -> "1,3,5-8"), used to sync checkboxes with the legacy field.
- ``aggregate_projection_character``: raw pymatgen projections -> dominant
  orbital character per band. Degrades gracefully if absent.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "compute_band_meta",
    "bands_crossing_ef",
    "format_band_indices",
    "aggregate_projection_character",
]


def compute_band_meta(
    bands: Sequence[Sequence[float]],
    *,
    ef_window: float = 0.0,
) -> list[dict[str, Any]]:
    """Metadata per band. ``bands`` = list of rows (energies over k).

    Energies are assumed already relative to E_F (efermi subtracted).
    ``crosses_ef`` is true if the band passes within +/-``ef_window`` of E=0
    (with ``ef_window`` <= 0: strict crossing test min<=0<=max).
    """
    out: list[dict[str, Any]] = []
    win = max(0.0, float(ef_window))
    for idx, row in enumerate(bands):
        arr = np.asarray(row, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out.append({"idx": idx, "e_min": float("nan"),
                        "e_max": float("nan"), "crosses_ef": False})
            continue
        e_min = float(np.min(finite))
        e_max = float(np.max(finite))
        crosses = (e_min <= win) and (e_max >= -win)
        out.append({"idx": idx, "e_min": e_min, "e_max": e_max,
                    "crosses_ef": bool(crosses)})
    return out


def bands_crossing_ef(
    band_meta: Sequence[dict[str, Any]],
    window: float = 0.0,
) -> list[int]:
    """Band indices crossing +/-``window`` around E=0.

    Recompute from e_min/e_max to stay consistent for any requested window
    (band_meta may have been computed without a window).
    """
    win = max(0.0, float(window))
    out: list[int] = []
    for m in band_meta:
        e_min = m.get("e_min")
        e_max = m.get("e_max")
        try:
            lo = float(e_min)
            hi = float(e_max)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo <= win and hi >= -win:
            out.append(int(m.get("idx", len(out))))
    return out


def format_band_indices(indices: Sequence[int]) -> str:
    """[1,3,5,6,7,8] -> "1,3,5-8". Inverse of ``parse_band_indices``.

    Deduplicate, sort, and compress consecutive runs into ``lo-hi``.
    Empty list -> "".
    """
    uniq = sorted({int(i) for i in indices if int(i) >= 0})
    if not uniq:
        return ""
    parts: list[str] = []
    start = prev = uniq[0]
    for cur in uniq[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cur
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def aggregate_projection_character(
    projections: Any,
    elements: Sequence[str] | None = None,
    *,
    top: int = 1,
) -> list[str]:
    """Dominant orbital character per band from pymatgen projections.

    ``projections``: accepted form = mapping {Spin: ndarray
    (n_band, n_k, n_orbital, n_ion)} (pymatgen BandStructureSymmLine
    .projections) OR already an ndarray (n_band, n_k, n_orbital[, n_ion]).
    ``elements``: ion symbol (len == n_ion) used to label "Ti-d". If absent
    or projections are empty -> list of "" (graceful degradation, no exception).

    Label = ``{element}-{orbital}`` for the channel with maximum cumulative
    weight. Orbitals are grouped s/p/d/f by pymatgen convention (0=s, 1-3=p,
    4-8=d, 9-15=f).
    """
    arr = _projection_array(projections)
    if arr is None or arr.size == 0:
        return []
    # arr -> (n_band, n_k, n_orbital, n_ion); complete missing dimensions
    while arr.ndim < 4:
        arr = arr[..., np.newaxis]
    n_band, _n_k, n_orb, n_ion = arr.shape
    weight = np.abs(arr) ** 2  # physical weight
    # sum over k -> (n_band, n_orb, n_ion)
    w = weight.sum(axis=1)
    sym = list(elements or [])
    out: list[str] = []
    for b in range(n_band):
        block = w[b]  # (n_orb, n_ion)
        if not np.isfinite(block).any() or block.sum() <= 0:
            out.append("")
            continue
        orb_i, ion_i = np.unravel_index(int(np.argmax(block)), block.shape)
        elem = sym[ion_i] if 0 <= ion_i < len(sym) else "?"
        out.append(f"{elem}-{_orbital_label(int(orb_i), n_orb)}")
    return out


def _projection_array(projections: Any) -> np.ndarray | None:
    if projections is None:
        return None
    obj = projections
    if isinstance(obj, dict):
        if not obj:
            return None
        # sum over available spin channels
        try:
            stacked = [np.asarray(v, dtype=float) for v in obj.values()]
        except (TypeError, ValueError):
            return None
        if not stacked:
            return None
        arr = stacked[0]
        for extra in stacked[1:]:
            if extra.shape == arr.shape:
                arr = arr + extra
        return arr
    try:
        return np.asarray(obj, dtype=float)
    except (TypeError, ValueError):
        return None


def _orbital_label(orb_index: int, n_orb: int) -> str:
    """Map pymatgen orbital index to s/p/d/f. Fallback "o{idx}" if schema is unknown."""
    if n_orb <= 4:  # compact s,p,d,f schema
        return "spdf"[orb_index] if 0 <= orb_index < 4 else f"o{orb_index}"
    if orb_index == 0:
        return "s"
    if 1 <= orb_index <= 3:
        return "p"
    if 4 <= orb_index <= 8:
        return "d"
    if 9 <= orb_index <= 15:
        return "f"
    return f"o{orb_index}"
