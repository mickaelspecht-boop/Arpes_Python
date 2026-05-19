"""Pure helpers for the visual DFT band picker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arpes.theory.models import (
    TheoryBandData,
    TheoryOverlayConfig,
    branch_display_names,
)

PATH_CONVENTION_MP_BULK = "mp_bulk"
PATH_CONVENTION_ARPES_PNICTIDES = "arpes_pnictides"


@dataclass(frozen=True)
class PickerBandCurve:
    band_index: int
    k: np.ndarray
    energy: np.ndarray
    crosses_ef_window: bool


@dataclass(frozen=True)
class PickerTick:
    x: float
    label: str


def validate_picker_data(data: TheoryBandData | dict[str, Any]) -> str:
    """Return an error message if ``data`` cannot be plotted by the picker."""
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    if not data.k_distance:
        return "DFT invalide: axe k vide."
    if not data.bands:
        return "DFT invalide: aucune bande."
    k = np.asarray(data.k_distance, dtype=float)
    if k.ndim != 1 or not np.isfinite(k).all():
        return "DFT invalide: axe k non fini."
    bands = np.asarray(data.bands, dtype=float)
    if bands.ndim != 2:
        return "DFT invalide: bands doit etre une matrice 2D."
    if bands.shape[1] != k.size:
        return (
            "DFT invalide: bands shape "
            f"{tuple(bands.shape)} incompatible avec k_distance len={k.size}."
        )
    return ""


def picker_band_curves(
    data: TheoryBandData | dict[str, Any],
    config: TheoryOverlayConfig | dict[str, Any],
    *,
    segment: str = "",
    ef_window: float = 0.0,
) -> list[PickerBandCurve]:
    """Return all source DFT curves for the picker.

    The picker intentionally uses the global band-structure path axis, like the
    Materials Project graph. Overlay-only transforms (branch-local k,
    ``k_scale``, ``k_shift``) are not applied here because they flatten the
    visual band diagram and make band identity hard to recognize.
    """
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    if validate_picker_data(data):
        return []
    k = picker_k_axis(data)
    bands = np.asarray(data.bands, dtype=float)
    if bands.ndim != 2 or bands.shape[1] != k.size:
        return []
    win = max(0.0, float(ef_window))
    out: list[PickerBandCurve] = []
    for idx, row in enumerate(bands):
        y = np.asarray(row, dtype=float)
        finite = y[np.isfinite(y)]
        crosses = bool(finite.size and float(np.nanmin(finite)) <= win and float(np.nanmax(finite)) >= -win)
        out.append(PickerBandCurve(idx, k.copy(), y.copy(), crosses))
    return out


def picker_k_axis(data: TheoryBandData | dict[str, Any]) -> np.ndarray:
    """Global band-structure axis for the picker.

    Prefer absolute MP distance when available; otherwise use stored
    ``k_distance``. Both represent the full high-symmetry path, unlike the
    ARPES overlay branch-local axis.
    """
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    k_abs = np.asarray(data.k_distance_abs, dtype=float)
    if k_abs.size == len(data.k_distance) and np.isfinite(k_abs).all():
        return k_abs
    return np.asarray(data.k_distance, dtype=float)


def picker_ticks(
    data: TheoryBandData | dict[str, Any],
    *,
    convention: str = PATH_CONVENTION_MP_BULK,
) -> list[PickerTick]:
    """High-symmetry tick marks on the picker axis."""
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    k = picker_k_axis(data)
    if k.size == 0:
        return []
    out: list[PickerTick] = []
    if data.branches:
        names = branch_display_names(data.branches)
        for i, (name, br) in enumerate(zip(names, data.branches)):
            try:
                start = max(0, min(int(br.get("start", 0)), k.size - 1))
                end = max(0, min(int(br.get("end", start)), k.size - 1))
            except (TypeError, ValueError):
                continue
            base = name.split(" (")[0]
            left, _, right = base.partition("-")
            if i == 0 and left:
                out.append(PickerTick(float(k[start]), path_label(left, convention)))
            if right:
                out.append(PickerTick(float(k[end]), path_label(right, convention)))
        return _merge_ticks(out)
    raw_k = np.asarray(data.k_distance, dtype=float)
    for item in data.labels:
        label = str(item.get("label") or "")
        pos = item.get("k")
        if not label or pos is None:
            continue
        try:
            idx = int(np.argmin(np.abs(raw_k - float(pos))))
        except (TypeError, ValueError):
            continue
        out.append(PickerTick(float(k[idx]), path_label(label, convention)))
    return _merge_ticks(out)


def path_label(label: str, convention: str = PATH_CONVENTION_MP_BULK) -> str:
    """Display label for the chosen path convention.

    ``arpes_pnictides`` keeps the raw MP label visible and appends the common
    ARPES 2D pnictide label when there is a useful low-energy alias. The alias
    is intentionally explicit because MP still supplies a 3D bulk path.
    """
    label = str(label or "")
    if convention != PATH_CONVENTION_ARPES_PNICTIDES:
        return label
    aliases = {
        "Y": "M",
        "P": "M/S",
        "Y₁": "M/S",
        "Y_1": "M/S",
        "N": "X/S",
    }
    alias = aliases.get(label)
    return f"{label}\n({alias})" if alias else label


def picker_segment_span(data: TheoryBandData | dict[str, Any], segment: str) -> tuple[float, float] | None:
    """Return selected segment span on the picker axis, if known."""
    data = TheoryBandData.from_dict(data) if isinstance(data, dict) else data
    if not segment or not data.branches:
        return None
    names = branch_display_names(data.branches)
    for name, br in zip(names, data.branches):
        if name != segment and name.split(" (")[0] != segment:
            continue
        k = picker_k_axis(data)
        try:
            start = max(0, min(int(br.get("start", 0)), k.size - 1))
            end = max(0, min(int(br.get("end", start)), k.size - 1))
        except (TypeError, ValueError):
            return None
        return tuple(sorted((float(k[start]), float(k[end]))))
    return None


def _merge_ticks(ticks: list[PickerTick]) -> list[PickerTick]:
    merged: list[PickerTick] = []
    for tick in ticks:
        if merged and abs(merged[-1].x - tick.x) <= 1e-9:
            if tick.label and tick.label not in merged[-1].label.split("|"):
                merged[-1] = PickerTick(tick.x, f"{merged[-1].label}|{tick.label}")
            continue
        merged.append(tick)
    return merged
