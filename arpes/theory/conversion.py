"""Conversion from pymatgen-like band structures to ARPES theory data."""
from __future__ import annotations

from typing import Any

import numpy as np

from arpes.theory.data import TheoryBandData, _finite_float
from arpes.theory.labels import _clean_label


def bandstructure_to_theory_data(
    bandstructure: Any,
    *,
    material_id: str,
    formula: str = "",
    source: str = "materials_project",
    path_type: str = "setyawan_curtarolo",
    with_projections: bool = False,
    crystal_system: str = "",
) -> TheoryBandData:
    """Convert a pymatgen-like band structure object to JSON-safe arrays."""
    efermi = _finite_float(getattr(bandstructure, "efermi", 0.0), 0.0)
    bands_obj = getattr(bandstructure, "bands", None)
    if isinstance(bands_obj, dict):
        channels = []
        for values in bands_obj.values():
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 2:
                channels.append(arr)
        if not channels:
            bands = np.asarray([], dtype=float)
        elif len(channels) == 1:
            bands = channels[0] - efermi
        else:
            bands = np.concatenate(channels, axis=0) - efermi
    else:
        bands = np.asarray(bands_obj, dtype=float) - efermi
    if bands.ndim != 2:
        raise ValueError("Band structure invalide: bandes DFT non matricielles.")

    k_distance = _k_distance_from_bandstructure(bandstructure, bands.shape[1])
    labels = _labels_from_bandstructure(bandstructure, k_distance)
    bands_list = bands.astype(float).tolist()
    from .band_select import aggregate_projection_character, compute_band_meta
    band_meta = compute_band_meta(bands_list)
    band_character: list[str] = []
    if with_projections:
        try:
            band_character = aggregate_projection_character(
                getattr(bandstructure, "projections", None),
                _structure_elements(bandstructure),
            )
        except Exception:
            band_character = []
        if band_character and len(band_character) != len(bands_list):
            band_character = []
    branches = _branches_from_bandstructure(bandstructure, bands.shape[1])
    k_abs = _k_distance_abs_from_bandstructure(bandstructure, bands.shape[1])
    return TheoryBandData(
        source=source,
        material_id=material_id,
        formula=formula,
        efermi=efermi,
        k_distance=[float(x) for x in k_distance],
        bands=bands_list,
        labels=labels,
        path_type=path_type,
        band_meta=band_meta,
        band_character=band_character,
        branches=branches,
        crystal_system=str(crystal_system or ""),
        k_distance_abs=[float(x) for x in k_abs],
    )


def _branches_from_bandstructure(bandstructure: Any, n_k: int) -> list[dict[str, Any]]:
    raw = getattr(bandstructure, "branches", None)
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for br in raw:
        try:
            name = str(br.get("name", "") if isinstance(br, dict) else getattr(br, "name", ""))
            s = int(br.get("start_index") if isinstance(br, dict) else getattr(br, "start_index"))
            e = int(br.get("end_index") if isinstance(br, dict) else getattr(br, "end_index"))
        except (TypeError, ValueError, AttributeError):
            continue
        s = max(0, min(s, n_k - 1))
        e = max(0, min(e, n_k - 1))
        if e < s:
            s, e = e, s
        out.append({"name": name, "start": s, "end": e})
    return out


def _structure_elements(bandstructure: Any) -> list[str]:
    struct = getattr(bandstructure, "structure", None)
    if struct is None:
        return []
    out: list[str] = []
    try:
        for site in struct:
            sp = getattr(site, "specie", None) or getattr(site, "species", None)
            sym = getattr(sp, "symbol", None)
            out.append(str(sym) if sym else str(sp))
    except Exception:
        return []
    return out


def _k_distance_from_bandstructure(bandstructure: Any, n_k: int) -> np.ndarray:
    dist = getattr(bandstructure, "distance", None)
    if dist is not None:
        arr = np.asarray(dist, dtype=float)
        if arr.size == n_k and np.all(np.isfinite(arr)):
            return _scaled_k_axis(arr)
    kpoints = getattr(bandstructure, "kpoints", None) or []
    coords = []
    for kp in kpoints:
        frac = getattr(kp, "frac_coords", None)
        coords.append(np.asarray(frac if frac is not None else kp, dtype=float))
    if len(coords) == n_k:
        out = [0.0]
        for prev, cur in zip(coords, coords[1:]):
            out.append(out[-1] + float(np.linalg.norm(cur - prev)))
        return _scaled_k_axis(np.asarray(out, dtype=float))
    return np.linspace(-1.0, 1.0, n_k)


def _k_distance_abs_from_bandstructure(bandstructure: Any, n_k: int) -> np.ndarray:
    dist = getattr(bandstructure, "distance", None)
    if dist is None:
        return np.empty(0, dtype=float)
    arr = np.asarray(dist, dtype=float)
    if arr.size == n_k and np.all(np.isfinite(arr)):
        return arr
    return np.empty(0, dtype=float)


def _scaled_k_axis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(hi - lo) or abs(hi - lo) <= 1e-12:
        return np.linspace(-1.0, 1.0, values.size)
    centered = values - 0.5 * (lo + hi)
    half = max(abs(float(np.nanmin(centered))), abs(float(np.nanmax(centered))), 1e-12)
    return centered / half


def _labels_from_bandstructure(bandstructure: Any, k_distance: np.ndarray) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    raw = getattr(bandstructure, "labels_dict", None) or {}
    if isinstance(raw, dict):
        for label, kp in raw.items():
            try:
                coord = np.asarray(getattr(kp, "frac_coords", kp), dtype=float)
                kpoints = getattr(bandstructure, "kpoints", None) or []
                idx = _nearest_kpoint_index(kpoints, coord)
                x = float(k_distance[idx]) if idx is not None else None
            except Exception:
                x = None
            labels.append({"label": _clean_label(label), "k": x})
    labels.sort(key=lambda item: float(item["k"]) if item.get("k") is not None else 1e9)
    return labels


def _nearest_kpoint_index(kpoints: list[Any], coord: np.ndarray) -> int | None:
    if not kpoints:
        return None
    distances = []
    for kp in kpoints:
        frac = np.asarray(getattr(kp, "frac_coords", kp), dtype=float)
        distances.append(float(np.linalg.norm(frac - coord)))
    return int(np.argmin(distances))
