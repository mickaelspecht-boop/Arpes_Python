"""Logique pure de sélection / caractérisation des bandes DFT.

Aucun Qt, aucun réseau, aucun I/O. Sert la liste cochable (UI) et le
loader MP. Testable isolément.

- ``compute_band_meta`` : par bande {idx, e_min, e_max, crosses_ef}.
- ``bands_crossing_ef`` : indices traversant ±window autour de E=0.
- ``format_band_indices`` : inverse de ``parse_band_indices``
  ([1,3,5,6,7,8] → "1,3,5-8"), pour synchroniser checkbox ↔ champ legacy.
- ``aggregate_projection_character`` : projections pymatgen brutes →
  caractère orbital dominant par bande. Dégrade gracieusement si absent.
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
    """Métadonnées par bande. ``bands`` = liste de lignes (énergies sur k).

    Les énergies sont supposées déjà relatives à E_F (efermi soustrait).
    ``crosses_ef`` vrai si la bande passe à ±``ef_window`` de E=0 (avec
    ``ef_window`` <= 0 : test strict de traversée min<=0<=max).
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
    """Indices des bandes traversant ±``window`` autour de E=0.

    Recalcule depuis e_min/e_max pour rester cohérent quelle que soit la
    fenêtre demandée (band_meta peut avoir été calculé sans fenêtre).
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
    """[1,3,5,6,7,8] → "1,3,5-8". Inverse de ``parse_band_indices``.

    Dédoublonne, trie, compresse les runs consécutifs en ``lo-hi``.
    Liste vide → "".
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
    """Caractère orbital dominant par bande depuis projections pymatgen.

    ``projections`` : forme tolérée = mapping {Spin: ndarray
    (n_band, n_k, n_orbital, n_ion)} (pymatgen BandStructureSymmLine
    .projections) OU déjà un ndarray (n_band, n_k, n_orbital[, n_ion]).
    ``elements`` : symbole par ion (len == n_ion) pour étiqueter
    "Ti-d". Si absent ou projections vides → liste de "" (dégradation
    gracieuse, aucune exception).

    Étiquette = ``{élément}-{orbitale}`` du canal de poids cumulé max.
    Orbitales regroupées s/p/d/f par convention pymatgen (0=s, 1-3=p,
    4-8=d, 9-15=f).
    """
    arr = _projection_array(projections)
    if arr is None or arr.size == 0:
        return []
    # arr -> (n_band, n_k, n_orbital, n_ion) ; compléter dims manquantes
    while arr.ndim < 4:
        arr = arr[..., np.newaxis]
    n_band, _n_k, n_orb, n_ion = arr.shape
    weight = np.abs(arr) ** 2  # poids physique
    # somme sur k -> (n_band, n_orb, n_ion)
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
        # somme sur les canaux de spin disponibles
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
    """Index orbitale pymatgen → s/p/d/f. Repli "o{idx}" si schéma inconnu."""
    if n_orb <= 4:  # schéma compact s,p,d,f
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
