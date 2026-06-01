"""Inférence de `scan_kind` depuis un dict metadata loader.

A.1 — Fiabilité `scan_kind`. Source de vérité unique pour le type de scan
(BM, FS, KZ, EDC, unknown). Pure, headless, testable.

Convention :
- "BM"   : band map 2D (cut angle vs énergie, polar fixe).
- "FS"   : Fermi surface 3D (kx ky E volume) ou 2D (kx ky map).
- "KZ"   : scan en énergie de photons (hv vs kpar).
- "EDC"  : energy distribution curve 1D (rare).
- "unknown" : impossible à déterminer.
"""
from __future__ import annotations


def infer_scan_kind(metadata: dict | None, *, data_ndim: int | None = None) -> str:
    """Renvoie le scan_kind canonique d'un dict metadata loader.

    Args:
        metadata: dict `data["metadata"]` retourné par un loader (CLS, Bessy,
            Solaris…). Peut contenir directement `scan_kind`, ou des clés qui
            permettent de l'inférer (`fs_data`, `kz_scan`, etc.).
        data_ndim: optionnel — dimension du tableau de données principal,
            pour disambiguer 2D (BM) vs 3D (FS) quand metadata est mince.

    Returns:
        Un str ∈ {"BM", "FS", "KZ", "EDC", "unknown"}.
    """
    meta = metadata or {}
    raw = str(meta.get("scan_kind") or "").strip()
    if raw in {"BM", "FS", "KZ", "EDC"}:
        return raw
    if meta.get("kz_scan") or meta.get("kz_data") or meta.get("kz_kind"):
        return "KZ"
    if meta.get("fs_data") is not None:
        return "FS"
    if data_ndim is not None:
        if data_ndim == 1:
            return "EDC"
        if data_ndim == 2:
            return "BM"
        if data_ndim == 3:
            return "FS"
    return "unknown"
