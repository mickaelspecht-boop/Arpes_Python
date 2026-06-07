"""Infer `scan_kind` from a loader metadata dict.

A.1 - `scan_kind` reliability. Single source of truth for scan type
(BM, FS, KZ, EDC, unknown). Pure, headless, testable.

Convention:
- "BM"   : 2D band map (cut angle vs energy, fixed polar).
- "FS"   : 3D Fermi surface (kx ky E volume) or 2D (kx ky map).
- "KZ"   : photon-energy scan (hv vs kpar).
- "EDC"  : 1D energy distribution curve (rare).
- "unknown" : impossible to determine.
"""
from __future__ import annotations


def infer_scan_kind(metadata: dict | None, *, data_ndim: int | None = None) -> str:
    """Return the canonical scan_kind from a loader metadata dict.

    Args:
        metadata: dict `data["metadata"]` returned by a loader (CLS, Bessy,
            Solaris...). May directly contain `scan_kind`, or keys that allow
            it to be inferred (`fs_data`, `kz_scan`, etc.).
        data_ndim: optional dimension of the main data array, used to
            disambiguate 2D (BM) vs 3D (FS) when metadata is thin.

    Returns:
        A str ∈ {"BM", "FS", "KZ", "EDC", "unknown"}.
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
