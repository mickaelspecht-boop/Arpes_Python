"""Consistent writes for entry.fit_result and per-zone fit results."""
from __future__ import annotations

from typing import Any


def set_fit_result(entry: Any, fr: dict | None, *, zone_id: str | None = None) -> None:
    """Write ``fr`` to ``entry`` with consistent multi-zone mirroring.

    Args:
        entry: FileEntry-like object (must expose ``fit_result``,
            ``fit_zones`` and ``active_zone_id``).
        fr: New fit_result payload (or ``None`` to clear).
        zone_id: Optional zone UUID to also update. If omitted, only
            updates the active zone (if one exists) to keep
            ``entry.fit_result`` in sync with the visible zone.
    """
    target_zone_id = zone_id or getattr(entry, "active_zone_id", None)
    zones = getattr(entry, "fit_zones", None) or []
    if target_zone_id:
        for z in zones:
            if z.get("id") == target_zone_id:
                z["fit_result"] = fr
                break
    # Mirror to the legacy single-fit slot only when the updated zone is
    # active (or no zone was named, falling back to legacy single-fit mode).
    if not target_zone_id or target_zone_id == getattr(entry, "active_zone_id", None):
        entry.fit_result = fr


def clear_fit_result(entry: Any) -> None:
    """Clear both the legacy slot and every per-zone fit_result."""
    entry.fit_result = None
    for z in getattr(entry, "fit_zones", None) or []:
        z["fit_result"] = None


def restore_fit_result(entry: Any, *, fit_result: dict | None, fit_zones: list | None) -> None:
    """Restore a full snapshot (legacy slot + per-zone) after a failed save.

    Bypasses the active-zone mirroring of ``set_fit_result``: the caller holds
    a coherent backup of BOTH ``entry.fit_result`` and the whole
    ``entry.fit_zones`` list and wants to roll both back verbatim.
    """
    entry.fit_result = fit_result
    entry.fit_zones = list(fit_zones) if fit_zones is not None else []
