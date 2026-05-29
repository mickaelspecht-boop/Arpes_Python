"""Multi-zone MDC fit orchestration.

Each zone owns a snapshot of FitParams (bounds + n_pairs + kF_init + Γ...)
plus its own fit_result. Zones are persisted on FileEntry.fit_zones.

Reads from / writes to:
- entry.fit_zones        list of dicts {id, label, color_idx, active, fit_params, fit_result}
- entry.active_zone_id   str (UUID of currently selected zone in UI)
- entry.fit_result       legacy single-fit payload — mirrors active zone for back-compat
- entry.fit_params       legacy single-fit params — mirrors active zone bounds for UI sync

Phase 1: synchronous loop; no per-zone gamma_center; no single_branch mode.
Asymmetric-zone detection emits a status warning but no auto-switch.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any

import numpy as np

from arpes.core.fit_result_store import clear_fit_result, set_fit_result
from arpes.core.session import FitParams


ZONE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


def _new_zone_id() -> str:
    return uuid.uuid4().hex[:8]


def _params_to_dict(fp: FitParams | dict) -> dict:
    return asdict(fp) if not isinstance(fp, dict) else dict(fp)


class FitZonesController:
    """CRUD over entry.fit_zones with a single dispatch verb."""

    def __init__(self, parent):
        self._parent = parent

    # ---------------------------------------------------------------- helpers
    def _entry(self):
        p = self._parent
        if not getattr(p, "_current_path", None):
            return None
        key = p._session.key_for_path(p._current_path)
        return p._session.get_or_create(key)

    def _status(self, msg: str) -> None:
        try:
            self._parent._status(msg)
        except Exception:
            pass

    def _next_color_idx(self, zones: list[dict]) -> int:
        used = {int(z.get("color_idx", -1)) for z in zones}
        for i in range(len(ZONE_PALETTE)):
            if i not in used:
                return i
        return len(zones) % len(ZONE_PALETTE)

    # ----------------------------------------------------------- public verb
    def fit_zone_action(self, verb: str, payload: dict | None = None) -> dict:
        """Single dispatch for zone CRUD ; keeps PROXY_MAP footprint to 1."""
        entry = self._entry()
        if entry is None:
            return {"ok": False, "error": "no_current_file"}
        payload = payload or {}
        handler = getattr(self, f"_v_{verb}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown_verb:{verb}"}
        return handler(entry, payload)

    # ------------------------------------------------------- individual verbs
    def _v_add(self, entry, payload: dict) -> dict:
        fp_src = payload.get("fit_params")
        if fp_src is None:
            # snapshot current UI spinboxes
            try:
                fp_src = self._parent._params.get_fit_params()
            except Exception:
                fp_src = FitParams()
        zone = {
            "id": _new_zone_id(),
            "label": str(payload.get("label") or f"Z{len(entry.fit_zones) + 1}"),
            "color_idx": self._next_color_idx(entry.fit_zones),
            "active": True,
            "fit_params": _params_to_dict(fp_src),
            "fit_result": None,
        }
        for key in ("k_min", "k_max", "ev_start", "ev_end"):
            if key in payload:
                zone["fit_params"][key] = float(payload[key])
        entry.fit_zones.append(zone)
        entry.active_zone_id = zone["id"]
        self._save()
        return {"ok": True, "zone_id": zone["id"]}

    def _v_remove(self, entry, payload: dict) -> dict:
        zid = payload.get("zone_id")
        before = len(entry.fit_zones)
        entry.fit_zones = [z for z in entry.fit_zones if z.get("id") != zid]
        if entry.active_zone_id == zid:
            entry.active_zone_id = entry.fit_zones[0]["id"] if entry.fit_zones else None
            self._sync_legacy_from_active(entry)
        self._save()
        return {"ok": True, "removed": before - len(entry.fit_zones)}

    def _v_set_active(self, entry, payload: dict) -> dict:
        zid = payload.get("zone_id")
        if not any(z.get("id") == zid for z in entry.fit_zones):
            return {"ok": False, "error": "zone_not_found"}
        entry.active_zone_id = zid
        self._sync_legacy_from_active(entry)
        self._save()
        return {"ok": True}

    def _v_toggle_active(self, entry, payload: dict) -> dict:
        zid = payload.get("zone_id")
        for z in entry.fit_zones:
            if z.get("id") == zid:
                z["active"] = bool(payload.get("value", not z.get("active", True)))
                self._save()
                return {"ok": True, "active": z["active"]}
        return {"ok": False, "error": "zone_not_found"}

    def _v_rename(self, entry, payload: dict) -> dict:
        zid = payload.get("zone_id")
        new_label = str(payload.get("label", "")).strip()
        if not new_label:
            return {"ok": False, "error": "empty_label"}
        for z in entry.fit_zones:
            if z.get("id") == zid:
                z["label"] = new_label
                self._save()
                return {"ok": True}
        return {"ok": False, "error": "zone_not_found"}

    def _v_clear_results(self, entry, payload: dict) -> dict:
        clear_fit_result(entry)
        self._save()
        return {"ok": True}

    def _v_list(self, entry, payload: dict) -> dict:
        return {"ok": True, "zones": list(entry.fit_zones), "active_id": entry.active_zone_id}

    # ---------------------------------------------------------- result update
    def store_result(self, zone_id: str, fr: dict) -> None:
        """Called by fit_runner after a per-zone fit completes.

        Also snapshots current FitParams into the zone so the spinboxes,
        overlay rect and fit_result stay coherent (HIGH-4 from audit).
        """
        entry = self._entry()
        if entry is None:
            return
        fp_snapshot: dict | None = None
        try:
            fp_snapshot = _params_to_dict(self._parent._params.get_fit_params())
        except Exception:
            fp_snapshot = None
        if fp_snapshot is not None:
            for z in entry.fit_zones:
                if z.get("id") == zone_id:
                    z["fit_params"] = fp_snapshot
                    break
        set_fit_result(entry, fr, zone_id=zone_id)
        self._save()

    def active_zone(self, entry=None) -> dict | None:
        entry = entry or self._entry()
        if entry is None or not entry.fit_zones:
            return None
        for z in entry.fit_zones:
            if z.get("id") == entry.active_zone_id:
                return z
        return entry.fit_zones[0]

    def _sync_legacy_from_active(self, entry) -> None:
        z = self.active_zone(entry)
        if z is None:
            return
        entry.fit_result = z.get("fit_result")
        try:
            fp = FitParams(**{
                k: v for k, v in z.get("fit_params", {}).items()
                if k in FitParams.__dataclass_fields__
            })
            entry.fit_params = fp
        except Exception:
            pass

    def asymmetric_warning(self, zone: dict, gamma_center: float) -> str | None:
        """Return a warning string if the zone lies entirely on one side of gamma_center."""
        fp = zone.get("fit_params", {})
        k_min = float(fp.get("k_min", 0.0))
        k_max = float(fp.get("k_max", 0.0))
        if k_min >= gamma_center or k_max <= gamma_center:
            return (
                f"Zone {zone.get('label')} entièrement d'un côté de Γ"
                f" (k∈[{k_min:.3f},{k_max:.3f}], Γ={gamma_center:.3f}) :"
                " le modèle ±kF peut produire un kF miroir artefact."
            )
        return None

    # -------------------------------------------------------------- internals
    def _save(self) -> None:
        try:
            self._parent._session.save()
        except Exception as exc:
            self._status(
                f"⚠ Sauvegarde session échouée ({exc}). Zones non persistées."
            )

    def color_for_zone(self, zone: dict) -> str:
        idx = int(zone.get("color_idx", 0)) % len(ZONE_PALETTE)
        return ZONE_PALETTE[idx]
