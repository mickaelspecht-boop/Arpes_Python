"""Pairing controller — gestion du pin FS pour overlay BM cuts.

A.4 du plan BM↔FS (cf BM_FS_ORGANIZATION_PLAN.md). Fallback simplifié
au lieu d'un vrai multi-actif `_current_fs_path` + `_current_bm_path`.

Concept : `_pinned_fs_path` mémorise la FS « contexte » lorsqu'on switche
sur une BM. Permet à l'overlay (Phase B) de savoir sur quelle FS dessiner
les lignes des BMs, sans casser le modèle `_current_path` unique.

Auto-pin : à chaque load d'une BM, trouve la FS compatible (manual ou
auto-discovery) et pin si trouvée.
"""
from __future__ import annotations

from arpes.io.file_pairing import (
    PairingCriteria,
    find_bms_for_fs,
    find_fs_for_bm,
)


class PairingController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    # ---------------------------------------------------------------
    # Pin / unpin
    # ---------------------------------------------------------------
    def _pin_fs_path(self, path: str | None) -> None:
        """Épingle une FS comme contexte pour l'overlay BM cuts."""
        self._parent._pinned_fs_path = path
        if hasattr(self._parent, "_status") and path:
            self._status(f"FS contexte épinglée : {path}")

    def _unpin_fs_path(self) -> None:
        """Retire l'épinglage."""
        self._parent._pinned_fs_path = None

    # ---------------------------------------------------------------
    # Auto-pin
    # ---------------------------------------------------------------
    def _auto_pin_fs_for_current_bm(
        self, *, criteria: PairingCriteria | None = None
    ) -> str | None:
        """Détecte la FS compatible avec le current_path (si BM) et pin.

        Retourne le path FS pinné ou None.
        """
        path = getattr(self._parent, "_current_path", None)
        if not path:
            return None
        entry = self._session.files.get(self._session.key_for_path(path))
        if entry is None:
            return None
        if getattr(entry.meta, "scan_kind", "") != "BM":
            return None
        matches = find_fs_for_bm(entry, path, self._session.files, criteria)
        if not matches:
            return None
        # Prend la première (manual prioritaire, sinon plus proche)
        chosen = matches[0].path
        self._pin_fs_path(chosen)
        return chosen

    # ---------------------------------------------------------------
    # Active FS path (pour overlay)
    # ---------------------------------------------------------------
    def _active_fs_path(self) -> str | None:
        """Retourne le path FS « actif » pour l'overlay.

        Logique : si le fichier courant est une FS, retourne current_path ;
        sinon retourne pinned_fs_path (peut être None).
        """
        path = getattr(self._parent, "_current_path", None)
        if path:
            entry = self._session.files.get(self._session.key_for_path(path))
            if entry is not None and getattr(entry.meta, "scan_kind", "") == "FS":
                return path
        return getattr(self._parent, "_pinned_fs_path", None)

    # ---------------------------------------------------------------
    # Bound BMs for current FS (pour overlay / tree)
    # ---------------------------------------------------------------
    def _bound_bms_for_active_fs(
        self, *, criteria: PairingCriteria | None = None
    ) -> list:
        """Retourne PairingMatch[] des BMs liées à la FS active."""
        fs_path = self._active_fs_path()
        if not fs_path:
            return []
        fs_entry = self._session.files.get(self._session.key_for_path(fs_path))
        if fs_entry is None:
            return []
        return find_bms_for_fs(fs_entry, fs_path, self._session.files, criteria)

    # ---------------------------------------------------------------
    # Verb-dispatch unique pour le proxy (CLAUDE.md plafond 150).
    # ---------------------------------------------------------------
    def _pairing_action(self, verb: str, payload: dict | None = None):
        """Dispatch verb-based pour exposer plusieurs actions via 1 entrée PROXY_MAP.

        Verbs supportés :
        - "pin"           payload={"path": str}        → _pin_fs_path
        - "unpin"         payload={}                   → _unpin_fs_path
        - "auto_pin_bm"   payload={}                   → _auto_pin_fs_for_current_bm
        - "active_fs"     payload={}                   → _active_fs_path
        - "bound_bms"     payload={}                   → _bound_bms_for_active_fs
        """
        payload = payload or {}
        if verb == "pin":
            self._pin_fs_path(payload.get("path"))
            return None
        if verb == "unpin":
            self._unpin_fs_path()
            return None
        if verb == "auto_pin_bm":
            return self._auto_pin_fs_for_current_bm()
        if verb == "active_fs":
            return self._active_fs_path()
        if verb == "bound_bms":
            return self._bound_bms_for_active_fs()
        raise ValueError(f"_pairing_action: verb inconnu '{verb}'")
