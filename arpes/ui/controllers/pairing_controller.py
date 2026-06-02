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
    build_pseudo_entries_from_logbook,
    find_bms_for_fs,
    find_fs_for_bm,
)
from arpes.physics.bm_cut_overlay import compute_bm_cut_in_fs_frame


def _augmented_files(session) -> dict:
    """Retourne session.files étendu avec les records logbook non chargés.

    Les vrais FileEntry priment via dict merge (session.files override
    les pseudo-entries du logbook s'il y a conflit de clé).
    """
    pseudo = build_pseudo_entries_from_logbook(session)
    return {**pseudo, **(session.files or {})}


def _normalized_augmented_files(session) -> dict:
    files = _augmented_files(session)
    for entry in files.values():
        parent = getattr(entry, "parent_fs_path", None)
        if parent:
            try:
                entry.parent_fs_path = session.key_for_path(parent)
            except Exception:
                pass
    return files


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
    def _session_key(self, path: str | None) -> str | None:
        if not path:
            return None
        try:
            return self._session.key_for_path(path)
        except Exception:
            return path

    def _pin_fs_path(self, path: str | None) -> None:
        """Épingle une FS comme contexte pour l'overlay BM cuts."""
        path = self._session_key(path)
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
        bm_key = self._session_key(path)
        entry = self._session.files.get(bm_key)
        if entry is None:
            return None
        if getattr(entry.meta, "scan_kind", "") != "BM":
            return None
        matches = find_fs_for_bm(
            entry, bm_key, _normalized_augmented_files(self._session), criteria
        )
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
            key = self._session_key(path)
            entry = self._session.files.get(key)
            if entry is not None and getattr(entry.meta, "scan_kind", "") == "FS":
                return key
        return getattr(self._parent, "_pinned_fs_path", None)

    # ---------------------------------------------------------------
    # Bound BMs for current FS (pour overlay / tree)
    # ---------------------------------------------------------------
    def _user_criteria(self) -> PairingCriteria:
        """Construit criteria depuis sliders panel FS si présents, sinon defaults."""
        c = PairingCriteria()
        ctrls = getattr(self._parent, "_fs_controls", None)
        if ctrls is None:
            return c
        sp_hv = getattr(ctrls, "sp_pairing_hv_tol", None)
        sp_az = getattr(ctrls, "sp_pairing_azi_tol", None)
        hv_pct = float(sp_hv.value()) if sp_hv is not None else 5.0
        az_deg = float(sp_az.value()) if sp_az is not None else 2.0
        return PairingCriteria(
            same_folder=c.same_folder, folder_depth=c.folder_depth,
            hv_tolerance_rel=hv_pct / 100.0, azi_tolerance_deg=az_deg,
            require_polarization=c.require_polarization,
            require_sample=c.require_sample,
        )

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
        return find_bms_for_fs(
            fs_entry, fs_path, _normalized_augmented_files(self._session),
            criteria or self._user_criteria(),
        )

    # ---------------------------------------------------------------
    # Collect BM cuts pour overlay Phase B (B.2).
    # ---------------------------------------------------------------
    def _collect_bm_cuts_for_active_fs(
        self,
        fs_metadata: dict | None = None,
        *,
        criteria: PairingCriteria | None = None,
        work_func: float | None = None,
        a_lattice: float = 3.96,
    ) -> list:
        """Pour la FS active, calcule la projection de toutes les BMs liées.

        Args:
            fs_metadata: dict raw_data["metadata"] de la FS active. Si None,
                tente self._raw_data["metadata"] (cas où la FS est current).
            criteria: filtre auto-discovery (defaults si None).
            work_func: φ. Si None, lu depuis self._params.sp_phi.

        Returns:
            list[BMCutLine].
        """
        fs_path = self._active_fs_path()
        if not fs_path:
            return []
        fs_entry = self._session.files.get(self._session.key_for_path(fs_path))
        if fs_entry is None:
            return []
        if fs_metadata is None:
            raw = getattr(self._parent, "_raw_data", None)
            fs_metadata = (raw or {}).get("metadata", {}) if raw else {}
        if work_func is None:
            try:
                work_func = float(self._params.sp_phi.value())
            except Exception:
                work_func = 4.031
        bms = find_bms_for_fs(
            fs_entry, fs_path, _normalized_augmented_files(self._session),
            criteria or self._user_criteria(),
        )
        cuts = []
        for match in bms:
            cut = compute_bm_cut_in_fs_frame(
                match.entry, match.path, fs_entry, fs_path, fs_metadata,
                work_func=float(work_func), a_lattice=float(a_lattice),
            )
            if cut is not None:
                cuts.append(cut)
        return cuts

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
        if verb == "collect_cuts":
            return self._collect_bm_cuts_for_active_fs(
                payload.get("fs_metadata"),
                work_func=payload.get("work_func"),
                a_lattice=float(payload.get("a_lattice", 3.96)),
            )
        if verb == "toggle_cuts":
            visible = bool(payload.get("visible", False))
            self._parent._show_bm_cuts = visible
            try:
                self._draw_fs_tab()
            except Exception:
                pass
            return None
        if verb == "diagnose":
            return self._diagnose_pairing()
        raise ValueError(f"_pairing_action: verb inconnu '{verb}'")

    # ---------------------------------------------------------------
    # Diagnostic : pourquoi telle BM ne se relie pas à la FS active.
    # ---------------------------------------------------------------
    def _diagnose_pairing(self) -> str:
        from PyQt6.QtWidgets import QMessageBox
        from arpes.io.file_pairing import _is_compatible_auto, _same_folder

        fs_path = self._active_fs_path()
        files = _normalized_augmented_files(self._session)
        lines: list[str] = []
        if not fs_path:
            txt = "Aucune FS active. Charge une FS ou pin une FS via clic droit."
            QMessageBox.information(self._parent, "Diagnostic pairing", txt)
            return txt
        fs_entry = files.get(fs_path) or self._session.files.get(
            self._session.key_for_path(fs_path)
        )
        if fs_entry is None:
            txt = f"FS active introuvable dans session : {fs_path}"
            QMessageBox.warning(self._parent, "Diagnostic pairing", txt)
            return txt
        c = PairingCriteria()
        sk_fs = getattr(fs_entry.meta, "scan_kind", "?")
        lines.append(f"FS : {fs_path}")
        lines.append(f"  scan_kind={sk_fs}  hv={fs_entry.meta.hv}  "
                     f"azi={fs_entry.meta.azi}  pol={fs_entry.meta.polarization}")
        lines.append(f"Critères : folder_depth={c.folder_depth} "
                     f"hv_tol={c.hv_tolerance_rel*100:.0f}%  azi_tol={c.azi_tolerance_deg}°")
        lines.append(f"Candidats (hors FS active) dans pool augmenté : {len(files)-1}")
        lines.append("")
        per_kind = {"BM": 0, "FS": 0, "unknown": 0, "other": 0}
        rejects = {"not_bm": 0, "folder": 0, "hv": 0, "azi": 0, "pol_sample": 0}
        matches = 0
        details: list[str] = []
        for path, entry in files.items():
            if path == fs_path:
                continue
            sk = getattr(entry.meta, "scan_kind", "") or "unknown"
            per_kind[sk] = per_kind.get(sk, 0) + 1 if sk in per_kind else per_kind["other"] + 1
            if sk != "BM":
                rejects["not_bm"] += 1
                continue
            folder_ok = _same_folder(path, fs_path, depth=c.folder_depth)
            if not folder_ok:
                rejects["folder"] += 1
                details.append(f"  REJET[folder] {path} (hors dossier FS, depth={c.folder_depth})")
                continue
            compat, dist = _is_compatible_auto(entry, path, fs_entry, fs_path, c)
            if not compat:
                from arpes.io.file_pairing import _relative_hv_diff, _azi_diff_deg
                hv_d = _relative_hv_diff(entry.meta.hv, fs_entry.meta.hv)
                azi_d = _azi_diff_deg(entry.meta.azi, fs_entry.meta.azi)
                if hv_d > c.hv_tolerance_rel:
                    rejects["hv"] += 1
                    details.append(f"  REJET[hv] {path} : Δhv/max = {hv_d*100:.2f}% > {c.hv_tolerance_rel*100:.0f}%")
                elif azi_d > c.azi_tolerance_deg:
                    rejects["azi"] += 1
                    details.append(f"  REJET[azi] {path} : Δazi = {azi_d:.2f}° > {c.azi_tolerance_deg}°")
                else:
                    rejects["pol_sample"] += 1
                    details.append(f"  REJET[pol/sample] {path}")
                continue
            matches += 1
            details.append(f"  OK distance={dist:.3f} : {path}")
        lines.append(f"scan_kind : BM={per_kind['BM']} FS={per_kind['FS']} "
                     f"unknown={per_kind['unknown']} other={per_kind.get('other',0)}")
        lines.append(f"Compatibles : {matches}")
        lines.append(f"Rejets : not_BM={rejects['not_bm']} folder={rejects['folder']} "
                     f"hv={rejects['hv']} azi={rejects['azi']} pol/sample={rejects['pol_sample']}")
        lines.append("")
        lines.extend(details[:30])
        if len(details) > 30:
            lines.append(f"... {len(details)-30} candidats supplémentaires omis.")
        text = "\n".join(lines)
        msg = QMessageBox(self._parent)
        msg.setWindowTitle("Diagnostic pairing FS ↔ BMs")
        msg.setText("Détail dans le contenu déroulant ci-dessous.")
        msg.setDetailedText(text)
        msg.exec()
        return text
