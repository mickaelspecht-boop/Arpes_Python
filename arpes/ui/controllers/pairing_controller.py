"""Pairing controller: FS pinning for BM cuts overlay.

A.4 of the BM<->FS plan. Simplified fallback instead of a true multi-active
`_current_fs_path` + `_current_bm_path` model.

Concept: `_pinned_fs_path` stores the context FS when switching to a BM. This
lets the overlay know which FS should receive BM lines without breaking the
single `_current_path` model.

Auto-pin: whenever a BM is loaded, find the compatible FS (manual or
auto-discovered) and pin it if found.
"""
from __future__ import annotations

from arpes.io.file_pairing import (
    PairingCriteria,
    build_pseudo_entries_from_logbook,
    find_bms_for_fs,
    find_fs_for_bm,
)
from arpes.core.sample import lattice_a_for_entry, work_function_for_entry
from arpes.physics.bm_cut_overlay import BZGeometry, compute_bm_cut_in_fs_frame


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
    # P3.1: writes through to parent are allow-listed (fail-loud on typo).
    _OWN_ATTRS = frozenset({"_parent"})
    _PARENT_WRITES = frozenset()

    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
        elif name in self._PARENT_WRITES:
            setattr(self._parent, name, value)
        else:
            raise AttributeError(
                f"{type(self).__name__} refuses to write '{name}': missing from "
                "_PARENT_WRITES (typo?). Add it to _PARENT_WRITES "
                "if the parent attribute is legitimate."
            )

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
        """Pin an FS as context for the BM cuts overlay."""
        path = self._session_key(path)
        self._parent._pinned_fs_path = path
        if hasattr(self._parent, "_status") and path:
            self._status(f"Context FS pinned: {path}")

    def _unpin_fs_path(self) -> None:
        """Remove the pin."""
        self._parent._pinned_fs_path = None

    # ---------------------------------------------------------------
    # Auto-pin
    # ---------------------------------------------------------------
    def _auto_pin_fs_for_current_bm(
        self, *, criteria: PairingCriteria | None = None
    ) -> str | None:
        """Detect and pin the FS compatible with current_path (if BM).

        Returns the pinned FS path or None.
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
        # Pick the first match (manual first, otherwise closest).
        chosen = matches[0].path
        self._pin_fs_path(chosen)
        return chosen

    # ---------------------------------------------------------------
    # Active FS path (for overlay)
    # ---------------------------------------------------------------
    def _active_fs_path(self) -> str | None:
        """Return the active FS path for the overlay.

        Logic: if the current file is an FS, return current_path; otherwise
        return pinned_fs_path (which may be None).
        """
        path = getattr(self._parent, "_current_path", None)
        if path:
            key = self._session_key(path)
            entry = self._session.files.get(key)
            if entry is not None and getattr(entry.meta, "scan_kind", "") == "FS":
                return key
        return getattr(self._parent, "_pinned_fs_path", None)

    # ---------------------------------------------------------------
    # Bound BMs for current FS (for overlay/tree)
    # ---------------------------------------------------------------
    def _user_criteria(self) -> PairingCriteria:
        """Build criteria from FS panel sliders if present, otherwise defaults."""
        c = PairingCriteria()
        ctrls = getattr(self._parent, "_fs_controls", None)
        if ctrls is None:
            return c
        sp_hv = getattr(ctrls, "sp_pairing_hv_tol", None)
        sp_az = getattr(ctrls, "sp_pairing_azi_tol", None)
        cmb_dir = getattr(ctrls, "cmb_direction", None)
        hv_pct = float(sp_hv.value()) if sp_hv is not None else 5.0
        az_deg = float(sp_az.value()) if sp_az is not None else 2.0
        direction = ""
        if cmb_dir is not None:
            txt = cmb_dir.currentText()
            direction = "" if txt.lower().startswith("all") else txt
        return PairingCriteria(
            same_folder=c.same_folder, folder_depth=c.folder_depth,
            hv_tolerance_rel=hv_pct / 100.0, azi_tolerance_deg=az_deg,
            require_polarization=c.require_polarization,
            require_sample=c.require_sample,
            direction_filter=direction,
        )

    def _bound_bms_for_active_fs(
        self, *, criteria: PairingCriteria | None = None
    ) -> list:
        """Return PairingMatch[] for BMs bound to the active FS."""
        fs_path = self._active_fs_path()
        if not fs_path:
            return []
        # fs_path is already a session key (from _active_fs_path); do NOT pass it
        # through key_for_path again — that is not idempotent on nested keys
        # ("BNA_S1/FS3" -> "FS3") and was why pairing returned nothing on
        # subfolder layouts (CLS2026) while working on flat ones (Ba122).
        fs_entry = self._fs_entry_for_key(fs_path)
        if fs_entry is None:
            return []
        return find_bms_for_fs(
            fs_entry, fs_path, _normalized_augmented_files(self._session),
            criteria or self._user_criteria(),
        )

    def _fs_entry_for_key(self, fs_path: str):
        """Look up an FS entry by its session key, tolerant of already-keyed or
        absolute paths (key_for_path is not idempotent on nested keys)."""
        files = self._session.files or {}
        return files.get(fs_path) or files.get(self._session.key_for_path(fs_path))

    # ---------------------------------------------------------------
    # Collect BM cuts pour overlay Phase B (B.2).
    # ---------------------------------------------------------------
    def _collect_bm_cuts_for_active_fs(
        self,
        fs_metadata: dict | None = None,
        *,
        criteria: PairingCriteria | None = None,
        work_func: float | None = None,
        a_lattice: float = 0.0,
    ) -> list:
        """For the active FS, compute projections of all bound BMs.

        Args:
            fs_metadata: dict raw_data["metadata"] for the active FS. If None,
                tries self._raw_data["metadata"] when the FS is current.
            criteria: auto-discovery filter (defaults if None).
            work_func: phi. If None, read from self._params.sp_phi.

        Returns:
            list[BMCutLine].
        """
        fs_path = self._active_fs_path()
        if not fs_path:
            return []
        fs_entry = self._fs_entry_for_key(fs_path)  # see _bound_bms note (no double key)
        if fs_entry is None:
            return []
        if fs_metadata is None:
            raw = getattr(self._parent, "_raw_data", None)
            fs_metadata = (raw or {}).get("metadata", {}) if raw else {}
        if work_func is None:
            fallback = float(getattr(self._session, "work_func", 0.0) or 0.0)
            try:
                fallback = float(self._params.sp_phi.value())
            except Exception:
                pass
            work_func = work_function_for_entry(
                self._session,
                fs_entry,
                fallback=fallback,
                entry_key=fs_path,
            )
        if float(work_func or 0.0) <= 0.0:
            # Fail loud instead of silently drawing nothing (was the #1 reason
            # "Show BM cuts" looked broken on samples without a logbook φ).
            raise ValueError(
                "BM cuts: set the work function φ first (needed to convert angles to k)."
            )
        if float(a_lattice or 0.0) <= 0.0:
            a_lattice = lattice_a_for_entry(self._session, fs_entry, fallback=0.0,
                                            entry_key=fs_path)
        if float(a_lattice or 0.0) <= 0.0:
            raise ValueError("BM cuts: set the lattice parameter a first.")
        criteria = criteria or self._user_criteria()
        bms = find_bms_for_fs(
            fs_entry, fs_path, _normalized_augmented_files(self._session),
            criteria,
        )
        geom = self._bz_geometry_for(fs_entry)
        cuts = []
        for match in bms:
            cut = compute_bm_cut_in_fs_frame(
                match.entry, match.path, fs_entry, fs_path, fs_metadata,
                work_func=float(work_func), a_lattice=float(a_lattice),
                overlay_max_hv_rel=float(criteria.hv_tolerance_rel),
                bz_geometry=geom,
            )
            if cut is not None:
                cuts.append(cut)
        return cuts

    def _bz_geometry_for(self, fs_entry) -> BZGeometry:
        """BZ shape/size from the FS panel + label convention from the entry.

        Headless or panel-less contexts fall back to the default square zone
        (the historical behaviour of the direction table).
        """
        overrides = dict(getattr(fs_entry, "fs_bz_label_overrides", {}) or {})
        ctrls = getattr(self._parent, "_fs_controls", None)
        params = None
        if ctrls is not None:
            try:
                params = ctrls.params()
            except Exception:
                params = None
        if params is None:
            return BZGeometry(label_overrides=overrides or None)
        return BZGeometry(
            shape=str(getattr(params, "bz_shape", "square") or "square"),
            half_x=float(getattr(params, "bz_half_x", 1.0) or 1.0),
            half_y=float(getattr(params, "bz_half_y", 1.0) or 1.0),
            angle_deg=float(getattr(params, "bz_angle_deg", 90.0) or 90.0),
            label_overrides=overrides or None,
        )

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
                a_lattice=float(payload.get("a_lattice", 0.0) or 0.0),
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
        raise ValueError(f"_pairing_action: unknown verb '{verb}'")

    # ---------------------------------------------------------------
    # Diagnostic: why a given BM does not bind to the active FS.
    # ---------------------------------------------------------------
    def _diagnose_pairing(self) -> str:
        from PyQt6.QtWidgets import QMessageBox
        from arpes.io.file_pairing import _is_compatible_auto, _same_folder

        fs_path = self._active_fs_path()
        files = _normalized_augmented_files(self._session)
        lines: list[str] = []
        if not fs_path:
            txt = "No active FS. Load an FS or pin one with the context menu."
            QMessageBox.information(self._parent, "Pairing diagnostics", txt)
            return txt
        fs_entry = files.get(fs_path) or self._session.files.get(
            self._session.key_for_path(fs_path)
        )
        if fs_entry is None:
            txt = f"Active FS not found in the session: {fs_path}"
            QMessageBox.warning(self._parent, "Pairing diagnostics", txt)
            return txt
        c = PairingCriteria()
        sk_fs = getattr(fs_entry.meta, "scan_kind", "?")
        lines.append(f"FS : {fs_path}")
        lines.append(f"  scan_kind={sk_fs}  hv={fs_entry.meta.hv}  "
                     f"azi={fs_entry.meta.azi}  pol={fs_entry.meta.polarization}")
        lines.append(f"Criteria: folder_depth={c.folder_depth} "
                     f"hv_tol={c.hv_tolerance_rel*100:.0f}%  azi_tol={c.azi_tolerance_deg}°")
        lines.append(f"Candidates (excluding active FS) in augmented pool: {len(files)-1}")
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
                details.append(f"  REJECT[folder] {path} (outside FS folder, depth={c.folder_depth})")
                continue
            compat, dist = _is_compatible_auto(entry, path, fs_entry, fs_path, c)
            if not compat:
                from arpes.io.file_pairing import _relative_hv_diff, _azi_diff_deg
                hv_d = _relative_hv_diff(entry.meta.hv, fs_entry.meta.hv)
                azi_d = _azi_diff_deg(entry.meta.azi, fs_entry.meta.azi)
                if hv_d > c.hv_tolerance_rel:
                    rejects["hv"] += 1
                    details.append(f"  REJECT[hv] {path}: delta hv/max = {hv_d*100:.2f}% > {c.hv_tolerance_rel*100:.0f}%")
                elif azi_d > c.azi_tolerance_deg:
                    rejects["azi"] += 1
                    details.append(f"  REJECT[azi] {path}: delta azi = {azi_d:.2f}° > {c.azi_tolerance_deg}°")
                else:
                    rejects["pol_sample"] += 1
                    details.append(f"  REJECT[pol/sample] {path}")
                continue
            matches += 1
            details.append(f"  OK distance={dist:.3f} : {path}")
        lines.append(f"scan_kind : BM={per_kind['BM']} FS={per_kind['FS']} "
                     f"unknown={per_kind['unknown']} other={per_kind.get('other',0)}")
        lines.append(f"Compatible: {matches}")
        lines.append(f"Rejected: not_BM={rejects['not_bm']} folder={rejects['folder']} "
                     f"hv={rejects['hv']} azi={rejects['azi']} pol/sample={rejects['pol_sample']}")
        lines.append("")
        lines.extend(details[:30])
        if len(details) > 30:
            lines.append(f"... {len(details)-30} additional candidates omitted.")
        text = "\n".join(lines)
        msg = QMessageBox(self._parent)
        msg.setWindowTitle("FS <-> BM Pairing Diagnostics")
        msg.setText("Details are available in the expandable section below.")
        msg.setDetailedText(text)
        msg.exec()
        return text
