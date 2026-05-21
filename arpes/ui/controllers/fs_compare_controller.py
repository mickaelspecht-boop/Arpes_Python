"""Controller compare FS polarisation (LV vs LH).

Charge 2 fichiers via le même `LoaderOrchestrator` que la chaîne principale,
sans toucher au `_current_path`/`_raw_data` actuels. Délègue le tracé au
``FsCompareCanvas``.
"""
from __future__ import annotations

from pathlib import Path

from arpes.io.loaders.common import load_arpes_file
from arpes.io.loader_orchestrator import LoaderOrchestrator
from arpes.physics.fs_ops import find_pol_partner, group_files_by_pol


class FsCompareController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    # ------------------------------------------------------------------
    #  Init & populate UI
    # ------------------------------------------------------------------

    def _refresh_fs_compare_selectors(self):
        """Re-rempli les combos depuis session.files (filtre FS uniquement)."""
        widget = getattr(self, "_fs_compare", None)
        if widget is None:
            return
        pairs: list[tuple[str, str]] = []
        session = getattr(self, "_session", None)
        folder = session.folder if session and session.folder else None
        for key in (session.files.keys() if session else []):
            full = str((folder / key) if folder else key)
            label = self._fs_compare_label_for(key)
            pairs.append((label, full))
        pairs.sort()
        widget.populate_selectors(pairs)

    def _fs_compare_label_for(self, key: str) -> str:
        """Étiquette combo : nom fichier + pol logbook si dispo."""
        session = getattr(self, "_session", None)
        if session is None:
            return key
        rec = self._fs_compare_record_for_key(key)
        pol = ""
        if rec:
            mapping = (session.logbook_mapping or {}).get("polarization", "")
            if mapping:
                pol = str(rec.get(mapping, "") or "").strip().upper()
        return f"{key}  [{pol}]" if pol else key

    def _fs_compare_record_for_key(self, key: str) -> dict | None:
        session = getattr(self, "_session", None)
        if not session:
            return None
        try:
            mgr = self._logbook_ctrl._mgr  # type: ignore[attr-defined]
        except Exception:
            mgr = None
        if mgr is not None:
            try:
                return mgr.find_record_for_path(
                    Path(session.folder) / key if session.folder else Path(key)
                )
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    #  User actions
    # ------------------------------------------------------------------

    def _on_fs_compare_pair_load(self):
        widget = getattr(self, "_fs_compare", None)
        if widget is None:
            return
        path_a, path_b = widget.selected_paths()
        if not path_a or not path_b:
            self._status("✗ Compare pol : choisir un fichier A et un fichier B.")
            return
        if path_a == path_b:
            self._status("⚠ Compare pol : A et B sont identiques.")
            return
        try:
            raw_a = self._fs_compare_load_minimal(path_a)
            raw_b = self._fs_compare_load_minimal(path_b)
        except Exception as exc:
            self._status(f"✗ Compare pol : chargement échec ({exc})")
            return
        label_a = self._fs_compare_label_for(Path(path_a).name)
        label_b = self._fs_compare_label_for(Path(path_b).name)
        diff_norm = widget.cmb_norm.currentText()
        msg = widget.draw_pair(
            raw_a, raw_b,
            label_a=label_a, label_b=label_b,
            diff_normalize=diff_norm,
        )
        widget.set_status(msg)
        self._status(msg)

    def _on_fs_compare_auto_suggest(self):
        """Auto-rempli B = partenaire pol opposée de A via logbook."""
        widget = getattr(self, "_fs_compare", None)
        if widget is None:
            return
        path_a, _ = widget.selected_paths()
        if not path_a:
            self._status("✗ Compare pol : sélectionner A d'abord.")
            return
        session = getattr(self, "_session", None)
        if session is None or not session.logbook_records:
            self._status("⚠ Compare pol : logbook absent — pas de suggestion auto.")
            return
        rec_a = self._fs_compare_record_for_key(Path(path_a).name)
        if not rec_a:
            self._status("⚠ Compare pol : A introuvable dans logbook.")
            return
        mapping = session.logbook_mapping or {}
        pol_col = mapping.get("polarization", "")
        pol_a = str(rec_a.get(pol_col, "") or "").strip().upper() if pol_col else ""
        if not pol_a:
            self._status("⚠ Compare pol : pol de A inconnue dans logbook.")
            return
        other_pol = "LH" if pol_a == "LV" else ("LV" if pol_a == "LH" else "")
        if not other_pol:
            self._status(f"⚠ Compare pol : pol={pol_a} non LV/LH — saisir B manuellement.")
            return
        # Reconstruit liste records avec path/filename peuplé (mapping['file']).
        file_col = mapping.get("file", "")
        records = []
        for r in session.logbook_records:
            rr = dict(r)
            if file_col and r.get(file_col):
                rr["path"] = str(r.get(file_col))
            records.append(rr)
        # Injecte material/run_id heuristiques si présents dans mapping.
        material_col = mapping.get("formula", "") or mapping.get("material", "")
        for rr in records:
            if material_col and rr.get(material_col):
                rr["material"] = str(rr.get(material_col))
            rr["run_id"] = str(rr.get("run_id", "") or rr.get("run", "") or "")
        grouped = group_files_by_pol(records, pol_key=pol_col or "Pol")
        cur_name = Path(path_a).name
        partner_name = find_pol_partner(grouped, cur_name, other_pol=other_pol)
        if not partner_name:
            self._status(f"⚠ Compare pol : aucun partenaire {other_pol} dans logbook.")
            return
        # Combo B contient des chemins complets ; partner_name = nom court.
        for i in range(widget.cmb_b.count()):
            data = widget.cmb_b.itemData(i)
            if data and Path(data).name == partner_name:
                widget.cmb_b.setCurrentIndex(i)
                self._status(f"✓ Compare pol : suggéré B = {partner_name} ({other_pol}).")
                return
        self._status(f"⚠ Compare pol : {partner_name} trouvé en logbook mais absent du dossier.")

    def _fs_compare_load_minimal(self, path: str) -> dict:
        """Charge un fichier en isolation (sans toucher état app courant)."""
        from arpes.ui.controllers.load_controller import LoadController  # cycle safe
        # Construit un entry temporaire compatible LoaderOrchestrator.build_context
        from arpes.core.session import FileEntry
        entry = FileEntry()
        wf = float(getattr(self._session, "work_func", 4.031))
        orch = LoaderOrchestrator(load_arpes_file, lambda fmt, md: str(fmt or ""))
        res = orch.load(
            path, entry,
            work_func=wf, ef_offset=entry.ef_offset,
            hv=None, angle_offsets=None, bessy_energy_reference="ef",
        )
        return res.data or {}
