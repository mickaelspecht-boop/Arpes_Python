"""Sample setup controller: the folder-load setup dialog (φ/a + logbook).

Single responsibility: decide when the SampleSetupDialog appears, persist its
results (session.sample_configs / current_sample / scoped logbooks) and the
Browse-only mode. Split out of LoadController (LOC cap + 1 controller = 1
responsibility).
"""
from __future__ import annotations

from pathlib import Path


class SampleSetupController:
    """Owns the Sample & logbook setup dialog lifecycle."""

    def __init__(self, parent):
        self._parent = parent

    @property
    def _session(self):
        return self._parent._session

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    def _sample_setup_action(self, verb: str, payload: dict | None = None):
        """Verb dispatch (PROXY_MAP budget): "folder_opened" | "open_dialog".

        folder_opened: auto-prompt once per folder — skipped when the session
        already has sample_configs (resume case) or is in Browse-only mode.
        open_dialog: manual edit via the browser "Samples…" button — always
        opens and clears Browse-only (the documented way out of that mode).
        """
        if verb == "folder_opened":
            if getattr(self._session, "browse_only", False):
                return  # user chose Browse only for this session
            if getattr(self._session, "sample_configs", None):
                self._warn_orphan_sample_keys()
                return
            return self._open_sample_setup(auto=True)
        if verb == "open_dialog":
            if getattr(self._session, "browse_only", False):
                self._session.browse_only = False
                self._session.save()
                self._status("Browse only disabled — full setup available again.")
            return self._open_sample_setup(auto=False)

    def _open_sample_setup(self, *, auto: bool) -> None:
        folder = getattr(self._session, "folder", None)
        if not folder:
            if not auto:
                self._status("Sample setup: open a data folder first.")
            return
        if getattr(self._parent, "_sample_dialog_open", False):
            return  # modal already showing (double folder-open guard)
        from arpes.core.sample_layout import detect_sample_layout
        layout = detect_sample_layout(folder)
        if auto and layout.mode == "single" and layout.n_root_files == 0:
            return  # nothing loadable yet — don't nag on empty folders
        from arpes.ui.widgets.dialogs.sample_setup_dialog import SampleSetupDialog
        self._parent._sample_dialog_open = True
        try:
            dialog = SampleSetupDialog(
                self._parent,
                folder_name=Path(folder).name,
                subfolders=[(s.key, s.n_files) for s in layout.subfolders],
                n_root_files=layout.n_root_files,
                detected_mode=layout.mode,
                existing=getattr(self._session, "sample_configs", {}) or {},
                session_default=getattr(self._session, "current_sample", {}) or {},
                folder_path=str(folder),
                existing_logbooks=getattr(self._session, "scoped_logbooks", {}) or {},
            )
            accepted = bool(dialog.exec())
        finally:
            self._parent._sample_dialog_open = False
        if not accepted:
            if getattr(dialog, "browse_only_requested", False):
                # Browse only still SAVES the φ/a the user typed (their
                # explicit request): they want to browse with the sample
                # parameters known, just without logbook/hν ceremony.
                n = self._save_dialog_configs(dialog)
                self._session.browse_only = True
                self._session.save()
                self._refresh_loaded_params()
                self._status(
                    "⚠ Browse only — raw-axis display, setup prompts disabled"
                    + (f" ({n} sample(s) φ/a saved)" if n else "")
                    + ". Use “Samples…” to leave this mode."
                )
            else:
                self._status("Sample setup skipped — φ/a will be requested at fit time.")
            return
        n = self._save_dialog_configs(dialog)
        n_logbooks = self._apply_dialog_logbooks(dialog.result_logbooks())
        if n or n_logbooks:
            self._status(
                f"✓ Sample setup — {n} sample(s) configured"
                + (f", {n_logbooks} logbook(s) attached" if n_logbooks else "")
            )
            self._refresh_loaded_params()
        else:
            self._status("Sample setup: no values entered — nothing saved.")

    def _save_dialog_configs(self, dialog) -> int:
        """Merge dialog φ/a into session.sample_configs / current_sample."""
        configs = dialog.result_configs()
        whole = configs.pop("", None)
        if whole is not None:
            merged = dict(getattr(self._session, "current_sample", {}) or {})
            merged.update(whole)
            self._session.current_sample = merged
        if configs:
            existing = dict(getattr(self._session, "sample_configs", {}) or {})
            for key, cfg in configs.items():
                base = dict(existing.get(key) or {})
                base.update(cfg)
                existing[key] = base
            self._session.sample_configs = existing
        if configs or whole is not None:
            self._session.save()
        return len(configs) + (1 if whole is not None else 0)

    def _apply_dialog_logbooks(self, wanted: list[dict]) -> int:
        """Attach each (rel, path, sheet) chosen in the dialog. Loud failures."""
        n_ok = 0
        ctrl = getattr(self._parent, "_logbook_ctrl", None)
        if ctrl is None or not wanted:
            return 0
        for item in wanted:
            rel, path, sheet = item["rel"], item["path"], item["sheet"]
            saved = (getattr(self._session, "scoped_logbooks", {}) or {}).get(rel) or {}
            mapping = saved.get("mapping") if (
                str(saved.get("path", "")) == str(path)
                and str(saved.get("sheet", "")) == str(sheet)
            ) else None
            try:
                ctrl.attach_scoped_silent(path, sheet, [rel],
                                          mapping_override=mapping)
                n_ok += 1
            except Exception as exc:
                self._status(
                    f"⚠ Logbook {Path(path).name} [{sheet}] → "
                    f"{rel or 'session'} failed: {exc}"
                )
        return n_ok

    def _refresh_loaded_params(self) -> None:
        """Mirror freshly saved φ/a into the visible parameter spinboxes.

        Without this, the dialog saves the values but the φ/Lattice fields in
        the side panel keep showing their old content — misleading. If a file
        is currently displayed, reload it so the saved φ/a actually apply to
        the axes on screen.
        """
        p = self._parent
        path = getattr(p, "_current_path", None)
        params = getattr(p, "_params", None)
        if not path or params is None:
            return
        from arpes.core.sample import sample_for_entry
        key = self._session.key_for_path(path)
        entry = self._session.get_or_create(key)
        sample = sample_for_entry(self._session, entry, entry_key=key)
        if sample.has_work_function:
            params.sp_phi.blockSignals(True)
            params.sp_phi.setValue(float(sample.work_function_eV))
            params.sp_phi.blockSignals(False)
        if sample.has_lattice_a:
            params.sp_crystal_a.blockSignals(True)
            params.sp_crystal_a.setValue(float(sample.a_angstrom))
            params.sp_crystal_a.blockSignals(False)
        if getattr(p, "_raw_data", None) is not None and (
            sample.has_work_function or sample.has_lattice_a
        ):
            self._status("Sample setup applied — recomputing axes with the new φ/a.")
            p._load_ctrl.load(path, force_reload=True)

    def _warn_orphan_sample_keys(self) -> None:
        """Resume case: warn when a saved sample key no longer matches a folder."""
        folder = getattr(self._session, "folder", None)
        configs = getattr(self._session, "sample_configs", {}) or {}
        if not folder or not configs:
            return
        root = Path(folder)
        orphans = [k for k in configs
                   if k and not (root / k).is_dir()]
        if orphans:
            self._status(
                "⚠ Sample config for "
                + ", ".join(f"'{k}'" for k in orphans)
                + " refers to subfolder(s) not found in this folder (renamed?)."
            )
