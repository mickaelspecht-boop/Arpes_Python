"""Save/Open/Recent pour fichiers session nommés et partageables.

Permet d'exporter l'état complet (`Session.to_payload`) vers un JSON arbitraire,
puis de le réimporter (relocate folder si absent) et repopuler le browser. Les
fichiers data ne sont PAS embarqués : le destinataire doit posséder son propre
dossier de données et l'app résout les chemins relatifs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from arpes.io.recent_sessions import add_recent, list_recent, remove_recent
from arpes.ui.widgets.dialogs.session_diff import SessionDiffDialog

SESSION_EXT = ".arpes-session.json"
SESSION_FILTER = f"ARPES Session (*{SESSION_EXT} *.json)"


class SessionIOController:
    """Backed by ``ArpesExplorer``: exposes save/open/recent slots."""

    def __init__(self, parent):
        self._parent = parent

    @property
    def _session(self):
        return self._parent._session

    @property
    def _browser(self):
        return self._parent._browser

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    def _save_session_as(self) -> None:
        if not self._session.folder:
            QMessageBox.warning(
                self._parent, "Save session",
                "No folder is open. Open a data folder before saving."
            )
            return
        default_name, ok = QInputDialog.getText(
            self._parent, "Save session as",
            "Session name:",
            text=self._session.folder.name,
        )
        if not ok or not default_name.strip():
            return
        name = default_name.strip()
        suggested_dir = str(Path.home() / "Documents")
        path, _ = QFileDialog.getSaveFileName(
            self._parent, "Save session as",
            str(Path(suggested_dir) / f"{name}{SESSION_EXT}"),
            SESSION_FILTER,
        )
        if not path:
            return
        if not path.endswith(SESSION_EXT) and not path.endswith(".json"):
            path += SESSION_EXT
        try:
            self._session.save_to(Path(path))
        except OSError as exc:
            QMessageBox.critical(self._parent, "Save session", f"Write failed: {exc}")
            return
        add_recent(path, name=name, folder_hint=self._session.folder.name)
        self._parent._refresh_recent_sessions_menu()
        self._status(f"Session \"{name}\" saved -> {path}")

    def _open_session_file(self) -> None:
        suggested_dir = str(Path.home() / "Documents")
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Open session", suggested_dir, SESSION_FILTER,
        )
        if not path:
            return
        self._open_session_path(Path(path))

    def _open_recent_session(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(
                self._parent, "Recent session",
                f"File not found:\n{path}\nIt will be removed from the list."
            )
            remove_recent(p)
            self._parent._refresh_recent_sessions_menu()
            return
        self._open_session_path(p)

    def _compare_sessions(self) -> None:
        dialog = SessionDiffDialog(self._parent)
        dialog.exec()

    def _open_session_path(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self._parent, "Open session", f"Cannot read file: {exc}")
            return
        folder = self._resolve_data_folder(payload)
        if folder is None:
            return
        self._apply_payload_to_session(folder, payload)
        self._refresh_browser_after_load(folder, payload)
        add_recent(path, name=payload.get("folder_hint") or path.stem,
                   folder_hint=str(folder.name))
        self._parent._refresh_recent_sessions_menu()
        missing = self._missing_files(folder, payload)
        if missing:
            self._status(
                f"Session opened. {len(missing)} missing file(s) "
                f"in {folder} (ignored)."
            )
        else:
            self._status(f"Session opened from {path.name}.")

    def _resolve_data_folder(self, payload: dict[str, Any]) -> Path | None:
        hint = payload.get("folder")
        candidate = Path(hint) if hint else None
        if candidate and candidate.is_dir():
            return candidate
        hint_name = payload.get("folder_hint") or (candidate.name if candidate else "")
        msg = (
            f"The original folder does not exist on this machine"
            + (f" (hint: {hint_name})." if hint_name else ".")
            + "\nLocate the matching data folder."
        )
        QMessageBox.information(self._parent, "Locate data folder", msg)
        chosen = QFileDialog.getExistingDirectory(
            self._parent, f"Data folder for {hint_name or 'session'}",
            str(Path.home()),
        )
        if not chosen:
            self._status("Session open cancelled (no folder provided).")
            return None
        return Path(chosen)

    def _apply_payload_to_session(self, folder: Path, payload: dict[str, Any]) -> None:
        self._session.folder = folder
        self._session.load_from_payload(payload)
        params = getattr(self._parent, "_params", None)
        if params is not None:
            try:
                params.apply_fit_section_states(self._session.fit_panel_sections)
                params.set_fit_preset_silent(self._session.fit_panel_preset)
            except Exception:
                pass
        notes = getattr(self._parent, "_notes_panel", None)
        if notes is not None:
            try:
                notes.refresh_from_session()
            except Exception:
                pass

    def _refresh_browser_after_load(self, folder: Path, payload: dict[str, Any]) -> None:
        b = self._browser
        b._folder = folder
        b._items_cache = None
        if hasattr(b, "_loader_label_cache"):
            b._loader_label_cache.clear()
        if hasattr(b, "_scan_kind_cache"):
            b._scan_kind_cache.clear()
        if hasattr(b, "_logbook_record_cache"):
            b._logbook_record_cache.clear()
        if hasattr(b, "_lbl_folder"):
            b._lbl_folder.setText(folder.name)
        if hasattr(b, "_populate"):
            b._populate()

    def _missing_files(self, folder: Path, payload: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for rel in (payload.get("files") or {}).keys():
            if not (folder / rel).exists():
                missing.append(rel)
        return missing
