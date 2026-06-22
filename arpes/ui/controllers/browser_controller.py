"""Controller for FileBrowserPanel selection, grouping, and navigation."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QListWidgetItem


class BrowserController:
    # Writes to the wrapped panel must be explicit.
    _OWN_ATTRS = frozenset({"_panel"})
    _PARENT_WRITES = frozenset()

    def __init__(self, panel):
        object.__setattr__(self, "_panel", panel)

    def __getattr__(self, name):
        return getattr(self._panel, name)

    def __setattr__(self, name, value):
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
        elif name in self._PARENT_WRITES:
            setattr(self._panel, name, value)
        else:
            raise AttributeError(
                f"{type(self).__name__} refuses to write '{name}': missing from "
                "_PARENT_WRITES (typo?). Add it to _PARENT_WRITES "
                "if the panel attribute is legitimate."
            )

    def _populate(self):
        selected_path = None
        cur = self._list.currentItem()
        if cur is not None:
            selected_path = cur.data(Qt.ItemDataRole.UserRole)

        self._list.clear()
        if not self._folder:
            self._update_summary([])
            self._refresh_selection_state()
            return

        discovered_paths = self._discover_items()
        all_paths = [
            p for p in discovered_paths
            if self._loaded_only_matches(p) and self._tag_filter_matches(p)
        ]
        groups: dict[str, list[Path]] = {}
        for p in all_paths:
            group = self._group_key_for_path(p)
            groups.setdefault(group, []).append(p)

        self._update_summary(all_paths)

        for group in sorted(groups, key=self._group_sort_key):
            paths = groups[group]
            self._add_header(group, len(paths))
            if group in self._collapsed_groups:
                continue

            for p in paths:
                rel = p.relative_to(self._folder)
                key = self._session.key_for_path(p)
                status = self._file_status(key)
                color  = self.STATUS_COLORS[status]
                item   = QListWidgetItem(self._item_label(p, status, key))
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                item.setData(Qt.ItemDataRole.UserRole + 1, key)
                item.setToolTip(str(rel))
                item.setForeground(QColor(color))
                self._list.addItem(item)

        if selected_path:
            self.select_file(selected_path)
        self._refresh_selection_state()

    def refresh_item(self, filename_or_key: str):
        """Updates the icon of a file in the list."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if not path:
                continue
            key = item.data(Qt.ItemDataRole.UserRole + 1) or self._session.key_for_path(path)
            if key == filename_or_key or Path(path).name == filename_or_key:
                status = self._file_status(key)
                color  = self.STATUS_COLORS[status]
                item.setText(self._item_label(path, status, key))
                item.setForeground(QColor(color))
                all_paths = self._discover_items()
                self._update_summary(all_paths)
                self._refresh_selection_state()
                break

    def _toggle_group(self, group: str):
        if group in self._collapsed_groups:
            self._collapsed_groups.remove(group)
        else:
            self._collapsed_groups.add(group)
        self._populate()

    def _on_double_click(self, item):
        group = item.data(Qt.ItemDataRole.UserRole + 2)
        if group is not None:
            self._toggle_group(group)
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.file_selected.emit(path)

    def _on_selection_change(self, current, _):
        self._refresh_selection_state()

    def _load_selected(self):
        item = self._list.currentItem()
        if item:
            group = item.data(Qt.ItemDataRole.UserRole + 2)
            if group is not None:
                self._toggle_group(group)
                return
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self.file_selected.emit(path)

    def navigate(self, delta: int):
        if self._list.count() == 0:
            return
        row = self._list.currentRow()
        if row < 0:
            row = 0 if delta >= 0 else self._list.count() - 1
        step = 1 if delta >= 0 else -1
        for new in range(row + step, self._list.count() if step > 0 else -1, step):
            item = self._list.item(new)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self._list.setCurrentRow(new)
                self.file_selected.emit(path)
                return

    def select_file(self, path: str) -> bool:
        """Visually selects the file in the list, expanding its folder group if needed."""
        if self._folder:
            try:
                rel = Path(path).relative_to(self._folder)
                group = str(rel.parent) if str(rel.parent) != "." else "."
                if group in self._collapsed_groups:
                    self._collapsed_groups.remove(group)
                    self._populate()
            except Exception:
                pass
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                self._list.setCurrentRow(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Right-click menu: similar-parameter flags + fit-parameter clipboard
    # ------------------------------------------------------------------
    HV_TOL_EV = 2.0  # |Δhν| ≤ this counts as the same photon energy

    @staticmethod
    def _to_float(value):
        try:
            f = float(value)
            return f if f == f else None  # reject NaN
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _norm_str(value) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _group_root(key: str) -> str:
        """Top-level subfolder of a session key ('' for root-level files)."""
        norm = str(key or "").replace("\\", "/")
        return norm.split("/")[0] if "/" in norm else ""

    def _status_msg(self, msg: str) -> None:
        win = self._list.window()
        try:
            win.statusBar().showMessage(msg, 6000)
            return
        except Exception:
            pass
        try:
            self._lbl_selection.setText(msg)
        except Exception:
            pass

    def _on_context_menu(self, pos):
        from PyQt6.QtWidgets import QMenu
        item = self._list.itemAt(pos)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        key = item.data(Qt.ItemDataRole.UserRole + 1)
        if not path or not key:
            return  # group header / non-file row
        entry = self._session.files.get(key)
        clip = self._fit_params_clipboard
        menu = QMenu(self._list)

        act_copy = menu.addAction(
            "Copy fit parameters", lambda *_: self._copy_fit_params(key))
        act_copy.setEnabled(entry is not None and getattr(entry, "fit_params", None) is not None)

        act_paste = menu.addAction(
            "Paste fit parameters here", lambda *_: self._paste_fit_params([key]))
        act_paste.setEnabled(clip is not None)

        if clip is not None and self._flagged_keys:
            menu.addAction(
                f"Paste fit parameters → {len(self._flagged_keys)} flagged",
                lambda *_: self._paste_fit_params(sorted(self._flagged_keys)),
            )

        menu.addSeparator()
        menu.addAction(
            "Flag files with similar parameters (other folders)",
            lambda *_: self._flag_similar(key, path),
        )
        if self._flagged_keys:
            menu.addAction(
                f"Clear similarity flags ({len(self._flagged_keys)})",
                lambda *_: self._clear_flags(),
            )
        menu.exec(self._list.mapToGlobal(pos))

    def _copy_fit_params(self, key: str) -> None:
        import copy
        entry = self._session.files.get(key)
        fp = getattr(entry, "fit_params", None) if entry else None
        if fp is None:
            self._status_msg("No fit parameters to copy on this file.")
            return
        self._panel._fit_params_clipboard = copy.deepcopy(fp)
        self._status_msg(f"Fit parameters copied from {Path(key).name}.")

    def _paste_fit_params(self, keys: list[str]) -> None:
        import copy
        clip = self._fit_params_clipboard
        if clip is None or not keys:
            return
        win = self._list.window()
        cur_key = None
        cur_path = getattr(win, "_current_path", None)
        if cur_path:
            try:
                cur_key = self._session.key_for_path(cur_path)
            except Exception:
                cur_key = None
        fitted = [
            k for k in keys
            if self._session.files.get(k) is not None
            and self._session.files[k].fit_result is not None
        ]
        if fitted:
            from PyQt6.QtWidgets import QMessageBox
            if QMessageBox.question(
                self._list, "Paste fit parameters",
                f"{len(fitted)} target file(s) already have a fit result. "
                "Overwrite their fit parameters?\n"
                "(Existing fit results are kept; only the parameters change.)",
            ) != QMessageBox.StandardButton.Yes:
                return
        for k in keys:
            entry = self._session.get_or_create(k)
            entry.fit_params = copy.deepcopy(clip)
        self._session.save()
        # If the currently-loaded file received new params, push them into the UI.
        if cur_key in keys and win is not None:
            params = getattr(win, "_params", None)
            if params is not None and hasattr(params, "load_fit_params"):
                try:
                    params.load_fit_params(self._session.files[cur_key].fit_params)
                except Exception:
                    pass
        try:
            from arpes.core import processing_history as ph
            for k in keys:
                ph.log_action(
                    win, ph.CAT_FIT, "fit params pasted",
                    entry=self._session.files.get(k),
                    summary="copied from another file",
                )
        except Exception:
            pass
        self._status_msg(f"Fit parameters pasted to {len(keys)} file(s).")

    def _flag_similar(self, ref_key: str, ref_path: str) -> None:
        ref_hv = self._to_float(self._meta_value_for_path(ref_path, "hv")[0])
        ref_dir = self._norm_str(self._meta_value_for_path(ref_path, "direction")[0])
        ref_pol = self._norm_str(self._meta_value_for_path(ref_path, "polarization")[0])
        ref_group = self._group_root(ref_key)
        if ref_hv is None:
            self._status_msg("Reference file has no hν (load it or its logbook first).")
            return
        paths = self._items_cache or self._discover_items()
        flagged: set[str] = set()
        for p in paths:
            k = self._session.key_for_path(p)
            if k == ref_key or self._group_root(k) == ref_group:
                continue  # only other subfolders
            hv = self._to_float(self._meta_value_for_path(p, "hv")[0])
            if hv is None or abs(hv - ref_hv) > self.HV_TOL_EV:
                continue
            if self._norm_str(self._meta_value_for_path(p, "direction")[0]) != ref_dir:
                continue
            if self._norm_str(self._meta_value_for_path(p, "polarization")[0]) != ref_pol:
                continue
            flagged.add(k)
        self._flagged_keys.clear()
        self._flagged_keys.update(flagged)
        self._populate()
        self._status_msg(
            f"{len(flagged)} similar file(s) flagged in other folders "
            f"(hν±{self.HV_TOL_EV:.0f} eV, same direction & polarization)."
        )

    def _clear_flags(self) -> None:
        self._flagged_keys.clear()
        self._populate()
        self._status_msg("Similarity flags cleared.")
