"""Controller for FileBrowserPanel selection, grouping, and navigation."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QListWidgetItem


class BrowserController:
    def __init__(self, panel):
        object.__setattr__(self, "_panel", panel)

    def __getattr__(self, name):
        return getattr(self._panel, name)

    def __setattr__(self, name, value):
        if name == "_panel":
            object.__setattr__(self, name, value)
        else:
            setattr(self._panel, name, value)

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
        """Met à jour l'icône d'un fichier dans la liste."""
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
        """Sélectionne visuellement le fichier dans la liste, en ouvrant son dossier si besoin."""
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
