"""Results-panel grouped file filter (Qt) — panel-first free functions.

Replaces the old flat checkbox list with a ``QTreeWidget`` that groups the
fitted files by metadata (compound / polarisation / direction) or by
user-defined manual groups, plus a "colour by group" toggle that recolours the
dispersion and Γ(E) plots so a whole compound / polarisation / manual group
shares one colour.

All grouping data logic is delegated to the pure ``analysis/result_groups.py``
layer; this module only builds the widgets, renders the tree, routes the
context-menu edits and resolves colours. Kept out of ``results.py`` for the
700-LOC cap.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
)

from arpes.analysis import result_groups as rg

_ROLE_FILE = Qt.ItemDataRole.UserRole       # leaf: filename (str); header: None
_ROLE_GROUP = Qt.ItemDataRole.UserRole + 1  # header: raw group label (str)
_PALETTE = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
_GREY = (0.6, 0.6, 0.6)


def _palette_color(idx):
    if idx is None or idx < 0:
        return _GREY
    return _PALETTE[int(idx) % len(_PALETTE)]


def _mpl_to_qcolor(rgb) -> QColor:
    r, g, b = rgb[:3]
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def _fitted_names(panel) -> list[str]:
    return [n for n, e in panel._session.files.items() if e.fit_result is not None]


# -- build -------------------------------------------------------------------

def build_group_filter(panel, layout) -> None:
    """Create the Group-by combo, All/None + colour toggle, and the tree."""
    panel._file_filter_unchecked = set()
    panel._group_color_cache = {}
    panel._groups_syncing = False
    panel._groups_refresh_pending = False

    header = QHBoxLayout()
    header.addWidget(QLabel("Group by:"))
    panel._cmb_group_by = QComboBox()
    panel._cmb_group_by.addItems(list(rg.GROUP_BY_OPTIONS))
    panel._cmb_group_by.setToolTip(
        "Organise the fitted files and (optionally) colour the plots:\n"
        "• None — flat list.\n"
        "• Compound / Polarisation / Direction — auto-grouped from metadata.\n"
        "• Manual groups — your named groups (right-click files to assign).")
    panel._cmb_group_by.currentTextChanged.connect(lambda *_: on_group_by_changed(panel))
    header.addWidget(panel._cmb_group_by, 1)
    panel._btn_new_group = QPushButton("+ Group")
    panel._btn_new_group.setMaximumWidth(74)
    panel._btn_new_group.setToolTip(
        "Create a manual group (from the selected files, or empty).")
    panel._btn_new_group.clicked.connect(lambda *_: new_group_from_selection(panel))
    header.addWidget(panel._btn_new_group)
    layout.addLayout(header)

    btn_row = QHBoxLayout()
    btn_all = QPushButton("All"); btn_all.setMaximumWidth(50)
    btn_all.clicked.connect(lambda: set_all(panel, True))
    btn_none = QPushButton("None"); btn_none.setMaximumWidth(50)
    btn_none.clicked.connect(lambda: set_all(panel, False))
    panel._chk_color_by_group = QCheckBox("Colour by group")
    panel._chk_color_by_group.setToolTip(
        "Colour the dispersion and Γ(E) curves by group instead of per file.")
    panel._chk_color_by_group.toggled.connect(panel.refresh)
    btn_row.addWidget(btn_all); btn_row.addWidget(btn_none)
    btn_row.addStretch(1)
    btn_row.addWidget(panel._chk_color_by_group)
    layout.addLayout(btn_row)

    tree = QTreeWidget()
    tree.setHeaderHidden(True)
    tree.setMaximumHeight(170)
    tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    tree.setStyleSheet("QTreeWidget{background:#222;color:#ddd;font-size:10px;}")
    tree.itemChanged.connect(lambda it, col: on_item_changed(panel, it, col))
    tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    tree.customContextMenuRequested.connect(lambda pos: on_context_menu(panel, pos))
    panel._tree_files = tree
    layout.addWidget(tree)
    _update_group_controls(panel)


def _update_group_controls(panel) -> None:
    by = panel._cmb_group_by.currentText()
    if getattr(panel, "_btn_new_group", None) is not None:
        panel._btn_new_group.setVisible(by == rg.GROUP_BY_MANUAL)
    if getattr(panel, "_chk_color_by_group", None) is not None:
        panel._chk_color_by_group.setEnabled(by != rg.GROUP_BY_NONE)


def on_group_by_changed(panel) -> None:
    _update_group_controls(panel)
    panel.refresh()


# -- tree sync ---------------------------------------------------------------

def sync_group_tree(panel) -> None:
    """Rebuild the tree from the session for the current grouping mode.

    Preserves the hidden-files set (``panel._file_filter_unchecked``) and the
    per-file colour cache used by "colour by group"."""
    tree = panel._tree_files
    session = panel._session
    by = panel._cmb_group_by.currentText()
    fitted = _fitted_names(panel)
    rg.prune_groups(session, fitted)
    cmap = rg.file_color_index(session, fitted, by)
    panel._group_color_cache = {n: _palette_color(i) for n, i in cmap.items()}
    unchecked = panel._file_filter_unchecked

    panel._groups_syncing = True
    tree.blockSignals(True)
    tree.clear()
    if by == rg.GROUP_BY_NONE:
        for n in fitted:
            _add_file_item(tree.invisibleRootItem(), n, n not in unchecked)
    else:
        for label, names in rg.grouped_files(session, fitted, by):
            head = QTreeWidgetItem([f"{label}  ({len(names)})"])
            head.setFlags(head.flags() | Qt.ItemFlag.ItemIsUserCheckable
                          | Qt.ItemFlag.ItemIsAutoTristate)
            head.setData(0, _ROLE_FILE, None)
            head.setData(0, _ROLE_GROUP, label)
            col = panel._group_color_cache.get(names[0]) if names else _GREY
            head.setForeground(0, _mpl_to_qcolor(col))
            tree.addTopLevelItem(head)
            for n in names:
                _add_file_item(head, n, n not in unchecked)
            head.setExpanded(True)
    tree.blockSignals(False)
    panel._groups_syncing = False


def _add_file_item(parent, name: str, checked: bool) -> None:
    it = QTreeWidgetItem([name])
    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    it.setData(0, _ROLE_FILE, name)
    it.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    parent.addChild(it)


# -- visibility / signals ----------------------------------------------------

def visible_files(panel) -> set[str]:
    unchecked = getattr(panel, "_file_filter_unchecked", set())
    return {n for n, e in panel._session.files.items()
            if e.fit_result is not None and n not in unchecked}


def set_all(panel, checked: bool) -> None:
    fitted = _fitted_names(panel)
    panel._file_filter_unchecked = set() if checked else set(fitted)
    panel.refresh()


def on_item_changed(panel, item, _col) -> None:
    if getattr(panel, "_groups_syncing", False):
        return
    _capture_unchecked(panel)
    _schedule_refresh(panel)


def _capture_unchecked(panel) -> None:
    unchecked: set[str] = set()

    def walk(it):
        for i in range(it.childCount()):
            c = it.child(i)
            fn = c.data(0, _ROLE_FILE)
            if fn:
                if c.checkState(0) != Qt.CheckState.Checked:
                    unchecked.add(str(fn))
            walk(c)

    walk(panel._tree_files.invisibleRootItem())
    panel._file_filter_unchecked = unchecked


def _schedule_refresh(panel) -> None:
    # A header toggle cascades to many child itemChanged signals; collapse them
    # into a single refresh on the next event-loop turn.
    if getattr(panel, "_groups_refresh_pending", False):
        return
    panel._groups_refresh_pending = True
    QTimer.singleShot(0, lambda: _flush_refresh(panel))


def _flush_refresh(panel) -> None:
    panel._groups_refresh_pending = False
    panel.refresh()


# -- colour by group ---------------------------------------------------------

def color_for_file(panel, name: str, ci: int, default_colors):
    chk = getattr(panel, "_chk_color_by_group", None)
    if chk is not None and chk.isChecked():
        cache = getattr(panel, "_group_color_cache", None) or {}
        if name in cache:
            return cache[name]
    if default_colors:
        return default_colors[ci % len(default_colors)]
    return (0.8, 0.8, 0.8)


# -- manual group context menu ----------------------------------------------

def _selected_files(panel) -> list[str]:
    out: list[str] = []
    for it in panel._tree_files.selectedItems():
        fn = it.data(0, _ROLE_FILE)
        if fn:
            out.append(str(fn))
    return out


def on_context_menu(panel, pos) -> None:
    if panel._cmb_group_by.currentText() != rg.GROUP_BY_MANUAL:
        return  # assignment only meaningful in manual mode
    tree = panel._tree_files
    item = tree.itemAt(pos)
    sel = _selected_files(panel)
    menu = QMenu(tree)
    if item is not None and item.data(0, _ROLE_FILE) is None:
        label = item.data(0, _ROLE_GROUP)
        if label and label != rg.UNGROUPED:
            menu.addAction("Rename group…", lambda: _rename_group(panel, label))
            menu.addAction("Next colour", lambda: _recolor_group(panel, label))
            menu.addAction("Delete group", lambda: _delete_group(panel, label))
            if sel:
                menu.addAction(f"Add {len(sel)} selected here",
                               lambda: _assign(panel, label, sel))
    if sel:
        if menu.actions():
            menu.addSeparator()
        sub = menu.addMenu(f"Add {len(sel)} file(s) to")
        for gname in rg.group_names(panel._session):
            sub.addAction(gname, lambda g=gname: _assign(panel, g, sel))
        sub.addAction("New group…", lambda: new_group_from_selection(panel))
        menu.addAction("Remove from group", lambda: _unassign(panel, sel))
    if menu.actions():
        menu.addSeparator()
    menu.addAction("New empty group",
                   lambda: (rg.add_group(panel._session), _save_refresh(panel)))
    menu.exec(tree.viewport().mapToGlobal(pos))


def _save_refresh(panel) -> None:
    try:
        panel._session.save()
    except Exception:
        pass
    panel.refresh()


def _assign(panel, group: str, files) -> None:
    rg.assign_to_group(panel._session, group, files)
    _save_refresh(panel)


def _unassign(panel, files) -> None:
    rg.unassign(panel._session, files)
    _save_refresh(panel)


def new_group_from_selection(panel) -> None:
    if panel._cmb_group_by.currentText() != rg.GROUP_BY_MANUAL:
        panel._cmb_group_by.setCurrentText(rg.GROUP_BY_MANUAL)
    files = _selected_files(panel)
    name, ok = QInputDialog.getText(panel, "New group", "Group name:")
    if not ok:
        return
    g = rg.add_group(panel._session, name.strip() or None)
    if files:
        rg.assign_to_group(panel._session, g["name"], files)
    _save_refresh(panel)


def _rename_group(panel, label: str) -> None:
    new, ok = QInputDialog.getText(panel, "Rename group", "New name:", text=label)
    if ok and rg.rename_group(panel._session, label, new.strip()):
        _save_refresh(panel)


def _recolor_group(panel, label: str) -> None:
    g = rg.find_group(panel._session, label)
    if g is not None:
        rg.set_group_color(panel._session, label, int(g.get("color_idx", 0)) + 1)
        _save_refresh(panel)


def _delete_group(panel, label: str) -> None:
    rg.remove_group(panel._session, label)
    _save_refresh(panel)
