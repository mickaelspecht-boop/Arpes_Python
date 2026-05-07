"""Menu builder for the ArpesExplorer main window."""
from __future__ import annotations

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu, QMenuBar


def build_menubar(window) -> QMenuBar:
    bar = window.menuBar()
    bar.clear()
    file_menu = bar.addMenu("&Fichier")

    act_save_as = QAction("Sauvegarder session sous…", window)
    act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
    act_save_as.triggered.connect(window._save_session_as)
    file_menu.addAction(act_save_as)

    act_open = QAction("Ouvrir session…", window)
    act_open.setShortcut(QKeySequence("Ctrl+O"))
    act_open.triggered.connect(window._open_session_file)
    file_menu.addAction(act_open)

    file_menu.addSeparator()

    recent_menu = file_menu.addMenu("Sessions récentes")
    window._recent_sessions_menu = recent_menu
    _populate_recent_menu(window, recent_menu)
    return bar


def _populate_recent_menu(window, menu: QMenu) -> None:
    from arpes.io.recent_sessions import list_recent
    menu.clear()
    items = list_recent(only_existing=False)
    if not items:
        empty = menu.addAction("(aucune)")
        empty.setEnabled(False)
        return
    for item in items:
        path = item.get("path", "")
        name = item.get("name") or item.get("folder_hint") or path
        action = QAction(name, window)
        action.setToolTip(path)
        action.triggered.connect(lambda _checked=False, p=path: window._open_recent_session(p))
        menu.addAction(action)
    menu.addSeparator()
    clear = menu.addAction("Effacer la liste")
    clear.triggered.connect(lambda: _clear_recent(window, menu))


def _clear_recent(window, menu: QMenu) -> None:
    from arpes.io.recent_sessions import _registry_path
    try:
        _registry_path().unlink(missing_ok=True)
    except OSError:
        pass
    _populate_recent_menu(window, menu)
