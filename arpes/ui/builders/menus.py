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

    act_compare = QAction("Comparer sessions...", window)
    act_compare.triggered.connect(window._compare_sessions)
    file_menu.addAction(act_compare)

    file_menu.addSeparator()

    recent_menu = file_menu.addMenu("Sessions récentes")
    window._recent_sessions_menu = recent_menu
    _populate_recent_menu(window, recent_menu)

    logbook_menu = bar.addMenu("&Logbook")
    act_logbook = QAction("Charger logbook (global)…", window)
    act_logbook.setToolTip("Logbook appliqué à tout le dossier de session.")
    act_logbook.triggered.connect(lambda: window._logbook_ctrl.open_dialog())
    logbook_menu.addAction(act_logbook)
    act_scoped = QAction("Ajouter logbook scopé sous-dossier…", window)
    act_scoped.setToolTip(
        "Attache un logbook à un sous-dossier précis (ex: CA041 vs CA046).\n"
        "Les records scopés ne matchent que les fichiers du sous-dossier visé."
    )
    act_scoped.triggered.connect(lambda: window._logbook_ctrl.add_scoped_logbook())
    logbook_menu.addAction(act_scoped)

    logbook_menu.addSeparator()
    attached_menu = logbook_menu.addMenu("Logbooks attachés")
    logbook_menu.aboutToShow.connect(
        lambda: _populate_attached_logbooks(window, attached_menu)
    )

    cache_menu = bar.addMenu("&Cache")
    act_reload = QAction("Recharger fichier courant (sans cache)", window)
    act_reload.setShortcut(QKeySequence("Ctrl+Shift+R"))
    act_reload.setToolTip(
        "Force le re-chargement du fichier courant en bypassant les caches RAM et disque."
    )
    act_reload.triggered.connect(window._reload_current_no_cache)
    cache_menu.addAction(act_reload)

    act_clear = QAction("Vider cache disque (.arpes_cache)", window)
    act_clear.setToolTip(
        "Supprime tous les artefacts npz mis en cache dans le dossier de session courant."
    )
    act_clear.triggered.connect(window._clear_disk_cache)
    cache_menu.addAction(act_clear)

    cache_menu.addSeparator()
    act_toggle = QAction("Activer cache disque", window)
    act_toggle.setCheckable(True)
    act_toggle.setChecked(bool(getattr(window, "_raw_disk_cache_enabled", True)))
    act_toggle.setToolTip(
        "Quand activé, les fichiers chargés sont sauvés en .arpes_cache/raw_artifacts/\n"
        "pour rechargement instant (~50 ms). Écriture asynchrone, quota 250 MB par dossier."
    )
    act_toggle.toggled.connect(window._toggle_disk_cache)
    cache_menu.addAction(act_toggle)
    window._cache_toggle_action = act_toggle
    return bar


def _populate_attached_logbooks(window, menu: QMenu) -> None:
    menu.clear()
    try:
        items = window._logbook_ctrl.attached_logbooks()
    except Exception:
        items = []
    if not items:
        empty = menu.addAction("(aucun)")
        empty.setEnabled(False)
        return
    for scope_label, filename, key in items:
        sub = menu.addMenu(f"{scope_label} — {filename}")
        act_detach = QAction("Détacher", window)
        act_detach.triggered.connect(lambda _c=False, k=key: window._logbook_ctrl.detach_logbook(k))
        sub.addAction(act_detach)


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
