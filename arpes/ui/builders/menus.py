"""Menu builder for the ArpesExplorer main window."""
from __future__ import annotations

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu, QMenuBar


def build_menubar(window) -> QMenuBar:
    bar = window.menuBar()
    bar.clear()
    file_menu = bar.addMenu("&File")

    act_save_as = QAction("Save session as…", window)
    act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
    act_save_as.triggered.connect(window._save_session_as)
    file_menu.addAction(act_save_as)

    act_open = QAction("Open session…", window)
    act_open.setShortcut(QKeySequence("Ctrl+O"))
    act_open.triggered.connect(window._open_session_file)
    file_menu.addAction(act_open)

    act_compare = QAction("Compare sessions...", window)
    act_compare.triggered.connect(window._compare_sessions)
    file_menu.addAction(act_compare)

    act_log = QAction("Processing log...", window)
    act_log.setToolTip(
        "Show a readable per-signal trace of loaded metadata, corrections, "
        "fit formulas, and FS/BM transformations."
    )
    act_log.triggered.connect(lambda: window._experience_log_ctrl.open_dialog())
    file_menu.addAction(act_log)

    file_menu.addSeparator()

    recent_menu = file_menu.addMenu("Recent sessions")
    window._recent_sessions_menu = recent_menu
    _populate_recent_menu(window, recent_menu)

    logbook_menu = bar.addMenu("&Logbook")
    act_logbook = QAction("Load logbook (global)…", window)
    act_logbook.setToolTip("Logbook applied to the whole session folder.")
    act_logbook.triggered.connect(lambda: window._logbook_ctrl.open_dialog())
    logbook_menu.addAction(act_logbook)
    act_scoped = QAction("Add subfolder-scoped logbook…", window)
    act_scoped.setToolTip(
        "Attaches a logbook to a specific subfolder (e.g. CA041 vs CA046).\n"
        "Scoped records only match files in the targeted subfolder."
    )
    act_scoped.triggered.connect(lambda: window._logbook_ctrl.add_scoped_logbook())
    logbook_menu.addAction(act_scoped)
    act_auto_scoped = QAction("Auto-attach scoped (Folder Name)…", window)
    act_auto_scoped.setToolTip(
        "Scans every sheet of an xlsx, reads the « Folder Name » cell\n"
        "of each sheet, and auto-attaches it to the matching subfolder.\n"
        "Faster than attaching each sheet one by one."
    )
    act_auto_scoped.triggered.connect(lambda: window._logbook_ctrl.auto_attach_scoped_logbooks_xlsx())
    logbook_menu.addAction(act_auto_scoped)

    logbook_menu.addSeparator()
    attached_menu = logbook_menu.addMenu("Attached logbooks")
    logbook_menu.aboutToShow.connect(
        lambda: _populate_attached_logbooks(window, attached_menu)
    )

    cache_menu = bar.addMenu("&Cache")
    act_reload = QAction("Reload current file (no cache)", window)
    act_reload.setShortcut(QKeySequence("Ctrl+Shift+R"))
    act_reload.setToolTip(
        "Forces a reload of the current file, bypassing the RAM and disk caches."
    )
    act_reload.triggered.connect(window._reload_current_no_cache)
    cache_menu.addAction(act_reload)

    act_clear = QAction("Clear disk cache (.arpes_cache)", window)
    act_clear.setToolTip(
        "Deletes all npz artifacts cached in the current session folder."
    )
    act_clear.triggered.connect(window._clear_disk_cache)
    cache_menu.addAction(act_clear)

    cache_menu.addSeparator()
    act_toggle = QAction("Enable disk cache", window)
    act_toggle.setCheckable(True)
    act_toggle.setChecked(bool(getattr(window, "_raw_disk_cache_enabled", True)))
    act_toggle.setToolTip(
        "When enabled, loaded files are saved to .arpes_cache/raw_artifacts/\n"
        "for instant reload (~50 ms). Asynchronous write, 250 MB quota per folder."
    )
    act_toggle.toggled.connect(window._toggle_disk_cache)
    cache_menu.addAction(act_toggle)
    window._cache_toggle_action = act_toggle

    view_menu = bar.addMenu("&View")
    act_proc_log = QAction("Processing log panel (live)", window)
    act_proc_log.setToolTip(
        "Show/hide the live processing-log dock: a timestamped journal of every\n"
        "data transform and fit operation applied to the current signal."
    )
    act_proc_log.triggered.connect(lambda: window._experience_log_ctrl.toggle_dock())
    view_menu.addAction(act_proc_log)

    settings_menu = bar.addMenu("&Settings")
    act_api = QAction("Materials Project API key…", window)
    act_api.setToolTip(
        "Enter and store your Materials Project API key (per-user, local).\n"
        "Also shows whether a key is in effect and if mp-api is available."
    )
    act_api.triggered.connect(lambda: _open_api_settings(window))
    settings_menu.addAction(act_api)
    return bar


def _open_api_settings(window) -> None:
    from arpes.ui.widgets.dialogs.api_settings_dialog import ApiSettingsDialog
    ApiSettingsDialog(window).exec()


def _populate_attached_logbooks(window, menu: QMenu) -> None:
    menu.clear()
    try:
        items = window._logbook_ctrl.attached_logbooks()
    except Exception:
        items = []
    if not items:
        empty = menu.addAction("(none)")
        empty.setEnabled(False)
        return
    for scope_label, filename, key in items:
        sub = menu.addMenu(f"{scope_label} — {filename}")
        act_detach = QAction("Detach", window)
        act_detach.triggered.connect(lambda _c=False, k=key: window._logbook_ctrl.detach_logbook(k))
        sub.addAction(act_detach)


def _populate_recent_menu(window, menu: QMenu) -> None:
    from arpes.io.recent_sessions import list_recent
    menu.clear()
    items = list_recent(only_existing=False)
    if not items:
        empty = menu.addAction("(none)")
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
    clear = menu.addAction("Clear list")
    clear.triggered.connect(lambda: _clear_recent(window, menu))


def _clear_recent(window, menu: QMenu) -> None:
    from arpes.io.recent_sessions import _registry_path
    try:
        _registry_path().unlink(missing_ok=True)
    except OSError:
        pass
    _populate_recent_menu(window, menu)
