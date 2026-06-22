"""Controller for the per-signal processing log: live dock + report dialog.

Owns the live ``ProcessingLogDock`` and exposes a thin ``log()`` helper that
controllers call at each data transform / fit operation. The actual append is
delegated to the pure ``core.processing_history`` single-setter, so the journal
is identical whether written from the UI or from a headless test.
"""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt

from arpes.core import processing_history as ph


class ExperienceLogController:
    def __init__(self, parent):
        self._parent = parent
        self._dock = None

    # ------------------------------------------------------------- resolution
    def _current_key(self) -> str | None:
        p = self._parent
        if not getattr(p, "_current_path", None):
            return None
        try:
            return p._session.key_for_path(p._current_path)
        except Exception:
            return None

    def _current_entry(self):
        key = self._current_key()
        if key is None:
            return None
        try:
            return self._parent._session.get_or_create(key)
        except Exception:
            return None

    # -------------------------------------------------------------- logging
    def log(
        self,
        category: str,
        action: str,
        *,
        entry: Any = None,
        summary: str = "",
        params: dict | None = None,
        coalesce: bool = False,
    ) -> None:
        """Record a provenance event for ``entry`` (default: current signal).

        Safe to call from any controller; never raises into the caller.
        """
        try:
            target = entry if entry is not None else self._current_entry()
            if target is None:
                return
            ph.log_event(
                target, category, action,
                summary=summary, params=params, coalesce=coalesce,
            )
            self.refresh_dock()
        except Exception:
            pass

    # ----------------------------------------------------------------- dock
    def ensure_dock(self):
        if self._dock is None:
            from arpes.ui.widgets.processing_log_dock import ProcessingLogDock
            self._dock = ProcessingLogDock(self._parent)
            self._parent.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self._dock
            )
        return self._dock

    def toggle_dock(self) -> None:
        dock = self.ensure_dock()
        dock.setVisible(not dock.isVisible())
        if dock.isVisible():
            dock.refresh()

    def refresh_dock(self) -> None:
        if self._dock is not None:
            try:
                self._dock.refresh()
            except Exception:
                pass

    # --------------------------------------------------------------- dialog
    def open_dialog(self) -> None:
        from arpes.ui.widgets.dialogs.experience_log import ExperienceLogDialog

        dlg = ExperienceLogDialog(
            self._parent._session, current_key=self._current_key(), parent=self._parent
        )
        dlg.exec()
