"""Controller for the per-signal processing log dialog."""
from __future__ import annotations


class ExperienceLogController:
    def __init__(self, parent):
        self._parent = parent

    def open_dialog(self) -> None:
        from arpes.ui.widgets.dialogs.experience_log import ExperienceLogDialog

        p = self._parent
        current_key = None
        if getattr(p, "_current_path", None):
            try:
                current_key = p._session.key_for_path(p._current_path)
            except Exception:
                current_key = None
        dlg = ExperienceLogDialog(p._session, current_key=current_key, parent=p)
        dlg.exec()
