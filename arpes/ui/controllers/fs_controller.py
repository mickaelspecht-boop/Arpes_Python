"""Fermi-surface UI controller for ArpesExplorer."""
from __future__ import annotations

from arpes.physics.fs import FermiSurfaceCanvas, FSControlPanel


class FSController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    def _current_is_fs(self) -> bool:
        meta = (self._raw_data or {}).get("metadata", {}) or {}
        return meta.get("fs_data") is not None

    def _on_fs_params_changed(self):
        self._save_current_fs_center()
        self._draw_fs_tab()

    def _choose_bz_preset(self):
        if not hasattr(self, "_fs_controls"):
            return
        from arpes.ui.widgets.dialogs import BZSelectorDialog
        dialog = BZSelectorDialog(self._parent)
        if dialog.exec():
            self._fs_controls.apply_bz_preset(dialog.selected_key)
            self._draw_fs_tab()
            self._status(f"ZDB appliquée : {dialog.selected_key}")

    def _save_current_fs_center(self):
        if self._raw_data is None or not self._current_path or not self._current_is_fs():
            return
        if FSControlPanel is None or not hasattr(self, "_fs_controls"):
            return
        entry = self._current_entry()
        if entry is None:
            return
        try:
            p = self._fs_controls.params()
            entry.fs_center_kx = float(p.kx_center)
            entry.fs_center_ky = float(p.ky_center)
            self._session.save()
        except Exception:
            pass

    def _draw_fs_tab(self):
        if not hasattr(self, "_fs_canvas") or FermiSurfaceCanvas is None:
            return
        if not hasattr(self, "_fs_controls") or FSControlPanel is None:
            return
        info = self._fs_canvas.draw_fs(self._raw_data, self._fs_controls.params())
        try:
            self._fs_controls.lbl_info.setText(info)
        except Exception:
            pass
