"""Normalisation and detector-grid UI controller for ArpesExplorer."""
from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from arpes.physics.plot_compute import display_grid_config as _plot_display_grid_config


class NormController:
    # P3.1: writes through to parent are allow-listed (fail-loud on typo).
    _OWN_ATTRS = frozenset({"_parent"})
    _PARENT_WRITES = frozenset({"_grid_display_info"})

    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
        elif name in self._PARENT_WRITES:
            setattr(self._parent, name, value)
        else:
            raise AttributeError(
                f"{type(self).__name__} refuses to write '{name}': missing from "
                "_PARENT_WRITES (typo?). Add it to _PARENT_WRITES "
                "if the parent attribute is legitimate."
            )

    def _load_grid_controls(self, cfg: dict | None):
        cfg = cfg or {}
        self._params.sp_grid_strength.setValue(self._display_grid_config(cfg)["strength"])
        if cfg.get("enabled"):
            self._params.lbl_grid.setText("BM correction active: automatic 2D Fourier mask.")
        else:
            self._params.lbl_grid.setText("BM correction: automatic 2D Fourier mask on display.")

    def _display_grid_config(self, cfg: dict | None) -> dict:
        return _plot_display_grid_config(cfg)

    def _grid_status_text(self, info: dict, target: str) -> str:
        info = info or {}
        method = info.get("method", "none")
        if info.get("error"):
            return f"Grid correction ({target}) impossible: {info.get('error')}"
        if method in {"fft2mask", "display_fft2mask"}:
            removed = int(info.get("removed_peak_count", 0) or 0)
            delta = float(info.get("rms_delta_percent", 0.0) or 0.0)
            view = info.get("view_mode")
            view_txt = f" {view}" if view else ""
            return (
                f"Grid correction ({target}{view_txt}): automatic 2D Fourier mask, "
                f"{removed} FFT peaks, Δ≈{delta:.1f}%, strength={float(info.get('strength', 1.0)):.2f}"
            )
        return f"Grid correction ({target}) active."

    def _apply_grid_correction(self):
        if self._raw_data is None or not self._current_path:
            QMessageBox.warning(self._parent, "Grid Effect", "Load a BM or FS first.")
            return
        cfg = self._params.grid_params()
        try:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.grid_correction = dict(cfg)
            self._session.save()
            self._update_display_data()
            self._draw_current_view()
            msg = self._grid_status_text(self._grid_display_info, "BM display")
            self._params.lbl_grid.setText(msg)
            self._status(msg)
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done("grid correction applied")
        except Exception as exc:
            QMessageBox.warning(self._parent, "Grid Effect", str(exc))
            self._status(f"Warning: grid effect: {exc}")

    def _reset_grid_correction(self):
        if not self._current_path:
            return
        entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
        entry.grid_correction = {}
        self._session.save()
        self._grid_display_info = {}
        self._update_display_data()
        self._draw_current_view()
        self._params.lbl_grid.setText("Grid correction disabled for this file.")
        self._status("Grid correction disabled for this file.")
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done("grid correction disabled")
