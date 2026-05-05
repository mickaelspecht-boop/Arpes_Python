"""Normalisation and detector-grid UI controller for ArpesExplorer."""
from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from arpes_plot_controller import display_grid_config as _plot_display_grid_config


class NormController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    def _load_grid_controls(self, cfg: dict | None):
        cfg = cfg or {}
        self._params.sp_grid_strength.setValue(self._display_grid_config(cfg)["strength"])
        if cfg.get("enabled"):
            self._params.lbl_grid.setText("Correction BM active : masque Fourier 2D automatique.")
        else:
            self._params.lbl_grid.setText("Correction BM : masque Fourier 2D automatique sur l'affichage.")

    def _display_grid_config(self, cfg: dict | None) -> dict:
        return _plot_display_grid_config(cfg)

    def _grid_status_text(self, info: dict, target: str) -> str:
        info = info or {}
        method = info.get("method", "none")
        if info.get("error"):
            return f"Correction grille ({target}) impossible : {info.get('error')}"
        if method in {"fft2mask", "display_fft2mask"}:
            removed = int(info.get("removed_peak_count", 0) or 0)
            delta = float(info.get("rms_delta_percent", 0.0) or 0.0)
            view = info.get("view_mode")
            view_txt = f" {view}" if view else ""
            return (
                f"Correction grille ({target}{view_txt}) : masque Fourier 2D auto, "
                f"{removed} pics FFT, Δ≈{delta:.1f}%, force={float(info.get('strength', 1.0)):.2f}"
            )
        return f"Correction grille ({target}) active."

    def _apply_grid_correction(self):
        if self._raw_data is None or not self._current_path:
            QMessageBox.warning(self._parent, "Effet grille", "Charge d'abord une BM ou une FS.")
            return
        cfg = self._params.grid_params()
        try:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.grid_correction = dict(cfg)
            self._session.save()
            self._update_display_data()
            self._draw_bm()
            if self._tabs.currentIndex() == 1:
                self._draw_mdc_edc()
            msg = self._grid_status_text(self._grid_display_info, "affichage BM")
            self._params.lbl_grid.setText(msg)
            self._status(msg)
        except Exception as exc:
            QMessageBox.warning(self._parent, "Effet grille", str(exc))
            self._status(f"⚠ Effet grille : {exc}")

    def _reset_grid_correction(self):
        if not self._current_path:
            return
        entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
        entry.grid_correction = {}
        self._session.save()
        self._grid_display_info = {}
        self._update_display_data()
        self._draw_bm()
        if self._tabs.currentIndex() == 1:
            self._draw_mdc_edc()
        self._params.lbl_grid.setText("Correction grille désactivée pour ce fichier.")
        self._status("Correction grille désactivée pour ce fichier.")
