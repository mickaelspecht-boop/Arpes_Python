"""Optional DFT overlay controller.

All Materials Project behavior is isolated here and in ``arpes.theory`` so the
feature can be removed without touching experimental loaders or fit logic.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from arpes.theory.materials_project import load_materials_project_band_data
from arpes.theory.models import available_segments, compare_fit_to_theory, segment_from_direction
from arpes.theory.plot import draw_theory_overlay


class TheoryOverlayController:
    def __init__(self, parent):
        self._parent = parent

    @property
    def _params(self):
        return self._parent._params

    def _current_overlay(self) -> dict:
        entry = self._parent._current_entry()
        if entry is not None:
            return entry.theory_overlay or {}
        return getattr(self._parent, "_theory_overlay", {}) or {}

    def _save_overlay(self, overlay: dict) -> None:
        self._parent._theory_overlay = overlay
        entry = self._parent._current_entry()
        if entry is not None:
            entry.theory_overlay = overlay
            self._parent._session.save()

    def _import_theory_overlay(self) -> None:
        cfg = self._params.theory_overlay_config()
        mpid = cfg.get("material_id", "").strip()
        if not mpid:
            self._parent._status("Attention: MP-ID vide pour overlay DFT.")
            return
        self._apply_mp_id(mpid, source="manuel", show_dialog_on_error=True)

    def _apply_mp_id(self, mpid: str, *, source: str = "manuel",
                     show_dialog_on_error: bool = False) -> bool:
        """Fetch MP + applique overlay. source ∈ {manuel, logbook}.

        Retourne True si succès, False sinon.
        """
        cfg = self._params.theory_overlay_config()
        cfg["material_id"] = mpid
        try:
            cache_root = self._cache_root()
            data = load_materials_project_band_data(mpid, cache_dir=cache_root)
            entry = self._parent._current_entry()
            direction = entry.meta.direction if entry is not None else ""
            segment = segment_from_direction(direction, data.labels)
            segments = available_segments(data.labels)
            if segment and not cfg.get("segment"):
                cfg["segment"] = segment
            overlay = {
                "enabled": True,
                "data": data.to_dict(),
                "config": cfg,
                "segments": segments,
                "status": "ok",
                "warning": (
                    "" if segment or not direction
                    else f"Direction logbook {direction} non trouvée dans le chemin DFT."
                ),
            }
            self._save_overlay(overlay)
            self._params.set_theory_overlay_state(overlay)
            self._params.txt_theory_mpid.setText(mpid)
            self._parent._draw_bm()
            label = "auto (logbook)" if source == "logbook" else "guide visuel, alignement manuel requis"
            self._parent._status(f"DFT MP importée: {mpid}  |  {label}.")
            return True
        except Exception as exc:
            self._parent._status(f"Attention: overlay DFT indisponible: {exc}")
            if show_dialog_on_error:
                QMessageBox.warning(self._parent, "Overlay DFT", str(exc))
            return False

    def _auto_fetch_theory_overlay_from_logbook(self) -> None:
        """Si entry.meta.mp_id present et overlay vide, tente fetch silencieux."""
        entry = self._parent._current_entry()
        if entry is None:
            return
        mpid = (getattr(entry.meta, "mp_id", "") or "").strip()
        if not mpid:
            return
        existing = entry.theory_overlay or {}
        existing_data = (existing.get("data") or {}).get("material_id", "")
        if existing_data == mpid:
            return  # déjà chargé
        self._apply_mp_id(mpid, source="logbook", show_dialog_on_error=False)

    def _clear_theory_overlay(self) -> None:
        self._save_overlay({})
        self._params.set_theory_overlay_state({})
        self._parent._draw_bm()
        self._parent._status("Overlay DFT vidé.")

    def _on_theory_overlay_changed(self) -> None:
        overlay = dict(self._current_overlay() or {})
        if not overlay:
            return
        cfg = self._params.theory_overlay_config()
        overlay["enabled"] = bool(cfg.get("enabled", False))
        overlay["config"] = cfg
        overlay.pop("comparison", None)
        self._save_overlay(overlay)
        self._parent._draw_bm()

    def _compare_theory_overlay(self) -> None:
        overlay = dict(self._current_overlay() or {})
        if not overlay.get("data"):
            self._parent._status("Attention: importer une DFT avant comparaison.")
            return
        if not self._parent._fit_res:
            self._parent._status("Attention: faire un fit MDC avant comparaison DFT.")
            return
        cfg = self._params.theory_overlay_config()
        overlay["enabled"] = bool(cfg.get("enabled", False))
        overlay["config"] = cfg
        results = compare_fit_to_theory(
            overlay.get("data") or {},
            cfg,
            self._parent._fit_res,
            max_results=6,
            min_points=3,
        )
        overlay["comparison"] = results
        self._save_overlay(overlay)
        self._params.set_theory_overlay_state(overlay)
        self._parent._draw_bm()
        if not results:
            self._parent._status(
                "Comparaison DFT: aucun recouvrement suffisant. Ajuster segment, dE, dk ou scale k."
            )
            return
        best = results[0]
        self._parent._status(
            "Comparaison DFT guide visuel: "
            f"bande {best['band_index']} {best['branch']} paire {best['pair_index'] + 1} "
            f"RMS={best['rms_e'] * 1000:.0f} meV sur {best['n_points']} points."
        )

    def _draw_theory_overlay(self, ax) -> None:
        try:
            count = draw_theory_overlay(ax, self._current_overlay())
            if count and hasattr(self._parent, "_mdc_map_canvas"):
                pass
        except Exception as exc:
            self._parent._status(f"Attention: overlay DFT non dessiné: {exc}")

    def _search_theory_mp(self) -> None:
        """Ouvre dialog recherche MP par formule. Pré-rempli depuis logbook si dispo."""
        from arpes.ui.widgets.dialogs import MPSearchDialog
        entry = self._parent._current_entry()
        initial = ""
        if entry is not None:
            initial = (
                getattr(entry.meta, "formula", "")
                or getattr(entry.meta, "material", "")
                or ""
            )
        dlg = MPSearchDialog(self._parent, initial_formula=str(initial or ""))
        dlg.mpid_selected.connect(self._params.txt_theory_mpid.setText)
        dlg.exec()

    def _restore_theory_overlay_for_entry(self) -> None:
        overlay = self._current_overlay()
        self._params.set_theory_overlay_state(overlay)

    def _cache_root(self) -> Path:
        folder = self._parent._session.folder
        if folder is not None:
            return Path(folder) / ".arpes_theory_cache"
        return Path(".arpes_theory_cache")
