"""Optional DFT overlay controller.

All Materials Project behavior is isolated here and in ``arpes.theory`` so the
feature can be removed without touching experimental loaders or fit logic.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from arpes.analysis.self_energy import real_self_energy
from arpes.theory.alignment import alignment_warnings
from arpes.theory.band_picker import validate_picker_data
from arpes.theory.band_select import format_band_indices
from arpes.theory.local_loaders import load_local_band_data
from arpes.theory.materials_project import load_materials_project_band_data
from arpes.theory.models import available_segments, compare_fit_to_theory, fit_mu_shift, parse_band_indices, segment_from_direction
from arpes.theory.plot import draw_theory_overlay
from arpes.ui.app_settings import resolve_mp_api_key


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
            self._parent._status("Warning: empty MP-ID for DFT overlay.")
            return
        self._apply_mp_id(mpid, source="manual", show_dialog_on_error=True)

    def _refresh_theory_overlay(self) -> None:
        """Ré-importe le MP-ID en ignorant le cache disque (récupère le
        vrai chemin de bandes MP même si import mis en cache avant)."""
        cfg = self._params.theory_overlay_config()
        mpid = cfg.get("material_id", "").strip()
        if not mpid:
            self._parent._status("Warning: empty MP-ID for DFT refresh.")
            return
        self._parent._status(f"Refreshing MP {mpid} (cache ignored)...")
        self._apply_mp_id(mpid, source="manual", show_dialog_on_error=True,
                          force_refresh=True)

    def _import_local_theory_overlay(self) -> None:
        start_dir = ""
        current = getattr(self._parent, "_current_path", None)
        if current:
            start_dir = str(Path(current).parent)
        path_s, _ = QFileDialog.getOpenFileName(
            self._parent,
            "Import local DFT",
            start_dir,
            "Local DFT (*.xml *.dat *.txt *.yaml *.yml *.json);;All files (*)",
        )
        if not path_s:
            return
        try:
            data = load_local_band_data(Path(path_s))
        except Exception as exc:
            self._parent._status(f"Warning: cannot import local DFT: {exc}")
            QMessageBox.warning(self._parent, "Import local DFT", str(exc))
            return

        cfg = self._params.theory_overlay_config()
        cfg["material_id"] = data.material_id
        entry = self._parent._current_entry()
        direction = entry.meta.direction if entry is not None else ""
        segment = segment_from_direction(direction, data.labels, data.branches)
        segments = available_segments(data.labels, data.branches)
        if segment and not cfg.get("segment"):
            cfg["segment"] = segment
        overlay = {
            "enabled": True,
            "data": data.to_dict(),
            "config": cfg,
            "segments": segments,
            "status": "ok",
            "warning": (
                data.warning
                or (
                    "" if segment or not direction
                    else f"Logbook direction {direction} not found in the DFT path."
                )
            ),
        }
        self._save_overlay(overlay)
        self._params.set_theory_overlay_state(overlay)
        self._params.txt_theory_mpid.setText(data.material_id)
        self._parent._draw_current_view(include_curves=False)
        self._parent._status(f"Local DFT imported: {Path(path_s).name} | manual alignment required.")

    def _apply_mp_id(self, mpid: str, *, source: str = "manuel",
                     show_dialog_on_error: bool = False,
                     force_refresh: bool = False) -> bool:
        """Fetch MP + applique overlay. source ∈ {manuel, logbook}.

        Retourne True si succès, False sinon.
        """
        cfg = self._params.theory_overlay_config()
        cfg["material_id"] = mpid
        try:
            cache_root = self._cache_root()
            data = load_materials_project_band_data(
                mpid, cache_dir=cache_root,
                with_projections=bool(cfg.get("with_projections", False)),
                force_refresh=force_refresh,
                api_key=resolve_mp_api_key(),
            )
            # Cache legacy (pré-branches) : re-fetch auto une seule fois
            # pour récupérer le vrai chemin MP au lieu d'afficher tout
            # le path tassé. Évite d'exiger le clic « Rafraîchir ».
            if not force_refresh and not data.branches:
                data = load_materials_project_band_data(
                    mpid, cache_dir=cache_root,
                    with_projections=bool(cfg.get("with_projections", False)),
                    force_refresh=True,
                    api_key=resolve_mp_api_key(),
                )
            entry = self._parent._current_entry()
            direction = entry.meta.direction if entry is not None else ""
            segment = segment_from_direction(direction, data.labels, data.branches)
            segments = available_segments(data.labels, data.branches)
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
                    else f"Logbook direction {direction} not found in the DFT path."
                ),
            }
            self._save_overlay(overlay)
            self._params.set_theory_overlay_state(overlay)
            self._params.txt_theory_mpid.setText(mpid)
            self._parent._draw_current_view(include_curves=False)
            label = "auto (logbook)" if source == "logbook" else "visual guide, manual alignment required"
            self._parent._status(f"MP DFT imported: {mpid}  |  {label}.")
            return True
        except Exception as exc:
            self._parent._status(f"Warning: DFT overlay unavailable: {exc}")
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
        self._parent._draw_current_view(include_curves=False)
        self._parent._status("DFT overlay cleared.")

    def _on_theory_overlay_changed(self) -> None:
        overlay = dict(self._current_overlay() or {})
        if not overlay:
            return
        cfg = self._params.theory_overlay_config()
        overlay["enabled"] = bool(cfg.get("enabled", False))
        overlay["config"] = cfg
        overlay.pop("comparison", None)
        self._save_overlay(overlay)
        # Overlay DFT = cosmétique : fast path (skip recompute mesh/couleur,
        # zoom préservé).
        self._parent._draw_current_view(include_curves=False, overlays_only=True)
        warnings = alignment_warnings(
            float(cfg.get("mu_shift", 0.0) or 0.0),
            float(cfg.get("z_scale", 1.0) or 1.0),
        )
        if warnings:
            self._parent._status("Warning: " + " ".join(warnings))

    def _open_theory_band_picker(self) -> None:
        overlay = dict(self._current_overlay() or {})
        data = overlay.get("data") or {}
        if not data:
            self._parent._status("Warning: import a DFT before choosing bands.")
            return
        validation_error = validate_picker_data(data)
        if validation_error:
            self._parent._status(validation_error)
            return

        from arpes.ui.widgets.dialogs import TheoryBandPickerDialog

        cfg = self._params.theory_overlay_config()
        n_bands = len(data.get("bands") or [])
        selected = parse_band_indices(str(cfg.get("band_indices") or ""), n_bands)
        signature = self._overlay_picker_signature(overlay)
        dialog = TheoryBandPickerDialog(
            data,
            cfg,
            segments=list(overlay.get("segments") or []),
            selected=selected,
            parent=self._parent,
        )
        applied: dict[str, object] = {}

        def _capture(indices, segment):
            applied["indices"] = list(indices or [])
            applied["segment"] = str(segment or "")

        dialog.selection_applied.connect(_capture)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        if self._overlay_picker_signature(self._current_overlay()) != signature:
            self._parent._status("DFT selection cancelled: overlay changed while the dialog was open.")
            return
        indices = list(applied.get("indices", dialog.selected_band_indices()))
        segment = str(applied.get("segment", dialog.selected_segment()) or "")
        spec = format_band_indices(indices)
        self._params.txt_theory_bands.blockSignals(True)
        self._params.txt_theory_bands.setText(spec)
        self._params.txt_theory_bands.blockSignals(False)
        self._params.cmb_theory_segment.blockSignals(True)
        self._params.cmb_theory_segment.setCurrentText(segment)
        self._params.cmb_theory_segment.blockSignals(False)
        self._params._on_theory_bands_text_edited()
        if not indices:
            self._parent._status("Empty DFT selection: keeping automatic top-N display.")
        else:
            self._parent._status(f"Selected DFT bands: {spec}.")

    def _overlay_picker_signature(self, overlay: dict | None) -> tuple:
        data = (overlay or {}).get("data") or {}
        cfg = (overlay or {}).get("config") or {}
        bands = data.get("bands") or []
        first = bands[0] if bands else []
        return (
            data.get("material_id") or "",
            len(bands),
            len(data.get("k_distance") or first or []),
            cfg.get("segment") or "",
        )

    def _compare_theory_overlay(self) -> None:
        overlay = dict(self._current_overlay() or {})
        if not overlay.get("data"):
            self._parent._status("Warning: import a DFT before comparison.")
            return
        if not self._parent._fit_res:
            self._parent._status("Warning: run an MDC fit before DFT comparison.")
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
        self._parent._draw_current_view(include_curves=False)
        if not results:
            self._parent._status(
                "DFT comparison: not enough overlap. Adjust segment, mu, Z, delta k, or k scale."
            )
            return
        best = results[0]
        self._parent._status(
            "DFT visual-guide comparison: "
            f"band {best['band_index']} {best['branch']} pair {best['pair_index'] + 1} "
            f"RMS={best['rms_e'] * 1000:.0f} meV over {best['n_points']} points."
        )

    def _fit_theory_mu_auto(self) -> None:
        overlay = dict(self._current_overlay() or {})
        if not overlay.get("data"):
            self._parent._status("Warning: import a DFT before fitting mu.")
            return
        if not self._parent._fit_res:
            self._parent._status("Warning: run an MDC fit before fitting mu.")
            return
        cfg = self._params.theory_overlay_config()
        res = fit_mu_shift(overlay.get("data") or {}, cfg, self._parent._fit_res)
        if res is None:
            self._parent._status(
                "Mu fit: not enough overlap. Adjust segment, Z, delta k, or k scale."
            )
            return
        self._params.sp_theory_mu.blockSignals(True)
        self._params.sp_theory_mu.setValue(float(res["mu"]))
        self._params.sp_theory_mu.blockSignals(False)
        self._params._schedule_theory_overlay_changed()
        self._parent._status(
            f"Mu fitted: {res['mu_before'] * 1000:+.0f} -> {res['mu'] * 1000:+.0f} meV "
            f"(band {res['band_index']} {res['branch']} P{res['pair_index'] + 1}, "
            f"RMS {res['rms_before'] * 1000:.0f} -> {res['rms_after'] * 1000:.0f} meV, "
            f"{res['n_points']} pts)"
        )

    def _calculate_self_energy(self) -> None:
        overlay = dict(self._current_overlay() or {})
        cfg = self._params.theory_overlay_config()
        overlay["config"] = cfg
        try:
            result = real_self_energy(self._parent._fit_res, overlay)
        except ValueError as exc:
            self._parent._status(f"Warning: Re Sigma unavailable: {exc}")
            return
        from arpes.ui.widgets.dialogs import SelfEnergyDialog
        dialog = SelfEnergyDialog(result, self._parent)
        dialog.exec()
        msg = (
            f"Re Sigma: band {result.band_index} {result.branch} "
            f"P{result.pair_index + 1}, RMS={result.rms_e * 1000:.0f} meV"
        )
        if result.kink_energy == result.kink_energy:
            msg += f", kink≈{result.kink_energy * 1000:.0f} meV"
        self._parent._status(msg)

    def _draw_theory_overlay(self, ax) -> None:
        try:
            count = draw_theory_overlay(ax, self._current_overlay())
            if count and hasattr(self._parent, "_mdc_map_canvas"):
                pass
        except Exception as exc:
            self._parent._status(f"Warning: DFT overlay not drawn: {exc}")

    def _search_theory_mp(self) -> None:
        """Open the MP formula-search dialog, prefilled from logbook when available."""
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
        dlg.mpid_selected.connect(self._import_selected_mpid)
        dlg.exec()

    def _import_selected_mpid(self, mpid: str) -> None:
        mpid = str(mpid or "").strip()
        if not mpid:
            return
        self._params.txt_theory_mpid.setText(mpid)
        self._apply_mp_id(mpid, source="search", show_dialog_on_error=True)

    def _align_theory_to_arpes(self) -> None:
        """Calcule scale + Δk pour mapper segment choisi sur [0, 1] (π/a).

        Premier label segment → 0, second → 1. Le sens vient du nom segment.
        """
        overlay = self._current_overlay()
        data_d = overlay.get("data") or {}
        labels = data_d.get("labels") or []
        if not labels:
            self._parent._status("Warning: import a DFT before alignment.")
            return
        segment = self._params.cmb_theory_segment.currentText().strip()
        if not segment:
            self._parent._status("Warning: choose a segment before aligning.")
            return
        # Real MP path: _branch_local_k already maps the branch onto
        # [0,1] (Γ→0, bord de zone→1 en π/a). Recalculer depuis les
        # positions de label sur l'axe global double-transformerait
        # l'overlay (cause du hors-cadre). → scale=1, Δk=0.
        branches = data_d.get("branches") or []
        if branches:
            from arpes.theory.models import _branch_index_for_segment
            if _branch_index_for_segment(branches, segment) is not None:
                self._params.sp_theory_kscale.blockSignals(True)
                self._params.sp_theory_dk.blockSignals(True)
                self._params.sp_theory_kscale.setValue(1.0)
                self._params.sp_theory_dk.setValue(0.0)
                self._params.sp_theory_kscale.blockSignals(False)
                self._params.sp_theory_dk.blockSignals(False)
                self._on_theory_overlay_changed()
                has_abs = bool(data_d.get("k_distance_abs"))
                a_val = float(self._params.sp_crystal_a.value())
                if has_abs and a_val > 0:
                    msg = (
                        f"Aligned {segment}: PHYSICAL scale "
                        f"(A^-1*a/pi, a={a_val:.4f} A). Gamma->0, X->1, M->sqrt(2). "
                        f"scale=1, delta k=0. Mirror Gamma for symmetric scans. "
                        f"mu still manual."
                    )
                else:
                    msg = (
                        f"Aligned {segment} (MP path): Gamma->0, edge->1 "
                        f"(normalized scale - enter crystal a "
                        f"for the exact physical scale). scale=1, delta k=0."
                    )
                self._parent._status(msg)
                return
        if "-" not in segment:
            self._parent._status("Warning: segment has no endpoints; cannot align.")
            return
        a, b = [s.strip() for s in segment.split("-", 1)]
        pos = {
            str(item.get("label") or "").upper().replace("GAMMA", "Γ"): item.get("k")
            for item in labels
        }
        a_key = a.upper().replace("GAMMA", "Γ")
        b_key = b.upper().replace("GAMMA", "Γ")
        pa, pb = pos.get(a_key), pos.get(b_key)
        if pa is None or pb is None:
            self._parent._status(
                f"Warning: segment {segment} not found in DFT labels."
            )
            return
        try:
            pa_f, pb_f = float(pa), float(pb)
        except (TypeError, ValueError):
            self._parent._status(f"Warning: {segment} positions are not numeric.")
            return
        if abs(pb_f - pa_f) <= 1e-9:
            self._parent._status(f"Warning: segment {segment} has zero width.")
            return
        scale = 1.0 / (pb_f - pa_f)
        shift = -pa_f / (pb_f - pa_f)
        self._params.sp_theory_kscale.blockSignals(True)
        self._params.sp_theory_dk.blockSignals(True)
        self._params.sp_theory_kscale.setValue(float(scale))
        self._params.sp_theory_dk.setValue(float(shift))
        self._params.sp_theory_kscale.blockSignals(False)
        self._params.sp_theory_dk.blockSignals(False)
        self._on_theory_overlay_changed()
        self._parent._status(
            f"Aligned {segment} to ARPES pi/a: scale={scale:.3f}, delta k={shift:+.3f} "
            f"({a}->0, {b}->1). mu still manual."
        )

    def _on_crystal_a_changed(self) -> None:
        a = float(self._params.sp_crystal_a.value())
        path = getattr(self._parent, "_current_path", None)
        if path:
            entry = self._parent._session.get_or_create(self._parent._session.key_for_path(path))
            entry.meta.crystal_a_angstrom = a
            self._parent._session.save()
        fs_controls = getattr(self._parent, "_fs_controls", None)
        sp_a = getattr(fs_controls, "sp_a", None)
        if sp_a is not None and abs(float(sp_a.value()) - a) > 1e-9:
            sp_a.blockSignals(True)
            sp_a.setValue(a)
            sp_a.blockSignals(False)
        self._parent._status(f"Crystal parameter a = {a:.4f} A saved.")
        if path and getattr(self._parent, "_raw_data", None) is not None:
            self._parent._status("Parameter a changed: recomputing k axes without cache.")
            self._parent._load_ctrl.load(path, force_reload=True)

    def _align_theory_efermi(self) -> None:
        self._params.sp_theory_mu.blockSignals(True)
        self._params.sp_theory_mu.setValue(0.0)
        self._params.sp_theory_mu.blockSignals(False)
        self._on_theory_overlay_changed()
        self._parent._status(
            "mu = 0 forced. Overlay: E = Z x E_DFT. "
            "Do not assume this alignment is physically optimal."
        )

    def _restore_theory_overlay_for_entry(self) -> None:
        overlay = self._current_overlay()
        self._params.set_theory_overlay_state(overlay)

    def _cache_root(self) -> Path:
        folder = self._parent._session.folder
        if folder is not None:
            return Path(folder) / ".arpes_theory_cache"
        return Path(".arpes_theory_cache")
