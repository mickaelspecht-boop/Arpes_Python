"""Controller UI pour pipeline de chargement d'un fichier ARPES.

Sort `arpes_explorer.ArpesExplorer._load_file` (~169 LOC) de la God class et
le découpe en 5 étapes lisibles :

1. `_prepare_entry`   : crée/restaure l'entrée session, applique l'EF par défaut
2. `_dispatch_loader` : appelle `LoaderOrchestrator` et récupère `load_result`
3. `_apply_post_load` : correction EF par colonne + stockage `_raw_data` parent
4. `_restore_session` : params UI restaurés depuis l'entrée session
5. `_refresh_ui`      : redraws BM/MDC-EDC/FS + statusbar

La couche métier (LoaderOrchestrator, apply_ef_correction_to_dict, etc.) reste
dans `arpes.io.*` ou dans `arpes_explorer` (sera déplacée en λ). Ce contrôleur
est purement orchestration Qt + état parent.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtWidgets import QApplication

from arpes.io.loader_orchestrator import LoaderOrchestrator
from arpes.physics.norm import remove_grid_artifact as remove_detector_grid_artifact
from arpes.physics.resolution import estimate_resolutions


@dataclass
class _PreparedEntry:
    key: str
    entry: Any
    is_new: bool
    fmt_guess: str
    hv_for_load: float
    hv_from_logbook: bool
    angle_offsets: dict | None
    logbook_hit: bool


class LoadController:
    """Pipeline de chargement d'un fichier ARPES (band map ou FS)."""

    def __init__(self, parent):
        self._parent = parent

    # ----------------------------------------------------- proxies parent
    @property
    def _session(self):
        return self._parent._session

    @property
    def _params(self):
        return self._parent._params

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    # ------------------------------------------------------------ public
    def load(self, path: str) -> None:
        """Pipeline complet — entrée unique appelée par le file browser."""
        self._ensure_arpes_plots()
        self._status(f"Chargement {Path(path).name} …")
        QApplication.processEvents()
        try:
            prepared = self._prepare_entry(path)
            load_result, orchestrator = self._dispatch_loader(path, prepared)
            d = load_result.data
            if d is None:
                self._status("⚠ erlab non disponible")
                return
            md = orchestrator.apply_loaded_metadata(d, prepared.entry)
            self._apply_post_load(d, prepared, load_result, orchestrator, path)
            self._restore_session(d, prepared.entry, md)
            self._refresh_ui(d, prepared, path)
        except Exception as e:
            self._status(f"⚠ {e}")
            traceback.print_exc()

    # ------------------------------------------------------------ steps
    def _ensure_arpes_plots(self) -> None:
        from arpes import app as _ae
        if _ae.AP is None:
            try:
                _ae.AP = _ae._load_ap()
            except Exception as e:
                self._status(f"⚠ arpes_plots : {e}")

    def _prepare_entry(self, path: str) -> _PreparedEntry:
        from arpes import app as _ae
        key = self._session.key_for_path(path)
        is_new_entry = key not in self._session.files
        entry = self._session.get_or_create(key)
        fmt_guess = ""
        try:
            fmt_guess = _ae.detect_format(path) if _ae.detect_format is not None else ""
        except Exception:
            fmt_guess = ""
        if fmt_guess == "bessy_ses_ibw" and (
            is_new_entry
            or (
                abs(float(entry.ef_offset) - 0.052) < 1e-9
                and not entry.ef_correction
                and not entry.fit_result
            )
        ):
            entry.ef_offset = 0.0
        self._params.sp_ef.blockSignals(True)
        self._params.sp_ef.setValue(entry.ef_offset)
        self._params.sp_ef.blockSignals(False)

        hv_before_load = float(self._params.sp_hv.value())
        logbook_hit = self._parent._logbook_ctrl.apply_to_controls(path)
        hv_from_logbook = (
            logbook_hit
            and float(self._params.sp_hv.value()) != hv_before_load
            and float(self._params.sp_hv.value()) > 0
        )
        if entry.meta.hv and entry.meta.hv > 0 and self._params.sp_hv.value() <= 0:
            self._params.set_hv_value_with_source(entry.meta.hv, "file")
        hv_for_load = float(self._params.sp_hv.value())
        angle_offsets = self._parent._angle_offsets_for_load(path, entry, hv_for_load)
        return _PreparedEntry(
            key=key,
            entry=entry,
            is_new=is_new_entry,
            fmt_guess=fmt_guess,
            hv_for_load=hv_for_load,
            hv_from_logbook=hv_from_logbook,
            angle_offsets=angle_offsets,
            logbook_hit=logbook_hit,
        )

    def _dispatch_loader(self, path: str, prepared: _PreparedEntry):
        from arpes import app as _ae
        orchestrator = LoaderOrchestrator(_ae.load_arpes_file, _ae._loader_label)
        load_result = orchestrator.load(
            path,
            prepared.entry,
            work_func=self._params.sp_phi.value(),
            ef_offset=self._params.sp_ef.value(),
            hv=prepared.hv_for_load,
            angle_offsets=prepared.angle_offsets,
            bessy_energy_reference=self._parent._bessy_energy_reference_mode(),
            best_angle_load_func=self._parent._load_with_best_angle_offsets,
        )
        return load_result, orchestrator

    def _apply_post_load(self, d, prepared, load_result, orchestrator, path):
        from arpes import app as _ae
        entry = prepared.entry
        if entry.ef_correction.get("mode") == "poly":
            d, ef_info = _ae.apply_ef_correction_to_dict(d, entry.ef_correction)
            self._parent._ef_correction_info = ef_info
        else:
            self._parent._ef_correction_info = {}
        # mutate dict slot in load_result for downstream consumers
        load_result.data = d
        self._parent._raw_data = d
        self._parent._current_path = path
        self._parent._fit_res = None

        hv_src = orchestrator.resolve_hv_after_load(
            d,
            entry,
            hv_for_load=prepared.hv_for_load,
            hv_from_logbook=prepared.hv_from_logbook,
        )
        if hv_src.source == "file" and hv_src.value is not None:
            self._params.set_hv_value_with_source(float(hv_src.value), "file")
        elif hv_src.value is not None:
            self._params.update_hv_source(hv_src.source)
        else:
            self._params.update_hv_source(None)

    def _restore_session(self, d, entry, md):
        self._params.sp_ef.blockSignals(True)
        self._params.sp_ef.setValue(entry.ef_offset)
        self._params.sp_ef.blockSignals(False)
        self._params.chk_norm.blockSignals(True)
        self._params.chk_norm.setChecked(entry.edcnorm)
        self._params.chk_norm.blockSignals(False)
        self._params.load_fit_params(entry.fit_params)
        saved_res = (entry.fit_result or {}).get("resolution", {}) if entry.fit_result else {}
        if saved_res:
            saved_source = str(saved_res.get("source", "") or "")
            if saved_source == "manual":
                res_kind = "manual"
            elif "defaut" in saved_source or saved_source == "default":
                res_kind = "default"
            else:
                res_kind = "estimated"
            self._params.set_resolution_with_source(
                float(saved_res.get("dE_meV", 15.0) or 15.0),
                float(saved_res.get("dk_inv_a", 0.005) or 0.005),
                res_kind,
                saved_source,
            )
        else:
            res = estimate_resolutions(md)
            res_kind = "default" if "defaut" in res.get("source", "") else "estimated"
            self._params.set_resolution_with_source(
                res["dE_meV"], res["dk_inv_a"], res_kind, res.get("source", "")
            )
        self._parent._cmb_view.blockSignals(True)
        self._parent._cmb_view.setCurrentText(entry.view_mode)
        self._parent._cmb_view.blockSignals(False)
        self._parent._load_grid_controls(entry.grid_correction)

        if entry.fit_result:
            self._parent._fit_res = entry.fit_result

        self._parent._apply_stored_gamma_to_current_file(save_entry=True)

    def _refresh_ui(self, d, prepared, path):
        entry = prepared.entry
        self._parent._update_display_data()
        grid_note = ""
        if entry.grid_correction.get("enabled"):
            grid_msg = self._parent._grid_status_text(self._parent._grid_display_info, "affichage BM")
            grid_note = "  |  " + grid_msg
            self._params.lbl_grid.setText(grid_msg)
        gamma_note = ""
        md_now = d.get("metadata", {}) or {}
        if md_now.get("angle_offsets_applied"):
            ao = md_now.get("angle_offsets_applied") or {}
            cand = md_now.get("angle_offset_candidate", ao.get("candidate", ""))
            cand_txt = f" {cand}" if cand else ""
            gamma_note = (
                f"  |  Γ offset angulaire{cand_txt} θ0={float(ao.get('theta0_deg', 0.0)):+.3f}°"
            )
        elif md_now.get("bm_gamma_axis_centered"):
            gamma_note = f"  |  Γ axe shift={float(md_now.get('bm_gamma_axis_shift', 0.0)):+.4f}"
        loader_note = ""
        loader_warnings = md_now.get("loader_warnings") or []
        if loader_warnings:
            loader_note = f"  |  ⚠ loader: {str(loader_warnings[0])[:120]}"
        elif md_now.get("energy_reference"):
            loader_note = f"  |  refE={md_now.get('energy_reference')}"
        self._parent._sel_ev = float(np.clip(-0.30, d["ev_arr"].min(), d["ev_arr"].max()))
        self._parent._sel_k = 0.0
        self._parent._sync_ev_spinbox()

        self._parent._draw_bm()
        self._parent._draw_mdc_edc()
        if self._parent._tabs.currentIndex() == 3:
            self._parent._draw_fs_tab()

        self._session.save()
        self._parent._browser.select_file(path)
        self._parent._browser.refresh_item(self._session.key_for_path(path))
        hv_txt = f"{d['hv']:.0f} eV" if d.get("hv") is not None else "—"
        lb_txt = "  |  logbook" if prepared.logbook_hit else ""
        self._status(
            f"Chargé : {Path(path).name}  hν={hv_txt}  |  "
            f"k {d['kpar'].min():.2f}→{d['kpar'].max():.2f} π/a  |  "
            f"E {d['ev_arr'].min():.3f}→{d['ev_arr'].max():.3f} eV"
            f"{lb_txt}{grid_note}{gamma_note}{loader_note}"
        )
        self._parent._refresh_helper_buttons()
