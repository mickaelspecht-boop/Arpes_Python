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
import copy
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtWidgets import QApplication

from arpes.core.session import normalize_tags, session_tags
from arpes.io.artifact_cache import (
    load_raw_artifact,
    save_raw_artifact,
    save_raw_artifact_async,
)
from arpes.io.loader_orchestrator import LoaderOrchestrator, LoaderOrchestratorResult
from arpes.physics.resolution import estimate_resolutions

try:
    from arpes.io.loaders import detect_format, load_arpes_file, loader_label
except ImportError:
    detect_format = None
    load_arpes_file = None
    loader_label = lambda *a, **k: ""  # noqa: E731

from arpes.physics.plot_compute import apply_ef_correction_to_dict


RAW_LOAD_CACHE_VERSION = 2


def _freeze_cache_value(value: Any) -> Any:
    """Transforme une valeur de contexte en clé hashable stable."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple((str(k), _freeze_cache_value(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(v) for v in value))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _clone_loaded_value(value: Any) -> Any:
    """Copie metadata/conteneurs, partage les gros tableaux numpy en lecture."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {k: _clone_loaded_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_loaded_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_loaded_value(v) for v in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


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
    def load(self, path: str, *, force_reload: bool = False) -> None:
        """Pipeline complet - entrée unique appelée par le file browser.

        Si ``force_reload`` est vrai, bypass les caches RAM + disque pour ce fichier.
        """
        self._ensure_arpes_plots()
        suffix = " (sans cache)" if force_reload else ""
        self._status(f"Chargement {Path(path).name}{suffix} ...")
        QApplication.processEvents()
        try:
            prepared = self._prepare_entry(path)
            entry_state_before_load = self._entry_state_token(prepared.entry)
            load_result, orchestrator = self._dispatch_loader(
                path, prepared, force_reload=force_reload,
            )
            d = load_result.data
            if d is None:
                self._status("Attention: erlab non disponible")
                return
            md = orchestrator.apply_loaded_metadata(d, prepared.entry)
            self._apply_post_load(d, prepared, load_result, orchestrator, path)
            self._restore_session(d, prepared.entry, md)
            entry_dirty = self._entry_state_token(prepared.entry) != entry_state_before_load
            self._refresh_ui(d, prepared, path, entry_dirty=entry_dirty)
        except Exception as e:
            self._status(f"Attention: {e}")
            traceback.print_exc()

    def _on_file_tags_changed(self) -> None:
        entry = self._parent._current_entry()
        if entry is None:
            return
        tags = normalize_tags(self._params.file_tags_text())
        entry.meta.tags = tags
        self._params.set_file_tags(tags)
        self._params.update_tag_completions(session_tags(self._session))
        self._session.save()
        if hasattr(self._parent, "_browser"):
            self._parent._browser.refresh_tag_completions()
            self._parent._browser._populate()
        tag_txt = ", ".join(tags) if tags else "aucun"
        self._status(f"Tags fichier enregistrés: {tag_txt}.")

    # ------------------------------------------------------------ steps
    def _ensure_arpes_plots(self) -> None:
        if self._parent.ap is None:
            from arpes.app import _load_ap
            self._parent.ap = _load_ap()
            if self._parent.ap is None:
                self._status("Attention: arpes_plots introuvable")

    def _prepare_entry(self, path: str) -> _PreparedEntry:
        key = self._session.key_for_path(path)
        is_new_entry = key not in self._session.files
        entry = self._session.get_or_create(key)
        fmt_guess = ""
        try:
            fmt_guess = detect_format(path) if detect_format is not None else ""
        except Exception:
            fmt_guess = ""
        if fmt_guess in ("bessy_ses_ibw", "cls_txt") and (
            is_new_entry
            or (
                abs(float(entry.ef_offset) - 0.052) < 1e-9
                and not entry.ef_correction
                and not entry.fit_result
            )
        ):
            # 0.052 eV : défaut historique sans sens pour ces formats — l'échelle
            # d'énergie vient déjà du fichier (Center/Central Energy).
            entry.ef_offset = 0.0
        self._params.sp_ef.blockSignals(True)
        self._params.sp_ef.setValue(entry.ef_offset)
        self._params.sp_ef.blockSignals(False)

        logbook_hit = self._parent._logbook_ctrl.apply_to_controls(path)
        logbook_values = getattr(self._parent._logbook_ctrl, "_last_applied_values", None)
        hv_from_logbook = (
            logbook_hit
            and getattr(logbook_values, "hv", None) is not None
            and float(self._params.sp_hv.value()) > 0
        )
        if not logbook_hit and entry.meta.hv and entry.meta.hv > 0:
            same_path = False
            current_path = getattr(self._parent, "_current_path", None)
            if current_path:
                try:
                    same_path = Path(current_path).resolve() == Path(path).resolve()
                except Exception:
                    same_path = str(current_path) == str(path)
            if (not same_path) or self._params.sp_hv.value() <= 0:
                self._params.set_hv_value_with_source(entry.meta.hv, "session")
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

    def _dispatch_loader(self, path: str, prepared: _PreparedEntry, *, force_reload: bool = False):
        orchestrator = LoaderOrchestrator(load_arpes_file, loader_label)
        cache_key = self._load_cache_key(path, prepared)
        cache = getattr(self._parent, "_raw_load_cache", None)
        session_folder = getattr(getattr(self._parent, "_session", None), "folder", None)
        disk_cache_enabled = bool(getattr(self._parent, "_raw_disk_cache_enabled", False))
        if force_reload and cache is not None:
            cache.pop(cache_key, None)
        if not force_reload and cache is not None and cache_key in cache:
            data_cached, offsets_cached = cache.pop(cache_key)
            cache[cache_key] = (data_cached, offsets_cached)
            self._parent._last_load_cache_hit = True
            self._parent._last_load_cache_source = "ram"
            self._parent._current_raw_load_cache_key = cache_key
            return (
                LoaderOrchestratorResult(
                    data=_clone_loaded_value(data_cached),
                    angle_offsets=dict(offsets_cached or {}),
                    context=orchestrator.build_context(
                        prepared.entry,
                        hv=prepared.hv_for_load,
                        angle_offsets=prepared.angle_offsets,
                        bessy_energy_reference=self._parent._bessy_energy_reference_mode(),
                    ),
                ),
                orchestrator,
            )

        disk_cached = (
            load_raw_artifact(path, cache_key, session_folder)
            if (disk_cache_enabled and not force_reload) else None
        )
        if disk_cached is not None:
            data_cached, offsets_cached = disk_cached
            if cache is not None:
                cache[cache_key] = (_clone_loaded_value(data_cached), dict(offsets_cached or {}))
                max_items = int(getattr(self._parent, "_raw_load_cache_max", 6) or 6)
                while len(cache) > max_items:
                    cache.popitem(last=False)
            self._parent._last_load_cache_hit = True
            self._parent._last_load_cache_source = "disque"
            self._parent._current_raw_load_cache_key = cache_key
            return (
                LoaderOrchestratorResult(
                    data=_clone_loaded_value(data_cached),
                    angle_offsets=dict(offsets_cached or {}),
                    context=orchestrator.build_context(
                        prepared.entry,
                        hv=prepared.hv_for_load,
                        angle_offsets=prepared.angle_offsets,
                        bessy_energy_reference=self._parent._bessy_energy_reference_mode(),
                    ),
                ),
                orchestrator,
            )

        self._parent._last_load_cache_hit = False
        self._parent._last_load_cache_source = ""
        self._parent._current_raw_load_cache_key = cache_key
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
        if cache is not None and load_result.data is not None:
            cache[cache_key] = (
                _clone_loaded_value(load_result.data),
                dict(load_result.angle_offsets or {}),
            )
            max_items = int(getattr(self._parent, "_raw_load_cache_max", 6) or 6)
            while len(cache) > max_items:
                cache.popitem(last=False)
            if disk_cache_enabled:
                quota_mb = float(getattr(self._parent, "_raw_disk_cache_quota_mb", 250) or 250)
                use_async = bool(getattr(self._parent, "_raw_disk_cache_async", True))
                if use_async:
                    save_raw_artifact_async(
                        path,
                        cache_key,
                        load_result.data,
                        load_result.angle_offsets or {},
                        session_folder,
                        quota_mb=quota_mb,
                    )
                else:
                    save_raw_artifact(
                        path,
                        cache_key,
                        load_result.data,
                        load_result.angle_offsets or {},
                        session_folder,
                    )
        return load_result, orchestrator

    def _load_cache_key(self, path: str, prepared: _PreparedEntry) -> tuple:
        entry = prepared.entry
        return (
            f"raw-loader-v{RAW_LOAD_CACHE_VERSION}",
            str(prepared.fmt_guess or ""),
            self._path_signature(Path(path)),
            round(float(self._params.sp_phi.value()), 8),
            round(float(self._params.sp_ef.value()), 8),
            round(float(prepared.hv_for_load or 0.0), 8),
            round(float(getattr(entry.meta, "temperature", 0.0) or 0.0), 8),
            round(float(getattr(entry.meta, "azi", 0.0) or 0.0), 8),
            str(getattr(entry.meta, "polarization", "") or ""),
            _freeze_cache_value(prepared.angle_offsets or {}),
            self._parent._bessy_energy_reference_mode(),
        )

    def _path_signature(self, path: Path) -> tuple:
        try:
            p = path.resolve()
        except Exception:
            p = path
        cache = getattr(self._parent, "_path_signature_cache", None)
        cache_key = str(p)
        quick_sig = self._quick_path_signature(p)
        if cache is not None and not p.is_dir():
            cached = cache.get(cache_key)
            if cached is not None and cached[0] == quick_sig:
                cache.move_to_end(cache_key)
                self._parent._last_path_signature_cache_hit = True
                return cached[1]
        self._parent._last_path_signature_cache_hit = False

        if p.is_dir():
            items = []
            try:
                for child in sorted(p.rglob("*")):
                    if not child.is_file():
                        continue
                    rel_parts = child.relative_to(p).parts
                    if rel_parts and rel_parts[0] in {".arpes_cache", ".arpes_theory_cache"}:
                        continue
                    st = child.stat()
                    items.append(("/".join(rel_parts), int(st.st_size), int(st.st_mtime_ns)))
            except Exception:
                try:
                    st = p.stat()
                    items.append((".", int(st.st_size), int(st.st_mtime_ns)))
                except Exception:
                    items.append((".", -1, -1))
            return ("dir", str(p), tuple(items))
        signature = self._file_signature_with_sidecars(p)
        if cache is not None:
            cache[cache_key] = (quick_sig, signature)
            max_items = int(getattr(self._parent, "_path_signature_cache_max", 128) or 128)
            while len(cache) > max_items:
                cache.popitem(last=False)
        return signature

    def _quick_path_signature(self, path: Path) -> tuple:
        if path.is_file():
            return self._file_signature_with_sidecars(path)
        try:
            st = path.stat()
            return (
                "dir" if path.is_dir() else "file",
                str(path),
                int(st.st_size),
                int(st.st_mtime_ns),
            )
        except Exception:
            return ("missing", str(path))

    def _file_signature_with_sidecars(self, path: Path) -> tuple:
        files = [path]
        cls_param = path.parent / f"{path.name}_param.txt"
        if cls_param.exists():
            files.append(cls_param)
        items = []
        for item in files:
            try:
                st = item.stat()
                items.append((item.name, int(st.st_size), int(st.st_mtime_ns)))
            except Exception:
                items.append((item.name, -1, -1))
        return ("file", str(path), tuple(items))

    def _apply_post_load(self, d, prepared, load_result, orchestrator, path):
        entry = prepared.entry
        if entry.ef_correction.get("mode") == "poly":
            d, ef_info = apply_ef_correction_to_dict(d, entry.ef_correction)
            self._parent._ef_correction_info = ef_info
        else:
            self._parent._ef_correction_info = {}
        # mutate dict slot in load_result for downstream consumers
        load_result.data = d
        self._parent._raw_data = d
        self._parent._current_path = path
        self._parent._fit_res = None
        self._parent._data_disp = None
        self._parent._disp_cache_key = None

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
        self._params.load_fit_params(entry.fit_params)
        tags = normalize_tags(getattr(entry.meta, "tags", []))
        entry.meta.tags = tags
        self._params.set_file_tags(tags)
        self._params.update_tag_completions(session_tags(self._session))
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
        if hasattr(self._parent, "_cmb_view_fit"):
            self._parent._cmb_view_fit.blockSignals(True)
            self._parent._cmb_view_fit.setCurrentText(entry.view_mode)
            self._parent._cmb_view_fit.blockSignals(False)
        self._parent._load_grid_controls(entry.grid_correction)

        if entry.fit_result:
            self._parent._fit_res = entry.fit_result

        if hasattr(self._params, "update_fit_quality"):
            try:
                threshold = float(self._params.sp_chi2_threshold.value())
            except Exception:
                threshold = 5.0
            fit_ctrl = getattr(self._parent, "_fit_runner_ctrl", None)
            current_hash = None
            try:
                if fit_ctrl is not None:
                    current_hash = fit_ctrl._current_fit_params_hash(entry)
            except Exception:
                current_hash = None
            self._params.update_fit_quality(
                entry.fit_result, threshold, current_hash=current_hash,
            )
            # G: titre onglet Fit MDC reflète l'état restauré
            try:
                if fit_ctrl is not None:
                    fit_ctrl._update_mdc_tab_label(entry.fit_result)
            except Exception:
                pass

        a_val = float(getattr(entry.meta, "crystal_a_angstrom", 0.0) or 0.0)
        if a_val <= 0.0:
            a_val = 4.143
        self._params.sp_crystal_a.blockSignals(True)
        self._params.sp_crystal_a.setValue(a_val)
        self._params.sp_crystal_a.blockSignals(False)
        self._parent._restore_theory_overlay_for_entry()
        # P1.1/P1.7 : restaure meta_gamma_state AVANT l'apply, et bloque les
        # signaux sp_cx pour éviter redraw partiel mid-load. L'apply voit
        # alors `previous_shift` correct → delta=0 → pas de drift fit_result.
        gamma_ctrl = getattr(self._parent, "_gamma_ctrl", None)
        if gamma_ctrl is not None and hasattr(gamma_ctrl, "_restore_gamma_meta_from_entry"):
            try:
                gamma_ctrl._restore_gamma_meta_from_entry()
            except Exception:
                pass
        sp_cx = getattr(self._params, "sp_cx", None)
        had_block = False
        if sp_cx is not None and hasattr(sp_cx, "blockSignals"):
            try:
                had_block = sp_cx.blockSignals(True)
            except Exception:
                had_block = False
        try:
            self._parent._apply_stored_gamma_to_current_file(save_entry=True)
        finally:
            if sp_cx is not None and hasattr(sp_cx, "blockSignals"):
                try:
                    sp_cx.blockSignals(had_block)
                except Exception:
                    pass
        # Restore BZ-crystal overlay settings from session entry into FSControlPanel
        try:
            self._parent._restore_fs_crystal_settings_from_entry(entry)
        except Exception:
            pass
        # Re-populate combo selectors du tab Compare pol (nouveau fichier dans session.files)
        try:
            self._parent._refresh_fs_compare_selectors()
        except Exception:
            pass
        try:
            self._parent._check_distortion_consistency_on_load()
            self._parent._apply_calib_for_current_if_any()
        except Exception:
            pass
        try:
            self._parent._refresh_band_analysis_panel()
        except Exception:
            pass
        try:
            self._parent._refresh_zones_strip()
        except Exception:
            pass
        # A.4 — auto-pin FS contexte si on charge une BM (overlay Phase B).
        try:
            self._parent._pairing_action("auto_pin_bm")
        except Exception:
            pass

    def _refresh_ui(self, d, prepared, path, *, entry_dirty: bool = False):
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
        cache_source = getattr(self._parent, "_last_load_cache_source", "")
        cache_note = f"  |  cache {cache_source}" if getattr(self._parent, "_last_load_cache_hit", False) else ""
        if loader_warnings:
            loader_note = f"  |  Attention: loader: {str(loader_warnings[0])[:120]}"
        elif md_now.get("energy_reference"):
            loader_note = f"  |  refE={md_now.get('energy_reference')}"
        self._parent._sel_ev = float(np.clip(-0.30, d["ev_arr"].min(), d["ev_arr"].max()))
        self._parent._sel_k = 0.0
        self._parent._sync_ev_spinbox()

        self._parent._draw_current_view()

        if entry_dirty or prepared.is_new or not getattr(self._parent, "_last_load_cache_hit", False):
            self._session.save()
        self._parent._browser.select_file(path)
        self._parent._browser.refresh_item(self._session.key_for_path(path))
        if getattr(self._parent, "_tabs", None) is not None and self._parent._tabs.currentIndex() == 2:
            results = getattr(self._parent, "_results", None)
            if results is not None and hasattr(results, "refresh"):
                results.refresh()
        hv_txt = f"{d['hv']:.0f} eV" if d.get("hv") is not None else "—"
        lb_txt = "  |  logbook" if prepared.logbook_hit else ""
        self._status(
            f"Chargé : {Path(path).name}  hν={hv_txt}  |  "
            f"k {d['kpar'].min():.2f}→{d['kpar'].max():.2f} π/a  |  "
            f"E {d['ev_arr'].min():.3f}→{d['ev_arr'].max():.3f} eV"
            f"{lb_txt}{cache_note}{grid_note}{gamma_note}{loader_note}"
        )
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"fichier chargé ({Path(path).name})")
        self._parent._refresh_helper_buttons()
        self._parent._auto_fetch_theory_overlay_from_logbook()

    @staticmethod
    def _entry_state_token(entry) -> str:
        try:
            payload = asdict(entry)
        except Exception:
            payload = getattr(entry, "__dict__", {})
        try:
            return json.dumps(_freeze_cache_value(payload), sort_keys=True)
        except Exception:
            return repr(payload)
