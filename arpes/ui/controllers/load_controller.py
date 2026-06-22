"""UI controller for the ARPES file-loading pipeline.

Moves `arpes_explorer.ArpesExplorer._load_file` (~169 LOC) out of the God class
and splits it into five readable steps:

1. `_prepare_entry`   : create/restore the session entry, apply default EF
2. `_dispatch_loader` : call `LoaderOrchestrator` and retrieve `load_result`
3. `_apply_post_load` : per-column EF correction + parent `_raw_data` storage
4. `_restore_session` : restore UI params from the session entry
5. `_refresh_ui`      : redraws BM/MDC-EDC/FS + statusbar

Business logic (LoaderOrchestrator, apply_ef_correction_to_dict, etc.) stays in
`arpes.io.*` or in `arpes_explorer` (to be moved later). This controller is
pure Qt orchestration plus parent state.
"""
from __future__ import annotations

import traceback
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from arpes.core import processing_history as ph
from arpes.core.sample import work_function_for_entry
from arpes.core.session import DEFAULT_EF_OFFSET_EV, normalize_tags, session_tags
from arpes.io.artifact_cache import (
    load_raw_artifact,
    save_raw_artifact,
    save_raw_artifact_async,
)
from arpes.io.loader_orchestrator import LoaderOrchestrator, LoaderOrchestratorResult
from arpes.physics.resolution import estimate_resolutions
from arpes.ui.controllers.load_cache_helpers import (
    RAW_LOAD_CACHE_VERSION,
    clone_loaded_value,
    entry_state_token,
    freeze_cache_value,
    path_signature,
)

try:
    from arpes.io.loaders import detect_format, load_arpes_file, loader_label
except ImportError:
    detect_format = None
    load_arpes_file = None
    loader_label = lambda *a, **k: ""  # noqa: E731

from arpes.physics.plot_compute import apply_ef_correction_to_dict


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
    hv_placeholder: bool = False


class LoadController:
    """Pipeline for loading an ARPES file (band map or FS)."""

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

    def _path_signature(self, path: Path) -> tuple:
        """Backward-compatible wrapper around the shared cache helper."""
        return path_signature(path, self._parent)

    # ------------------------------------------------------------ public
    def load(self, path: str, *, force_reload: bool = False) -> None:
        """Complete pipeline, the single entry point called by the file browser.

        If ``force_reload`` is true, bypass RAM and disk caches for this file.
        """
        self._ensure_arpes_plots()
        suffix = " (without cache)" if force_reload else ""
        self._status(f"Loading {Path(path).name}{suffix} ...")
        QApplication.processEvents()
        try:
            prepared = self._prepare_entry(path)
            entry_state_before_load = self._entry_state_token(prepared.entry)
            load_result, orchestrator = self._dispatch_loader(
                path, prepared, force_reload=force_reload,
            )
            d = load_result.data
            if d is None:
                self._status("Warning: erlab unavailable")
                return
            md = orchestrator.apply_loaded_metadata(d, prepared.entry)
            self._apply_post_load(d, prepared, load_result, orchestrator, path)
            self._restore_session(d, prepared.entry, md, prepared.key)
            entry_dirty = self._entry_state_token(prepared.entry) != entry_state_before_load
            self._refresh_ui(d, prepared, path, entry_dirty=entry_dirty)
            # Record the load once per signal (first load + explicit no-cache
            # reloads); plain file switches must not clutter the journal.
            entry = prepared.entry
            if force_reload or not getattr(entry, "processing_history", None):
                meta = getattr(entry, "meta", None)
                loader = (
                    getattr(meta, "loader_label", "")
                    or getattr(meta, "source_format", "")
                    or "loader"
                )
                hv = float(getattr(meta, "hv", 0.0) or 0.0)
                ph.log_action(self._parent,
                    ph.CAT_LOAD,
                    "reloaded (no cache)" if force_reload else "loaded",
                    entry=entry,
                    summary=f"{loader}" + (f", hν={hv:.0f} eV" if hv else ""),
                )
        except Exception as e:
            self._status(f"Warning: {e}")
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

    def _on_work_function_changed(self) -> None:
        path = getattr(self._parent, "_current_path", None)
        if not path:
            return
        try:
            phi = float(self._params.sp_phi.value())
        except Exception:
            return
        entry = self._session.get_or_create(self._session.key_for_path(path))
        entry.meta.work_function_eV = float(phi)
        try:
            self._session.save()
        except Exception:
            pass
        self._status(f"φ = {phi:.3f} eV saved.")
        if phi > 0 and getattr(self._parent, "_raw_data", None) is not None:
            self._status("φ changed: recomputing energy/k without cache.")
            self.load(path, force_reload=True)
        if hasattr(self._parent, "_browser"):
            self._parent._browser.refresh_tag_completions()
            self._parent._browser._populate()

    # ------------------------------------------------------------ steps
    def _ensure_arpes_plots(self) -> None:
        if self._parent.ap is None:
            from arpes.app import _load_ap
            self._parent.ap = _load_ap()
            if self._parent.ap is None:
                self._status("Warning: arpes_plots not found")

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
                abs(float(entry.ef_offset) - DEFAULT_EF_OFFSET_EV) < 1e-9
                and not entry.ef_correction
                and not entry.fit_result
            )
        ):
            # Historical default has no meaning for these formats because the
            # energy scale already comes from the file (Center/Central Energy).
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
        hv_for_loader = float(prepared.hv_for_load or 0.0)
        if (
            getattr(self._session, "browse_only", False)
            and hv_for_loader <= 0
            and str(prepared.fmt_guess or "") == "cls_txt"
        ):
            # CLS refuses hv=None. Browse-only loads it with a placeholder;
            # the flag lets _maybe_apply_raw_axes strip the fake hv and the
            # consistency warnings it triggers before anything is shown/saved.
            hv_for_loader = 21.2
            prepared.hv_placeholder = True
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
                    data=clone_loaded_value(data_cached),
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
                cache[cache_key] = (clone_loaded_value(data_cached), dict(offsets_cached or {}))
                max_items = int(getattr(self._parent, "_raw_load_cache_max", 6) or 6)
                while len(cache) > max_items:
                    cache.popitem(last=False)
            self._parent._last_load_cache_hit = True
            self._parent._last_load_cache_source = "disk"
            self._parent._current_raw_load_cache_key = cache_key
            return (
                LoaderOrchestratorResult(
                    data=clone_loaded_value(data_cached),
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
        a_lattice = self._lattice_a_for_load(prepared.entry, prepared.key)
        work_func = self._work_function_for_load(path, prepared)
        if getattr(self._session, "browse_only", False):
            # Browse-only: load with neutral placeholders when φ/a are
            # missing. The k axis they produce is NEVER displayed — the view
            # swaps to raw θ/E axes (see _maybe_apply_raw_axes) and every
            # physics feature is guarded against axes_raw_view.
            if work_func <= 0:
                work_func = 4.5
            if a_lattice is None:
                a_lattice = 1.0
        if work_func <= 0:
            raise ValueError(
                "Work function φ missing. "
                "Enter φ in the UI or SampleConfig.work_function_eV before loading."
            )
        if a_lattice is None and str(prepared.fmt_guess or "") in {"cls_txt", "bessy_ses_ibw", "solaris_da30"}:
            raise ValueError(
                "Lattice parameter a missing. "
                "Enter crystal a (Å) in Lattice and units before loading; "
                "otherwise the kx/ky axes in π/a collapse to 0 and the map looks empty."
            )
        load_result = orchestrator.load(
            path,
            prepared.entry,
            work_func=work_func,
            ef_offset=self._params.sp_ef.value(),
            a_lattice=a_lattice,
            hv=hv_for_loader,
            angle_offsets=prepared.angle_offsets,
            bessy_energy_reference=self._parent._bessy_energy_reference_mode(),
            best_angle_load_func=self._parent._load_with_best_angle_offsets,
        )
        if cache is not None and load_result.data is not None:
            cache[cache_key] = (
                clone_loaded_value(load_result.data),
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
        work_func = self._work_function_for_load(path, prepared)
        a_lattice = self._lattice_a_for_load(entry, prepared.key)
        return (
            f"raw-loader-v{RAW_LOAD_CACHE_VERSION}",
            str(prepared.fmt_guess or ""),
            path_signature(Path(path), self._parent),
            round(float(work_func), 8),
            round(float(self._params.sp_ef.value()), 8),
            round(float(a_lattice or 0.0), 8),
            round(float(prepared.hv_for_load or 0.0), 8),
            round(float(getattr(entry.meta, "temperature", 0.0) or 0.0), 8),
            round(float(getattr(entry.meta, "azi", 0.0) or 0.0), 8),
            str(getattr(entry.meta, "polarization", "") or ""),
            freeze_cache_value(prepared.angle_offsets or {}),
            self._parent._bessy_energy_reference_mode(),
            bool(getattr(self._session, "browse_only", False)),
        )

    def _lattice_a_for_load(self, entry, entry_key: str | None = None) -> float | None:
        from arpes.ui.controllers.load_lattice_sync import lattice_a_for_load

        return lattice_a_for_load(self, entry, entry_key=entry_key)

    def _work_function_for_load(self, path: str, prepared: _PreparedEntry) -> float:
        entry = prepared.entry
        work_func = work_function_for_entry(
            self._session,
            entry,
            fallback=self._params.sp_phi.value(),
        )
        if work_func > 0:
            return float(work_func)
        inferred = self._infer_cls_work_function(path, prepared)
        if inferred is None:
            return float(work_func)
        entry.meta.work_function_eV = float(inferred)
        self._params.sp_phi.blockSignals(True)
        self._params.sp_phi.setValue(float(inferred))
        self._params.sp_phi.blockSignals(False)
        try:
            self._session.save()
        except Exception:
            pass
        self._status(f"CLS-inferred φ = {float(inferred):.3f} eV (hν - Central Energy).")
        return float(inferred)

    def _infer_cls_work_function(self, path: str, prepared: _PreparedEntry) -> float | None:
        fmt = str(prepared.fmt_guess or "")
        if fmt and fmt != "cls_txt":
            return None
        hv = float(prepared.hv_for_load or 0.0)
        if hv <= 0:
            return None
        p = Path(path)
        param_files: list[Path]
        if p.is_file():
            param_files = [p.parent / f"{p.name}_param.txt"]
        elif p.is_dir():
            param_files = sorted(p.glob("*_param.txt"))
        else:
            param_files = []
        for param_file in param_files:
            try:
                text = param_file.read_text(errors="ignore")
            except OSError:
                continue
            match = re.search(r"Central Energy:\s*([-\d.]+)", text)
            if not match:
                continue
            try:
                central = float(match.group(1))
            except ValueError:
                continue
            phi = hv - central
            if 0.0 < phi <= 7.0:
                return float(phi)
        return None

    def _maybe_apply_raw_axes(self, d, prepared) -> None:
        """Browse-only raw view: swap k/E−EF axes for instrument θ/E axes.

        Active only when session.browse_only is set AND at least one of
        φ/a/hν is unknown. The swapped arrays come from loader metadata
        (theta_par_deg / energy_raw) so no physics is recomputed — the
        meta flag ``axes_raw_view`` tells every physics feature to refuse.
        """
        meta = d.get("metadata", {}) or {}
        meta.pop("axes_raw_view", None)
        if not getattr(self._session, "browse_only", False):
            return
        phi_known = self._work_function_for_load("", prepared) > 0
        a_known = self._lattice_a_for_load(prepared.entry, prepared.key) is not None
        hv_known = float(prepared.hv_for_load or 0.0) > 0
        if phi_known and a_known and hv_known:
            return  # everything calibrated: normal axes even in browse-only
        theta = meta.get("theta_par_deg")
        e_raw = meta.get("energy_raw")
        data = d.get("data")
        if theta is None or e_raw is None or data is None:
            raise ValueError(
                "Browse-only raw axes unavailable for this loader (no θ/E "
                "axes stored in the file metadata). Set φ, a and hν to load "
                "with calibrated axes."
            )
        theta = np.asarray(theta, dtype=float)
        e_raw = np.asarray(e_raw, dtype=float)
        if data.shape[0] != len(theta) or data.shape[-1] != len(e_raw):
            raise ValueError(
                f"Browse-only raw axes shape mismatch: data {data.shape} vs "
                f"θ[{len(theta)}] × E[{len(e_raw)}] — loader metadata bug."
            )
        d["kpar"] = theta
        d["ev_arr"] = e_raw
        scale = str(meta.get("energy_axis_original", "kinetic") or "kinetic")
        meta["axes_raw_view"] = True
        meta["axes_raw_xlabel"] = "θ (°) [raw, instrument frame]"
        meta["axes_raw_ylabel"] = (
            "E binding (eV) [raw]" if scale.startswith("bind")
            else "E kinetic (eV) [raw]"
        )
        if prepared.hv_placeholder:
            # Strip the CLS placeholder hν: never display, never persist
            # (resolve_hv_after_load runs after this and reads d["hv"]),
            # and drop the consistency warnings the fake value triggered.
            d["hv"] = None
            meta["hv_warning"] = None
            meta["loader_warnings"] = [
                w for w in (meta.get("loader_warnings") or [])
                if "hν" not in str(w)
            ]
        d["metadata"] = meta

    def _apply_post_load(self, d, prepared, load_result, orchestrator, path):
        self._maybe_apply_raw_axes(d, prepared)
        entry = prepared.entry
        if entry.ef_correction.get("mode") == "poly":
            d, ef_info = apply_ef_correction_to_dict(d, entry.ef_correction)
            self._parent._ef_correction_info = ef_info
        else:
            self._parent._ef_correction_info = {}
        # mutate dict slot in load_result for downstream consumers
        load_result.data = d
        self._parent._raw_data = d
        # Offset currently baked into ev_arr — baseline for the live EF shift.
        self._parent._ef_offset_applied = float(getattr(entry, "ef_offset", 0.0) or 0.0)
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

    def _restore_session(self, d, entry, md, entry_key=None):
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

        from arpes.ui.controllers.load_lattice_sync import sync_lattice_widgets_for_entry
        sync_lattice_widgets_for_entry(self, entry, entry_key)
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
        # FS Explorer: stop any sweep animation on the old volume and refresh
        # the tab if it is the one currently visible.
        try:
            self._parent._fs_explorer_action("file_changed")
        except Exception:
            pass

    def _refresh_ui(self, d, prepared, path, *, entry_dirty: bool = False):
        entry = prepared.entry
        self._parent._update_display_data()
        grid_note = ""
        if entry.grid_correction.get("enabled"):
            grid_msg = self._parent._grid_status_text(self._parent._grid_display_info, "BM display")
            grid_note = "  |  " + grid_msg
            self._params.lbl_grid.setText(grid_msg)
        gamma_note = ""
        md_now = d.get("metadata", {}) or {}
        if md_now.get("angle_offsets_applied"):
            ao = md_now.get("angle_offsets_applied") or {}
            cand = md_now.get("angle_offset_candidate", ao.get("candidate", ""))
            cand_txt = f" {cand}" if cand else ""
            gamma_note = (
                f"  |  Γ angular offset{cand_txt} θ0={float(ao.get('theta0_deg', 0.0)):+.3f}°"
            )
        elif md_now.get("bm_gamma_axis_centered"):
            gamma_note = f"  |  Γ axis shift={float(md_now.get('bm_gamma_axis_shift', 0.0)):+.4f}"
        loader_note = ""
        loader_warnings = md_now.get("loader_warnings") or []
        cache_source = getattr(self._parent, "_last_load_cache_source", "")
        cache_note = f"  |  cache {cache_source}" if getattr(self._parent, "_last_load_cache_hit", False) else ""
        if loader_warnings:
            loader_note = f"  |  Warning: loader: {str(loader_warnings[0])[:120]}"
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
        from arpes.ui.tab_index import IDX_RESULTS
        if getattr(self._parent, "_tabs", None) is not None and self._parent._tabs.currentIndex() == IDX_RESULTS:
            results = getattr(self._parent, "_results", None)
            if results is not None and hasattr(results, "refresh"):
                results.refresh()
        hv_txt = f"{d['hv']:.0f} eV" if d.get("hv") is not None else "—"
        lb_txt = "  |  logbook" if prepared.logbook_hit else ""
        if md_now.get("axes_raw_view"):
            axes_txt = (
                f"θ {d['kpar'].min():.2f}→{d['kpar'].max():.2f} °  |  "
                f"E {d['ev_arr'].min():.3f}→{d['ev_arr'].max():.3f} eV (raw)"
                "  |  ⚠ browse-only raw axes"
            )
        else:
            axes_txt = (
                f"k {d['kpar'].min():.2f}→{d['kpar'].max():.2f} π/a  |  "
                f"E {d['ev_arr'].min():.3f}→{d['ev_arr'].max():.3f} eV"
            )
        self._status(
            f"Loaded: {Path(path).name}  hν={hv_txt}  |  {axes_txt}"
            f"{lb_txt}{cache_note}{grid_note}{gamma_note}{loader_note}"
        )
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"file loaded ({Path(path).name})")
        self._parent._refresh_helper_buttons()
        self._parent._auto_fetch_theory_overlay_from_logbook()

    @staticmethod
    def _entry_state_token(entry) -> str:
        return entry_state_token(entry)
